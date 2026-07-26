"""DP-MLM differentially private text-rewriting defense (Meisenbacher et al., ACL Findings 2024).

DP-MLM (`github.com/sjmeis/DPMLM`) rewrites text one *word* at a time using an encoder-only
masked language model (RoBERTa) and the **exponential mechanism**, giving a per-word ε-DP
guarantee. For each content word we mask it, feed RoBERTa the ``(original_window, masked_window)``
sentence pair (the paper's "contextualization" trick), read the vocab logits at the mask slot,
**clip** them to a calibrated ``[clip_min, clip_max]`` (sensitivity ``Δu = |clip_max − clip_min|``),
scale by ``1/(2·Δu/ε)``, softmax, and **sample** a replacement token. This is a faithful port of
the reference ``src/dpmlm/core.py`` (``privatize_batch`` / ``dpmlm_rewrite_batch``); the math and
defaults match. Unlike the other defenses here (heuristic/LM rewriters), DP-MLM comes with a formal
DP guarantee -- it is the reason to add it.

Privacy accounting is per word: a text with ``n`` privatized words spends ``n·ε`` by sequential
composition (reported, not enforced -- as in the paper). We use the authors' published roberta-base
clip bounds.

Two deliberate deviations from the reference, both required by this framework:

* **Determinism for cache correctness.** :class:`~prompt_anonymity.defenses.base.CachedDefense`
  caches rewrites keyed on :meth:`params`, so genuine ``np.random.choice`` sampling would make one
  input yield different outputs across runs and corrupt the cache. We seed NumPy *per turn* from the
  turn text + a configurable ``seed`` (in ``params()``) before privatizing. The exponential-mechanism
  *distribution* is unchanged -- still true DP sampling -- only reproducibility is added. Consequence
  to keep in mind: an identical turn always maps to the same output, which is fine for offline
  evaluation but is not fresh randomness per release.
* **Scope.** We port the paper's headline mode (rewrite every content word) plus the optional Presidio
  PII toggle (``pii=True`` == the reference ``PII=True, hybrid=False``: detected entities become kept
  ``<ENTITY_TYPE>`` placeholders, everything else is still DP-rewritten). The reference's IPI/NER and
  ``hybrid_budget`` modes and ``dpmlm_rewrite_plus`` (add/delete tokens) are omitted.

Needs the ``[dpmlm]`` extra (adds ``nltk`` on top of torch/transformers); the ``pii=True`` variant
also needs the ``[dpmlm-pii]`` extra (``presidio-analyzer`` + spaCy). NLTK data is fetched lazily on
first backend build. Like every defense here, this only rewrites text; features are recomputed by the
featurize stage.
"""

from __future__ import annotations

import hashlib
import os
import string

from ._backends import PerTurnBatchRewriteDefense, gpu_dtype

#: Encoder MLM used for privatization (the paper's default).
DPMLM_MODEL = "FacebookAI/roberta-base"

#: Published roberta-base logit clip bounds from the reference implementation. Sensitivity is
#: ``|clip_max − clip_min|``; these bound the exponential mechanism's utility range.
DPMLM_CLIP_MIN = -3.2093127
DPMLM_CLIP_MAX = 16.304797887802124

#: Per-word privacy budget (a text with n privatized words spends n·ε). The README's worked
#: example uses ε=25; sweep it by registering / instantiating with different values.
DPMLM_EPSILON = float(os.environ.get("DPMLM_EPSILON", "25"))

#: Seed folded into the per-turn RNG so identical turns rewrite reproducibly (cache-safe).
DPMLM_SEED = int(os.environ.get("DPMLM_SEED", "0"))

#: MLM forward-pass batch size over masked positions (output-neutral; not in params()).
DPMLM_BATCH_SIZE = int(os.environ.get("DPMLM_BATCH_SIZE", "16"))


class _DPMLMBackend:
    """Faithful port of the reference ``DPMLM`` class, trimmed to core + optional PII.

    Loads ``AutoModelForMaskedLM`` once and privatizes words through the exponential mechanism in
    GPU batches. Heavy imports (torch/transformers/nltk/presidio) live in ``__init__`` so importing
    this module never requires them.
    """

    def __init__(self, *, model, clip_min, clip_max, epsilon, seed, concat, stop, pii, batch_size):
        import nltk
        import torch
        from nltk.corpus import stopwords
        from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        # NLTK data needed for word tokenization + stopwords (mirrors the reference setup_resources).
        for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
            nltk.download(pkg, quiet=True)

        self._torch = torch
        self._nltk = nltk
        self.detokenizer = TreebankWordDetokenizer()
        self.word_tokenizer = TreebankWordTokenizer()
        self.stop = set(stopwords.words("english"))

        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self.concat = bool(concat)
        self.stop_flag = bool(stop)  # STOP=True means "also privatize stopwords".
        self.batch_size = int(batch_size)

        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.sensitivity = abs(self.clip_max - self.clip_min)
        if self.sensitivity <= 0:
            raise ValueError("DP-MLM sensitivity must be > 0; check clip_min/clip_max.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # Sliding-window budget: MLM context is halved by the CONCAT sentence pair, minus special
        # tokens (matches the reference's ``(model_max_length // 2) - 32``).
        self.max_context = (self.tokenizer.model_max_length // 2) - 32

        self.lm_model = AutoModelForMaskedLM.from_pretrained(
            model, torch_dtype=gpu_dtype(torch)
        ).to(self.device)
        self.lm_model.eval()
        torch.set_grad_enabled(False)

        self.analyzer = None
        if pii:
            from presidio_analyzer import AnalyzerEngine

            self.analyzer = AnalyzerEngine()

    # -- context windowing (verbatim from reference) ----------------------------------------------

    def sliding_window(self, tokens, target_idx, max_len):
        length = len(tokens)
        if length <= max_len:
            return 0, length

        half_window = max_len // 2
        lower = target_idx - half_window
        upper = target_idx + half_window

        if lower < 0:
            upper -= lower
            lower = 0
        elif upper > length:
            lower -= (upper - length)
            upper = length

        lower = max(0, lower)
        upper = min(length, upper)
        return int(lower), int(upper)

    def _encode_masked(self, clean_sent, masked_sent):
        """Build the MLM input ids for one masked position, never raising and keeping the ``<mask>``.

        The reference uses ``truncation="only_first"`` on the ``(clean, masked)`` pair so the masked
        (second) segment is protected -- but that raises "Sequence to truncate too short" when the
        masked segment alone exceeds ``model_max_length`` (real on WildChat, where one NLTK "word"
        can be a long URL / raw ``<svg ...>`` blob that explodes into hundreds of subwords). We keep
        the reference pair path for the normal case and, on overflow, fall back to a single sequence
        truncated to a window *centered on the mask* so the mask always survives.
        """
        tok = self.tokenizer
        max_len = tok.model_max_length

        if self.concat:
            try:
                return tok.encode(
                    " " + clean_sent, " " + masked_sent, add_special_tokens=True,
                    truncation="only_first", max_length=max_len,
                )
            except Exception:
                pass  # masked segment alone too long for the pair -> mask-centered fallback below.

        core = tok.encode(" " + masked_sent, add_special_tokens=False)
        budget = max_len - tok.num_special_tokens_to_add(pair=False)  # room for the model's specials.
        if len(core) > budget:
            try:
                mpos = core.index(tok.mask_token_id)
            except ValueError:
                core = core[:budget]  # no mask (shouldn't happen) -> plain head truncation.
            else:
                hi = min(len(core), mpos + budget // 2)
                lo = max(0, hi - budget)
                core = core[lo:hi]
        return tok.build_inputs_with_special_tokens(core)  # adds bos/eos (RoBERTa) or cls/sep (BERT).

    # -- exponential-mechanism sampling over masked positions (verbatim math) ---------------------

    def _privatize_batch(self, tokens, indices):
        """Return ``{"{word}_{idx}": replacement}`` for each word index, sampled via the
        exponential mechanism over the MLM logits at each masked position."""
        import numpy as np

        torch = self._torch
        predictions: dict[str, str] = {}

        for k in range(0, len(indices), self.batch_size):
            batch_indices = indices[k:k + self.batch_size]
            batch_input_ids = []
            batch_mask_positions = []

            for idx in batch_indices:
                lower_w, upper_w = self.sliding_window(tokens, idx, self.max_context)
                chunk_tokens = tokens[lower_w:upper_w]
                rel_idx = idx - lower_w

                masked_chunk = list(chunk_tokens)
                masked_chunk[rel_idx] = self.tokenizer.mask_token

                clean_sent = self.detokenizer.detokenize(chunk_tokens)
                masked_sent = self.detokenizer.detokenize(masked_chunk)

                input_ids = self._encode_masked(clean_sent, masked_sent)

                try:
                    m_pos = input_ids.index(self.tokenizer.mask_token_id)
                except ValueError:
                    m_pos = 0  # fallback; word is left unchanged below.
                batch_input_ids.append(input_ids)
                batch_mask_positions.append(m_pos)

            inputs = self.tokenizer.pad(
                {"input_ids": batch_input_ids}, padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.lm_model(**inputs).logits  # [batch, seq, vocab]

            for i, idx in enumerate(batch_indices):
                target_word = tokens[idx]
                m_pos = batch_mask_positions[i]
                if m_pos == 0:  # mask token was truncated away -> keep original.
                    predictions[f"{target_word}_{idx}"] = target_word
                    continue

                mask_logits = logits[i, m_pos].float().cpu().numpy()
                mask_logits = np.clip(mask_logits, self.clip_min, self.clip_max)
                # Exponential mechanism: Pr[v] ∝ exp(ε·u(v) / (2·Δu)).
                mask_logits = mask_logits / (2 * self.sensitivity / self.epsilon)
                shifted = mask_logits - np.max(mask_logits)  # softmax stability
                scores = np.exp(shifted)
                scores /= scores.sum()
                chosen_idx = np.random.choice(len(scores), p=scores)
                predictions[f"{target_word}_{idx}"] = self.tokenizer.decode(chosen_idx).strip()

        return predictions

    # -- PII detection (Presidio) -----------------------------------------------------------------

    def _pii_scrub(self, sentence):
        """Replace detected PII spans with ``<ENTITY_TYPE>`` placeholders (matches the reference
        ``PII=True`` path). Returns ``(sentence, placeholder_char_ranges)``."""
        sentence = sentence.replace("<", "").replace(">", "")
        results = self.analyzer.analyze(text=sentence, language="en")
        placeholder_ranges = []
        for x in sorted(results, key=lambda r: r.start, reverse=True):
            rep = "<" + x.entity_type + ">"
            sentence = sentence[:x.start] + rep + sentence[x.end:]
            placeholder_ranges.append((x.start, x.start + len(rep)))
        return sentence, placeholder_ranges

    # -- per-turn rewriting -----------------------------------------------------------------------

    def _rewrite_one(self, text):
        import numpy as np

        sentence = " ".join(str(text).split("\n"))

        pii_ranges = []
        if self.analyzer is not None:
            sentence, pii_ranges = self._pii_scrub(sentence)

        tokens = self._nltk.word_tokenize(sentence)
        if not tokens:
            return sentence

        # Tokens overlapping a PII placeholder are kept verbatim (not DP-rewritten).
        pii_mask = [False] * len(tokens)
        if pii_ranges:
            spans = list(self.word_tokenizer.span_tokenize(sentence))
            for i, (t_start, t_end) in enumerate(spans):
                if i >= len(pii_mask):  # word_tokenize vs span_tokenize can disagree on token count.
                    break
                if any(max(t_start, p_start) < min(t_end, p_end) for p_start, p_end in pii_ranges):
                    pii_mask[i] = True

        indices = []
        for i, tok in enumerate(tokens):
            if pii_mask[i]:
                continue
            if tok in string.punctuation:
                continue
            if not self.stop_flag and tok.lower() in self.stop:
                continue
            indices.append(i)

        # Reproducible per-turn randomness: identical turn -> identical rewrite (cache-safe), while
        # still sampling from the exponential-mechanism distribution. A *stable* hash is required --
        # Python's builtin hash() is per-process randomized (PYTHONHASHSEED) and would desync the
        # cache across runs/machines, so derive the seed from SHA-256 of the turn text + self.seed.
        digest = hashlib.sha256(f"{self.seed}:{sentence}".encode("utf-8")).digest()
        np.random.seed(int.from_bytes(digest[:4], "big"))
        res = self._privatize_batch(tokens, indices)

        out = []
        for i, tok in enumerate(tokens):
            r = res.get(f"{tok}_{i}")
            if r is None:  # skipped (stopword/punct/PII) -> keep as-is.
                out.append(tok)
                continue
            # Restore the original word's capitalization.
            out.append(r.capitalize() if tok[:1].isupper() else r.lower())
        return self.detokenizer.detokenize(out)

    def rewrite_batch(self, texts):
        return [self._rewrite_one(t) for t in texts]


class DPMLMDefense(PerTurnBatchRewriteDefense):
    """Differentially private per-word rewriting with RoBERTa + the exponential mechanism.

    The backend is built lazily, so a fully-cached run loads no model. Set ``pii=True`` to also
    scrub Presidio-detected entities to ``<ENTITY_TYPE>`` placeholders before rewriting the rest.
    """

    name = "dp_mlm"
    version = "1"

    def __init__(self, *, model: str = DPMLM_MODEL, epsilon: float = DPMLM_EPSILON,
                 clip_min: float = DPMLM_CLIP_MIN, clip_max: float = DPMLM_CLIP_MAX,
                 seed: int = DPMLM_SEED, concat: bool = True, stop: bool = False,
                 pii: bool = False, batch_size: int = DPMLM_BATCH_SIZE):
        self.model = model
        self.epsilon = float(epsilon)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.seed = int(seed)
        self.concat = bool(concat)
        self.stop = bool(stop)
        self.pii = bool(pii)
        self.batch_size = int(batch_size)
        self._backend = None

    def params(self) -> dict:
        # Every knob that changes the (seeded) output goes in the cache key; batch size does not.
        return {
            "model": self.model,
            "epsilon": self.epsilon,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "sensitivity": abs(self.clip_max - self.clip_min),
            "seed": self.seed,
            "concat": self.concat,
            "stop": self.stop,
            "pii": self.pii,
        }

    def _get_backend(self) -> _DPMLMBackend:
        if self._backend is None:
            self._backend = _DPMLMBackend(
                model=self.model, clip_min=self.clip_min, clip_max=self.clip_max,
                epsilon=self.epsilon, seed=self.seed, concat=self.concat, stop=self.stop,
                pii=self.pii, batch_size=self.batch_size,
            )
        return self._backend

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        return self._get_backend().rewrite_batch(texts)
