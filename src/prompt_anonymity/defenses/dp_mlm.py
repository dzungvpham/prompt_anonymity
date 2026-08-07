"""DP-MLM differentially private text-rewriting defense (Meisenbacher et al., ACL Findings 2024).

DP-MLM (`github.com/sjmeis/DPMLM`) rewrites text one *word* at a time using an encoder-only
masked language model (RoBERTa) and the **exponential mechanism**, giving a per-word ε-DP
guarantee. For each content word we mask it, feed RoBERTa the ``(original_window, masked_window)``
sentence pair (the paper's "contextualization" trick), read the vocab logits at the mask slot,
**clip** them to a calibrated ``[clip_min, clip_max]`` (sensitivity ``Δu = |clip_max − clip_min|``),
scale by ``1/(2·Δu/ε)``, softmax, and **sample** a replacement token. This is a faithful port of
the reference ``src/dpmlm/core.py`` (``privatize_batch`` / ``dpmlm_rewrite_batch``); the math and
the clip bounds match. Unlike the other defenses here (heuristic/LM rewriters), DP-MLM comes with a
formal DP guarantee -- it is the reason to add it.

Privacy accounting is per word: a text with ``n`` privatized words spends ``n·ε`` by sequential
composition (reported, not enforced -- as in the paper). We use the authors' published roberta-base
clip bounds. The one default we do *not* inherit is ε: this defense runs at **ε=100** per word,
because below ~ε=100 the sampled words are mostly unrelated to the originals and the defended text
stops being usable prompt text (see ``DPMLM_EPSILON``). The ``dp_mlm_eps<ε>`` registry entries sweep
around it.

Two deliberate deviations from the reference, both required by this framework:

* **Determinism for cache correctness.** :class:`~prompt_anonymity.defenses.base.CachedDefense`
  caches rewrites keyed on :meth:`params`, so genuinely random sampling would make one input yield
  different outputs across runs and corrupt the cache. We seed a per-conversation ``torch.Generator``
  from the conversation text + a configurable ``seed`` (in ``params()``) before sampling its words.
  The exponential-mechanism *distribution* is unchanged -- still true DP sampling -- only
  reproducibility is added. Consequence to keep in mind: an identical conversation always maps to the
  same output, which is fine for offline evaluation but is not fresh randomness per release.
* **Scope.** We port the paper's headline mode (rewrite every content word) plus the optional Presidio
  PII toggle (``pii=True`` == the reference ``PII=True, hybrid=False``: detected entities become kept
  ``<ENTITY_TYPE>`` placeholders, everything else is still DP-rewritten). The reference's IPI/NER and
  ``hybrid_budget`` modes are omitted.

**Adaptive length (the paper's Algorithm 3, "Text Rewriting +-").** Plain DP-MLM emits exactly one
word per input word, so the rewrite preserves word count and text length perfectly -- which the paper
calls its "primary limitation" (§7) and which matters here because length and word count are literal
features in the stylometric vectors the attacks use. Setting ``add_prob`` (the paper's ``A``) and/or
``del_prob`` (``D``) above zero enables the fix: each eligible word is **deleted** with probability
``D`` (no MLM call, no budget), and each surviving privatized word is followed by an **added** word
with probability ``A``, drawn by inserting a ``<mask>`` into the context and running DP-MLM as usual
at the same ε. Both default to ``0.0``, so ``dp_mlm`` and the ``dp_mlm_eps<ε>`` sweep are unaffected;
the ``dp_mlm_var_a<A>`` registry entries turn it on. Budget becomes ``2·n·ε`` worst case (an addition
for every word) and ``(1 − D + A)·n·ε`` in expectation -- the paper's stated "``(A − D)nε``" appears
to drop the base ``n`` term. The add/delete coins are data-independent Bernoulli draws, so they spend
no budget themselves.

Deviations from the reference ``dpmlm_rewrite_plus`` in this mode:

1. **Context is the original (deletion-applied) text, never the running privatized copy.** The
   reference's plus path privatizes sequentially and mutates ``working_tokens``, so later words see
   earlier *replacements*. Batching forbids that -- and this is the same deviation already taken for
   the plain path, since we port the reference's own batch path (its non-batch ``dpmlm_rewrite``
   likewise defaults to ``REPLACE=False``, i.e. original context). Deletions *are* reflected in the
   context, as in the reference.
2. **No ``<mask>`` ever appears in the CONCAT clean segment.** The reference leaves one there for an
   addition (a side effect of passing the mutated list to both sides); we keep the clean side as
   unmasked context, so exactly one mask exists per encoded input and the mask-position lookup is
   unambiguous. This matches the paper's Algorithm 1, whose context segment is the original tokens.
3. **The coins are drawn over every non-punctuation, non-PII token** -- function words included, as in
   the reference -- while *privatization* still follows the usual eligible set. Deleting stopwords is
   deliberate: function-word frequency is the classic authorship signal. Note the consequence: the
   coin set is larger than the ``n`` we privatize, so the perturbation rate is not a like-for-like
   match with a fixed-length run at the same ε.
4. **The last surviving token is never deleted**, so a turn can never be emptied. (The reference only
   guards the final *token*, usually punctuation, so its guard rarely fires.)
5. **Added words are lowercased and empty decodes dropped**; the reference appends the decoded token
   raw. An addition has no original word to inherit case from, and capitalization ratios are
   themselves stylometric features.
6. **An addition whose mask is truncated away is dropped**, so pathological turns realize a slightly
   lower effective ``A``.
7. **Determinism**, as above: the coins come from a ``random.Random`` seeded per turn on a stream
   *separate* from the sampling generator, so ``add_prob=del_prob=0`` reproduces fixed-length output
   exactly and ``A``/``D`` act as nested thresholds on a fixed uniform stream (the A=0.1 additions are
   a subset of the A=0.25 ones). Both coins are drawn for every eligible token regardless of the
   delete outcome -- distributionally identical to the reference, but it decouples the two knobs.

Needs the ``[dpmlm]`` extra (adds ``nltk`` on top of torch/transformers); the ``pii=True`` variant
also needs the ``[dpmlm-pii]`` extra (``presidio-analyzer`` + spaCy). NLTK data is fetched lazily on
first backend build. Like every defense here, this only rewrites text; features are recomputed by the
featurize stage.
"""

from __future__ import annotations

import hashlib
import os
import random
import string

from ._backends import PerTurnBatchRewriteDefense, gpu_dtype

#: Encoder MLM used for privatization (the paper's default).
DPMLM_MODEL = "FacebookAI/roberta-base"

#: Published roberta-base logit clip bounds from the reference implementation. Sensitivity is
#: ``|clip_max − clip_min|``; these bound the exponential mechanism's utility range.
DPMLM_CLIP_MIN = -3.2093127
DPMLM_CLIP_MAX = 16.304797887802124

#: Per-word privacy budget (a text with n privatized words spends n·ε). Default ε=100: at the
#: paper's ε=25 the clip floor sits near the logit mean, so the exponential mechanism's tail over the
#: 50k vocab draws mostly unrelated words and the rewrite is not usable as text. ε=100 is the lowest
#: value in the paper's set {10,25,50,100,250} whose output stays readable -- weaker privacy, which
#: is the tradeoff the sweep exists to measure. Sweep it via the registered ``dp_mlm_eps<ε>``
#: defenses, this env var, or by instantiating with a different value.
DPMLM_EPSILON = float(os.environ.get("DPMLM_EPSILON", "100"))

#: Seed folded into the per-turn RNG so identical turns rewrite reproducibly (cache-safe).
DPMLM_SEED = int(os.environ.get("DPMLM_SEED", "0"))

#: Adaptive-length probabilities (the paper's Algorithm 3 ``A`` and ``D``): per eligible word, chance
#: of adding a word after it / of deleting it. Both ``0.0`` == the fixed-length headline mode, so the
#: plain ``dp_mlm`` and the ε sweep keep one-word-in-one-word-out. The paper evaluates A ∈ {0.1, 0.25}
#: with D = 0.05 (Appendix C); the reference code defaults to A=0.15. In ``params()``, so each
#: setting caches separately.
DPMLM_ADD_PROB = float(os.environ.get("DPMLM_ADD_PROB", "0.0"))
DPMLM_DEL_PROB = float(os.environ.get("DPMLM_DEL_PROB", "0.0"))

#: The two kinds of masked-position job. A REPLACE job masks the word at the anchor index (plain
#: DP-MLM); an INSERT job masks a *new* slot immediately after the anchor (adaptive length). Both run
#: through the same batching, and ``(anchor, kind)`` ordering puts an added word right after the word
#: it follows.
_JOB_REPLACE = 0
_JOB_INSERT = 1

#: MLM forward-pass batch size over masked positions (output-neutral; not in params()). Default is
#: sized for an A100 -- masked positions from all turns are pooled and length-sorted, and only the
#: masked position is projected through the vocab head, so large batches fit easily.
DPMLM_BATCH_SIZE = int(os.environ.get("DPMLM_BATCH_SIZE", "128"))

#: Masked positions are processed in chunks of whole turns totalling ~this many positions, so peak
#: memory stays bounded on a large unknown side while GPU batches still fill. Output-neutral.
DPMLM_FLUSH_POSITIONS = int(os.environ.get("DPMLM_FLUSH_POSITIONS", "8192"))

#: Flush finished conversations to the cache every this many, so this multi-hour defense is crash-
#: safe and resumable (a killed run picks up from the last checkpoint). Output-neutral.
DPMLM_CHECKPOINT_EVERY = int(os.environ.get("DPMLM_CHECKPOINT_EVERY", "200"))


class _DPMLMBackend:
    """Faithful port of the reference ``DPMLM`` class, trimmed to core + optional PII.

    Loads ``AutoModelForMaskedLM`` once and privatizes words through the exponential mechanism in
    GPU batches. Heavy imports (torch/transformers/nltk/presidio) live in ``__init__`` so importing
    this module never requires them.
    """

    def __init__(self, *, model, clip_min, clip_max, epsilon, seed, concat, stop, pii, batch_size,
                 add_prob=0.0, del_prob=0.0, flush_positions=DPMLM_FLUSH_POSITIONS):
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
        self.flush_positions = int(flush_positions)

        self.add_prob = float(add_prob)
        self.del_prob = float(del_prob)
        for label, prob in (("add_prob", self.add_prob), ("del_prob", self.del_prob)):
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"DP-MLM {label} must be in [0, 1]; got {prob}.")
        #: True when the adaptive-length mode is on; plain mode short-circuits every coin draw.
        self.varlen = self.add_prob > 0.0 or self.del_prob > 0.0
        #: Diagnostic counters for the current rewrite_batch (cache misses in this process only).
        self._stats = {"total": 0, "perturbed": 0, "added": 0, "deleted": 0}

        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.sensitivity = abs(self.clip_max - self.clip_min)
        if self.sensitivity <= 0:
            raise ValueError("DP-MLM sensitivity must be > 0; check clip_min/clip_max.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # Right-pad so a mask position computed on the unpadded ids stays valid after padding.
        self.tokenizer.padding_side = "right"
        # Sliding-window budget: MLM context is halved by the CONCAT sentence pair, minus special
        # tokens (matches the reference's ``(model_max_length // 2) - 32``).
        self.max_context = (self.tokenizer.model_max_length // 2) - 32

        self.lm_model = AutoModelForMaskedLM.from_pretrained(
            model, torch_dtype=gpu_dtype(torch)
        ).to(self.device)
        self.lm_model.eval()
        torch.set_grad_enabled(False)

        # For the mask-position-only projection: run the encoder trunk, then send just the masked
        # position's hidden state through the vocab head (RoBERTa ``lm_head`` / BERT ``cls``) --
        # avoids projecting all ~512 positions through the 50k-vocab head and the multi-GB logits
        # tensor. Falls back to full logits at runtime if a model doesn't fit this split.
        self._trunk = self.lm_model.base_model
        self._head = getattr(self.lm_model, "lm_head", None) or getattr(self.lm_model, "cls", None)
        self._use_fast_head = self._head is not None

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

        # Leading/trailing special-token ids: RoBERTa uses bos/eos, BERT uses cls/sep.
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
        eos = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
        prefix = [bos] if bos is not None else []
        suffix = [eos] if eos is not None else []

        core = tok.encode(" " + masked_sent, add_special_tokens=False)
        budget = max_len - len(prefix) - len(suffix)  # room for the specials we add back.
        if len(core) > budget:
            try:
                mpos = core.index(tok.mask_token_id)
            except ValueError:
                core = core[:budget]  # no mask (shouldn't happen) -> plain head truncation.
            else:
                hi = min(len(core), mpos + budget // 2)
                lo = max(0, hi - budget)
                core = core[lo:hi]
        return prefix + core + suffix

    # -- building masked inputs + batched MLM forward ---------------------------------------------

    def _build_input(self, tokens, idx, clean_full=None, insert=False):
        """Encoded input ids for masking word ``idx`` of ``tokens`` (sliding window + CONCAT).

        ``clean_full`` is the whole turn already detokenized; reused as the clean segment when the
        sliding window spans the whole turn (short/medium turns), so the clean side is detokenized
        once per turn instead of once per word.

        With ``insert=True`` the mask does not replace word ``idx`` but takes a *new* slot right
        after it (adaptive length): the window is still taken around ``idx`` on the unmodified
        ``tokens``, and the mask is spliced into the window copy only. Keeping the window mask-free
        is what preserves the ``clean_full`` fast path and guarantees the encoded pair holds exactly
        one mask -- in the masked segment, never in the clean one.
        """
        lower_w, upper_w = self.sliding_window(tokens, idx, self.max_context)
        chunk_tokens = tokens[lower_w:upper_w]
        rel_idx = idx - lower_w

        masked_chunk = list(chunk_tokens)
        if insert:
            # One token over max_context, which its 32-token margin absorbs.
            masked_chunk.insert(rel_idx + 1, self.tokenizer.mask_token)
        else:
            masked_chunk[rel_idx] = self.tokenizer.mask_token

        if clean_full is not None and lower_w == 0 and upper_w == len(tokens):
            clean_sent = clean_full
        else:
            clean_sent = self.detokenizer.detokenize(chunk_tokens)
        masked_sent = self.detokenizer.detokenize(masked_chunk)
        return self._encode_masked(clean_sent, masked_sent)

    def _forward_mask_logits(self, input_ids_list, mask_positions):
        """Vocab logits at each masked position for a batch, as a GPU tensor ``[len(batch), vocab]``.

        Runs the encoder trunk once, then projects ONLY the masked position through the vocab head
        (``lm_head``/``cls``) -- far cheaper than materializing ``[batch, seq, vocab]`` logits. Falls
        back to a full forward (indexing the mask position out) if the trunk/head split doesn't fit
        the loaded model. Stays on-device so the softmax/sampling can run on the GPU too.
        """
        torch = self._torch
        inputs = self.tokenizer.pad(
            {"input_ids": input_ids_list}, padding=True, return_tensors="pt"
        ).to(self.device)
        mpos = torch.as_tensor(mask_positions, device=self.device)
        rows = torch.arange(mpos.shape[0], device=self.device)

        with torch.inference_mode():
            if self._use_fast_head:
                try:
                    hidden = self._trunk(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                    )[0]                                   # [batch, seq, hidden]
                    return self._head(hidden[rows, mpos])  # [batch, vocab]
                except Exception:
                    self._use_fast_head = False  # one-time fallback for models that don't fit the split.
            logits = self.lm_model(**inputs).logits      # [batch, seq, vocab]
            return logits[rows, mpos]

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

    # -- per-turn planning ------------------------------------------------------------------------

    def _plan_turn(self, text):
        """``(sentence, tokens, jobs)`` for a turn: the word tokens to emit and the masked-position
        jobs to run over them.

        ``jobs`` is a list of ``(anchor, kind)`` in ascending order -- ``_JOB_REPLACE`` to privatize
        ``tokens[anchor]`` (the content words: stopwords, punctuation and ``pii`` placeholders are
        skipped), and, in adaptive-length mode, ``_JOB_INSERT`` to draw an extra word after it. That
        order is the sampling order in :meth:`_process_chunk`, so it must not depend on batching.

        In adaptive-length mode ``tokens`` is the turn *minus its deleted words*: the coins are
        resolved here, before any batching, and deletions are applied to the token list itself so
        everything downstream just sees a shorter turn.
        """
        sentence = " ".join(str(text).split("\n"))

        pii_ranges = []
        if self.analyzer is not None:
            sentence, pii_ranges = self._pii_scrub(sentence)

        tokens = self._nltk.word_tokenize(sentence)
        if not tokens:
            return sentence, [], []

        # Tokens overlapping a PII placeholder are kept verbatim (not DP-rewritten).
        pii_mask = [False] * len(tokens)
        if pii_ranges:
            spans = list(self.word_tokenizer.span_tokenize(sentence))
            for i, (t_start, t_end) in enumerate(spans):
                if i >= len(pii_mask):  # word_tokenize vs span_tokenize can disagree on token count.
                    break
                if any(max(t_start, p_start) < min(t_end, p_end) for p_start, p_end in pii_ranges):
                    pii_mask[i] = True

        indices, coin_indices = [], []
        for i, tok in enumerate(tokens):
            if pii_mask[i]:
                continue
            if tok in string.punctuation:
                continue
            # The add/delete coins cover function words too (as the reference plus path does), even
            # though we only ever *privatize* the content words below.
            coin_indices.append(i)
            if not self.stop_flag and tok.lower() in self.stop:
                continue
            indices.append(i)

        if not self.varlen:
            return sentence, tokens, [(i, _JOB_REPLACE) for i in indices]

        # -- adaptive length (Algorithm 3) ---------------------------------------------------------
        # The coins are data-independent Bernoulli draws, so they can be resolved here, before any
        # batching -- which is what lets an added word ride through the GPU pass as an ordinary job
        # instead of forcing the reference's sequential per-word loop. Own RNG stream, seeded from
        # this turn's text: independent of chunking, and it leaves the sampling stream untouched so
        # add_prob=del_prob=0 still reproduces fixed-length output exactly.
        rng = random.Random(int.from_bytes(
            hashlib.sha256(f"{self.seed}:varlen:{sentence}".encode("utf-8")).digest()[:8], "big"
        ))
        delete_at, add_after = set(), set()
        last_coin = coin_indices[-1] if coin_indices else None
        for i in coin_indices:
            # Draw both coins for every token whatever the outcome, so A and D stay independent
            # thresholds on one fixed stream (the A=0.1 additions are a subset of the A=0.25 ones).
            u_del, u_add = rng.random(), rng.random()
            if u_del < self.del_prob and i != last_coin:  # last one stays: never empty a turn.
                delete_at.add(i)
                continue
            if self.add_prob > 0.0 and u_add <= self.add_prob:  # `<=`: random() can return 0.0.
                add_after.add(i)

        # Apply the deletions to the token list and re-base the jobs onto what survives.
        if delete_at:
            base, remap = [], {}
            for i, tok in enumerate(tokens):
                if i in delete_at:
                    continue
                remap[i] = len(base)
                base.append(tok)
        else:
            base, remap = tokens, None

        privatize_at = {i for i in indices if i not in delete_at}
        jobs = []
        for i in sorted(privatize_at | add_after):  # add_after and delete_at are disjoint.
            b = i if remap is None else remap[i]
            if i in privatize_at:
                jobs.append((b, _JOB_REPLACE))
            if i in add_after:
                jobs.append((b, _JOB_INSERT))

        self._stats["deleted"] += len(delete_at)
        return sentence, base, jobs

    # -- chunked, GPU-saturating rewriting --------------------------------------------------------

    def _process_chunk(self, chunk, outputs, progress=None):
        """Rewrite a chunk of planned turns, writing results into ``outputs`` by turn position.

        Phase 1 (CPU): build every masked input in the chunk. Phase 2 (GPU): length-sort and run the
        MLM in full batches, keeping only the masked position's logits (on-device). Phase 3 (GPU):
        per-conversation seeded exponential-mechanism sampling -- clip/softmax/multinomial run on the
        A100 in one call per conversation, so the 50k-vocab math never touches the CPU. Output does
        not depend on batching; each conversation is seeded from its own text so the cache stays
        consistent.
        """
        torch = self._torch
        mask_id = self.tokenizer.mask_token_id

        # Phase 1: build masked inputs for every job in the chunk (replacements and additions alike).
        chunk_positions = sum(len(jobs) for _ti, _s, _tok, jobs in chunk)
        flat_ids, flat_mpos, flat_meta = [], [], []
        for pos, (_ti, _sentence, tokens, jobs) in enumerate(chunk):
            clean_full = self.detokenizer.detokenize(tokens) if len(tokens) <= self.max_context else None
            for anchor, kind in jobs:
                input_ids = self._build_input(tokens, anchor, clean_full, insert=(kind == _JOB_INSERT))
                try:
                    m_pos = input_ids.index(mask_id)
                except ValueError:
                    # Mask truncated away -> no forward, no draw. A replacement keeps the original
                    # word; an addition simply does not happen.
                    continue
                flat_ids.append(input_ids)
                flat_mpos.append(m_pos)
                flat_meta.append((pos, anchor, kind))
        if progress is not None:  # count the kept-original positions (never forwarded) up front.
            progress.update(chunk_positions - len(flat_ids))

        # Phase 2: length-sorted batches through the MLM; keep masked-position logits on-device.
        n = len(flat_ids)
        row_of: dict = {}
        chunk_logits = None
        order = sorted(range(n), key=lambda j: len(flat_ids[j]))
        for b in range(0, n, self.batch_size):
            sel = order[b:b + self.batch_size]
            logits = self._forward_mask_logits([flat_ids[j] for j in sel], [flat_mpos[j] for j in sel])
            if chunk_logits is None:  # allocate now that we know the head's vocab width.
                chunk_logits = torch.empty((n, logits.shape[-1]), dtype=torch.float16, device=self.device)
            for row, j in enumerate(sel):
                chunk_logits[j] = logits[row].to(torch.float16)
                row_of[flat_meta[j]] = j
            if progress is not None:
                progress.update(len(sel))

        # Phase 3: per-conversation seeded sampling on the GPU (exponential mechanism).
        scale = 2 * self.sensitivity / self.epsilon
        gen = torch.Generator(device=self.device)
        for pos, (ti, sentence, tokens, jobs) in enumerate(chunk):
            repl: dict = {}
            live = [job for job in jobs if (pos, *job) in row_of]  # job order == draw order.
            if live:
                # Stable per-conversation seed (builtin hash() is per-process randomized).
                seed = int.from_bytes(
                    hashlib.sha256(f"{self.seed}:{sentence}".encode("utf-8")).digest()[:8], "big"
                ) & 0x7FFFFFFFFFFFFFFF
                gen.manual_seed(seed)

                rows = torch.as_tensor([row_of[(pos, *job)] for job in live], device=self.device)
                # Pr[v] ∝ exp(ε·u(v) / (2·Δu)): clip, scale, softmax, sample -- all on-device.
                block = chunk_logits[rows].float().clamp_(self.clip_min, self.clip_max).div_(scale)
                probs = torch.softmax(block, dim=-1)
                chosen = torch.multinomial(probs, 1, generator=gen).squeeze(1).tolist()
                for job, cid in zip(live, chosen):
                    repl[job] = self.tokenizer.decode(cid).strip()

            out = []
            for i, tok in enumerate(tokens):  # deleted words are already gone from `tokens`.
                r = repl.get((i, _JOB_REPLACE))
                if r is None:  # skipped/kept -> original word, original case.
                    out.append(tok)
                else:  # restore the original word's capitalization.
                    out.append(r.capitalize() if tok[:1].isupper() else r.lower())
                    self._stats["total"] += 1
                    if r.lower() != tok.lower():
                        self._stats["perturbed"] += 1
                added = repl.get((i, _JOB_INSERT))
                if added:  # an added word has no original case to inherit; empty decode -> drop it.
                    out.append(added.lower())
                    self._stats["added"] += 1
            outputs[ti] = self.detokenizer.detokenize(out)

    def rewrite_batch(self, texts):
        from tqdm.auto import tqdm

        outputs: list = [None] * len(texts)
        self._stats = {"total": 0, "perturbed": 0, "added": 0, "deleted": 0}

        # Plan all turns; accumulate whole turns into a chunk until it holds ~flush_positions masked
        # positions, so peak memory is bounded but GPU batches still fill. Counting *jobs* (not
        # eligible words) keeps that bound honest when additions are switched on.
        plans = [self._plan_turn(t) for t in texts]
        total_positions = sum(len(jobs) for _s, _tok, jobs in plans)

        progress = tqdm(total=total_positions, desc="DP-MLM words", unit="word", leave=False)
        chunk, chunk_positions = [], 0
        for ti, (sentence, tokens, jobs) in enumerate(plans):
            if not tokens:
                outputs[ti] = sentence
                continue
            chunk.append((ti, sentence, tokens, jobs))
            chunk_positions += len(jobs)
            if chunk_positions >= self.flush_positions:
                self._process_chunk(chunk, outputs, progress)
                chunk, chunk_positions = [], 0
        if chunk:
            self._process_chunk(chunk, outputs, progress)
        progress.close()

        if self.varlen:
            # Realized rates for the turns actually rewritten here (cache misses only, and
            # rewrite_batch sees *distinct* turns) -- a diagnostic, not a privacy audit trail.
            s = self._stats
            spent = (s["total"] + s["added"]) * self.epsilon
            print(f"[dp_mlm] A={self.add_prob} D={self.del_prob}: {s['total']} words privatized "
                  f"({s['perturbed']} changed), {s['added']} added, {s['deleted']} deleted; "
                  f"budget spent {spent:.0f} (= {s['total'] + s['added']} draws x eps={self.epsilon:g})")
        return outputs


class DPMLMDefense(PerTurnBatchRewriteDefense):
    """Differentially private per-word rewriting with RoBERTa + the exponential mechanism.

    The backend is built lazily, so a fully-cached run loads no model. Set ``pii=True`` to also
    scrub Presidio-detected entities to ``<ENTITY_TYPE>`` placeholders before rewriting the rest.
    Set ``add_prob``/``del_prob`` above zero for the paper's adaptive-length mode (Algorithm 3), in
    which the rewrite no longer preserves the input's word count -- see the module docstring.
    """

    name = "dp_mlm"
    version = "1"
    checkpoint_every = DPMLM_CHECKPOINT_EVERY  # crash-safe/resumable: flush every N conversations.

    def __init__(self, *, model: str = DPMLM_MODEL, epsilon: float = DPMLM_EPSILON,
                 clip_min: float = DPMLM_CLIP_MIN, clip_max: float = DPMLM_CLIP_MAX,
                 seed: int = DPMLM_SEED, concat: bool = True, stop: bool = False,
                 pii: bool = False, batch_size: int = DPMLM_BATCH_SIZE,
                 add_prob: float = DPMLM_ADD_PROB, del_prob: float = DPMLM_DEL_PROB):
        self.model = model
        self.epsilon = float(epsilon)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.seed = int(seed)
        self.concat = bool(concat)
        self.stop = bool(stop)
        self.pii = bool(pii)
        self.batch_size = int(batch_size)
        self.add_prob = float(add_prob)
        self.del_prob = float(del_prob)
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
            "add_prob": self.add_prob,
            "del_prob": self.del_prob,
        }

    def _get_backend(self) -> _DPMLMBackend:
        if self._backend is None:
            self._backend = _DPMLMBackend(
                model=self.model, clip_min=self.clip_min, clip_max=self.clip_max,
                epsilon=self.epsilon, seed=self.seed, concat=self.concat, stop=self.stop,
                pii=self.pii, batch_size=self.batch_size,
                add_prob=self.add_prob, del_prob=self.del_prob,
            )
        return self._backend

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        return self._get_backend().rewrite_batch(texts)
