r"""Label every conversation of a built split with a topic, through Claude on Microsoft Foundry.

Companion to :mod:`prompt_anonymity.data.compute_features`: where that script writes one feature
vector per document, this one writes three labels per document, read off its user turns by an LLM:

* ``topic`` -- one of the 24 conversation-topic categories of Chatterji et al. (2025), *How People
  Use ChatGPT* (NBER w34255, Appendix A.3), with its ``topic_group`` -- one of their seven coarse
  topics (Table 3) -- derived from the taxonomy file rather than asked for, so the two levels can
  never disagree;
* ``intent`` -- their Asking / Doing / Expressing classification (Appendix A.2);
* ``description`` and ``subject`` -- **our addition**, two free-text English labels for where 24
  categories are too coarse (nearly all of swe-chat is ``computer_programming``):

  - ``description``: one sentence of at most 25 words saying what the user wants, **specifics
    kept** -- the names of their projects, files, characters, people. Shaped like the one-sentence
    precis Chatterji et al. gave their human annotators (Appendix B).
  - ``subject``: a noun phrase of at most eight words naming the subject matter **in general
    terms**. Widely known names stay (languages, libraries, public figures, events, well-known
    works); names from the user's own world are replaced by what kind of thing they are ("a
    personal web app", "an original fantasy character").

  The split is the point. A user-specific name -- a repository, a recurring character -- is
  identity, not topic, so ``subject`` is the topic-only representation (a clustering attack on it
  measures how far topic alone links a user), and the gap between attacks on ``description`` and on
  ``subject`` measures how much shared *names* add beyond shared topics.

Why a published taxonomy: it is citable, it was validated against human labels on WildChat --
coarse topic Cohen's kappa 0.56 model vs. human plurality against 0.47 human vs. human (149
messages; confusions concentrated between Seeking Information and Practical Guidance), intent 0.74
against 0.60 -- and it puts our label distribution on the same scale as theirs. Those agreement
figures are what to quote for the label noise, and they are moderate: a topic label is a noisy
measurement even for people.

**Where this departs from their classifier**, each a deliberate adaptation to report:

1. **Unit: the whole conversation, not the last message.** They label the last user message with
   up to 10 earlier messages as context; a document here is a conversation, so the model is asked
   for the capability most of its user messages seek. Their "if several, the one most relevant to
   the LAST message" rule becomes "the one most of the user's messages are seeking".
2. **User turns only.** The dataset holds no assistant replies, so their context's assistant turns
   are absent.
3. **Model**: Claude Haiku 5.5, not gpt-5-mini.
4. **Output**: a structured JSON reply holding ``description`` and ``subject`` too, not a bare
   category name.
5. **Truncation**: their 5,000 characters per message is kept, plus a cap on the whole conversation
   (see ``--max-chars``) that their 11-message window imposed implicitly.

The 24 categories, their descriptions and their examples are theirs verbatim, including the three
image/media categories this text-only data will rarely fill.

Where the output goes
---------------------

The split is READ from ``data/hf`` and the labels are WRITTEN to ``dist/<split>_topics.parquet``,
one row per document in split order, keyed by ``doc_id`` -- joined back exactly like a feature file
(``author_id`` is carried for the same reason). That file holds the labels and nothing else
(:data:`DATASET_COLUMNS`: the ids, ``topic``, ``topic_group``, ``intent``, ``subject``,
``description`` and
``label_status``, which says why a topic is missing). **What the labels cost is not data** and is
kept out of ``data/`` altogether, under ``tmp/topic_labels/`` at the project root (see "Caching,
cost and re-runs"). Topics are deliberately **not** added as columns of
``<split>.parquet``: they are one model's reading of the text under one version of the taxonomy,
not a property of the corpus, and the split's schema is a contract
(``validate_dataset.EXPECTED_COLUMNS``) that a rebuild would silently drop extra columns from. The
name shares the ``<split>_<name>.parquet`` namespace with feature and defended files, so ``topics``
must never become the name of a featurizer or a defense.

Topics are labelled on the **undefended** text, so every defense of a split shares one set of
topics and a per-topic comparison across defenses compares the same conversations.

The model and the request
-------------------------

Claude Haiku 5.5, through the project's Foundry deployment (``claude-haiku-5-5``) with the Anthropic
SDK's :class:`~anthropic.AnthropicFoundry` client. Credentials are :data:`FOUNDRY_ENDPOINT_ENV`
and :data:`FOUNDRY_API_KEY_ENV`, from the environment or the project's ``.env``, and **both are
required**: with the key alone the SDK would post an Azure key to Anthropic's own API.

**Foundry has no Message Batches API**, so there is no half-price batch tier here: every request is
a standard Messages call at the standard rate, fanned out over a thread pool (``--workers``). The
work is network-bound, so threads scale it; the pool width is a rate-limit knob, not a CPU one.

Each request is shaped by what this model accepts:

* **Thinking is turned off** (``thinking={"type": "disabled"}``, accepted at effort ``high`` or
  below). Haiku 5.5 thinks by default and bills thinking as output tokens, and telling it to answer
  directly does not stop it -- only the parameter does. ``--thinking adaptive`` turns it back on for
  a pilot comparing the two.
* **Structured output**: the reply is constrained to a JSON schema whose ``topic`` and ``intent``
  fields are enums of the taxonomy's ids, so an answer outside the label set cannot come back.
  The schema's order is ``description``, ``subject``, ``topic``, ``intent``: the model states the
  specifics, then generalizes them, then categorizes -- each step written before the next.
* **No sampling parameters and no prefill** -- both are a 400 on this model. The consequence is
  that **labels are not deterministic**: two uncached runs can disagree on a borderline
  conversation, and the cache is what makes a re-run reproducible.
* **The instructions are a cached system block.** ``cache_control`` sits on that block alone. The
  top-level automatic form would instead cache the *last* block -- the conversation -- paying the
  cache-write premium on every conversation for a prefix that is never read again.

The input is the conversation's user turns, in order, each wrapped in ``<turn_k>`` tags inside one
``<conversation>`` element (assistant replies are not in the dataset). The text is not
XML-escaped, because the corpora are full of literal ``<URL>``/``<PATH>`` placeholders. Each turn is
cut at ``--max-turn-chars`` (Chatterji et al.'s 5,000) and the whole at ``--max-chars``, so a few
pasted files cannot crowd out the turns after them, and no request comes near the 100,000-token
prompt length past which this model is billed at five times the rate.

Statuses
--------

``label_status`` says what happened to each document, and every non-``labeled`` row is kept rather
than dropped:

* ``labeled`` -- a valid topic, intent, subject and description.
* ``refused`` -- the model's safety classifiers declined (``stop_reason == "refusal"``), with the
  classifier's ``refusal_category``. There is no server-side fallback for this model. **Refusals
  are not random**: they concentrate on particular content, so dropping these rows would remove
  exactly those topics from any per-topic analysis.
* ``max_tokens`` -- the reply ran out of budget before the answer finished.
* ``invalid`` -- a reply that did not parse to the schema (structured output should make this
  impossible; the raw reply is kept in the cache entry to find out why).
* ``rejected`` -- the endpoint refused the request itself with a 4xx (an Azure content filter, an
  input it will not take), with the status and message in ``rejection_reason``. **Not cached**, so
  the next run tries it again, and normally not billed. A rejection that is really a configuration
  error stops the run instead: a 401/403/404 at once, anything else after
  :data:`MAX_CONSECUTIVE_REJECTIONS` in a row, since a wrong deployment or a parameter the endpoint
  does not accept would otherwise be "recorded" for every document.
* ``empty`` -- no non-blank turn, so no request was made.

Caching, cost and re-runs
-------------------------

Every label is cached as it arrives (:meth:`~prompt_anonymity.caching.TransformCache.apply_streaming`),
content-addressed by the rendered input, under ``<cache-dir>/topics/topic_labels``. A killed run
loses at most the requests in flight and resumes where it stopped; a re-run of a finished split is
free and needs no credentials. Refusals are cached too -- they are the model's answer, and asking
again would be billed again.

**Cache invalidation is deliberate, as for the utility judge, not automatic as for a defense.** The
namespace is :data:`TOPIC_LABEL_VERSION` plus :meth:`FoundryTopicLabeler.params` -- the deployment,
the full system prompt (instructions *and* every label), the response schema, thinking, effort and
the output budget. Editing ``topic_taxonomy.yaml`` therefore re-labels everything on the next run,
which is correct: it is a different question. Editing *this module* does not, since hashing its
source would discard paid labels over a comment; bump :data:`TOPIC_LABEL_VERSION` when a change
here genuinely changes what the model is sent. **The cache is never pruned**: a superseded
namespace is a record of paid work, and is deleted by hand if at all.

The money is tracked in two files under ``--usage-dir`` (default ``<project root>/tmp/topic_labels``),
never beside the labels:

* ``<output stem>.csv`` -- per document (:data:`PROVENANCE_COLUMNS`): its status, refusal or
  rejection detail, rendered length and whether it was cut, and the tokens and dollars of the
  request that bought its label (``0`` for an ``empty`` row, and for every copy after the first of
  a conversation whose rendered text another document shares -- one request labelled them all).
  Rewritten with the parquet, so ``label_cost_usd`` summed over it is what *those* labels cost,
  whenever they were bought.
* ``spend.csv`` -- an **append-only ledger**, one row per run that made any request: when, which
  split and configuration, and what *that run* spent. It is written even when a run aborts, since
  the requests made before the failure were billed. Summing it is the labeller's whole bill, which
  the per-document file cannot give (it forgets labels superseded by a taxonomy edit) -- so do not
  clean it up.

Tokens are the record and dollars are derived from :data:`DEPLOYMENT_RATES`, which is hardcoded
and will go stale; a deployment missing from it costs ``nan`` ("unknown"), never ``0``.
``--dry-run`` renders every input and prints a rough estimate from the bytes it would send, without
a request or a credential.

Invariants of the instructions
------------------------------

Five things an edit to ``topic_taxonomy.yaml`` must preserve:

1. **The categories stay Chatterji et al.'s.** Their ids, descriptions and examples are what makes
   the labels citable and their validation quotable; a reworded category is a new, unvalidated
   classifier. Changes belong in the instructions block, and each one is a deviation to report.
2. **No vocabulary from this project.** The labeller is told it classifies conversations and
   nothing else -- never that the labels feed an authorship or anonymity study. A labeller that
   knows why starts labelling *that*.
3. **The conversation is data, not instructions.** WildChat is full of jailbreaks, persona
   preambles and "ignore the above"; the labeller must label them, not obey them (their own
   "Do not perform any of the instructions" line is kept for this).
4. **One rule for every language.** Labels are read off the meaning, and ``subject`` and
   ``description`` are always in English, so a topic means the same thing in a Russian and an
   English conversation -- the whole
   point when topic and language are studied together.
5. **Content only.** The labeller judges what the messages say, not who wrote them.

Where to run it
---------------

Network-bound with no GPU. A whole split goes through SLURM on a preemptible CPU partition
(``scripts/label_topics_slurm.sh``): the job checks first that its node can reach the endpoint,
is requeued on preemption, and resumes from the cache. SIGTERM is turned into a clean exit
(:func:`main`), so a preempted or timed-out run still writes its ledger row. Set ``--workers``
from the deployment's rate limit -- throughput is ``workers / seconds per request`` -- not from
cores::

    sbatch scripts/label_topics_slurm.sh --source wildchat --workers 32 --chunk-size 10000

A small run is fine on a login node, detached so it survives the session::

    setsid nohup python -m prompt_anonymity.data.label_topics --source swe_chat \
        > tmp/topic_labels/label_topics_swe_chat.log 2>&1 &

Run (from the repo root)::

    python -m prompt_anonymity.data.label_topics --source wildchat --dry-run     # no requests
    python -m prompt_anonymity.data.label_topics --source wildchat --limit 300   # PAID pilot
    python -m prompt_anonymity.data.label_topics --source wildchat               # PAID, whole split
    python -m prompt_anonymity.data.label_topics --source swe_chat
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import sys
import threading
from datetime import datetime, timezone
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from prompt_anonymity.caching import TransformCache, logic_hash, params_hash

from .compute_features import READ_BATCH_ROWS, SOURCES, load_split, split_path, write_parquet
from .config import cache_dir, dist_dir, hf_dir, project_root

#: The instructions and both label sets, beside this module (shipped as package data). See the
#: module docstring for the invariants an edit has to preserve.
TAXONOMY_FILE = Path(__file__).with_name("topic_taxonomy.yaml")

#: Manual logic version of the label cache. Since this module's source is deliberately not hashed
#: (see "Caching, cost and re-runs"), this is the only lever that re-labels on a code change: bump
#: it when an edit here changes what the model is sent or how its reply is read.
TOPIC_LABEL_VERSION = "1"

#: Environment variables holding the Foundry resource's endpoint (base URL) and key. Both required.
FOUNDRY_ENDPOINT_ENV = "ANTHROPIC_ENDPOINT"
FOUNDRY_API_KEY_ENV = "MODEL_API_KEY"

#: The deployment labels are requested from. **A Foundry deployment name, not an Anthropic model
#: id** -- on Foundry the ``model`` field names the deployment. Part of the cache key.
DEFAULT_DEPLOYMENT = "claude-haiku-5-5"

#: Effort level sent in ``output_config``. ``low`` because choosing from a fixed list is the
#: simple, high-volume kind of request it is meant for; a pilot should check it against ``medium``.
DEFAULT_EFFORT = "low"

#: Effort levels accepted with thinking off. Haiku 5.5 rejects disabled thinking at ``xhigh`` and
#: ``max`` with a 400, so those two are only offered with ``--thinking adaptive``.
EFFORTS_WITHOUT_THINKING = ("low", "medium", "high")
EFFORTS_WITH_THINKING = ("low", "medium", "high", "xhigh", "max")

#: Output budget per request. Thinking off, the reply is one short JSON object (~40 tokens), so 512
#: is ample; thinking on, it shares the budget with the reasoning, and a reply cut off by the budget
#: is a ``max_tokens`` row rather than a label.
MAX_TOKENS_WITHOUT_THINKING = 512
MAX_TOKENS_WITH_THINKING = 8192

#: Input caps, in characters: each user turn, then the rendered conversation as a whole. The
#: per-turn cap is Chatterji et al.'s per-message truncation. The total -- four such turns, a few
#: thousand tokens -- is ours, standing in for the bound their 11-message window set implicitly,
#: and stays far below the 100,000-token prompt length past which Haiku 5.5 is billed at its
#: long-context rate.
DEFAULT_MAX_TURN_CHARS = 5_000
DEFAULT_MAX_CHARS = 20_000

#: What replaces text cut from a turn, and the line closing a conversation cut after some turn.
#: The instructions tell the model what ``[...]`` means.
TRUNCATION_MARKER = " [...]"
OMITTED_TURNS_LINE = "[... {count} more messages not shown]"

#: Concurrent requests. A rate-limit knob: raise it if the deployment's quota allows and 429s are
#: rare, lower it if they dominate (the SDK retries them, but each retry is wall-clock).
DEFAULT_WORKERS = 8

#: SDK retries for connection errors and 408/409/429/5xx, with the SDK's own backoff -- well above
#: its default of 2, because a long run will meet transient rate limiting and the alternative is
#: aborting it.
FOUNDRY_MAX_RETRIES = 8

#: Seconds per request. The reply is short; this only has to outlast a slow queue.
FOUNDRY_TIMEOUT_SECONDS = 120.0

#: HTTP statuses that mean the client is misconfigured -- bad key, no access, no such deployment --
#: and so stop the run at once instead of being recorded against one document.
CONFIGURATION_ERROR_STATUSES = (401, 403, 404)

#: Consecutive per-document rejections (other 4xx) after which the run stops: a scattering is a
#: content filter doing its job, an unbroken run of them is a request every document would fail.
MAX_CONSECUTIVE_REJECTIONS = 20

#: Documents rendered and labelled per round. Each round's labels are cached as they arrive, so this
#: only bounds how much rendered text is held at once and how often the progress line refreshes.
DEFAULT_CHUNK_SIZE = 2_000

#: Seed for ``--limit`` sampling; the project-wide default.
DEFAULT_SEED = 47

#: The label cache: ``<cache-dir>/<CACHE_SUBDIR>/<CACHE_NAME>/<logic hash>/<params hash>/``.
CACHE_SUBDIR = "topics"
CACHE_NAME = "topic_labels"

#: The output's name part: ``<split>_topics.parquet``.
OUTPUT_LABEL = "topics"

#: Where the cost records go, relative to the project root: kept out of ``data/`` because what a
#: label cost is not part of the dataset. Holds ``<output stem>.csv`` and the ``spend.csv`` ledger.
USAGE_SUBDIR = Path("tmp") / "topic_labels"

#: The append-only spend ledger's file name under the usage directory.
SPEND_LEDGER = "spend.csv"

#: Prompt length (all input tokens, cache reads and writes included) above which Haiku 5.5 bills a
#: request at its long-context rate card.
LONG_PROMPT_TOKENS = 100_000

#: Rough UTF-8 bytes per token, for ``--dry-run``'s estimate only. Measured **2.49** over the full
#: swe-chat run (6.7 MB rendered -> 2.69M uncached input tokens, per-request overhead included;
#: mostly English and code) and 2.44 on the system prompt, so the low end is that. The high end
#: allows for Cyrillic, which packs more bytes into each token. The real count comes back in each
#: response's ``usage``.
DRY_RUN_BYTES_PER_TOKEN = (2.5, 3.5)

#: Output tokens per labelled conversation assumed by ``--dry-run``: one JSON object. Measured 43
#: with ``subject`` alone; the ~25-word ``description`` roughly doubles it.
DRY_RUN_OUTPUT_TOKENS = 80

#: An id in the taxonomy file: lower snake case, since ids are what the labels are stored as.
LABEL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

#: The token counts a response's ``usage`` reports, as stored in a cache entry and the provenance.
#: In the Anthropic API they are **disjoint buckets** -- ``input_tokens`` excludes the tokens read
#: from or written to the prompt cache -- unlike the OpenAI-compatible ``cached_tokens``, which is a
#: subset of ``prompt_tokens``.
USAGE_FIELDS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                "output_tokens")

#: The columns of the dataset file, ``dist/<split>_topics.parquet``: the labels and nothing else.
DATASET_COLUMNS = ["doc_id", "author_id", "topic", "topic_group", "intent", "subject",
                   "description", "label_status"]

#: The columns of the per-document provenance file under the usage directory: how each label was
#: obtained and what it cost. Never written to ``data/``.
PROVENANCE_COLUMNS = ["doc_id", "label_status", "refusal_category", "rejection_reason",
                      "input_chars", "input_truncated", *USAGE_FIELDS, "label_cost_usd"]

#: Every column :func:`label_documents` produces, before the split into the two files.
LABEL_COLUMNS = list(dict.fromkeys(DATASET_COLUMNS + PROVENANCE_COLUMNS))

#: The spend ledger's columns: one row per run that made at least one request.
SPEND_COLUMNS = ["finished_at_utc", "outcome", "source", "output_stem", "deployment", "thinking",
                 "effort", "n_documents", "requests", *USAGE_FIELDS, "cost_usd"]


# --- prices -------------------------------------------------------------------

@dataclass(frozen=True)
class TokenRates:
    """One rate card, in US dollars per million tokens.

    ``cache_write`` is the 5-minute cache write -- the only kind this module requests.
    """

    input: float
    cache_write: float
    cache_read: float
    output: float


#: Rate cards per deployment: ``(prompt <= LONG_PROMPT_TOKENS, prompt > LONG_PROMPT_TOKENS)``.
#: Anthropic's list prices, which Foundry bills at through the Azure Marketplace, as read off the
#: pricing page on 2026-11-07. **Add a deployment here rather than guessing a price**: a missing one
#: costs ``nan`` ("unknown"), which nobody mistakes for free.
DEPLOYMENT_RATES: dict[str, tuple[TokenRates, TokenRates]] = {
    "claude-haiku-5-5": (TokenRates(input=0.10, cache_write=0.125, cache_read=0.01, output=0.50),
                         TokenRates(input=0.50, cache_write=0.625, cache_read=0.05, output=2.50)),
}


def request_cost(deployment: str, usage: dict) -> float:
    """Dollars one request cost from its token counts, or ``nan`` when the deployment is unpriced.

    The rate card is chosen by the request's whole prompt length -- uncached input plus cache reads
    plus cache writes -- as the price list defines it, even where most of the prompt is a cache hit.
    """
    cards = DEPLOYMENT_RATES.get(deployment)
    if cards is None:
        return float("nan")
    prompt_tokens = (usage["input_tokens"] + usage["cache_read_input_tokens"]
                     + usage["cache_creation_input_tokens"])
    rates = cards[1] if prompt_tokens > LONG_PROMPT_TOKENS else cards[0]
    return (usage["input_tokens"] * rates.input
            + usage["cache_creation_input_tokens"] * rates.cache_write
            + usage["cache_read_input_tokens"] * rates.cache_read
            + usage["output_tokens"] * rates.output) / 1e6


@dataclass
class RunUsage:
    """What the requests *this run* made cost, for the end-of-run summary.

    Counts requests made, not documents labelled: a label served from the cache costs this run
    nothing and appears nowhere here, so a re-run of a finished split ends at ``requests == 0``.
    What each label cost when it was bought is in its row of the output instead.
    """

    requests: int = 0
    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, usage: dict, cost: float) -> None:
        """Fold one request's token counts and cost in."""
        self.requests += 1
        for name in USAGE_FIELDS:
            setattr(self, name, getattr(self, name) + usage[name])
        self.cost_usd += cost

    def summary(self, deployment: str) -> str:
        """One line: requests, tokens per bucket, and the cost or a note that none is known."""
        if not self.requests:
            return "API usage this run: 0 requests (everything served from the cache) -- $0.00"
        line = (f"API usage this run: {self.requests:,} requests  in={self.input_tokens:,}  "
                f"cache_read={self.cache_read_input_tokens:,}  "
                f"cache_write={self.cache_creation_input_tokens:,}  out={self.output_tokens:,} tok")
        if math.isnan(self.cost_usd):
            return line + (f"  cost=unknown ({deployment!r} has no entry in DEPLOYMENT_RATES; the "
                           "token counts are exact either way)")
        return line + f"  est. cost=${self.cost_usd:.4f}"


# --- the taxonomy -------------------------------------------------------------

@dataclass(frozen=True)
class Label:
    """One intent the model may answer with: its stored id and the description it is shown."""

    id: str
    description: str


@dataclass(frozen=True)
class Topic:
    """One topic category: its stored id, its coarse group, and what the model is shown for it --
    a one-line description and a few example requests (Chatterji et al.'s, in several languages)."""

    id: str
    group: str
    description: str
    examples: tuple[str, ...]


@dataclass(frozen=True)
class TopicGroup:
    """A coarse topic (Chatterji et al.'s Table 3). Used to aggregate labels; never shown to the
    model, which -- as in their pipeline -- picks among the fine categories only."""

    id: str
    name: str


@dataclass(frozen=True)
class Taxonomy:
    """The labeller's instructions and label sets, as read from ``topic_taxonomy.yaml``.

    Everything the model is told is rendered from this object (:meth:`system_prompt`), and so is
    the schema its reply is held to (:meth:`response_schema`), so the list the model reads and the
    enum it is constrained to cannot drift apart.
    """

    instructions: str
    topics: tuple[Topic, ...]
    groups: tuple[TopicGroup, ...]
    intents: tuple[Label, ...]

    @property
    def topic_ids(self) -> list[str]:
        """Every topic id, in file order."""
        return [topic.id for topic in self.topics]

    @property
    def intent_ids(self) -> list[str]:
        """Every intent id, in file order."""
        return [intent.id for intent in self.intents]

    def group_of(self) -> dict[str, str]:
        """Map each topic id to the id of its coarse group."""
        return {topic.id: topic.group for topic in self.topics}

    def system_prompt(self) -> str:
        """The full system prompt, laid out as Chatterji et al.'s classifier prompt is.

        The instructions; the categories with their descriptions; the examples under each
        category; then the intent categories, whose examples are part of their descriptions.
        """
        lines = [self.instructions.strip(), "", "Topic categories:", ""]
        lines += [f"- **{topic.id}**: {topic.description}" for topic in self.topics]
        lines += ["", "Examples:"]
        for topic in self.topics:
            if topic.examples:
                lines.append(f"**{topic.id}**:")
                lines += [f'- "{example}"' for example in topic.examples]
        lines += ["", "Intent categories:", ""]
        lines += [f"- **{intent.id}**: {intent.description}" for intent in self.intents]
        return "\n".join(lines)

    def response_schema(self) -> dict:
        """The JSON schema a reply is constrained to: a description and a subject, then a topic and
        an intent by id -- in that order, so the specifics are written before they are generalized
        and the generalization before it is categorized."""
        return {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "subject": {"type": "string"},
                "topic": {"type": "string", "enum": self.topic_ids},
                "intent": {"type": "string", "enum": self.intent_ids},
            },
            "required": ["description", "subject", "topic", "intent"],
            "additionalProperties": False,
        }


def _entries(document: dict, section: str, path: Path) -> list[dict]:
    """The non-empty list of mappings under ``section``, each with a snake-case ``id``."""
    entries = document.get(section)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"topic taxonomy at {path} needs a non-empty {section!r} list.")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: every {section!r} entry must be a mapping.")
        if not LABEL_ID_PATTERN.match(str(entry.get("id") or "")):
            raise ValueError(f"{path}: {section!r} id {entry.get('id')!r} is not lower snake case.")
    return entries


def _text(entry: dict, field: str, section: str, path: Path) -> str:
    """A required, non-blank text field of one entry."""
    value = str(entry.get(field) or "").strip()
    if not value:
        raise ValueError(f"{path}: {section!r} entry {entry['id']!r} has no {field!r}.")
    return value


def load_taxonomy(path=None) -> Taxonomy:
    """Read and validate the taxonomy file (default :data:`TAXONOMY_FILE`).

    Failures are loud rather than defaulted, as for the utility judge's rubric: a missing file,
    unreadable YAML, a missing section, a malformed or duplicated id, a topic in no known group or
    a group with no topic all raise. A labeller running without its instructions, or with two
    topics sharing an id, would return labels that look like measurements and are not -- each paid
    for and then cached.
    """
    path = Path(path) if path is not None else TAXONOMY_FILE
    try:
        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"topic taxonomy not found at {path}.") from None
    except yaml.YAMLError as error:
        raise ValueError(f"topic taxonomy at {path} is not valid YAML: {error}") from None
    if not isinstance(document, dict):
        raise ValueError(f"topic taxonomy at {path} must be a mapping.")

    instructions = str(document.get("instructions") or "").strip()
    if not instructions:
        raise ValueError(f"topic taxonomy at {path} has no non-empty 'instructions' key.")
    groups = tuple(TopicGroup(entry["id"], _text(entry, "name", "topic_groups", path))
                   for entry in _entries(document, "topic_groups", path))
    group_ids = {group.id for group in groups}
    topics = []
    for entry in _entries(document, "topics", path):
        group = str(entry.get("group") or "")
        if group not in group_ids:
            raise ValueError(f"{path}: topic {entry['id']!r} names unknown group {group!r}.")
        examples = entry.get("examples") or []
        if not isinstance(examples, list) or not all(str(e).strip() for e in examples):
            raise ValueError(f"{path}: topic {entry['id']!r} examples must be non-blank strings.")
        topics.append(Topic(entry["id"], group, _text(entry, "description", "topics", path),
                            tuple(str(example).strip() for example in examples)))
    intents = tuple(Label(entry["id"], _text(entry, "description", "intents", path))
                    for entry in _entries(document, "intents", path))
    taxonomy = Taxonomy(instructions, tuple(topics), groups, intents)

    empty_groups = sorted(group_ids - set(taxonomy.group_of().values()))
    if empty_groups:
        raise ValueError(f"{path}: topic groups with no topic: {', '.join(empty_groups)}.")
    # Ids are unique across groups, topics and intents, so a combined analysis can never confuse
    # one kind of label with another.
    every_id = [group.id for group in groups] + taxonomy.topic_ids + taxonomy.intent_ids
    duplicated = sorted(label for label, count in Counter(every_id).items() if count > 1)
    if duplicated:
        raise ValueError(f"{path}: ids used more than once: {', '.join(duplicated)}.")
    return taxonomy


# --- the input ----------------------------------------------------------------

def render_conversation(turns, *, max_turn_chars: int | None = DEFAULT_MAX_TURN_CHARS,
                        max_chars: int | None = DEFAULT_MAX_CHARS) -> tuple[str, bool]:
    """``(text, truncated)``: a conversation's user turns as the model sees them.

    ::

        <conversation>
        <turn_1>first user message</turn_1>
        <turn_2>second user message [...]</turn_2>
        [... 3 more messages not shown]
        </conversation>

    Blank turns are dropped and the rest numbered in order. Each turn is cut at ``max_turn_chars``
    and the rendering stops once ``max_chars`` of turn text is spent, so a single pasted file cannot
    crowd out every turn after it; either cap is off when ``None``. ``truncated`` says whether
    anything was cut. A conversation with no non-blank turn renders as ``""`` -- nothing to label,
    and no request is made for it. The text is the label cache's key, so changing this rendering
    re-labels everything; bump :data:`TOPIC_LABEL_VERSION` if it ever has to change.
    """
    kept = [str(turn).strip() for turn in turns]
    kept = [turn for turn in kept if turn]
    if not kept:
        return "", False
    budget = math.inf if max_chars is None else int(max_chars)
    per_turn = math.inf if max_turn_chars is None else int(max_turn_chars)
    parts = ["<conversation>"]
    truncated = False
    for number, turn in enumerate(kept, start=1):
        if budget <= 0:
            parts.append(OMITTED_TURNS_LINE.format(count=len(kept) - number + 1))
            truncated = True
            break
        limit = min(per_turn, budget)
        if len(turn) > limit:
            turn = turn[:int(limit)] + TRUNCATION_MARKER
            truncated = True
        budget -= len(turn)
        parts.append(f"<turn_{number}>{turn}</turn_{number}>")
    parts.append("</conversation>")
    return "\n".join(parts), truncated


def iter_turns(path: Path, positions):
    """``(position, turns)`` for the wanted row positions of a split, in file order.

    Streams the ``turns`` column a batch at a time, the same way
    :func:`~prompt_anonymity.data.apply_defenses.read_turns` does and for the same reason --
    ``turns`` is essentially the whole file -- but yields as it goes, so the caller can render and
    label one chunk while the rest is still on disk.
    """
    wanted = {int(position) for position in positions}
    first_row = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=READ_BATCH_ROWS, columns=["turns"]):
        column = batch.column(0)
        for offset in range(len(batch)):
            if first_row + offset in wanted:
                yield first_row + offset, [str(turn) for turn in (column[offset].as_py() or [])]
        first_row += len(batch)


def cache_key(conversation: str) -> str:
    """The cache key of one rendered conversation: its SHA-256.

    The cache stores each entry's key beside its output, so keying by the hash rather than the
    text keeps the entries small; the namespace already pins everything else the label depends on.
    """
    return hashlib.sha256(conversation.encode("utf-8")).hexdigest()


# --- the client ---------------------------------------------------------------

def foundry_credential(name: str) -> str:
    """One credential, from the environment or from the project's ``.env``.

    Reads the single key with ``dotenv_values`` rather than ``load_dotenv()``, which would write
    every key in the file into this process's environment (see
    :func:`prompt_anonymity.attacks.llm._foundry.foundry_credential`, which this mirrors so the
    data pipeline does not import the attack stack).
    """
    value = os.environ.get(name)
    if value:
        return value
    from dotenv import dotenv_values, find_dotenv

    for path in (find_dotenv(), find_dotenv(usecwd=True)):
        if path:
            found = (dotenv_values(path) or {}).get(name)
            if found:
                return found
    raise RuntimeError(
        f"{name} is not set and is not in the project's .env. The topic labeller needs both "
        f"{FOUNDRY_ENDPOINT_ENV} (the Foundry endpoint) and {FOUNDRY_API_KEY_ENV} (its key)."
    )


class FoundryTopicLabeler:
    """Label rendered conversations with Claude on Microsoft Foundry: one cacheable record each.

    The client is built on the first real request, so a run whose labels are all cached needs no
    credentials and makes no connection.

    Parameters
    ----------
    taxonomy : Taxonomy
        Instructions and label sets; the system prompt and the response schema are rendered from it.
    deployment : str
        Foundry deployment name (default :data:`DEFAULT_DEPLOYMENT`).
    thinking : bool
        ``False`` (default) sends ``thinking={"type": "disabled"}``; ``True`` sends adaptive
        thinking. Haiku 5.5 thinks unless told not to with this parameter.
    effort : str
        ``output_config.effort``. With thinking off it must be one of
        :data:`EFFORTS_WITHOUT_THINKING`.
    max_tokens : int, optional
        Output budget per request; defaults by ``thinking`` (:data:`MAX_TOKENS_WITHOUT_THINKING` /
        :data:`MAX_TOKENS_WITH_THINKING`).
    workers : int
        Concurrent requests in :meth:`stream`.

    Attributes
    ----------
    usage : RunUsage
        Tokens and dollars of the requests this object has made; read it after :meth:`stream`
        has drained.
    """

    def __init__(self, taxonomy: Taxonomy, *, deployment: str = DEFAULT_DEPLOYMENT,
                 thinking: bool = False, effort: str = DEFAULT_EFFORT,
                 max_tokens: int | None = None, workers: int = DEFAULT_WORKERS):
        allowed = EFFORTS_WITH_THINKING if thinking else EFFORTS_WITHOUT_THINKING
        if effort not in allowed:
            raise ValueError(f"effort {effort!r} is not accepted with thinking "
                             f"{'on' if thinking else 'off'}; choose one of {', '.join(allowed)}.")
        self.taxonomy = taxonomy
        self.deployment = deployment
        self.thinking = thinking
        self.effort = effort
        self.max_tokens = int(max_tokens or (MAX_TOKENS_WITH_THINKING if thinking
                                             else MAX_TOKENS_WITHOUT_THINKING))
        self.workers = max(1, int(workers))
        self.system_prompt = taxonomy.system_prompt()
        self.response_schema = taxonomy.response_schema()
        self._topic_ids = set(taxonomy.topic_ids)
        self._intent_ids = set(taxonomy.intent_ids)
        self._client = None
        self._client_lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self.usage = RunUsage()
        # Rejections since the last request that got a reply; see MAX_CONSECUTIVE_REJECTIONS.
        self._consecutive_rejections = 0

    def params(self) -> dict:
        """Everything that changes what the model is sent or asked for -- the cache namespace.

        Concurrency, retries and timeouts are left out: they change how requests are made, not
        what any of them returns.
        """
        return {
            "deployment": self.deployment,
            "system_prompt": self.system_prompt,
            "response_schema": self.response_schema,
            "thinking": self.thinking,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
        }

    def _ensure_client(self):
        """The Foundry client, built once on first use (thread-safe)."""
        with self._client_lock:
            if self._client is None:
                from anthropic import AnthropicFoundry

                self._client = AnthropicFoundry(
                    api_key=foundry_credential(FOUNDRY_API_KEY_ENV),
                    base_url=foundry_credential(FOUNDRY_ENDPOINT_ENV),
                    max_retries=FOUNDRY_MAX_RETRIES,
                    timeout=FOUNDRY_TIMEOUT_SECONDS,
                )
                print(f"[topics] Foundry deployment {self.deployment!r}, thinking "
                      f"{'adaptive' if self.thinking else 'off'}, effort {self.effort!r}")
        return self._client

    def request(self, conversation: str) -> dict:
        """The ``messages.create`` arguments for one rendered conversation.

        No ``temperature``/``top_p``/``top_k`` and no assistant prefill: both are rejected by this
        model. ``cache_control`` marks the system block only (see the module docstring).
        """
        return {
            "model": self.deployment,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": conversation}],
            "thinking": {"type": "adaptive"} if self.thinking else {"type": "disabled"},
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": self.response_schema},
            },
        }

    def label(self, conversation: str) -> dict:
        """One conversation's label record (see :meth:`read_reply`), from one request.

        Transient failures (connection errors, 408/409/429, 5xx) are retried by the SDK
        (:data:`FOUNDRY_MAX_RETRIES`) and raise if they outlast it. Any other 4xx is either a
        configuration error, which raises (see :meth:`_rejected`), or a rejection of this one
        input, which comes back as a ``rejected`` record.
        """
        import anthropic

        client = self._ensure_client()
        try:
            message = client.messages.create(**self.request(conversation))
        except anthropic.APIStatusError as err:
            if 400 <= err.status_code < 500 and err.status_code not in (408, 409, 429):
                return self._rejected(conversation, err)
            raise
        with self._usage_lock:
            self._consecutive_rejections = 0
        return self.read_reply(message)

    def _rejected(self, conversation: str, err) -> dict:
        """The record of a request the endpoint refused, or a raise if it is a configuration error.

        Raises at once on :data:`CONFIGURATION_ERROR_STATUSES`, and on the
        :data:`MAX_CONSECUTIVE_REJECTIONS`-th rejection in a row, with the input's length and
        opening attached -- the same request would fail the same way for every other document.
        """
        snippet = conversation[:200].replace("\n", " ")
        problem = (f"Foundry rejected the request (HTTP {err.status_code}) for deployment "
                   f"{self.deployment!r}. Input: {len(conversation)} chars, starts {snippet!r}. "
                   f"Error: {err.message}")
        if err.status_code in CONFIGURATION_ERROR_STATUSES:
            raise RuntimeError(f"{problem} -- check {FOUNDRY_ENDPOINT_ENV}, {FOUNDRY_API_KEY_ENV} "
                               "and --deployment.") from err
        with self._usage_lock:
            self._consecutive_rejections += 1
            streak = self._consecutive_rejections
        if streak >= MAX_CONSECUTIVE_REJECTIONS:
            raise RuntimeError(f"{streak} requests in a row rejected; stopping, since this looks "
                               f"like a request no document can pass. Last one: {problem}") from err
        record = empty_record()
        record.update(status="rejected",
                      rejection_reason=f"HTTP {err.status_code}: {str(err.message)[:500]}")
        return record

    def read_reply(self, message) -> dict:
        """The cacheable record of one response: status, labels, token counts and cost.

        Keys: ``status`` (see the module docstring), ``description``/``subject``/``topic``/
        ``intent`` (``None`` unless labelled), ``stop_reason``, ``refusal_category``, ``rejection_reason`` (always
        ``None`` here: a rejected request has no reply), ``raw_reply`` (kept only when the reply did
        not parse, to see why), the four :data:`USAGE_FIELDS` and ``cost_usd``. The spend is
        recorded first: the tokens were billed whatever the reply turns out to be.
        """
        usage_block = getattr(message, "usage", None)
        usage = {name: int(getattr(usage_block, name, 0) or 0) for name in USAGE_FIELDS}
        cost = request_cost(self.deployment, usage)
        with self._usage_lock:
            self.usage.add(usage, cost)

        stop_reason = getattr(message, "stop_reason", None)
        record = {"status": "labeled", "description": None, "subject": None, "topic": None,
                  "intent": None,
                  "stop_reason": stop_reason, "refusal_category": None, "rejection_reason": None,
                  "raw_reply": None, **usage, "cost_usd": cost}
        if stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            record.update(status="refused", refusal_category=getattr(details, "category", None))
            return record
        if stop_reason == "max_tokens":
            record["status"] = "max_tokens"
            return record

        text = "".join(block.text for block in (getattr(message, "content", None) or [])
                       if getattr(block, "type", None) == "text")
        try:
            reply = json.loads(text)
        except ValueError:
            reply = None
        valid = (isinstance(reply, dict) and reply.get("topic") in self._topic_ids
                 and reply.get("intent") in self._intent_ids
                 and isinstance(reply.get("subject"), str)
                 and isinstance(reply.get("description"), str))
        if not valid:
            record.update(status="invalid", raw_reply=text[:2000])
            return record
        record.update(description=reply["description"].strip(), subject=reply["subject"].strip(),
                      topic=reply["topic"], intent=reply["intent"])
        return record

    def stream(self, conversations: list[str]):
        """Label ``conversations`` concurrently, yielding ``(index, record)`` as each completes.

        The shape :meth:`~prompt_anonymity.caching.TransformCache.apply_streaming` consumes, so
        every label is cached the moment it lands. A request that still fails after the SDK's
        retries, a configuration error, or a SIGTERM (see :func:`main`) raises here; the requests
        not yet started are cancelled and the ones in flight are **abandoned, not waited for** --
        their labels are lost (at most ``workers`` of them) and their tokens may miss this run's
        ledger row, but everything yielded before is already cached. Not waiting is what lets a
        preempted job write its ledger row inside the scheduler's grace period: one in-flight
        request can sit in SDK retry backoff for minutes.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        pool = ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(conversations))))
        try:
            futures = {pool.submit(self.label, conversation): index
                       for index, conversation in enumerate(conversations)}
            for future in as_completed(futures):
                yield futures[future], future.result()
        finally:
            # On normal completion every future is already done, so not waiting changes nothing.
            pool.shutdown(wait=False, cancel_futures=True)


def open_label_cache(labeler: FoundryTopicLabeler, root) -> TransformCache:
    """The label cache under ``<root>/topics/topic_labels``, namespaced by version and params.

    ``prune_stale=False``: :class:`~prompt_anonymity.caching.TransformCache` would otherwise delete
    every other logic-version namespace on open, i.e. all the labels bought before the last
    :data:`TOPIC_LABEL_VERSION` bump.
    """
    return TransformCache(Path(root) / CACHE_SUBDIR, CACHE_NAME,
                          logic_hash([], version=TOPIC_LABEL_VERSION),
                          params_hash(labeler.params()), prune_stale=False)


# --- the run ------------------------------------------------------------------

def sample_documents(documents: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    """A seeded random sample of ``limit`` documents, in split order (all of them when ``None``).

    Random rather than a head slice: the splits are grouped by identity, so the first N rows can be
    one or two people. The index stays each document's row position in the split, which is what
    :func:`iter_turns` reads against.
    """
    if limit is None or limit >= len(documents):
        return documents
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(documents), size=int(limit), replace=False))
    return documents.iloc[positions]


def output_stem(source: str, *, deployment: str, thinking: bool, effort: str,
                max_turn_chars: int | None, max_chars: int | None, taxonomy_path: Path,
                limit: int | None, seed: int) -> str:
    """The output's file name without ``.parquet``: ``<split>_topics`` plus any non-default choice.

    A non-default deployment, thinking, effort, input caps or taxonomy file is appended, and so is
    a sample (``_n300_seed47``), so a pilot can never overwrite -- or be mistaken for -- the labels
    of the whole split under the default configuration, and two arms of a comparison sit side by
    side.
    """
    parts = [source, OUTPUT_LABEL]
    if deployment != DEFAULT_DEPLOYMENT:
        parts.append(deployment)
    if thinking:
        parts.append(f"think-{effort}")
    elif effort != DEFAULT_EFFORT:
        parts.append(f"effort-{effort}")
    if (max_turn_chars, max_chars) != (DEFAULT_MAX_TURN_CHARS, DEFAULT_MAX_CHARS):
        parts.append(f"chars{max_turn_chars or 'all'}-{max_chars or 'all'}")
    if taxonomy_path.resolve() != TAXONOMY_FILE.resolve():
        parts.append(f"tax-{taxonomy_path.stem}")
    if limit is not None:
        parts.append(f"n{limit}_seed{seed}")
    return "_".join(parts)


def empty_record() -> dict:
    """The record of a document with no non-blank turn: no request, no cost."""
    return {"status": "empty", "description": None, "subject": None, "topic": None, "intent": None,
            "stop_reason": None, "refusal_category": None, "rejection_reason": None,
            "raw_reply": None, **{name: 0 for name in USAGE_FIELDS}, "cost_usd": 0.0}


def label_documents(documents: pd.DataFrame, path: Path, labeler: FoundryTopicLabeler,
                    cache: TransformCache, *, max_turn_chars: int | None, max_chars: int | None,
                    chunk_size: int = DEFAULT_CHUNK_SIZE) -> pd.DataFrame:
    """Label ``documents`` (rows of the split at ``path``) and return the output frame.

    Renders and labels ``chunk_size`` documents at a time as their turns stream off disk; within a
    chunk, only inputs missing from the cache are sent, each identical input once, and each label
    is cached as it arrives. Rows come back in split order with :data:`LABEL_COLUMNS`, which the
    caller splits into the dataset file and the provenance file.
    """
    group_of = labeler.taxonomy.group_of()
    author_of = dict(zip(documents.index, documents["author_id"]))
    doc_id_of = dict(zip(documents.index, documents["doc_id"]))
    rows: list[dict] = []
    progress = tqdm(total=len(documents), desc="  labelling", unit="doc")
    # Documents with identical rendered text share one request and one cache entry. Its tokens and
    # dollars are charged to the first such document only, so the provenance file sums to what
    # the labels cost rather than counting a shared request once per copy.
    charged: set[str] = set()

    def label_chunk(chunk: list[tuple[int, list[str]]]) -> None:
        rendered = [render_conversation(turns, max_turn_chars=max_turn_chars,
                                        max_chars=max_chars) for _, turns in chunk]
        to_label = [text for text, _ in rendered if text]

        def on_progress(done: int, total: int) -> None:
            # The bar advances a chunk at a time; this shows the requests moving inside one.
            progress.set_postfix_str(f"chunk: {done:,}/{total:,} sent", refresh=True)

        # A rejected request is reported in this run's output but not cached, so the next run
        # retries it.
        records = iter(cache.apply_streaming(
            to_label, labeler.stream, key=cache_key, on_progress=on_progress,
            persist=lambda record: record["status"] != "rejected"))
        for (position, _), (text, truncated) in zip(chunk, rendered):
            record = next(records) if text else empty_record()
            if text:
                key = cache_key(text)
                if key in charged:
                    record = {**record, **{name: 0 for name in USAGE_FIELDS}, "cost_usd": 0.0}
                charged.add(key)
            rows.append({
                "doc_id": doc_id_of[position],
                "author_id": author_of[position],
                "topic": record["topic"],
                "topic_group": group_of.get(record["topic"]) if record["topic"] else None,
                "intent": record["intent"],
                "subject": record["subject"],
                "description": record["description"],
                "label_status": record["status"],
                "refusal_category": record["refusal_category"],
                "rejection_reason": record["rejection_reason"],
                "input_chars": len(text),
                "input_truncated": truncated,
                **{name: record[name] for name in USAGE_FIELDS},
                "label_cost_usd": record["cost_usd"],
            })
        progress.update(len(chunk))

    chunk: list[tuple[int, list[str]]] = []
    for position, turns in iter_turns(path, documents.index):
        chunk.append((position, turns))
        if len(chunk) >= max(1, int(chunk_size)):
            label_chunk(chunk)
            chunk = []
    if chunk:
        label_chunk(chunk)
    progress.close()
    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def dry_run(documents: pd.DataFrame, path: Path, labeler: FoundryTopicLabeler,
            cache: TransformCache, *, max_turn_chars: int | None, max_chars: int | None) -> None:
    """Render every input and print what a real run would send, with a rough cost estimate.

    Makes no request and needs no credential. The estimate converts the bytes of every uncached
    input at :data:`DRY_RUN_BYTES_PER_TOKEN`, prices the system prompt as a cache read on every
    request but the first, and assumes :data:`DRY_RUN_OUTPUT_TOKENS` per reply -- an order of
    magnitude to decide on, not a quote. The first pilot's ``usage`` is the real number.
    """
    n_empty = n_truncated = n_cached = 0
    pending: set[str] = set()
    pending_bytes = 0
    for _, turns in tqdm(iter_turns(path, documents.index), total=len(documents),
                         desc="  rendering", unit="doc"):
        text, truncated = render_conversation(turns, max_turn_chars=max_turn_chars,
                                              max_chars=max_chars)
        n_truncated += truncated
        if not text:
            n_empty += 1
            continue
        key = cache_key(text)
        if cache.get(key) is not None:
            n_cached += 1
        elif key not in pending:
            pending.add(key)
            pending_bytes += len(text.encode("utf-8"))

    n_requests = len(pending)
    print(f"[dry run] {len(documents):,} documents: {n_cached:,} already labelled, {n_empty:,} "
          f"empty, {n_requests:,} distinct inputs to send; {n_truncated:,} cut by the input caps")
    cards = DEPLOYMENT_RATES.get(labeler.deployment)
    if not n_requests:
        print("[dry run] nothing to send -- $0.00")
        return
    if cards is None:
        print(f"[dry run] {labeler.deployment!r} has no entry in DEPLOYMENT_RATES; no estimate.")
        return
    rates = cards[0]
    system_bytes = len(labeler.system_prompt.encode("utf-8"))
    estimates = []
    for bytes_per_token in sorted(DRY_RUN_BYTES_PER_TOKEN, reverse=True):
        input_tokens = pending_bytes / bytes_per_token
        system_tokens = system_bytes / bytes_per_token
        estimates.append((input_tokens * rates.input
                          + system_tokens * rates.cache_write
                          + system_tokens * (n_requests - 1) * rates.cache_read
                          + n_requests * DRY_RUN_OUTPUT_TOKENS * rates.output) / 1e6)
    low, high = estimates
    print(f"[dry run] {pending_bytes / 1e6:,.1f} MB of conversation text + a "
          f"{system_bytes:,}-byte cached system prompt per request")
    print(f"[dry run] rough cost ${low:,.2f}-${high:,.2f} at "
          f"{DRY_RUN_BYTES_PER_TOKEN[0]:g}-{DRY_RUN_BYTES_PER_TOKEN[1]:g} bytes/token, thinking "
          f"{'adaptive (output not estimated: thinking tokens are extra)' if labeler.thinking else 'off'}")


def write_provenance(frame: pd.DataFrame, path: Path) -> None:
    """Write the per-document provenance CSV atomically: a temp file beside it, then a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    frame[PROVENANCE_COLUMNS].to_csv(tmp, index=False)
    os.replace(tmp, path)


def append_spend(path: Path, row: dict) -> None:
    """Append one run's row to the spend ledger, writing the header when the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPEND_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def report(frame: pd.DataFrame, dataset_path: Path, provenance_path: Path,
           labeler: FoundryTopicLabeler) -> None:
    """Log what was written: statuses, the topic groups, this run's spend and the labels' total."""
    print(f"wrote {len(frame):,} topic labels -> {dataset_path} "
          f"({dataset_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote their provenance and cost -> {provenance_path}")
    statuses = frame["label_status"].value_counts()
    print("  status: " + ", ".join(f"{status} {count:,}" for status, count in statuses.items()))
    refused = frame.loc[frame["label_status"] == "refused", "refusal_category"]
    if len(refused):
        categories = refused.fillna("none").value_counts()
        print("  refusal categories: " + ", ".join(f"{c} {n:,}" for c, n in categories.items()))
    groups = frame["topic_group"].value_counts(normalize=True)
    print("  topic groups: " + ", ".join(f"{group} {share:.1%}" for group, share in groups.items()))
    print(f"  {labeler.usage.summary(labeler.deployment)}")
    total = frame["label_cost_usd"].sum(min_count=1)
    unpriced = int(frame["label_cost_usd"].isna().sum())
    print(f"  these labels cost ${total:,.4f} in all, whenever bought"
          + (f" ({unpriced:,} rows unpriced)" if unpriced else ""))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="swe_chat", choices=sorted(SOURCES),
                   help="which built split to label (default: swe_chat)")
    p.add_argument("--limit", type=int, default=None,
                   help="label only a seeded random sample of this many documents (a pilot); the "
                        "output name records the sample, so it never overwrites the full labels")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"seed for --limit (default: {DEFAULT_SEED})")
    p.add_argument("--dry-run", action="store_true",
                   help="render every input and print what a real run would send and roughly "
                        "cost; no request, no credential, nothing written")
    p.add_argument("--deployment", default=DEFAULT_DEPLOYMENT,
                   help=f"Foundry deployment name (default: {DEFAULT_DEPLOYMENT})")
    p.add_argument("--thinking", choices=("off", "adaptive"), default="off",
                   help="'off' (default) disables thinking, which this model otherwise does by "
                        "default and bills as output; 'adaptive' turns it on")
    p.add_argument("--effort", default=DEFAULT_EFFORT, choices=EFFORTS_WITH_THINKING,
                   help=f"output_config effort (default: {DEFAULT_EFFORT}); xhigh and max need "
                        "--thinking adaptive")
    p.add_argument("--max-tokens", type=int, default=None,
                   help=f"output budget per request (default: {MAX_TOKENS_WITHOUT_THINKING} with "
                        f"thinking off, {MAX_TOKENS_WITH_THINKING} with it on)")
    p.add_argument("--max-turn-chars", type=int, default=DEFAULT_MAX_TURN_CHARS,
                   help=f"cut each user turn to this many characters; 0 keeps them whole "
                        f"(default: {DEFAULT_MAX_TURN_CHARS:,})")
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                   help=f"stop rendering a conversation after this many characters of turn text; "
                        f"0 keeps it whole (default: {DEFAULT_MAX_CHARS:,})")
    p.add_argument("--taxonomy", default=None,
                   help="another taxonomy YAML, for an A/B of the instructions or the label sets "
                        "(default: the packaged topic_taxonomy.yaml); named in the output file")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"concurrent requests (default: {DEFAULT_WORKERS})")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help=f"documents rendered and labelled per round (default: {DEFAULT_CHUNK_SIZE:,})")
    p.add_argument("--dist-dir", default=None,
                   help="directory holding the built split to read (default: the project's data/hf)")
    p.add_argument("--out-dir", default=None,
                   help="where to write the labels (default: the project's data/dist)")
    p.add_argument("--cache-dir", default=None,
                   help="label cache root (default: data/.cache); holds paid labels, never pruned")
    p.add_argument("--usage-dir", default=None,
                   help=f"where the per-document provenance CSV and the {SPEND_LEDGER} ledger go "
                        f"(default: <project root>/{USAGE_SUBDIR}); kept out of data/")
    args = p.parse_args()

    thinking = args.thinking == "adaptive"
    max_turn_chars = args.max_turn_chars or None
    max_chars = args.max_chars or None
    taxonomy_path = Path(args.taxonomy) if args.taxonomy else TAXONOMY_FILE
    taxonomy = load_taxonomy(taxonomy_path)
    try:
        labeler = FoundryTopicLabeler(taxonomy, deployment=args.deployment, thinking=thinking,
                                      effort=args.effort, max_tokens=args.max_tokens,
                                      workers=args.workers)
    except ValueError as error:
        raise SystemExit(str(error)) from None

    # Paths default to the project's data/ folder (see prompt_anonymity.data.config).
    dist = Path(args.dist_dir) if args.dist_dir else hf_dir()
    out_dir = Path(args.out_dir) if args.out_dir else dist_dir()
    usage_dir = Path(args.usage_dir) if args.usage_dir else project_root() / USAGE_SUBDIR
    cache = open_label_cache(labeler, Path(args.cache_dir) if args.cache_dir else cache_dir())
    path = split_path(args.source, dist)

    # Only the key columns: `turns` is streamed back a batch at a time by `iter_turns`.
    documents = sample_documents(load_split(args.source, dist, columns=["doc_id", "author_id"]),
                                 args.limit, args.seed)
    print(f"[{args.source}] {len(documents):,} documents selected"
          + (f" (seeded sample, seed {args.seed})" if args.limit is not None else "")
          + f"; {len(taxonomy.topic_ids)} topics in {len(taxonomy.groups)} groups, "
            f"{len(taxonomy.intent_ids)} intents")
    print(f"[cache] {cache.dir}")

    if args.dry_run:
        dry_run(documents, path, labeler, cache, max_turn_chars=max_turn_chars,
                max_chars=max_chars)
        return

    stem = output_stem(args.source, deployment=args.deployment, thinking=thinking,
                       effort=args.effort, max_turn_chars=max_turn_chars, max_chars=max_chars,
                       taxonomy_path=taxonomy_path, limit=args.limit, seed=args.seed)
    outcome = "aborted"
    # SIGTERM (a SLURM preemption, a time limit, scancel) would otherwise end the process without
    # running the `finally` below, losing this run's row in the spend ledger. Raising SystemExit
    # turns it into an ordinary abort; the labels bought so far are in the cache either way.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(f"stopped by signal {signum}"))
    try:
        frame = label_documents(documents, path, labeler, cache, max_turn_chars=max_turn_chars,
                                max_chars=max_chars, chunk_size=args.chunk_size)
        dataset_path = out_dir / f"{stem}.parquet"
        provenance_path = usage_dir / f"{stem}.csv"
        write_parquet(frame[DATASET_COLUMNS], dataset_path)
        write_provenance(frame, provenance_path)
        outcome = "completed"
        report(frame, dataset_path, provenance_path, labeler)
    finally:
        # Requests made before a failure were billed, so the ledger is written either way.
        if labeler.usage.requests:
            append_spend(usage_dir / SPEND_LEDGER, {
                "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "outcome": outcome, "source": args.source, "output_stem": stem,
                "deployment": labeler.deployment, "thinking": labeler.thinking,
                "effort": labeler.effort, "n_documents": len(documents),
                "requests": labeler.usage.requests,
                **{name: getattr(labeler.usage, name) for name in USAGE_FIELDS},
                "cost_usd": labeler.usage.cost_usd,
            })
            print(f"[spend] this run's ${labeler.usage.cost_usd:.4f} recorded in "
                  f"{usage_dir / SPEND_LEDGER}")


if __name__ == "__main__":
    main()
