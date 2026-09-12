"""OpenRouter's asynchronous Batch API: the same prompts, at roughly half the price.

:class:`~prompt_anonymity.attacks.llm._openrouter.OpenRouterChat` answers prompts in real time by
fanning a thread pool at the chat-completions endpoint. That is the right shape for a defense
rewriting turns inside a job, and the wrong one for a reranking attack: a reranker's prompts are all
known up front, nothing downstream is waiting on any individual verdict, and the corpus is large
enough that the bill is the binding constraint. OpenRouter bills batched requests at **~50% of the
model's standard per-token price** with a 24-hour completion window, which is exactly that trade.

This client is a sibling of ``OpenRouterChat`` rather than a mode of it. ``OpenRouterChat`` is
imported by both existing judge attacks *and* by the Frame Shift defense across package boundaries,
and its contract -- call it, get an answer -- should not quietly grow a path that blocks for a day.

The API
-------
::

    POST /api/beta/batches       {endpoint, model, requests: [{custom_id, body}, ...]}
    GET  /api/beta/batches/:id   -> validating -> in_progress -> finalizing -> completed
                                    (terminal: completed | failed | expired | cancelled)

``endpoint`` and ``model`` must be serialized **before** ``requests`` in the JSON body, so the
payload is built in that order and handed over as a pre-serialized string rather than a dict.
Results come back inline in the completed batch object's ``results`` array, **in arbitrary order**,
each keyed by the ``custom_id`` it was submitted under -- never by position.

Surviving the wall clock
------------------------
A 24-hour window is longer than any SLURM allocation this project runs in, so a submitted batch must
outlive the process that submitted it. Before submitting, this client writes a *ticket* -- the batch
id, keyed by a hash of the exact prompts it covers -- under the cache directory. On re-entry with
the same prompts it resumes polling that batch instead of submitting a second copy of it. Nothing
reaches the :class:`~prompt_anonymity.caching.TransformCache` until results land, so a killed job
re-enters with an identical set of cache misses and therefore an identical ticket hash. Without
this, every re-run of an interrupted batch pays for the whole thing again.

``wait=False`` submits, records the tickets and stops, which is what a login node wants: come back
tomorrow and run the same command to collect.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

#: OpenRouter's batch endpoints (beta namespace, distinct from the v1 chat-completions URL).
OPENROUTER_BATCH_URL = "https://openrouter.ai/api/beta/batches"
#: The per-request endpoint each batched body is executed against.
OPENROUTER_CHAT_ENDPOINT = "/v1/chat/completions"
#: Environment variable holding the key used for the batched reranker (loaded from a ``.env``).
#: Deliberately *not* ``OPENROUTER_API_KEY``: the rerank attack's spend is meant to be separable
#: from the defenses' and the other judges', which is easier with its own key than with an invoice.
SONNET_API_KEY_ENV = "SONNET_OR_KEY"

#: Statuses that mean the batch is finished, one way or another.
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled", "canceled"}

#: Exit status for "the batch is queued; come back and collect it" under ``wait=False``.
#:
#: Distinct from 1 on purpose. A submit job that exits 1 is indistinguishable from one that crashed,
#: so SLURM would mark every successful submission FAILED and a script could not tell a queued batch
#: from a broken build -- which is the difference between "wait an hour" and "something is wrong and
#: you are about to pay for it again". Callers branch: 0 = nothing to submit (all cached), 3 =
#: queued, anything else = a real failure.
BATCH_QUEUED_EXIT = 3

#: Variant suffix selecting OpenRouter's batch-priced copy of a model.
#:
#: **This is not cosmetic, and it is not implied by posting to the batch endpoint.**
#: ``anthropic/claude-sonnet-5`` and ``anthropic/claude-sonnet-5:batch`` are two separate slugs in
#: ``GET /api/v1/models``, priced $2/$10 and $1/$5 per MTok respectively -- exactly the 50% the
#: batch discount is supposed to be. The quickstart's own example uses a plain slug, so submitting
#: one is accepted; it just runs the batch at full price. That is a silent overcharge rather than an
#: error, which is why the suffix is applied here instead of being left to whoever names the model.
#:
#: The variant keeps ``reasoning`` and ``reasoning_effort`` among its supported parameters, so
#: extended thinking survives the swap -- verified against the live model list, not assumed.
BATCH_VARIANT_SUFFIX = ":batch"


def batch_slug(model: str, suffix: str | None = BATCH_VARIANT_SUFFIX) -> str:
    """``model`` as its batch-priced variant, or unchanged when it already names one.

    A slug that already carries a variant (``:batch``, ``:free``, ``:nitro``, ...) is left alone --
    the caller has been explicit -- as is every model when ``suffix`` is ``None``, the escape hatch
    for a model that has no batch variant and would 400 on one.
    """
    if not suffix or ":" in model.rsplit("/", 1)[-1]:
        return model
    return model + suffix


class OpenRouterBatch:
    """Submit many prompts as one OpenRouter batch, then collect them in input order.

    Built lazily by its caller on first real need, so a fully-cached run constructs no client and
    needs no API key.

    Parameters
    ----------
    model : str
        OpenRouter model slug (e.g. ``"anthropic/claude-sonnet-5"``).
    system_prompt : str
        System message prepended to every request in the batch.
    temperature, top_p : float or None
        Sampling controls, **omitted from the payload when ``None``, which is the default**.
        Unlike :class:`~prompt_anonymity.attacks.llm._openrouter.OpenRouterChat`, which always sends
        them, this client leaves them out: Claude Sonnet 5 removed the sampling controls entirely
        (``temperature`` and ``top_p`` are absent from its ``supported_parameters`` on OpenRouter,
        and Anthropic answers a 400 for them), and a batch that 400s is a day lost rather than a
        retry. A thinking model is not made deterministic by ``temperature=0`` anyway, so nothing is
        given up; set them explicitly for a model that does take them.
    max_tokens : int
        Output budget. Thinking tokens count against it.
    reasoning_effort : str or None
        ``"low"``/``"medium"``/``"high"``/``"xhigh"``/``"max"``, sent as ``reasoning.effort``.
        OpenRouter maps this onto Anthropic's ``output_config.effort`` for Claude 4.6 and newer, so
        this is how extended thinking is requested on Sonnet 5. ``None`` omits the field.

        The older fixed thinking budget is **not** an option here: ``budget_tokens`` is rejected
        with a 400 on Sonnet 5, which uses adaptive thinking steered by effort instead.
    requests_per_batch : int
        Chunk size. Each chunk is its own batch with its own ticket, so a large corpus makes
        progress in pieces rather than as one all-or-nothing submission.
    poll_interval, poll_cap : float
        Seconds between status checks, growing geometrically to ``poll_cap``.
    max_retries, backoff_cap, timeout : int / float / float
        Retry budget for transient HTTP failures, backoff ceiling, and per-request timeout.
    wait : bool
        ``True`` polls until every chunk is terminal. ``False`` submits, writes the tickets and
        raises :class:`SystemExit` naming the batch ids.
    ticket_dir : str or pathlib.Path or None
        Where resume tickets live. ``None`` disables resumption entirely -- every run submits fresh,
        which is only safe for a batch small enough to lose.
    """

    def __init__(self, model: str, system_prompt: str = "", *, temperature: float | None = None,
                 top_p: float | None = None, max_tokens: int = 1024,
                 reasoning_effort: str | None = None,
                 requests_per_batch: int = 10_000, poll_interval: float = 30.0,
                 poll_cap: float = 300.0, max_retries: int = 8, backoff_cap: float = 30.0,
                 timeout: float = 120.0, wait: bool = True, ticket_dir=None,
                 variant_suffix: str | None = BATCH_VARIANT_SUFFIX,
                 base_url: str = OPENROUTER_BATCH_URL, api_key_env: str = SONNET_API_KEY_ENV):
        import requests
        from dotenv import load_dotenv

        self._requests = requests
        #: The model as configured -- what the cache is namespaced by, so a real-time smoke run and
        #: a batched run of the same prompts share their verdicts.
        self.model = model
        #: The slug actually submitted: the batch-priced variant of :attr:`model`. Different from it
        #: is the normal case, and is what earns the discount.
        self.batch_model = batch_slug(model, variant_suffix)
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.requests_per_batch = requests_per_batch
        self.poll_interval = poll_interval
        self.poll_cap = poll_cap
        self.max_retries = max_retries
        self.backoff_cap = backoff_cap
        self.timeout = timeout
        self.wait = wait
        self.base_url = base_url
        self.ticket_dir = None if ticket_dir is None else Path(ticket_dir)

        #: Token and dollar totals reported by OpenRouter for the batches this client waited on.
        #: Tokens are the record; ``cost`` is OpenRouter's own figure for what was actually billed,
        #: so a cached re-run correctly adds nothing.
        self.total_tokens = 0
        self.total_cost = 0.0

        # Walks UP from the working directory: a .env at the repo root (or above it) is found, one
        # in a SUBdirectory is not -- `DS_env/.env` does not work when the job runs from the root.
        load_dotenv()
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env} not set. Add it to a .env file at the REPO ROOT "
                f"(e.g. '{api_key_env}=sk-or-...') so the batched rerank attack can call OpenRouter."
            )
        # Printed rather than silent: the submitted slug is the one that sets the price, and a run
        # that quietly fell back to the full-price model looks identical to one that did not.
        print(f"listwise rerank: OpenRouter batch client submitting as '{self.batch_model}'"
              + (f" (configured '{model}')" if self.batch_model != model else "")
              + (f" at reasoning effort '{reasoning_effort}'." if reasoning_effort else "."))

    # --- HTTP -----------------------------------------------------------------

    def _backoff(self, attempt: int) -> None:
        delay = min(self.backoff_cap, 2 ** attempt)
        time.sleep(random.uniform(0, delay))  # full jitter de-synchronizes concurrent workers

    def _call(self, method: str, url: str, payload: str | None = None) -> dict:
        """One HTTP call with the same retry policy as the synchronous client: 429 and 5xx and
        network errors are retried with jittered exponential backoff, other 4xx fail fast with
        OpenRouter's error body attached (``raise_for_status`` alone carries only the status line).
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = self._requests.request(method, url, headers=headers, data=payload,
                                                  timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except self._requests.exceptions.HTTPError as err:
                status = err.response.status_code
                body = err.response.text
                last_err = RuntimeError(f"{status} {err.response.reason}: {body}")
                if 400 <= status < 500 and status != 429:
                    raise RuntimeError(
                        f"OpenRouter rejected the batch request (HTTP {status}) for model "
                        f"{self.batch_model!r}; not retrying. If it names the model, that slug may "
                        f"have no batch variant -- pass variant_suffix=None to submit "
                        f"{self.model!r} verbatim. Response body: {body}"
                    ) from err
                if attempt < self.max_retries - 1:
                    self._backoff(attempt)
            except Exception as err:  # noqa: BLE001 - network/JSON errors are all retryable
                last_err = err
                if attempt < self.max_retries - 1:
                    self._backoff(attempt)
        raise RuntimeError(
            f"OpenRouter batch request failed after {self.max_retries} attempts: {last_err}")

    # --- tickets --------------------------------------------------------------

    def _ticket_path(self, texts: list[str]) -> Path | None:
        """Where this exact chunk's resume ticket lives, or ``None`` when resumption is disabled.

        Keyed by the prompts themselves plus everything that changes what was submitted, so a
        ticket can only ever be reused for a byte-identical resubmission.
        """
        if self.ticket_dir is None:
            return None
        digest = hashlib.sha256("\x00".join(
            [self.batch_model, self.system_prompt, str(self.reasoning_effort),
             str(self.max_tokens), *texts]
        ).encode("utf-8")).hexdigest()
        return self.ticket_dir / f"{digest}.json"

    def _read_ticket(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)["batch_id"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError):
            return None  # a corrupt ticket is a missing ticket: submit again rather than crash

    def _write_ticket(self, path: Path | None, batch_id: str, n: int) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"batch_id": batch_id, "model": self.batch_model, "n_requests": n,
                       "submitted_at": time.time()}, handle)

    def _clear_ticket(self, path: Path | None) -> None:
        if path is not None:
            path.unlink(missing_ok=True)

    # --- batch lifecycle ------------------------------------------------------

    def _body(self, text: str) -> dict:
        """One request body. Every optional knob is omitted rather than sent as a default, because
        a parameter a model does not accept is a 400 on the whole batch -- see ``temperature``."""
        body = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    def _submit(self, texts: list[str]) -> str:
        # Serialized by hand, and in this order: OpenRouter requires `endpoint` and `model` to
        # appear before `requests` in the JSON body.
        payload = json.dumps({
            "endpoint": OPENROUTER_CHAT_ENDPOINT,
            "model": self.batch_model,
            "requests": [{"custom_id": str(index), "body": self._body(text)}
                         for index, text in enumerate(texts)],
        })
        batch = self._call("POST", self.base_url, payload)
        batch_id = batch.get("id")
        if not batch_id:
            raise RuntimeError(f"OpenRouter accepted the batch but returned no id: {batch}")
        return batch_id

    def _poll(self, batch_id: str) -> dict:
        """Block until this batch reaches a terminal status, returning the final batch object."""
        delay = self.poll_interval
        while True:
            batch = self._call("GET", f"{self.base_url}/{batch_id}")
            status = batch.get("status", "")
            if status in TERMINAL_STATUSES:
                return batch
            counts = batch.get("request_counts") or {}
            print(f"  batch {batch_id}: {status} "
                  f"({counts.get('completed', 0)}/{counts.get('total', '?')} done); "
                  f"next check in {delay:.0f}s")
            time.sleep(delay)
            delay = min(self.poll_cap, delay * 1.5)

    def _collect(self, batch: dict, n: int) -> list[str]:
        """The batch's replies in submission order, keyed back through ``custom_id``.

        A request that errored yields ``""``. The callers treat an unparseable reply as a refusal
        and fall back to the base attack's own ordering for that row, so one failed request costs
        that row's rerank and nothing else -- unlike the synchronous client, where a single failure
        aborts the whole batch. At batch scale that is the right trade: losing a day's work to one
        bad row is not.
        """
        status = batch.get("status")
        if status != "completed":
            raise RuntimeError(
                f"OpenRouter batch {batch.get('id')} ended as {status!r} rather than 'completed'. "
                f"Counts: {batch.get('request_counts')}. Re-run to resubmit."
            )
        usage = batch.get("usage") or {}
        self.total_tokens += int(usage.get("total_tokens") or 0)
        self.total_cost += float(usage.get("cost") or 0.0)

        replies = [""] * n
        n_errors = 0
        for result in batch.get("results") or []:
            try:
                index = int(result["custom_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= index < n:
                continue
            response = result.get("response") or {}
            if result.get("error") or response.get("status_code") != 200:
                n_errors += 1
                continue
            try:
                replies[index] = response["body"]["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                n_errors += 1
        if n_errors:
            print(f"  batch {batch.get('id')}: {n_errors}/{n} requests returned no usable content; "
                  f"those rows fall back to the base attack's ordering")
        return replies

    def complete_batch(self, texts: list[str]) -> list[str]:
        """Answer every prompt in ``texts`` through the Batch API, preserving input order.

        Chunked into batches of ``requests_per_batch``, each with its own resume ticket. With
        ``wait=False`` this submits everything and then raises :class:`SystemExit`.
        """
        texts = list(texts)
        if not texts:
            return []

        chunks = [texts[start:start + self.requests_per_batch]
                  for start in range(0, len(texts), self.requests_per_batch)]
        tickets = [self._ticket_path(chunk) for chunk in chunks]

        # Submit (or recover) every chunk first, so `wait=False` leaves the whole run queued rather
        # than one chunk of it, and a wait=True run has all its work in flight while it polls.
        batch_ids = []
        for chunk, ticket in zip(chunks, tickets):
            batch_id = self._read_ticket(ticket)
            if batch_id:
                print(f"  resuming existing batch {batch_id} for {len(chunk):,} prompts "
                      f"(not resubmitting)")
            else:
                batch_id = self._submit(chunk)
                self._write_ticket(ticket, batch_id, len(chunk))
                print(f"  submitted batch {batch_id} with {len(chunk):,} prompts")
            batch_ids.append(batch_id)

        if not self.wait:
            # Printed rather than carried on the exception, because the exit code is the part a
            # calling script reads and a SystemExit(int) prints nothing.
            print(
                f"Submitted {len(batch_ids)} batch(es) ({len(texts):,} prompts) and stopped, as "
                f"asked: {', '.join(batch_ids)}. They complete within 24h. Re-run the SAME command "
                f"without --no-wait to collect the results -- the tickets under "
                f"{self.ticket_dir} make it resume these batches rather than pay for them twice."
            )
            raise SystemExit(BATCH_QUEUED_EXIT)

        replies: list[str] = []
        for chunk, ticket, batch_id in zip(chunks, tickets, batch_ids):
            batch = self._poll(batch_id)
            replies.extend(self._collect(batch, len(chunk)))
            # Only once the results are in hand: a cleared ticket is an unrecoverable batch.
            self._clear_ticket(ticket)
        if self.total_cost:
            print(f"  listwise rerank: OpenRouter billed ${self.total_cost:.4f} "
                  f"for {self.total_tokens:,} tokens across {len(batch_ids)} batch(es)")
        return replies


# --- self-test ---------------------------------------------------------------

def _stub_transport(state: dict):
    """A fake ``requests`` module for the checks below: no network, scripted responses.

    Results come back **out of order**, with one errored request and one that is fine, because that
    is the shape the real API returns and the shape that a positional implementation would silently
    scramble.
    """
    import types

    class _HTTPError(Exception):
        def __init__(self, response):
            self.response = response

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def request(method, url, headers=None, data=None, timeout=None):
        state["calls"].append((method, url, data))
        if method == "POST":
            state["submitted"] += 1
            return _Response({"id": f"batch_{state['submitted']}"})
        state["polls"] += 1
        batch_id = url.rsplit("/", 1)[-1]
        if state.get("hold_once") and state["polls"] == 1:
            return _Response({"id": batch_id, "status": "in_progress",
                              "request_counts": {"total": 3, "completed": 1}})
        return _Response({
            "id": batch_id, "status": "completed",
            "request_counts": {"total": 3, "completed": 2, "failed": 1},
            "usage": {"total_tokens": 1234, "cost": 0.0075},
            "results": [
                {"custom_id": "2", "response": {"status_code": 200, "body": {
                    "choices": [{"message": {"content": "third"}}]}}},
                {"custom_id": "0", "response": {"status_code": 200, "body": {
                    "choices": [{"message": {"content": "first"}}]}}},
                {"custom_id": "1", "response": {"status_code": 500}, "error": "boom"},
            ],
        })

    module = types.ModuleType("requests")
    module.request = request
    module.exceptions = types.SimpleNamespace(HTTPError=_HTTPError)
    return module


def _selftest(ticket_dir) -> None:
    """Wire shape, result keying and ticket bookkeeping -- offline, no key, no spend.

    The three things checked here are the three that fail *expensively*: a payload OpenRouter
    rejects, results reassembled by position instead of ``custom_id`` (which silently attributes
    every reason to the wrong document), and a resume that resubmits instead of resuming (which
    pays for the same 24-hour batch twice). Run with
    ``python -m prompt_anonymity.attacks.llm._openrouter_batch --selftest``.
    """
    import sys
    import types

    state = {"calls": [], "submitted": 0, "polls": 0, "hold_once": False}
    sys.modules["requests"] = _stub_transport(state)
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv
    os.environ[SONNET_API_KEY_ENV] = "sk-or-selftest"

    tickets = Path(ticket_dir)
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if condition else 'FAIL'}  {name}{'' if condition else f'  -- {detail}'}")
        if not condition:
            failures.append(name)

    client = OpenRouterBatch("anthropic/claude-sonnet-5", "SYSTEM", max_tokens=3000,
                             reasoning_effort="high", ticket_dir=tickets, poll_interval=0)
    replies = client.complete_batch(["p0", "p1", "p2"])
    check("results land in input order, keyed by custom_id",
          replies == ["first", "", "third"], repr(replies))
    check("an errored request becomes an empty reply, not a crash", replies[1] == "")
    check("usage is recorded", (client.total_cost, client.total_tokens) == (0.0075, 1234))

    body = json.loads(state["calls"][0][2])
    check("endpoint and model are serialized before requests",
          list(body) == ["endpoint", "model", "requests"], str(list(body)))
    check("endpoint is the chat-completions one", body["endpoint"] == OPENROUTER_CHAT_ENDPOINT)
    # The one failure here that costs money instead of raising: the plain slug is accepted by the
    # batch endpoint and billed at FULL price, so the discount is only earned by the :batch variant.
    check("a plain slug is submitted as its :batch variant",
          body["model"] == "anthropic/claude-sonnet-5:batch", body["model"])
    check("the configured (logical) model is left alone for the cache",
          client.model == "anthropic/claude-sonnet-5", client.model)
    check("custom_id is the input index",
          [r["custom_id"] for r in body["requests"]] == ["0", "1", "2"])
    check("reasoning effort is sent as reasoning.effort",
          body["requests"][0]["body"]["reasoning"] == {"effort": "high"})
    # budget_tokens is rejected with a 400 on Sonnet 5; effort is the only thinking control.
    check("no budget_tokens is emitted", "budget_tokens" not in state["calls"][0][2])
    # Sonnet 5 removed the sampling controls; sending them 400s the batch a day after submission.
    check("no temperature/top_p is emitted by default",
          "temperature" not in body["requests"][0]["body"]
          and "top_p" not in body["requests"][0]["body"])
    check("the system prompt rides every request",
          all(r["body"]["messages"][0] == {"role": "system", "content": "SYSTEM"}
              for r in body["requests"]))
    check("the ticket is cleared once results are in hand", not any(tickets.glob("*.json")))

    state["calls"].clear()
    OpenRouterBatch("anthropic/claude-sonnet-5", "S", reasoning_effort=None,
                    ticket_dir=tickets, poll_interval=0).complete_batch(["x"])
    check("no reasoning field when effort is None",
          "reasoning" not in json.loads(state["calls"][0][2])["requests"][0]["body"])

    check("an explicit :batch slug is not double-suffixed",
          batch_slug("anthropic/claude-sonnet-5:batch") == "anthropic/claude-sonnet-5:batch")
    check("another variant is left alone", batch_slug("qwen/qwen3-8b:free") == "qwen/qwen3-8b:free")
    check("variant_suffix=None submits verbatim",
          batch_slug("qwen/qwen3-8b", None) == "qwen/qwen3-8b")
    check("a dash in the model name is not mistaken for a variant",
          batch_slug("anthropic/claude-sonnet-5") == "anthropic/claude-sonnet-5:batch")

    # --no-wait leaves a ticket; re-entering with the same prompts must RESUME, not resubmit.
    state["submitted"] = 0
    queued = OpenRouterBatch("anthropic/claude-sonnet-5", "SYSTEM", max_tokens=3000,
                             reasoning_effort="high", ticket_dir=tickets, wait=False,
                             poll_interval=0)
    import contextlib
    import io

    said = io.StringIO()
    try:
        with contextlib.redirect_stdout(said):
            queued.complete_batch(["p0", "p1", "p2"])
        check("wait=False stops instead of polling", False, "no SystemExit raised")
    except SystemExit as stop:
        check("wait=False exits with the distinct queued status",
              stop.code == BATCH_QUEUED_EXIT, f"exit code {stop.code!r}")
        check("wait=False says how to collect", "Re-run the SAME command" in said.getvalue())
    check("wait=False submitted exactly once", state["submitted"] == 1)
    check("wait=False left a ticket behind", len(list(tickets.glob("*.json"))) == 1)

    state["submitted"] = 0
    resumed = OpenRouterBatch("anthropic/claude-sonnet-5", "SYSTEM", max_tokens=3000,
                              reasoning_effort="high", ticket_dir=tickets, poll_interval=0)
    check("the same prompts resume the same batch",
          resumed.complete_batch(["p0", "p1", "p2"]) == ["first", "", "third"])
    check("resuming costs no new submission", state["submitted"] == 0)

    state["submitted"] = 0
    OpenRouterBatch("anthropic/claude-sonnet-5", "SYSTEM", max_tokens=3000,
                    reasoning_effort="high", ticket_dir=tickets, poll_interval=0
                    ).complete_batch(["p0", "p1", "DIFFERENT"])
    check("different prompts do not reuse a ticket", state["submitted"] == 1)

    state.update(submitted=0, polls=0, hold_once=True)
    chunked = OpenRouterBatch("anthropic/claude-sonnet-5", "S", ticket_dir=None,
                              requests_per_batch=2, poll_interval=0, poll_cap=0)
    check("chunks are concatenated in order",
          len(chunked.complete_batch(["a", "b", "c", "d"])) == 4)
    check("one batch per chunk", state["submitted"] == 2)
    check("a non-terminal status is polled again", state["polls"] > 2)

    del os.environ[SONNET_API_KEY_ENV]
    try:
        OpenRouterBatch("anthropic/claude-sonnet-5")
        check("a missing key fails loudly", False, "no RuntimeError raised")
    except RuntimeError as err:
        check("a missing key names the variable and the repo root",
              SONNET_API_KEY_ENV in str(err) and "REPO ROOT" in str(err))

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    if failures:
        raise SystemExit(1)


def main() -> None:
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(
        description="Wire-shape and resume checks for the OpenRouter batch client "
                    "(offline; no key, no spend).")
    parser.add_argument("--selftest", action="store_true", help="run the checks")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do; pass --selftest")
    print("openrouter batch self-test")
    with tempfile.TemporaryDirectory() as tickets:
        _selftest(tickets)


if __name__ == "__main__":
    main()
