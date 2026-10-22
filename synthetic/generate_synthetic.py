"""Part 3a: synthetic query datasets from the real independent claims (Part 1 corpus).

  .venv/bin/python3.14 synthetic/generate_synthetic.py --pilot    # 10 per dataset, synchronous calls, printed
  .venv/bin/python3.14 synthetic/generate_synthetic.py --small 50 # first 50 per dataset, synchronous calls
  .venv/bin/python3.14 synthetic/generate_synthetic.py --full     # one Message Batch per dataset

Synthetic text is only ever a QUERY; each example's `answer` is the real claim(s) it was derived from.
  paraphrase     one per independent claim: same scope, different words
  element_reuse  one per independent claim: a new invention reusing one core element
  combination    N per split (N = independent claims in the split): core elements of two claims from different
                 patents in the same split, combined into one claim

The key is read from ANTHROPIC_API_KEY (or from a gitignored .env at the repo root). Budget: every run computes a
worst-case cost first (input chars / CHARS_PER_TOKEN + per-request overhead, plus n_requests x max_tokens of
output) and aborts if it exceeds what is left of MAX_BUDGET_USD; actual usage is appended to
outputs/spend_log.jsonl. Raw responses are cached in outputs/synthetic/cache/, keyed by
(dataset, claim ids, prompt version, model, attempt, thinking settings, full prompt text), and never requested twice.
Outputs: outputs/split.json, outputs/synthetic/{paraphrase,element_reuse,combination}.jsonl, failed.jsonl
(pilot: outputs/synthetic/pilot/). No query is rejected for its content (length, overlap, element ids, near-
duplicates): those checks are recorded in meta.flags. failed.jsonl only lists calls that produced no usable claim.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from search.data import load_corpus  # noqa: E402

MODEL = "claude-haiku-5-5"
MAX_BUDGET_USD = 15.00
PRICE_IN, PRICE_OUT = 0.05, 0.25  # USD per million tokens, Message Batches; synchronous calls cost SYNC_MULTIPLIER x
SYNC_MULTIPLIER = 2
CHARS_PER_TOKEN = 2.5  # worst-case estimate of input tokens (measured 3.07 chars/token on these prompts; prose is ~4)
REQUEST_OVERHEAD_TOKENS = 400  # structured-output schema + framing per request (measured ~200)
# max_tokens per request = base + per_token x source tokens (sources' chars / CHARS_PER_TOKEN; both claims for a
# combination). Fixed caps don't fit: claims run from ~50 to ~1,300 tokens, a paraphrase is about as long as its
# source, element lists restate the source, and element_reuse / combination also think (0-1.3k tokens in the pilot).
# Only generated tokens are billed, so generous caps cost nothing unless used; they do raise the worst-case estimate.
MAX_TOKENS = {"paraphrase": (300, 2.0), "element_reuse": (2500, 3.0), "combination": (3000, 2.5)}
# element_reuse v3 / combination v4: one or two key parts per claim are enough (infringement-like overlap, not all core
# elements), one cohesive invention around them; combination never declines and returns `mix`.
# Shared rules v2: every query is a complete claim (preamble, "comprising:", elements, period).
PROMPT_VERSIONS = {"paraphrase": "v1", "element_reuse": "v3", "combination": "v4"}
SHARED_RULES_VERSION = "v2"
# Haiku 5.5 thinks by default and spent all of max_tokens on it in the first pilot (empty text, stop_reason
# max_tokens), so paraphrase does not think. element_reuse and combination think a little: this model only takes
# adaptive thinking, where effort "low" chose not to think at all and "medium" thought ~1k tokens.
# Both settings are part of the cache key, as is the full prompt text.
THINKING = {"paraphrase": {"type": "disabled"}, "element_reuse": {"type": "adaptive"}, "combination": {"type": "adaptive"}}
EFFORT = {"element_reuse": "medium", "combination": "medium"}
SPLIT = {"seed": 0, "fractions": {"train": 0.8, "val": 0.1, "test": 0.1}}
PAIR_SEED = 0
PILOT_N = 10
SYNC_WORKERS = 16  # parallel synchronous calls (the SDK retries rate-limited calls with backoff)
DUP_JACCARD = 0.9  # queries this similar (token sets) to an earlier query of the same dataset get a flag

DATASETS = ["paraphrase", "element_reuse", "combination"]
OUT = os.path.join(ROOT, "outputs")
SYN = os.path.join(OUT, "synthetic")
CACHE = os.path.join(SYN, "cache")
SPEND_LOG = os.path.join(OUT, "spend_log.jsonl")
BATCHES = os.path.join(SYN, "batches.json")  # submitted Message Batches, so a rerun of --full resumes them
PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

ELEMENT = {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "core": {"type": "boolean"}},
           "required": ["id", "text", "core"], "additionalProperties": False}
ELEMENTS = {"type": "array", "items": ELEMENT}
IDS = {"type": "array", "items": {"type": "string"}}
SCHEMAS = {  # structured outputs: every field required
    "paraphrase": {"paraphrase": {"type": "string"}},
    "element_reuse": {"elements": ELEMENTS, "chosen_elements": IDS, "new_claim": {"type": "string"}},
    "combination": {"elements_a": ELEMENTS, "elements_b": ELEMENTS, "used_from_a": IDS, "used_from_b": IDS,
                    "mix": {"type": "string", "enum": ["mostly_a", "mostly_b", "equal"]}, "new_claim": {"type": "string"}},
}


def schema(dataset: str) -> dict:
    props = SCHEMAS[dataset]
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


# ---------------------------------------------------------------- inputs

def load_claims() -> list[dict]:
    patents, _ = load_corpus(os.path.join(ROOT, "data/patent_data_small"), os.path.join(ROOT, "data/scraped/claims.jsonl"))
    return [{"claim_id": f"{p['doc_number']}#{c['num']}", "doc_number": p["doc_number"], "num": c["num"],
             "text": c["text"], "title": p["title"], "abstract": p["abstract"], "cpc": p["path"][-1],
             "cpc_path": list(p["path"])}
            for p in patents for c in p["claims"] if c["independent"]]


def load_split(docs: list[str]) -> dict[str, str]:
    """doc_number -> train/val/test, by patent; reused from outputs/split.json if it exists."""
    path = os.path.join(OUT, "split.json")
    if os.path.exists(path):
        s = json.load(open(path))
        missing = set(docs) - {d for name in s["fractions"] for d in s[name]}
        if missing:
            sys.exit(f"{path} does not cover {len(missing)} patents; delete it to make a new split")
    else:
        d = sorted(set(docs))
        random.Random(SPLIT["seed"]).shuffle(d)
        n_train = round(SPLIT["fractions"]["train"] * len(d))
        n_val = round(SPLIT["fractions"]["val"] * len(d))
        s = {**SPLIT, "train": sorted(d[:n_train]), "val": sorted(d[n_train:n_train + n_val]), "test": sorted(d[n_train + n_val:])}
        os.makedirs(OUT, exist_ok=True)
        json.dump(s, open(path, "w"), indent=1)
    return {d: name for name in s["fractions"] for d in s[name]}


def prompt(dataset: str, sources: list[dict]) -> str:
    text = open(os.path.join(PROMPTS, f"{dataset}_{PROMPT_VERSIONS[dataset]}.txt")).read().strip()
    if dataset == "combination":
        a, b = sources
        fields = {"title_a": a["title"], "claim_a": a["text"], "abstract_a": a["abstract"],
                  "title_b": b["title"], "claim_b": b["text"], "abstract_b": b["abstract"]}
    else:
        fields = {"title": sources[0]["title"], "claim": sources[0]["text"], "abstract": sources[0]["abstract"]}
    for k, v in fields.items():  # not str.format: the templates contain JSON braces
        text = text.replace("{" + k + "}", v)
    return text + "\n\n" + open(os.path.join(PROMPTS, f"shared_rules_{SHARED_RULES_VERSION}.txt")).read().strip()


def make_requests(dataset: str, claims: list[dict], split_of: dict, attempt: int = 0) -> list[dict]:
    if dataset == "combination":
        groups = []
        for i, name in enumerate(SPLIT["fractions"]):
            cs = [c for c in claims if split_of[c["doc_number"]] == name]
            rng, seen, pairs = random.Random(PAIR_SEED * 100 + i), set(), []
            while len(pairs) < len(cs) and len({c["doc_number"] for c in cs}) > 1:
                a, b = rng.sample(cs, 2)
                key = frozenset((a["claim_id"], b["claim_id"]))
                if a["doc_number"] != b["doc_number"] and key not in seen:
                    seen.add(key)
                    pairs.append([a, b])
            groups += pairs
    else:
        groups = [[c] for c in claims]
    return [request(dataset, g, split_of[g[0]["doc_number"]], attempt) for g in groups]


def request(dataset: str, sources: list[dict], split: str, attempt: int = 0) -> dict:
    ids, text = [c["claim_id"] for c in sources], prompt(dataset, sources)
    key = hashlib.sha1(json.dumps([dataset, ids, PROMPT_VERSIONS[dataset], MODEL, attempt, THINKING[dataset],
                                   EFFORT.get(dataset), hashlib.sha1(text.encode()).hexdigest()]).encode()).hexdigest()
    return {"key": key, "custom_id": key[:40], "dataset": dataset, "split": split, "sources": sources,
            "attempt": attempt, "prompt": text, "max_tokens": max_tokens(dataset, sources)}


def max_tokens(dataset: str, sources: list[dict]) -> int:
    base, per_token = MAX_TOKENS[dataset]
    return int(base + per_token * sum(len(c["text"]) for c in sources) / CHARS_PER_TOKEN)


def version(dataset: str) -> str:
    return f"{PROMPT_VERSIONS[dataset]}+rules-{SHARED_RULES_VERSION}"


def params(r: dict) -> dict:
    d = r["dataset"]
    return {"model": MODEL, "max_tokens": r["max_tokens"], "messages": [{"role": "user", "content": r["prompt"]}],
            "output_config": {"format": {"type": "json_schema", "schema": schema(d)},
                              **({"effort": EFFORT[d]} if d in EFFORT else {})},
            "thinking": THINKING[d]}


# ---------------------------------------------------------------- budget and cache

def cost(input_tokens: float, output_tokens: float, sync: bool) -> float:
    return (input_tokens * PRICE_IN + output_tokens * PRICE_OUT) / 1e6 * (SYNC_MULTIPLIER if sync else 1)


def spent() -> float:
    return sum(json.loads(line)["cost_usd"] for line in open(SPEND_LOG)) if os.path.exists(SPEND_LOG) else 0.0


def check_budget(reqs: list[dict], sync: bool, what: str, reserved: float = 0.0) -> float:
    """Abort if the worst case exceeds what is left (minus `reserved`: worst cases of batches submitted but not yet
    billed); -> the worst case."""
    tin = sum(len(r["prompt"]) / CHARS_PER_TOKEN + REQUEST_OVERHEAD_TOKENS for r in reqs)
    tout = sum(r["max_tokens"] for r in reqs)
    worst, left = cost(tin, tout, sync), MAX_BUDGET_USD - spent() - reserved
    print(f"{what}: {len(reqs)} requests, worst case {tin / 1e6:.2f}M in + {tout / 1e6:.2f}M out tokens "
          f"= ${worst:.3f} ({'sync' if sync else 'batch'} prices); budget left ${left:.3f} of ${MAX_BUDGET_USD:.2f}")
    if worst > left:
        sys.exit(f"ABORT: worst-case cost ${worst:.3f} exceeds the remaining budget ${left:.3f}")
    return worst


def log_spend(mode: str, dataset: str, responses: list[dict]) -> None:
    tin = sum(r["usage"]["input_tokens"] for r in responses)
    tout = sum(r["usage"]["output_tokens"] for r in responses)
    row = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": mode, "dataset": dataset,
           "model": MODEL, "n_requests": len(responses), "input_tokens": tin, "output_tokens": tout,
           "cost_usd": round(cost(tin, tout, mode == "sync"), 6)}
    os.makedirs(OUT, exist_ok=True)
    with open(SPEND_LOG, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  spent ${row['cost_usd']:.4f} ({tin} in / {tout} out tokens); total so far ${spent():.4f}")


def cache_path(r: dict) -> str:
    return os.path.join(CACHE, r["dataset"], r["key"] + ".json")


def cached(r: dict) -> dict | None:
    p = cache_path(r)
    return json.load(open(p)) if os.path.exists(p) else None


def save(r: dict, message) -> dict:
    resp = {"key": r["key"], "dataset": r["dataset"], "claim_ids": [c["claim_id"] for c in r["sources"]],
            "prompt_version": version(r["dataset"]), "model": message.model, "attempt": r["attempt"],
            "stop_reason": message.stop_reason, "text": "".join(b.text for b in message.content if b.type == "text"),
            "usage": {"input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens}}
    os.makedirs(os.path.dirname(cache_path(r)), exist_ok=True)
    json.dump(resp, open(cache_path(r), "w"))
    return resp


def parse(resp: dict | None) -> dict | None:
    """The response's JSON, or None if missing, truncated or not the expected shape."""
    if resp is None or resp["stop_reason"] == "max_tokens":
        return None
    try:
        data = json.loads(resp["text"])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and set(SCHEMAS[resp["dataset"]]) <= set(data) else None


# ---------------------------------------------------------------- API calls

def run_sync(client, reqs: list[dict]) -> list[dict]:
    def one(r):
        try:
            return save(r, client.messages.create(**params(r)))
        except Exception as e:  # not cached: a later run retries it
            print(f"  request {r['custom_id']} failed: {e!r}")
            return None
    with ThreadPoolExecutor(SYNC_WORKERS) as pool:
        return [x for x in pool.map(one, reqs) if x]


def run_full_batches(client, reqs_by_dataset: dict[str, list[dict]], deadline_min: float) -> None:
    """One Message Batch per dataset, all submitted at once; ids saved to BATCHES so a rerun resumes instead of paying
    twice. After `deadline_min` minutes unfinished batches are cancelled; whatever they finished is kept and the rest
    is left uncached for the synchronous pass in generate() (bounded wall time)."""
    state = json.load(open(BATCHES)) if os.path.exists(BATCHES) else {}
    pending, reserved = [], 0.0
    for d, reqs in reqs_by_dataset.items():
        todo = [r for r in reqs if cached(r) is None]
        ids = {r["custom_id"] for r in todo}
        old = [b for b in state.get(d, []) if b["status"] != "collected" and ids & set(b["custom_ids"])]
        if old:
            print(f"{d}: resuming batch(es) {[b['id'] for b in old]}")
            pending += [(d, b) for b in old]
            continue
        if not todo:
            continue
        reserved += check_budget(todo, False, f"{d} batch", reserved)
        batch = client.messages.batches.create(requests=[{"custom_id": r["custom_id"], "params": params(r)} for r in todo])
        entry = {"id": batch.id, "custom_ids": sorted(ids), "status": "submitted",
                 "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        state.setdefault(d, []).append(entry)
        json.dump(state, open(BATCHES, "w"), indent=1)
        print(f"{d}: batch {batch.id} submitted ({len(todo)} requests)", flush=True)
        pending.append((d, entry))
    t0, cancelled = time.time(), False
    while pending:
        live = [(d, e, client.messages.batches.retrieve(e["id"])) for d, e in pending]
        print(f"  [{(time.time() - t0) / 60:4.1f} min] " + " | ".join(
            f"{d}: {b.request_counts.succeeded} done, {b.request_counts.processing} processing ({b.processing_status})"
            for d, _, b in live), flush=True)
        for d, e, b in live:
            if b.processing_status != "ended":
                continue
            by_id = {r["custom_id"]: r for r in reqs_by_dataset[d]}
            got = [save(by_id[x.custom_id], x.result.message) for x in client.messages.batches.results(e["id"])
                   if x.result.type == "succeeded" and x.custom_id in by_id]
            log_spend("batch", d, got)
            e["status"] = "collected"
            json.dump(state, open(BATCHES, "w"), indent=1)
            print(f"{d}: collected {len(got)} of {len(e['custom_ids'])} from batch {e['id']}", flush=True)
            pending.remove((d, e))
        if pending and not cancelled and time.time() - t0 > deadline_min * 60:
            print(f"deadline of {deadline_min} min reached: cancelling unfinished batches; the rest runs synchronously")
            for _, e in pending:
                client.messages.batches.cancel(e["id"])
            cancelled = True
        if pending:
            time.sleep(30)


def generate(client, dataset: str, reqs: list[dict]) -> None:
    """Synchronously request everything not cached; then retry once (attempt 1) whatever errored or came back malformed.
    In --full mode this only handles what the batches did not finish, and the retries."""
    for attempt in (0, 1):
        if attempt:
            reqs = [request(dataset, r["sources"], r["split"], 1) for r in reqs if parse(cached(r)) is None]
        todo = [r for r in reqs if cached(r) is None]
        if not todo:
            continue
        check_budget(todo, True, f"{dataset}{' retry' if attempt else ''} (sync)")
        log_spend("sync", dataset, run_sync(client, todo))


# ---------------------------------------------------------------- checks (flags, never rejections)

CLAIM_NUMBER = re.compile(r"^\s*\d+\s*[.)]\s*")  # "1. A wheel ..."
REF_NUMERALS = re.compile(r"\s*\(\s*\d+[a-z']?(?:\s*,\s*\d+[a-z']?)*\s*\)")  # "( 12 )", "(12, 14a)"
CLAIM_FORM = re.compile(r"^(?:An?|The|One or more)\s[^;]{1,300}?\b(?:comprising|consisting(?: essentially)? of|including|"
                        r"having|characterized|wherein|which comprises)\b.*\.$", re.S)
FIGURES = re.compile(r"\s*\(?\b(?:FIGS?|Figs?)\.?\s*\d+[A-Za-z]?\)?")
WORDS = re.compile(r"[a-z0-9]+")


def strip(text: str) -> tuple[str, list[str]]:
    """Remove a claim number, parenthesised reference numerals and figure references; -> (text, what was removed).
    The preamble stays: every query is a complete claim."""
    removed = []
    for name, pattern in (("claim number", CLAIM_NUMBER), ("reference numerals", REF_NUMERALS),
                          ("figure references", FIGURES)):
        text, n = pattern.subn(" " if name != "claim number" else "", text)
        if n and name not in removed:
            removed.append(name)
    return re.sub(r"\s+([,;.:])", r"\1", re.sub(r"\s+", " ", text)).strip(), removed


def tokens(text: str) -> list[str]:
    return WORDS.findall(text.lower())


def jaccard(a: list[str], b: list[str]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def rouge_l(a: list[str], b: list[str]) -> float:
    """ROUGE-L F1: longest common subsequence of tokens."""
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(a), lcs / len(b)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def build(dataset: str, reqs: list[dict], paraphrase_flag_rouge: float | None) -> tuple[list[dict], list[dict]]:
    """Cached responses -> (examples, failures). Nothing is rejected for its content: checks become `meta.flags`.
    Only a call with no usable claim (API error, or unparseable JSON after the retry) is a failure."""
    rows, failed = [], []
    for r in reqs:
        retry = request(dataset, r["sources"], r["split"], 1)
        resp = cached(r) if parse(cached(r)) is not None else cached(retry)
        data = parse(resp)
        if data is None or not (data.get("paraphrase") or data.get("new_claim") or "").strip():
            failed.append({"dataset": dataset, "split": r["split"], "claim_ids": [c["claim_id"] for c in r["sources"]],
                           "reason": "no response (API error)" if resp is None else
                           f"no usable claim (stop_reason {resp['stop_reason']})", "response": resp and resp["text"][:2000]})
            continue
        meta, flags = {}, []
        if dataset == "paraphrase":
            query = data["paraphrase"]
        elif dataset == "element_reuse":
            query = data["new_claim"]
            meta = {"elements": data["elements"], "chosen_elements": data["chosen_elements"]}
            bad = [i for i in data["chosen_elements"] if i not in {e["id"] for e in data["elements"]}]
            if bad:
                flags.append(f"chosen_elements not in elements: {bad}")
            if not 1 <= len(data["chosen_elements"]) <= 2:
                flags.append(f"{len(data['chosen_elements'])} chosen elements (asked for 1-2)")
        else:
            query = data["new_claim"]
            meta = {k: data[k] for k in ("elements_a", "elements_b", "used_from_a", "used_from_b", "mix")}
            bad = ([i for i in data["used_from_a"] if i not in {e["id"] for e in data["elements_a"]}] +
                   [i for i in data["used_from_b"] if i not in {e["id"] for e in data["elements_b"]}])
            if bad:
                flags.append(f"used ids not in the element lists: {bad}")
            for side in "ab":
                if not 1 <= len(data[f"used_from_{side}"]) <= 2:
                    flags.append(f"{len(data[f'used_from_{side}'])} elements used from {side.upper()} (asked for 1-2)")
        query, removed = strip(query)
        if removed:
            meta["stripped"] = removed
        if not CLAIM_FORM.match(query):
            flags.append("not in claim form (preamble + transitional phrase, ending in a period)")
        meta["length_ratio"] = round(len(query.split()) / max(len(c["text"].split()) for c in r["sources"]), 3)
        if dataset == "paraphrase":
            q, s = tokens(query), tokens(r["sources"][0]["text"])
            meta["jaccard"], meta["rouge_l"] = round(jaccard(q, s), 4), round(rouge_l(q, s), 4)
            if paraphrase_flag_rouge is not None and meta["rouge_l"] > paraphrase_flag_rouge:
                flags.append(f"close to the source: ROUGE-L {meta['rouge_l']} > {paraphrase_flag_rouge}")
        meta["flags"] = flags
        rows.append({"query_id": f"{dataset}-{r['key'][:12]}", "dataset": dataset, "split": r["split"],
                     "query_text": query,
                     "answer": [{"claim_id": c["claim_id"], "doc_number": c["doc_number"], "cpc": c["cpc"]} for c in r["sources"]],
                     "meta": meta, "model": resp["model"], "prompt_version": version(dataset)})
    seen = []
    for row in rows:  # near-duplicates: token sets compared with every earlier query of this dataset
        t = set(tokens(row["query_text"]))
        dup = next((k for k, s in seen if len(t & s) / max(1, len(t | s)) >= DUP_JACCARD), None)
        if dup:
            row["meta"]["flags"].append(f"near-duplicate of {dup}")
        seen.append((row["query_id"], t))
    return rows, failed


# ---------------------------------------------------------------- report

def report(results: dict, claims_by_id: dict, n_requests: dict) -> None:
    print("\n" + "=" * 100 + "\nREPORT")
    for dataset, (rows, failed) in results.items():
        before = Counter(r for r in n_requests[dataset])
        after = Counter(r["split"] for r in rows)
        print(f"\n{dataset}: " + ", ".join(f"{s} {before[s]} -> {after[s]}" for s in SPLIT["fractions"])
              + f"  (requests -> queries; total {sum(before.values())} -> {len(rows)})")
        for reason, n in Counter(re.sub(r"\d+(\.\d+)?", "#", re.sub(r"[:(].*", "", r["reason"])).strip() for r in failed).most_common():
            print(f"  failed  {n:4d}: {reason}")
        for flag, n in Counter(re.sub(r"\d+(\.\d+)?|'[^']*'|\[.*\]|-[0-9a-f]{12}", "#", re.sub(r":.*", "", f))
                               for r in rows for f in r["meta"]["flags"]).most_common():
            print(f"  flagged {n:4d}: {flag}")
        if rows:
            ratio = np.array([r["meta"]["length_ratio"] for r in rows])
            print("  length vs source: " + "  ".join(f"p{q} {np.percentile(ratio, q):.2f}x" for q in (0, 10, 50, 90, 100)))
        if dataset == "combination" and rows:
            print("  mix:", dict(Counter(r["meta"]["mix"] for r in rows)))
        stripped = Counter(x for r in rows for x in r["meta"].get("stripped", []))
        if stripped:
            print("  stripped from kept queries:", dict(stripped))
        if dataset == "paraphrase" and rows:
            for m in ("jaccard", "rouge_l"):
                v = np.array([r["meta"][m] for r in rows])
                print(f"  {m:8s} vs source: " + "  ".join(f"p{q} {np.percentile(v, q):.3f}" for q in (0, 10, 25, 50, 75, 90, 100)))
                hist, edges = np.histogram(v, bins=10, range=(0, 1))
                print("    " + "  ".join(f"{edges[i]:.1f}-{edges[i + 1]:.1f}: {h}" for i, h in enumerate(hist) if h))
    print(f"\nspend: ${spent():.4f} of ${MAX_BUDGET_USD:.2f} (outputs/spend_log.jsonl)")


def show(row: dict, claims_by_id: dict, width: int = 100) -> None:
    import textwrap
    print("\n" + "-" * width + f"\n{row['query_id']}  [{row['split']}]")
    for a in row["answer"]:
        c = claims_by_id[a["claim_id"]]
        print(f"SOURCE {a['claim_id']} ({c['cpc']}) {c['title']}")
        print(textwrap.indent(textwrap.fill(c["text"], width - 4), "    "))
    meta = {k: v for k, v in row["meta"].items() if not k.startswith("elements")}
    for k in ("elements", "elements_a", "elements_b"):
        for e in row["meta"].get(k, []):
            print(f"    {e['id']:4s} {'core' if e['core'] else 'conv'}  {textwrap.shorten(e['text'], width - 16)}")
    print(f"QUERY  {meta}")
    print(textwrap.indent(textwrap.fill(row["query_text"], width - 4), "    "))


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help=f"{PILOT_N} per dataset, synchronous, printed; then stop")
    mode.add_argument("--small", type=int, metavar="N", help="first N per dataset (same order as the pilot), synchronous")
    mode.add_argument("--full", action="store_true", help="everything, one Message Batch per dataset")
    ap.add_argument("--deadline-min", type=float, default=40,
                    help="--full: cancel batches still running after this many minutes and finish synchronously")
    ap.add_argument("--paraphrase-flag-rouge", type=float, default=None,
                    help="flag paraphrases with ROUGE-L above this (unset: only report the distribution)")
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY") and os.path.exists(os.path.join(ROOT, ".env")):
        for line in open(os.path.join(ROOT, ".env")):
            k, _, v = line.strip().partition("=")
            if k == "ANTHROPIC_API_KEY":
                os.environ[k] = v
    import anthropic
    client = anthropic.Anthropic(max_retries=8)

    claims = load_claims()
    claims_by_id = {c["claim_id"]: c for c in claims}
    split_of = load_split([c["doc_number"] for c in claims])
    print(f"{len(claims)} independent claims of {len(split_of)} patents; split "
          + ", ".join(f"{s} {sum(v == s for v in split_of.values())} patents / "
                      f"{sum(split_of[c['doc_number']] == s for c in claims)} claims" for s in SPLIT["fractions"]))
    n = PILOT_N if args.pilot else args.small
    sync = n is not None
    out_dir = os.path.join(SYN, "pilot" if args.pilot else f"small_{n}") if sync else SYN
    os.makedirs(out_dir, exist_ok=True)
    results, n_requests, all_reqs = {}, {}, {}
    for dataset in DATASETS:
        reqs = make_requests(dataset, claims, split_of)
        if sync:  # a fixed shuffled order, so the pilot is the first PILOT_N of every small run (cache reused)
            random.Random(PAIR_SEED).shuffle(reqs)
            reqs = reqs[:n]
        all_reqs[dataset] = reqs
    if not sync:
        run_full_batches(client, all_reqs, args.deadline_min)
    for dataset, reqs in all_reqs.items():
        n_requests[dataset] = [r["split"] for r in reqs]
        generate(client, dataset, reqs)
        results[dataset] = build(dataset, reqs, args.paraphrase_flag_rouge)
        rows = results[dataset][0]
        with open(os.path.join(out_dir, f"{dataset}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(os.path.join(out_dir, "failed.jsonl"), "w") as f:
        f.writelines(json.dumps(x) + "\n" for _, failed in results.values() for x in failed)

    if args.pilot:
        for dataset, (rows, failed) in results.items():
            print("\n" + "#" * 100 + f"\n# {dataset.upper()}: {len(rows)} queries, {len(failed)} failed calls")
            for row in rows:
                show(row, claims_by_id)
            for x in failed:
                print(f"\nFAILED {x['claim_ids']}: {x['reason']}")
    report(results, claims_by_id, n_requests)
    if not args.pilot:
        rng = random.Random(0)
        for dataset, (rows, _) in results.items():
            print("\n" + "#" * 100 + f"\n# {dataset}: 5 random examples")
            for row in rng.sample(rows, min(5, len(rows))):
                show(row, claims_by_id)
    print(f"\nwritten to {os.path.relpath(out_dir, ROOT)}/")


if __name__ == "__main__":
    main()
