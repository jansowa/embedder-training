#!/usr/bin/env python
"""Judge (query, passage) pairs with the exact MS MARCO-pl audit prompt.

Reproduces the prompt that produced the 348 487 `task06-night-jobs-pl-v2-prefix-first`
verdicts (see docs/llm_judge_prompts.md): system message = the bytes of
docs/judge_prompts/SYS_SFT_AUDIT_V2.txt, user message = DATA_SFT_AUDIT.txt with
{passage} and {query} substituted (plain replacement, so braces in the text are
safe), temperature 0, JSON response format, thinking disabled. Nothing is
paraphrased; use --show-prompt to print the first request verbatim.

Input : our training layout, one JSON per line:
        {"query_id": ..., "query": ..., "pos": [...], "neg": [...]}
        By default only pos[0] is judged (the positive the training uses);
        --all-positives judges every positive, --negatives K also judges the
        first K negatives (to estimate the false-negative rate).
Journal: one line per judged pair {"key","verdict","prompt_version","model"},
        appended and fsync'd; a rerun skips keys already present, so the job is
        safe to interrupt. Pairs that fail after the retries are not journaled
        and are retried next time.
Output: the input rows with "judge": {"pos": [verdict...], "neg": [verdict...]}
        and "clean" (weekend definition: odpowiadalne, polszczyzna,
        sensowne_zapytanie true and zbyt_ogolne false for pos[0]), plus a
        summary printed at the end.

Server: any OpenAI-compatible /chat/completions (vLLM). API key from the
        variable named by --api-key-env (default JUDGE_API_KEY), may be empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "docs" / "judge_prompts"
PROMPT_VERSION = "task06-night-jobs-pl-v2-prefix-first"
AXES = ("odpowiadalne", "polszczyzna", "sensowne_zapytanie", "zbyt_ogolne")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def load_prompts(system_file: Path, user_file: Path) -> tuple[str, str]:
    system = system_file.read_text(encoding="utf-8")
    user = user_file.read_text(encoding="utf-8")
    for ph in ("{passage}", "{query}"):
        if ph not in user:
            sys.exit(f"{user_file} lacks the {ph} placeholder")
    return system, user


def build_messages(system: str, user_tpl: str, passage: str, query: str, passage_chars: int) -> list[dict]:
    if passage_chars and len(passage) > passage_chars:
        passage = passage[:passage_chars]
    user = user_tpl.replace("{passage}", passage).replace("{query}", query)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_verdict(text: str) -> dict | None:
    text = THINK_RE.sub("", text)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(text[i : j + 1])
    except json.JSONDecodeError:
        return None
    if not all(a in obj for a in AXES):
        return None
    out = {a: bool(obj[a]) if not isinstance(obj[a], str) else obj[a].strip().lower() == "true" for a in AXES}
    out["glowny_problem"] = str(obj.get("glowny_problem", ""))
    return out


class Judge:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float, max_tokens: int):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout, self.max_tokens = model, api_key, timeout, max_tokens

    def call(self, messages: list[dict]) -> dict | None:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_completion_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"] or ""
                verdict = parse_verdict(text)
                if verdict is not None:
                    return verdict
                time.sleep(1)  # malformed JSON: ask again
            except urllib.error.HTTPError as exc:
                if exc.code in (429,) or exc.code >= 500:
                    time.sleep(min(60, 3 * 2 ** attempt))
                else:
                    detail = exc.read().decode("utf-8", errors="replace")[:300]
                    print(f"[judge] HTTP {exc.code}: {detail}", file=sys.stderr)
                    return None
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                print(f"[judge] {exc!r}; retry {attempt + 1}", file=sys.stderr)
                time.sleep(min(60, 3 * 2 ** attempt))
        return None


def pair_key(query_id: str, role: str, idx: int, passage: str) -> str:
    return f"{query_id}::{role}{idx}::{hashlib.sha1(passage.encode('utf-8')).hexdigest()[:12]}"


def load_journal(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated last line from a killed run; the pair is judged again
            out[r["key"]] = r["verdict"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--journal", required=True, type=Path)
    ap.add_argument("--base-url", required=True, help="e.g. http://server:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="JUDGE_API_KEY")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-completion-tokens", type=int, default=256)
    ap.add_argument("--timeout-seconds", type=float, default=120.0)
    ap.add_argument("--passage-chars", type=int, default=4000, help="truncate the passage in the prompt only")
    ap.add_argument("--all-positives", action="store_true")
    ap.add_argument("--negatives", type=int, default=0, help="also judge the first K negatives of every row")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of pending pairs this run")
    ap.add_argument("--system-file", type=Path, default=PROMPT_DIR / "SYS_SFT_AUDIT_V2.txt")
    ap.add_argument("--user-file", type=Path, default=PROMPT_DIR / "DATA_SFT_AUDIT.txt")
    ap.add_argument("--show-prompt", action="store_true", help="print the first request's messages verbatim and exit")
    args = ap.parse_args()

    system, user_tpl = load_prompts(args.system_file, args.user_file)
    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    jobs: list[tuple[str, str, str, int, str]] = []  # key, query_id, role, idx, passage
    for r in rows:
        qid = str(r.get("query_id"))
        poss = r.get("pos", []) if args.all_positives else r.get("pos", [])[:1]
        for i, p in enumerate(poss):
            jobs.append((pair_key(qid, "pos", i, p), qid, "pos", i, p))
        for i, n in enumerate(r.get("neg", [])[: args.negatives]):
            jobs.append((pair_key(qid, "neg", i, n), qid, "neg", i, n))
    queries = {str(r.get("query_id")): r["query"] for r in rows}

    if args.show_prompt:
        k, qid, role, idx, passage = jobs[0]
        print(json.dumps(build_messages(system, user_tpl, passage, queries[qid], args.passage_chars), ensure_ascii=False, indent=2))
        return

    journal = load_journal(args.journal)
    pending = [j for j in jobs if j[0] not in journal]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"rows {len(rows)}, pairs {len(jobs)}, already judged {len(jobs) - len(pending)}, pending {len(pending)}", flush=True)

    judge = Judge(args.base_url, args.model, os.environ.get(args.api_key_env, ""), args.timeout_seconds, args.max_completion_tokens)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failed = 0
    with args.journal.open("a", encoding="utf-8") as jf:
        if args.journal.stat().st_size > 0:
            # make sure a truncated last line from a killed run does not swallow the next record
            with args.journal.open("rb") as chk:
                chk.seek(-1, os.SEEK_END)
                if chk.read(1) != b"\n":
                    jf.write("\n")

        def work(job):
            key, qid, role, idx, passage = job
            verdict = judge.call(build_messages(system, user_tpl, passage, queries[qid], args.passage_chars))
            if verdict is None:
                return key, None
            rec = {"key": key, "verdict": verdict, "prompt_version": PROMPT_VERSION, "model": args.model}
            with lock:
                jf.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n"); jf.flush(); os.fsync(jf.fileno())
                journal[key] = verdict
            return key, verdict

        done = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for fut in as_completed([ex.submit(work, j) for j in pending]):
                key, verdict = fut.result()
                done += 1
                if verdict is None:
                    failed += 1
                if done % 200 == 0 or done == len(pending):
                    rate = done / max(1e-9, time.time() - t0)
                    print(f"  {done}/{len(pending)} judged ({failed} failed), {rate:.1f} pairs/s", flush=True)

    # output: input rows + verdicts; clean = weekend definition on pos[0]
    n_clean = n_judged = 0
    counts = {a: 0 for a in AXES}
    with args.output.open("w", encoding="utf-8") as out:
        for r in rows:
            qid = str(r.get("query_id"))
            poss = r.get("pos", []) if args.all_positives else r.get("pos", [])[:1]
            jp = [journal.get(pair_key(qid, "pos", i, p)) for i, p in enumerate(poss)]
            jn = [journal.get(pair_key(qid, "neg", i, n)) for i, n in enumerate(r.get("neg", [])[: args.negatives])]
            v0 = jp[0] if jp else None
            clean = None
            if v0 is not None:
                n_judged += 1
                for a in AXES:
                    counts[a] += int(v0[a])
                clean = v0["odpowiadalne"] and v0["polszczyzna"] and v0["sensowne_zapytanie"] and not v0["zbyt_ogolne"]
                n_clean += int(clean)
            r = dict(r)
            r["judge"] = {"pos": jp, "neg": jn}
            r["clean"] = clean
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}: {n_judged}/{len(rows)} rows judged, clean {n_clean} ({100 * n_clean / max(1, n_judged):.1f}%), "
          + ", ".join(f"{a}={100 * counts[a] / max(1, n_judged):.1f}%" for a in AXES) + f", failed pairs this run: {failed}")


if __name__ == "__main__":
    main()
