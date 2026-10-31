#!/usr/bin/env python
"""Annotate (query, passage) pairs with an LLM behind an OpenAI-compatible server (vLLM).

Generic successor of judge_pairs.py: the prompt (system file + user template with {query}
and {passage}) and the output schema (a JSON schema file with enum fields) are parameters, the
server is asked for structured output, every answer is validated against the schema, and a
journal makes the run resumable (kill it any time; rerun with the same --journal).

Input:  pairs jsonl from scripts/build_annotation_pairs.py (pair_id, source, query_id, side,
        index, query, passage, ...). Processed in file order (the file is priority-ordered).
Output: --output jsonl, one record per pair: the pair's id fields plus the schema fields plus
        model / prompt_version; the sidecar format of docs/llm-annotation-schema.md §6.
        Written at the end of the run (and on Ctrl-C) from the journal; --write-output-only
        rebuilds it from an existing journal without calling the server.

Structured output: --structured json_schema (response_format type json_schema), guided_json
(vLLM extra field), json_object (format only; enums are then enforced by validation and
retry) or auto (try them in that order, fall back on HTTP 400).

Usage:
  JUDGE_API_KEY=... .venv/bin/python scripts/annotate_pairs.py \
      --pairs datasets/annotation/pairs_v1.jsonl --output results/annotation/pairs_v1.llm.jsonl \
      --journal results/annotation/pairs_v1.journal.jsonl \
      --base-url http://server:8000/v1 --model <served name> --concurrency 8
"""

from __future__ import annotations

import argparse
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
PROMPT_VERSION = "annotate-v1"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def load_schema(path: Path) -> dict:
    schema = json.loads(path.read_text(encoding="utf-8"))
    for k in schema["required"]:
        if "enum" not in schema["properties"][k]:
            sys.exit(f"{path}: field {k} has no enum; this tool validates enums only")
    return schema


def validate(obj: dict, schema: dict) -> dict | None:
    """Return the normalised record or None when a required field is missing or out of its enum."""
    out = {}
    for k in schema["required"]:
        if k not in obj:
            return None
        v = obj[k]
        prop = schema["properties"][k]
        if prop.get("type") == "integer":
            try:
                v = int(v)
            except (TypeError, ValueError):
                return None
        elif isinstance(v, str):
            v = v.strip()
        if v not in prop["enum"]:
            return None
        out[k] = v
    return out


def parse_json(text: str) -> dict | None:
    text = THINK_RE.sub("", text)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class Client:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float, max_tokens: int,
                 schema: dict, structured: str, no_thinking: bool):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout, self.max_tokens = model, api_key, timeout, max_tokens
        self.schema, self.no_thinking = schema, no_thinking
        self.modes = ["json_schema", "guided_json", "json_object"] if structured == "auto" else [structured]
        self.lock = threading.Lock()

    def payload(self, messages: list[dict], mode: str) -> dict:
        p: dict = {"model": self.model, "messages": messages, "temperature": 0.0, "max_completion_tokens": self.max_tokens}
        if self.no_thinking:
            p["chat_template_kwargs"] = {"enable_thinking": False}
        if mode == "json_schema":
            p["response_format"] = {"type": "json_schema", "json_schema": {"name": "annotation", "schema": self.schema, "strict": True}}
        elif mode == "guided_json":
            p["guided_json"] = self.schema
        else:
            p["response_format"] = {"type": "json_object"}
        return p

    def post(self, payload: dict) -> str:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"] or ""

    def call(self, messages: list[dict]) -> tuple[dict | None, str]:
        """(validated record or None, reason)."""
        last = "no attempt"
        for attempt in range(6):
            mode = self.modes[0]
            try:
                text = self.post(self.payload(messages, mode))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                if exc.code == 400 and len(self.modes) > 1:
                    with self.lock:
                        if self.modes[0] == mode:
                            self.modes.pop(0)
                            print(f"[annotate] server rejected {mode} ({detail[:120]}); falling back to {self.modes[0]}", file=sys.stderr, flush=True)
                    continue
                if exc.code == 429 or exc.code >= 500:
                    time.sleep(min(60, 3 * 2 ** attempt)); last = f"HTTP {exc.code}"; continue
                return None, f"HTTP {exc.code}: {detail}"
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                time.sleep(min(60, 3 * 2 ** attempt)); last = repr(exc); continue
            obj = parse_json(text)
            rec = validate(obj, self.schema) if obj else None
            if rec is not None:
                return rec, "ok"
            last = f"invalid answer: {text[:160]!r}"
            time.sleep(0.5)
        return None, last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--journal", type=Path, required=True)
    ap.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL"))
    ap.add_argument("--api-key-env", default="JUDGE_API_KEY")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap the number of pending pairs this run")
    ap.add_argument("--sources", nargs="*", default=None, help="only these sources")
    ap.add_argument("--tier", type=int, default=None, help="only this tier")
    ap.add_argument("--system-file", type=Path, default=PROMPT_DIR / "SYS_ANNOTATE_V1.txt")
    ap.add_argument("--user-file", type=Path, default=PROMPT_DIR / "DATA_ANNOTATE_V1.txt")
    ap.add_argument("--schema-file", type=Path, default=PROMPT_DIR / "SCHEMA_ANNOTATE_V1.json")
    ap.add_argument("--prompt-version", default=PROMPT_VERSION)
    ap.add_argument("--passage-chars", type=int, default=2000, help="truncate the passage in the prompt only")
    ap.add_argument("--max-completion-tokens", type=int, default=200)
    ap.add_argument("--timeout-seconds", type=float, default=180.0)
    ap.add_argument("--structured", choices=["auto", "json_schema", "guided_json", "json_object"], default="auto")
    ap.add_argument("--allow-thinking", action="store_true", help="do not pass enable_thinking=false")
    ap.add_argument("--show-prompt", action="store_true", help="print the first request's messages and exit")
    ap.add_argument("--write-output-only", action="store_true", help="rebuild --output from the journal, no server calls")
    a = ap.parse_args()

    system = a.system_file.read_text(encoding="utf-8")
    user_tpl = a.user_file.read_text(encoding="utf-8")
    for ph in ("{query}", "{passage}"):
        if ph not in user_tpl:
            sys.exit(f"{a.user_file} lacks the {ph} placeholder")
    schema = load_schema(a.schema_file)

    pairs = []
    with a.pairs.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                p = json.loads(line)
                if a.sources and p.get("source") not in a.sources:
                    continue
                if a.tier is not None and p.get("tier") != a.tier:
                    continue
                pairs.append(p)

    def messages(p: dict) -> list[dict]:
        passage = p["passage"]
        if a.passage_chars and len(passage) > a.passage_chars:
            passage = passage[:a.passage_chars]
        return [{"role": "system", "content": system},
                {"role": "user", "content": user_tpl.replace("{query}", p["query"]).replace("{passage}", passage)}]

    if a.show_prompt:
        print(json.dumps(messages(pairs[0]), ensure_ascii=False, indent=2)); return

    journal: dict[str, dict] = {}
    if a.journal.exists():
        with a.journal.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated last line of a killed run
                if r.get("record") is not None:
                    journal[r["key"]] = r["record"]

    def write_output() -> None:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with a.output.open("w", encoding="utf-8") as fh:
            for p in pairs:
                rec = journal.get(p["pair_id"])
                if rec is None:
                    continue
                out = {k: p[k] for k in ("pair_id", "source", "query_id", "side", "index") if k in p}
                out.update(rec); out["model"] = a.model; out["prompt_version"] = a.prompt_version
                fh.write(json.dumps(out, ensure_ascii=False) + "\n"); n += 1
        print(f"wrote {n} annotated pairs to {a.output} ({len(pairs) - n} pending)", flush=True)

    if a.write_output_only:
        write_output(); return
    if not a.base_url or not a.model:
        sys.exit("--base-url and --model are required (or JUDGE_BASE_URL / JUDGE_MODEL)")

    pending = [p for p in pairs if p["pair_id"] not in journal]
    if a.limit is not None:
        pending = pending[:a.limit]
    print(f"{len(pairs)} pairs, {len(journal)} done earlier, {len(pending)} to annotate now", flush=True)
    client = Client(a.base_url, a.model, os.environ.get(a.api_key_env, ""), a.timeout_seconds,
                    a.max_completion_tokens, schema, a.structured, not a.allow_thinking)
    a.journal.parent.mkdir(parents=True, exist_ok=True)
    jlock = threading.Lock()
    done = failed = 0
    t0 = time.time()
    failures: dict[str, int] = {}

    def work(p: dict) -> tuple[str, dict | None, str]:
        rec, reason = client.call(messages(p))
        return p["pair_id"], rec, reason

    try:
        with a.journal.open("a", encoding="utf-8") as jf, ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futures = [ex.submit(work, p) for p in pending]
            for fut in as_completed(futures):
                key, rec, reason = fut.result()
                with jlock:
                    if rec is not None:
                        journal[key] = rec; done += 1
                        jf.write(json.dumps({"key": key, "record": rec, "ts": round(time.time())}, ensure_ascii=False) + "\n")
                    else:
                        failed += 1; failures[reason[:80]] = failures.get(reason[:80], 0) + 1
                    jf.flush()
                    total = done + failed
                    if total % 100 == 0 or total == len(pending):
                        el = time.time() - t0
                        rate = total / el if el else 0
                        eta = (len(pending) - total) / rate / 60 if rate else float("inf")
                        print(f"[annotate] {total}/{len(pending)} ({done} ok, {failed} failed), {rate * 60:.0f}/min, eta {eta:.0f} min", flush=True)
    except KeyboardInterrupt:
        print("[annotate] interrupted; writing what is done", file=sys.stderr, flush=True)
    write_output()
    if failures:
        print("failures:", json.dumps(failures, ensure_ascii=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
