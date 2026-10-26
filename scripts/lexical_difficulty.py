#!/usr/bin/env python
"""Lexical difficulty of (query, positive passage) pairs for Polish retrieval data.

The measure answers one question: how much of the query is literally copied from
its positive passage?  Everything is computed on Stanza lemmas (CPU only),
NFKC-normalised, lower-cased, alphanumeric tokens only.

Per-pair metrics (see ``pair_metrics``):

* ``idf_coverage``         IDF-weighted share of query content lemmas found in the passage
* ``coverage``             unweighted share of query content lemmas found in the passage
* ``longest_copied_ngram`` longest contiguous lemma n-gram shared by query and passage
* ``copy_density``         ``longest_copied_ngram`` / number of query lemmas
* ``entity_preservation``  share of entity-like query lemmas (capitalised and not
                           sentence-initial, or numeric) found in the passage;
                           ``null`` when the query has no such token
* ``has_rare_match``       the query shares a rare lemma with the passage (rare = idf
                           above the 90th df-weighted percentile of the pool)
* ``max_matched_idf``      highest idf among the shared content lemmas
* ``q_len_content``        number of distinct query content lemmas

CLI, all datasets in one call so that they share a single IDF pool::

    CUDA_VISIBLE_DEVICES= python scripts/lexical_difficulty.py \\
        --input fiqa=datasets/arms/fiqa-2073.jsonl --input trivia=/path/train_pl.jsonl \\
        --max-lines 20000 --sample 300 --seed 42 \\
        --idf-pool datasets/synth/chunks_pool.jsonl --pool-field text --pool-max-lines 3000 \\
        --output-dir results/lexical_difficulty --summary results/calibration.md \\
        --contrast fiqa:trivia

Single dataset::

    python scripts/lexical_difficulty.py --input data.jsonl --output metrics.jsonl \\
        --idf-pool pool.jsonl --sample 300 --seed 42

Library::

    nlp = load_pipeline()
    queries, passages = analyze_texts(nlp, qs), analyze_texts(nlp, ps)
    idf = build_idf(passages + analyze_texts(nlp, pool_texts))
    pair_metrics(queries[0], passages[0], idf)

Input rows need ``query`` and either ``pos`` (list of passages, the first one is
used) or ``texts`` plus a parallel ``labels`` list (the first text with a truthy
label is the positive).  The positives of every input always join the IDF pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

DEFAULT_MODEL_DIR = os.environ.get(
    "STANZA_RESOURCES_DIR", str(Path.home() / ".cache" / "stanza" / "1.14.0" / "resources")
)

# Lemma-level stopwords: the short doc2query list plus pronouns, prepositions,
# conjunctions, particles, auxiliaries and question words (with common inflected
# forms in case the lemmatiser leaves them untouched).
POLISH_STOPWORDS = frozenset(
    """
    a aby ale albo ani aż bo bowiem by być był była było były co czy dla do gdy gdzie i ich
    jak jako jego jej jest jeśli już kiedy która które który lecz lub ma mają mieć na nad nie
    o od oraz po pod przez przy się są ta tak ten to tu w we z za ze że
    ja ty on ona ono my wy oni one mnie mi mną ciebie cię tobie tobą jemu mu go nim niego niej
    nią ją nich nimi im nam nas wam was je siebie sobie sobą swój mój twój nasz wasz
    tamten tamta tamto ów owa owo taki taka takie sam sama samo każdy każda każde wszystek
    wszyscy wszystko wszystkie żaden żadna żadne nikt nic ktoś coś jakiś jakaś jakieś któryś
    inny inna inne tego temu tym tej tych tymi tę te którego której których którym którzy którą
    jakiego jakiej jakich jakim czego czemu czym kogo komu kim
    kto czyj jaki jaka jakie ile ilu dlaczego skąd dokąd dotąd
    przed bez ku między pomiędzy u wśród wobec względem podczas według mimo pomimo oprócz
    około obok wokół poza spod sprzed znad zza ponad poprzez wewnątrz wzdłuż dzięki
    żeby iż ponieważ gdyż jeżeli choć chociaż więc zatem toteż też także również natomiast
    jednak jednakże bądź tudzież ni niż czyli zaś albowiem aczkolwiek jakby jakoby niby skoro
    dopóki dopóty zanim póki oto
    jeszcze tylko nawet właśnie może bardzo tam tutaj teraz wtedy zawsze nigdy często czasem
    dość dosyć prawie zbyt jedynie chyba raczej niemal jeden
    zostać móc musieć chcieć powinien powinna powinno winien
    jestem jesteś jesteśmy jesteście będzie będą byłem byłeś byliśmy byłyśmy byłby byłaby byłoby
    byliby mam masz mamy macie miał miała miało mieli miały mogą mogę możesz możemy możecie mógł
    mogła mogło mogli mogły można musi muszą muszę musisz musimy musiał
    byś bym byśmy byście niech np itp itd tzn
    """.split()
)

_NONWORD = re.compile(r"[\W_]+")
_DIGITS = re.compile(r"\d+")


# --------------------------------------------------------------------------- text analysis


@dataclass
class Analyzed:
    """Lemma view of one text."""

    lemmas: list[str]  # NFKC lower-cased alphanumeric lemmas, in text order
    surfaces: list[str]  # the matching surface forms, cleaned the same way
    content: list[str]  # lemmas minus stopwords minus length-1 lemmas
    entity_like: list[str]  # content lemmas of capitalised (non sentence-initial) or numeric tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Analyzed":
        return cls(
            list(data["lemmas"]), list(data["surfaces"]), list(data["content"]), list(data["entity_like"])
        )

    def forms(self) -> set[str]:
        """Every lemma and surface form; proper nouns are lemmatised inconsistently."""
        return set(self.lemmas) | set(self.surfaces)

    def surface_of(self) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = {}
        for lemma, surface in zip(self.lemmas, self.surfaces):
            mapping.setdefault(lemma, set()).add(surface)
        return mapping


def _clean(token: str) -> str:
    return _NONWORD.sub("", unicodedata.normalize("NFKC", token)).lower()


def is_content(lemma: str) -> bool:
    return len(lemma) > 1 and lemma not in POLISH_STOPWORDS


def analyzed_from_stanza(doc: Any) -> Analyzed:
    lemmas: list[str] = []
    surfaces: list[str] = []
    entity_like: list[str] = []
    for sentence in doc.sentences:
        initial_seen = False
        for word in sentence.words:
            surface = unicodedata.normalize("NFKC", word.text or "")
            lemma = _clean(word.lemma or surface)
            if not lemma:
                continue
            lemmas.append(lemma)
            surfaces.append(_clean(surface) or lemma)
            capitalised = surface[:1].isupper() and initial_seen
            numeric = _DIGITS.search(lemma) is not None
            if (capitalised or numeric) and lemma not in POLISH_STOPWORDS:
                entity_like.append(lemma)
            initial_seen = True
    return Analyzed(lemmas, surfaces, [l for l in lemmas if is_content(l)], entity_like)


def load_pipeline(model_dir: str = DEFAULT_MODEL_DIR, threads: int | None = None) -> Any:
    """CPU-only Stanza pipeline (tokenize, pos, lemma) for Polish."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import stanza  # noqa: WPS433 (lazy so summarising does not need stanza)

    if threads:
        import torch

        torch.set_num_threads(threads)
    return stanza.Pipeline(
        "pl",
        model_dir=model_dir,
        processors="tokenize,pos,lemma",
        use_gpu=False,
        download_method=None,
        verbose=False,
    )


class LemmaCache:
    """Append-only JSONL cache keyed by the sha1 of the text."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.entries: dict[str, Analyzed] = {}
        self._handle = None
        if path and path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    self.entries[record["h"]] = Analyzed.from_dict(record["a"])
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Analyzed | None:
        return self.entries.get(self.key(text))

    def put(self, text: str, analyzed: Analyzed) -> None:
        key = self.key(text)
        self.entries[key] = analyzed
        if self._handle:
            self._handle.write(json.dumps({"h": key, "a": analyzed.to_dict()}, ensure_ascii=False) + "\n")
            self._handle.flush()


def analyze_texts(
    nlp: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    cache: LemmaCache | None = None,
    label: str = "",
) -> list[Analyzed]:
    """Lemmatise ``texts`` with Stanza's bulk processing, in batches."""
    import stanza

    results: list[Analyzed | None] = [cache.get(t) if cache else None for t in texts]
    todo = [i for i, r in enumerate(results) if r is None]
    started = time.time()
    for start in range(0, len(todo), batch_size):
        indices = todo[start : start + batch_size]
        docs = nlp([stanza.Document([], text=texts[i]) for i in indices])
        for i, doc in zip(indices, docs):
            analyzed = analyzed_from_stanza(doc)
            results[i] = analyzed
            if cache:
                cache.put(texts[i], analyzed)
        done = start + len(indices)
        print(f"  [{label}] {done}/{len(todo)} lemmatised, {time.time() - started:.0f}s", file=sys.stderr)
    return [r for r in results if r is not None]


# --------------------------------------------------------------------------- idf


@dataclass
class Idf:
    n_docs: int
    df: dict[str, int]
    rare_threshold: float  # idf above this = rare (90th df-weighted percentile)

    def __call__(self, lemma: str) -> float:
        return math.log(self.n_docs / max(self.df.get(lemma, 0), 1))

    def is_rare(self, lemma: str) -> bool:
        return self(lemma) > self.rare_threshold


def build_idf(passages: Iterable[Analyzed], rare_percentile: float = 90.0) -> Idf:
    """Passage-level document frequencies -> idf = log(N / df)."""
    df: dict[str, int] = {}
    n_docs = 0
    for passage in passages:
        n_docs += 1
        for lemma in set(passage.lemmas):
            df[lemma] = df.get(lemma, 0) + 1
    idf = Idf(n_docs, df, 0.0)
    # Threshold over lemma occurrences (weight = df): "rarer than 90% of what one reads".
    pairs = sorted((idf(lemma), count) for lemma, count in df.items())
    total = sum(count for _, count in pairs)
    cumulative = 0
    threshold = pairs[-1][0] if pairs else 0.0
    for value, count in pairs:
        cumulative += count
        if cumulative >= rare_percentile / 100.0 * total:
            threshold = value
            break
    idf.rare_threshold = threshold
    return idf


# --------------------------------------------------------------------------- metrics


def _longest_contiguous(left: Sequence[str], right: Sequence[str]) -> int:
    row = [0] * (len(right) + 1)
    longest = 0
    for ltoken in left:
        previous_row = row
        row = [0] * (len(right) + 1)
        for index, rtoken in enumerate(right, start=1):
            if ltoken == rtoken:
                row[index] = previous_row[index - 1] + 1
                if row[index] > longest:
                    longest = row[index]
    return longest


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def pair_metrics(q: Analyzed, p: Analyzed, idf: Idf) -> dict[str, Any]:
    """Lexical difficulty metrics of one (query, passage) pair.

    A query lemma counts as present in the passage when the lemma or any of its
    surface forms occurs among the passage lemmas or surface forms.
    """
    p_set = set(p.lemmas)
    p_forms = p.forms()
    q_surfaces = q.surface_of()

    def present(lemma: str) -> bool:
        return lemma in p_forms or any(s in p_forms for s in q_surfaces.get(lemma, ()))

    q_content = _unique(q.content)
    matched = [l for l in q_content if present(l)]
    missing = [l for l in q_content if not present(l)]
    weight_all = sum(idf(l) for l in q_content)
    weight_hit = sum(idf(l) for l in matched)
    entities = _unique(q.entity_like)
    entities_hit = [l for l in entities if present(l)]
    longest = _longest_contiguous(q.lemmas, p.lemmas)
    rare_hits = [l for l in matched if idf.is_rare(l)]
    return {
        "n_surface_matched": sum(1 for l in matched if l not in p_set),
        "idf_coverage": weight_hit / weight_all if weight_all > 0 else None,
        "coverage": len(matched) / len(q_content) if q_content else None,
        "longest_copied_ngram": longest,
        "copy_density": longest / len(q.lemmas) if q.lemmas else 0.0,
        "entity_preservation": len(entities_hit) / len(entities) if entities else None,
        "n_entity_like": len(entities),
        "has_rare_match": bool(rare_hits),
        "n_rare_matched": len(rare_hits),
        "max_matched_idf": max((idf(l) for l in matched), default=0.0),
        "q_len_content": len(q_content),
        "q_len_lemmas": len(q.lemmas),
        "p_len_lemmas": len(p.lemmas),
        "q_matched": matched,
        "q_missing": missing,
        "q_entity_like": entities,
    }


# --------------------------------------------------------------------------- summary

SUMMARY_METRICS = (
    "idf_coverage",
    "coverage",
    "longest_copied_ngram",
    "copy_density",
    "entity_preservation",
    "max_matched_idf",
    "q_len_content",
)
FLAG_METRICS = ("has_rare_match",)


def percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def auc(low: Sequence[float], high: Sequence[float]) -> float | None:
    """P(value from ``high`` > value from ``low``), ties count one half (Mann-Whitney)."""
    if not low or not high:
        return None
    ordered = sorted(low)
    import bisect

    wins = 0.0
    for value in high:
        wins += bisect.bisect_left(ordered, value) + 0.5 * (
            bisect.bisect_right(ordered, value) - bisect.bisect_left(ordered, value)
        )
    return wins / (len(low) * len(high))


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows)}
    for metric in SUMMARY_METRICS:
        values = [float(r[metric]) for r in rows if r.get(metric) is not None]
        summary[metric] = {
            "n": len(values),
            "mean": sum(values) / len(values) if values else None,
            **{f"p{p}": percentile(values, p) for p in (10, 25, 50, 75, 90)},
        }
    for flag in FLAG_METRICS:
        summary[flag] = sum(1 for r in rows if r.get(flag)) / len(rows) if rows else None
    covered = [r["idf_coverage"] for r in rows if r.get("idf_coverage") is not None]
    for cut in (0.5, 0.8, 0.9):
        summary[f"idf_coverage_gt_{cut}"] = (
            sum(1 for v in covered if v > cut) / len(covered) if covered else None
        )
    summary["entity_any"] = sum(1 for r in rows if r.get("n_entity_like")) / len(rows) if rows else None
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_summary(
    summaries: dict[str, dict[str, Any]],
    per_dataset_rows: dict[str, Sequence[dict[str, Any]]],
    contrasts: Sequence[tuple[str, str]] = (),
    meta: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = ["# Lexical difficulty calibration", ""]
    if meta:
        for key, value in meta.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines += [
        "## IDF-weighted query coverage",
        "",
        "| dataset | n | p10 | p25 | median | p75 | p90 | mean | >0.5 | >0.8 | >0.9 | has_rare_match |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        m = s["idf_coverage"]
        lines.append(
            f"| {name} | {s['n']} | {_fmt(m['p10'])} | {_fmt(m['p25'])} | {_fmt(m['p50'])} | "
            f"{_fmt(m['p75'])} | {_fmt(m['p90'])} | {_fmt(m['mean'])} | "
            f"{_fmt(s['idf_coverage_gt_0.5'])} | {_fmt(s['idf_coverage_gt_0.8'])} | "
            f"{_fmt(s['idf_coverage_gt_0.9'])} | {_fmt(s['has_rare_match'])} |"
        )
    for metric in SUMMARY_METRICS[1:]:
        lines += [
            "",
            f"## {metric}",
            "",
            "| dataset | n | p10 | p25 | median | p75 | p90 | mean |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, s in summaries.items():
            m = s[metric]
            extra = f" (queries with entity-like tokens: {_fmt(s['entity_any'])})" if metric == "entity_preservation" else ""
            lines.append(
                f"| {name}{extra} | {m['n']} | {_fmt(m['p10'])} | {_fmt(m['p25'])} | {_fmt(m['p50'])} | "
                f"{_fmt(m['p75'])} | {_fmt(m['p90'])} | {_fmt(m['mean'])} |"
            )
    if contrasts:
        lines += [
            "",
            "## Separation (AUC = P(metric is higher in the second dataset), 0.5 = no separation)",
            "",
            "| metric | " + " | ".join(f"{a} vs {b}" for a, b in contrasts) + " |",
            "|---|" + "---|" * len(contrasts),
        ]
        for metric in SUMMARY_METRICS + FLAG_METRICS:
            cells = []
            for a, b in contrasts:
                low = [float(r[metric]) for r in per_dataset_rows.get(a, []) if r.get(metric) is not None]
                high = [float(r[metric]) for r in per_dataset_rows.get(b, []) if r.get(metric) is not None]
                cells.append(_fmt(auc(low, high)))
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- io


def extract_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    query = row.get("query") or row.get("question")
    passage: Any = None
    if "pos" in row:
        pos = row["pos"]
        passage = pos[0] if isinstance(pos, list) and pos else pos if isinstance(pos, str) else None
    elif "texts" in row and "labels" in row:
        for text, label in zip(row["texts"], row["labels"]):
            if label in (1, True, "1", "true", "True"):
                passage = text
                break
    elif "positive" in row:
        passage = row["positive"]
    if not isinstance(query, str) or not isinstance(passage, str) or not query.strip() or not passage.strip():
        return None
    return query, passage


def read_pairs(path: Path, *, max_lines: int | None, sample: int | None, seed: int) -> list[dict[str, Any]]:
    """Read ``(query, pos)`` rows; sample ``sample`` of the first ``max_lines`` rows."""
    pairs: list[dict[str, Any]] = []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line in islice(handle, max_lines):
            if not line.strip():
                continue
            row = json.loads(line)
            pair = extract_pair(row)
            if pair is None:
                skipped += 1
                continue
            pairs.append({"query_id": row.get("query_id", row.get("id")), "query": pair[0], "pos": pair[1]})
    if skipped:
        print(f"  {path.name}: skipped {skipped} rows without a usable positive", file=sys.stderr)
    if sample and sample < len(pairs):
        pairs = random.Random(seed).sample(pairs, sample)
    return pairs


def read_pool_texts(path: Path, field: str, max_lines: int | None) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in islice(handle, max_lines):
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get(field)
            if value is None:
                pair = extract_pair(row)
                value = pair[1] if pair else None
            if isinstance(value, list):
                value = value[0] if value else None
            if isinstance(value, str) and value.strip():
                texts.append(value)
    return texts


def _parse_input(spec: str) -> tuple[str, Path]:
    if "=" in spec and not Path(spec).exists():
        name, _, path = spec.partition("=")
        return name, Path(path)
    return Path(spec).stem, Path(spec)


# --------------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", required=True, help="NAME=path.jsonl (repeatable)")
    parser.add_argument("--output", help="metrics JSONL (single --input only)")
    parser.add_argument("--output-dir", help="directory for <NAME>.jsonl metrics (any number of inputs)")
    parser.add_argument("--idf-pool", action="append", default=[], help="JSONL of reference passages (repeatable)")
    parser.add_argument("--pool-field", default="text")
    parser.add_argument("--pool-max-lines", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None, help="rows sampled per input")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lines", type=int, default=None, help="only read the first N rows of each input")
    parser.add_argument("--summary", help="write the markdown summary here")
    parser.add_argument("--contrast", action="append", default=[], help="LOW:HIGH dataset names for AUC")
    parser.add_argument("--cache", help="lemma cache JSONL (optional)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    inputs = [_parse_input(spec) for spec in args.input]
    if args.output and len(inputs) != 1:
        parser.error("--output works with exactly one --input; use --output-dir otherwise")
    if not args.output and not args.output_dir:
        parser.error("give --output or --output-dir")

    started = time.time()
    datasets = {
        name: read_pairs(path, max_lines=args.max_lines, sample=args.sample, seed=args.seed)
        for name, path in inputs
    }
    for name, rows in datasets.items():
        print(f"{name}: {len(rows)} pairs", file=sys.stderr)
    pool_texts: list[str] = []
    for pool_path in args.idf_pool:
        texts = read_pool_texts(Path(pool_path), args.pool_field, args.pool_max_lines)
        print(f"pool {pool_path}: {len(texts)} passages", file=sys.stderr)
        pool_texts.extend(texts)

    nlp = load_pipeline(args.model_dir, args.threads)
    cache = LemmaCache(Path(args.cache)) if args.cache else None
    lemma_started = time.time()
    analyzed: dict[str, tuple[list[Analyzed], list[Analyzed]]] = {}
    for name, rows in datasets.items():
        queries = analyze_texts(nlp, [r["query"] for r in rows], batch_size=args.batch_size, cache=cache, label=f"{name} queries")
        passages = analyze_texts(nlp, [r["pos"] for r in rows], batch_size=args.batch_size, cache=cache, label=f"{name} passages")
        analyzed[name] = (queries, passages)
    pool_analyzed = analyze_texts(nlp, pool_texts, batch_size=args.batch_size, cache=cache, label="idf pool")
    lemma_seconds = time.time() - lemma_started

    # One IDF pool shared by all inputs: every positive plus the external pool, deduplicated by text.
    seen: set[str] = set()
    pool: list[Analyzed] = []
    for name, rows in datasets.items():
        for row, passage in zip(rows, analyzed[name][1]):
            if row["pos"] not in seen:
                seen.add(row["pos"])
                pool.append(passage)
    for text, passage in zip(pool_texts, pool_analyzed):
        if text not in seen:
            seen.add(text)
            pool.append(passage)
    idf = build_idf(pool)
    print(f"idf pool: {idf.n_docs} passages, {len(idf.df)} lemmas, rare idf > {idf.rare_threshold:.2f}", file=sys.stderr)

    per_dataset_rows: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name, rows in datasets.items():
        queries, passages = analyzed[name]
        metrics_rows = []
        for row, q, p in zip(rows, queries, passages):
            record = {"dataset": name, "query_id": row["query_id"], "query": row["query"]}
            record.update(pair_metrics(q, p, idf))
            metrics_rows.append(record)
        per_dataset_rows[name] = metrics_rows
        summaries[name] = summarize(metrics_rows)
        out_path = Path(args.output) if args.output else Path(args.output_dir) / f"{name}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for record in metrics_rows:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{name}: wrote {out_path}", file=sys.stderr)

    contrasts = [tuple(c.split(":", 1)) for c in args.contrast]
    meta = {
        "backend": f"stanza (pl, tokenize+pos+lemma, CPU) model_dir={args.model_dir}",
        "inputs": ", ".join(f"{name}={path}" for name, path in inputs),
        "sample": f"{args.sample} rows per input, seed {args.seed}, first {args.max_lines or 'all'} rows as candidates",
        "idf_pool": f"{idf.n_docs} passages ({len(seen)} unique texts: all sampled positives + {len(pool_texts)} external), "
        f"{len(idf.df)} lemmas, rare = idf > {idf.rare_threshold:.2f} (90th df-weighted percentile)",
        "stopwords": f"{len(POLISH_STOPWORDS)} lemma-level stopwords",
        "wall_time": f"{time.time() - started:.0f}s total, {lemma_seconds:.0f}s lemmatising",
    }
    text = render_summary(summaries, per_dataset_rows, contrasts, meta)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
