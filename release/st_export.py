#!/usr/bin/env python
"""Package a SPLADE document encoder and a static query table as one Sentence Transformers ``SparseEncoder``.

The result is an ordinary model directory for ``push_to_hub``: a ``StaticQueryRouter`` with the route ``query`` =
``SparseStaticExpansion`` (the table, no transformer) and ``document`` = the SPLADE checkpoint's own modules
(``MLMTransformer`` + ``SpladePooling``). The two custom classes travel as ``static_query_expansion.py`` in the
model root, so loading needs ``trust_remote_code=True``:

    model = SparseEncoder("out-dir-or-repo", trust_remote_code=True)
    model.encode_query([...]); model.encode_document([...])

Usage:
  .venv/bin/python -m release.st_export --doc-model models/arm-dense-kl-n5kj-ens400-r2v12-bias005 \
      --table models/static-r2v12b/credit-k8-kd --out models/st/r2v12b-static-credit-k8-kd
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REMOTE_MODULE = "static_query_expansion"


def _as_remote_code(out: Path) -> None:
    """The custom classes as remote code: one file in the root, every reference renamed to it."""
    shutil.copy(Path(__file__).with_name("st_static.py"), out / f"{REMOTE_MODULE}.py")
    modules = json.loads((out / "modules.json").read_text())
    for m in modules:
        m["type"] = m["type"].replace("release.st_static.", f"{REMOTE_MODULE}.")
    (out / "modules.json").write_text(json.dumps(modules, indent=2) + "\n")
    for cfg in out.rglob("router_config.json"):
        data = json.loads(cfg.read_text())
        data["types"] = {k: v.replace("release.st_static.", f"{REMOTE_MODULE}.") for k, v in data["types"].items()}
        cfg.write_text(json.dumps(data, indent=4) + "\n")


def export_query_only(table: str | Path, out: str | Path, device: str = "cpu") -> Path:
    """The query side alone (a few MB, no transformer): for services that only encode queries against an index
    built elsewhere; ST also skips the per-call walk over the document transformer's modules."""
    from sentence_transformers import SparseEncoder

    from release.st_static import SparseStaticExpansion

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    model = SparseEncoder(modules=[SparseStaticExpansion.from_table_dir(str(table))], similarity_fn_name="dot", device=device)
    model.save(str(out))
    _as_remote_code(out)
    return out


def export(doc_model: str | Path, table: str | Path, out: str | Path, device: str = "cpu") -> Path:
    from sentence_transformers import SparseEncoder

    from release.st_static import SparseStaticExpansion, StaticQueryRouter

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    doc = SparseEncoder(str(doc_model), device=device)
    query = SparseStaticExpansion.from_table_dir(str(table), tokenizer=doc.tokenizer)
    vocab = len(doc.tokenizer.get_vocab())
    if query.output_dim != vocab:
        raise SystemExit(f"table output_dim {query.output_dim} != document vocabulary {vocab}")
    router = StaticQueryRouter.for_query_document(query_modules=[query], document_modules=list(doc))
    model = SparseEncoder(modules=[router], similarity_fn_name="dot", device=device)
    model.save(str(out))

    _as_remote_code(out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doc-model", required=True, help="SPLADE checkpoint in the sentence-transformers layout")
    p.add_argument("--table", required=True, help="static-sparse-query/1 directory (distill.static_query output)")
    p.add_argument("--out", required=True)
    p.add_argument("--query-only-out", default=None, help="also write the query side alone (no document encoder) here")
    args = p.parse_args(argv)
    out = export(args.doc_model, args.table, args.out)
    if args.query_only_out:
        export_query_only(args.table, args.query_only_out)
    print(json.dumps({"out": str(out), "files": sorted(str(f.relative_to(out)) for f in out.rglob("*") if f.is_file())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
