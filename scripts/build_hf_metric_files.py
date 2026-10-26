#!/usr/bin/env python
"""Assemble the per-row difficulty files meant for a Hugging Face dataset upload.

One JSONL per arm under results/lexical_difficulty/hf_upload/, joining:
  - the training row itself (query_id, query, positive text, negatives count)
  - the Stanza lexical metrics (results/lexical_difficulty/full/<arm>.jsonl)
  - the base-model difficulty (results/base_loss/<arm>.jsonl)
plus a README.md documenting every field. Arms whose metric files are missing
get whatever is available; the README lists what each file contains.
The upload itself is left to the user.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LEXICAL_FIELDS = ["idf_coverage", "coverage", "n_surface_matched", "longest_copied_ngram", "copy_density",
                  "entity_preservation", "n_entity_like", "has_rare_match", "n_rare_matched", "max_matched_idf",
                  "q_len_content", "q_len_lemmas", "p_len_lemmas", "q_matched", "q_missing", "q_entity_like"]
LOSS_FIELDS = ["loss_own", "margin_own", "loss_inbatch", "margin_inbatch", "pos_score", "query_active_dims", "pos_active_dims"]

README = """# Trudność par (zapytanie, pozytyw) w zbiorach do dostrajania polish-SPLADE

Każdy plik `<arm>.jsonl` ma jeden wiersz na pytanie treningowe z danego ramienia
eksperymentu (2073 wierszy po filtrze score rerankera > 23,50). Pola:

Wiersz treningowy
- `dataset` - nazwa ramienia (fiqa-2073, squadv2-2073, msmarco-clean-2073, trivia-2073, wikisynth-2073)
- `query_id`, `query` - identyfikator i treść pytania
- `pos` - tekst pozytywu (pierwszy pozytyw, ten używany w treningu)
- `n_neg` - liczba twardych negatywów w wierszu (w treningu używano 6)

Pokrycie leksykalne (Stanza pl, lematy; słowa treści = bez stopwordów i interpunkcji)
- `coverage` - udział lematów treści pytania obecnych w pozytywie
- `idf_coverage` - jak wyżej, ale ważony idf lematu (pula idf: pozytywy trzech ramion naturalnych + 3000 chunków Wikipedii); `null`, gdy pytanie nie ma lematów treści
- `n_surface_matched` - lematy pytania, które w pozytywie występują w identycznej formie powierzchniowej
- `longest_copied_ngram`, `copy_density` - najdłuższy wspólny n-gram lematów i udział skopiowanych n-gramów (sygnał "pytanie przepisane z pasażu")
- `entity_preservation`, `n_entity_like`, `q_entity_like` - czy tokeny wyglądające na nazwy własne/liczby z pytania są w pozytywie
- `has_rare_match`, `n_rare_matched`, `max_matched_idf` - czy pytanie dzieli z pozytywem lemat rzadki (wysoki idf) i jaki najwyższy
- `q_len_content`, `q_len_lemmas`, `p_len_lemmas` - długości w lematach
- `q_matched`, `q_missing` - listy lematów pytania obecnych/nieobecnych w pozytywie

Trudność dla modelu bazowego (sdadas/polish-splade, bez dostrajania, długość 256)
- `pos_score` - iloczyn skalarny zapytanie·pozytyw
- `loss_own`, `margin_own` - entropia krzyżowa i margines (pozytyw minus najlepszy z własnych 6 negatywów) liczone tylko wobec negatywów tego wiersza; niezależne od batcha, nadają się do rankingu trudności
- `loss_inbatch`, `margin_inbatch` - to samo liczone tak, jak widzi to trening (batch 4 pytań, negatywy in-batch + własne; zależy od losowego składu batcha)
- `query_active_dims`, `pos_active_dims` - liczba niezerowych wymiarów reprezentacji rzadkiej

Uwaga: w kalibracji (FiQA, które działa w dostrajaniu, vs trivia, które psuje model)
`idf_coverage` nie rozdziela tych zbiorów (AUC 0,54), więc nie jest to walidowany
filtr jakości, tylko opis danych.

Pliki w tym katalogu:
{files}
"""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["fiqa-2073", "squadv2-2073", "msmarco-clean-2073", "trivia-2073", "wikisynth-2073"])
    ap.add_argument("--arms-dir", type=Path, default=Path("datasets/arms"))
    ap.add_argument("--lexical-dir", type=Path, default=Path("results/lexical_difficulty/full"))
    ap.add_argument("--loss-dir", type=Path, default=Path("results/base_loss"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/lexical_difficulty/hf_upload"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for arm in args.arms:
        rows = read_jsonl(args.arms_dir / f"{arm}.jsonl")
        if not rows:
            lines.append(f"- `{arm}.jsonl`: brak pliku ramienia, pominięto"); continue
        lex = {str(r["query_id"]): r for r in read_jsonl(args.lexical_dir / f"{arm}.jsonl")}
        loss = {str(r["query_id"]): r for r in read_jsonl(args.loss_dir / f"{arm}.jsonl")}
        n_lex = n_loss = 0
        with (args.out_dir / f"{arm}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                qid = str(r["query_id"])
                rec = {"dataset": arm, "query_id": qid, "query": r["query"], "pos": r["pos"][0], "n_neg": len(r.get("neg", []))}
                if qid in lex:
                    n_lex += 1
                    rec.update({k: lex[qid].get(k) for k in LEXICAL_FIELDS})
                if qid in loss:
                    n_loss += 1
                    rec.update({k: loss[qid].get(k) for k in LOSS_FIELDS})
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        lines.append(f"- `{arm}.jsonl`: {len(rows)} wierszy, metryki leksykalne dla {n_lex}, trudność bazowa dla {n_loss}")
        print(lines[-1])
    (args.out_dir / "README.md").write_text(README.format(files="\n".join(lines)), encoding="utf-8")
    print("wrote", args.out_dir / "README.md")


if __name__ == "__main__":
    main()
