# LLM-as-a-judge prompts used on the MS MARCO-pl pairs (verbatim)

Every prompt below is extracted programmatically (Python `ast`) from
`scripts/task06_night_jobs_remote.py` in the doc2query project at
`/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/doc2query`, at the git
revisions named in each section. The raw strings are in `docs/judge_prompts/`
next to this file; the SHA-256 of each is given so a copy can be checked. The
earlier hand-typed version of this document is superseded; nothing here is
retyped.

## Which prompt produced the 384 576 verdicts

`artifacts/task06/night_jobs_v3/verdicts/night_jobs.journal.jsonl`, job
`sft_full_audit`, run 2026-08-31 19:36 to 21:54. Counted over the journal:

| verdicts | `prompt_version` in the journal | layout | script revision |
|---:|---|---|---|
| 36 089 | `task06-night-jobs-pl-v1` | system = `SYSTEM`; user = `SFT_AUDIT_V2_TEMPLATE` filled with the passage and query (axes inside the user message) | `a957041` (19:39) |
| 348 487 | `task06-night-jobs-pl-v2-prefix-first` | system = `SYS_SFT_AUDIT_V2` (axes in the system message); user = `DATA_SFT_AUDIT` filled with the passage and query | `a456fb2` (20:55) then `25d35ad` (21:10) = HEAD |

The axes text is byte-identical in all three revisions; only the placement
(user vs system message) and one detail differ: between 20:55 and 21:10 the
system prompt still carried doubled braces `{{` `}}` around the JSON
example (a leftover from `.format` templates, removed in `25d35ad`). The journal
has no timestamps, so the share of verdicts made in that 15-minute window is
unknown; at the observed rate it is of the order of a few tens of thousands.
The file is unchanged since `25d35ad`, so HEAD is the prompt that produced the
large majority of the verdicts.

Protocol (from the script): `temperature = 0`, JSON response format, thinking
disabled, model served by vLLM on the owner's inference server.

## 1. Prefix-first layout (348 487 verdicts) — HEAD = `25d35ad`

**system** (`docs/judge_prompts/SYS_SFT_AUDIT_V2.txt`, sha256 `5424665f48c3031300a67df748839c5d8128b045560e92293847ebf25e540c9d`)

```
Jesteś precyzyjnym asystentem do budowy danych treningowych dla polskiego systemu wyszukiwania. Piszesz WYŁĄCZNIE po polsku i zwracasz wyłącznie JSON.

Dostaniesz pasaż i zapytanie z danych treningowych (tłumaczone maszynowo
z angielskiego). Oceń tę parę na czterech osiach, niezależnie od siebie:

1. `odpowiadalne` — czy na to zapytanie da się odpowiedzieć WYŁĄCZNIE na
   podstawie tego pasażu? false, gdy pasaż jest tylko w temacie albo odpowiada
   częściowo. Uwaga na synonimy i warianty terminów (np. odma/rozedma) — licz
   treść, nie dosłowne dopasowanie słów.
2. `polszczyzna` — ścisła definicja. WADĄ SĄ: uszkodzone znaki (â, Ã),
   przekręcone lub sklejone nazwy własne, kalki składniowe dające bezsens,
   urwane zdania, słowa z języka innego niż polski i angielski. WADĄ NIE SĄ:
   angielskie nazwy własne i terminy (walking dead, selenium, paracord),
   skróty stanów i miast (nj, ca, de), mała litera, styl telegraficzny
   zapytań, brak znaku zapytania.
3. `sensowne_zapytanie` — czy to realna potrzeba informacyjna użytkownika
   wyszukiwarki? false dla fragmentów zdań, testów z lukami, bełkotu.
4. `zbyt_ogolne` — czy zapytanie pasowałoby do tysięcy różnych pasaży?
   Krótkie nie znaczy ogólne: „czy DNA jest widoczne" jest konkretne.

Zwróć wyłącznie JSON:
{"odpowiadalne": <true|false>, "polszczyzna": <true|false>,
 "sensowne_zapytanie": <true|false>, "zbyt_ogolne": <true|false>,
 "glowny_problem": "<brak|nieodpowiadalne|tlumaczenie|niesensowne|zbyt_ogolne>"}
```

**user** (`docs/judge_prompts/DATA_SFT_AUDIT.txt`, sha256 `eb1711c76cd2550fef45f61bb0b4f125705f8d0fd5dc563c0f28e367bfaf1b12`), filled with `.format(passage=..., query=...)`:

```
Pasaż:
{passage}

Zapytanie (z danych treningowych, tłumaczone maszynowo z angielskiego):
{query}
```

Variant between 20:55 and 21:10 (`a456fb2`,
`docs/judge_prompts/prefix_first_escaped_a456fb2/SYS_SFT_AUDIT_V2.txt`, sha256
`8146c4792967382c9da028dd971d807cce336a454d66416f85388612128ba885`): identical except the JSON example reads

```
{"odpowiadalne": <true|false>, "polszczyzna": <true|false>,
 "sensowne_zapytanie": <true|false>, "zbyt_ogolne": <true|false>,
 "glowny_problem": "<brak|nieodpowiadalne|tlumaczenie|niesensowne|zbyt_ogolne>"}
```

## 2. Legacy layout (first 36 089 verdicts) — `a957041`

**system** (`docs/judge_prompts/SYSTEM.txt`, sha256 `0814cb122792251f6879d0c95ff524a303a8a3157f04b66d979874c9da923caf`)

```
Jesteś precyzyjnym asystentem do budowy danych treningowych dla polskiego systemu wyszukiwania. Piszesz WYŁĄCZNIE po polsku i zwracasz wyłącznie JSON.
```

**user** (`docs/judge_prompts/v1_era_a957041/SFT_AUDIT_V2_TEMPLATE.txt`, sha256
`6c74e039d587e85e600afb4b41c56e9bdd4568cdad53e5bc37d9ce35616052f7`; it is a `.format` template, so the doubled braces below reach
the model as single braces):

```
Pasaż:
{passage}

Zapytanie (z danych treningowych, tłumaczone maszynowo z angielskiego):
{query}

Oceń tę parę na czterech osiach, niezależnie od siebie:

1. `odpowiadalne` — czy na to zapytanie da się odpowiedzieć WYŁĄCZNIE na
   podstawie tego pasażu? false, gdy pasaż jest tylko w temacie albo odpowiada
   częściowo. Uwaga na synonimy i warianty terminów (np. odma/rozedma) — licz
   treść, nie dosłowne dopasowanie słów.
2. `polszczyzna` — ścisła definicja. WADĄ SĄ: uszkodzone znaki (â, Ã),
   przekręcone lub sklejone nazwy własne, kalki składniowe dające bezsens,
   urwane zdania, słowa z języka innego niż polski i angielski. WADĄ NIE SĄ:
   angielskie nazwy własne i terminy (walking dead, selenium, paracord),
   skróty stanów i miast (nj, ca, de), mała litera, styl telegraficzny
   zapytań, brak znaku zapytania.
3. `sensowne_zapytanie` — czy to realna potrzeba informacyjna użytkownika
   wyszukiwarki? false dla fragmentów zdań, testów z lukami, bełkotu.
4. `zbyt_ogolne` — czy zapytanie pasowałoby do tysięcy różnych pasaży?
   Krótkie nie znaczy ogólne: „czy DNA jest widoczne" jest konkretne.

Zwróć wyłącznie JSON:
{{"odpowiadalne": <true|false>, "polszczyzna": <true|false>,
 "sensowne_zapytanie": <true|false>, "zbyt_ogolne": <true|false>,
 "glowny_problem": "<brak|nieodpowiadalne|tlumaczenie|niesensowne|zbyt_ogolne>"}}
```

## 3. Answerability-only judge (other night jobs, not the SFT audit)

**system** (`docs/judge_prompts/SYS_ANSWERABLE.txt`, sha256 `7504b3d6b9be4b8af520dead2f6e3c688e2b93e0e3a5222ca78f5298c9c0db99`)

```
Jesteś precyzyjnym asystentem do budowy danych treningowych dla polskiego systemu wyszukiwania. Piszesz WYŁĄCZNIE po polsku i zwracasz wyłącznie JSON.

Dostaniesz pasaż i zapytanie. Czy pasaż ZAWIERA odpowiedź na zapytanie? true
tylko wtedy, gdy odpowiedź
da się wskazać w treści pasażu; false, gdy pasaż jest w temacie, ale odpowiedzi
nie podaje.

Zwróć wyłącznie JSON: {"answerable": <true|false>, "dowod": "<fragment pasażu albo pusty>"}
```

**user** (`docs/judge_prompts/DATA_ANSWERABLE.txt`, sha256 `b36c1cea9d61d86475b80094839629d3f09e37d120a44a226f4274ad47607712`)

```
Pasaż:
{passage}

Zapytanie:
{query}
```

## Reading the verdicts

"Clean" in the weekend experiments: `odpowiadalne`, `polszczyzna`,
`sensowne_zapytanie` true and `zbyt_ogolne` false; 80.5 % of pairs. Per the
doc2query notes the `polszczyzna` axis alone has low precision and is not worth
filtering on by itself; `odpowiadalne` carries most of the signal.

## Ready pipeline for next week (inference server, OpenAI-compatible)

`scripts/dataset_annotate_judge.py` in doc2query re-uses the four axes as a
standalone, resumable annotator. Note that its system message opens with a
different sentence ("Jesteś precyzyjnym audytorem danych treningowych dla
wyszukiwarki. Odpowiadasz wyłącznie obiektem JSON, bez komentarza.") and omits
`glowny_problem` and two parenthetical examples, so it is not byte-identical
to the prompt above. To reproduce the MS MARCO verdicts exactly, point the
annotator at `SYS_SFT_AUDIT_V2.txt` + `DATA_SFT_AUDIT.txt` instead, or run the
night-jobs script itself.

```bash
cd /media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/doc2query
# 1/3 local annotations, nothing removed, 23.5 gate recorded as a label
uv run python scripts/dataset_annotate_local.py \
    --dataset speakleash/squadv2 --split train \
    --output data/annot/squadv2.local.jsonl --min-positive-score 23.5
# 2/3 judge verdicts, journal is the source of truth, safe to interrupt
JUDGE_API_KEY=... uv run python scripts/dataset_annotate_judge.py \
    --input data/annot/squadv2.local.jsonl \
    --output data/annot/squadv2.judged.jsonl \
    --journal data/annot/squadv2.judge.journal.jsonl \
    --base-url http://<server>:8000/v1 --model <model> --concurrency 8 \
    --skip-rule quiz_query --skip-rule quiz_dump_passage
# 3/3 pick a policy, get a per-rule report
uv run python scripts/dataset_apply_filters.py --list-rules
uv run python scripts/dataset_apply_filters.py \
    --input data/annot/squadv2.judged.jsonl \
    --output data/train/squadv2.clean.jsonl \
    --rule not_answerable --rule too_general --rule quiz_query \
    --report data/annot/squadv2_filter_report.json
```

Local rules in step 3 (`src/doc2query/dataquality/rules.py`): record-level
`quiz_query`, `mojibake_query`, `duplicate_query`; positive-level
`quiz_dump_passage`, `citation_dump_passage`, `short_passage`,
`mojibake_passage`, `copy_risk`, `below_score_threshold`, `duplicate_passage`,
`not_answerable`, `nonsense_query`, `bad_polish`, `too_general`; negative-level
`negative_positive_elsewhere`, `negative_mojibake`. Column names for other
layouts: `--col-example-id`, `--col-query`, `--col-positive-texts`,
`--col-positive-ids` on step 1.

Caveat for non-translated data: both messages say "tłumaczone maszynowo z
angielskiego". For native or synthetic Polish queries that sentence primes the
judge for translation artefacts; dropping it makes a new prompt version.
