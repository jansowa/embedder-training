# Adnotacja par (zapytanie, pasaż) modelem LLM: schemat v1

Projekt z 31.10.2026, do wdrożenia w repo kuchni danych (patrz PLAN-2026-10-31 §10).
Model wykonawczy: Qwen3 27B przez vLLM (OpenAI-compatible), dekodowanie ograniczone do
schematu JSON (`guided_json` / structured outputs), temperatura 0.

## 1. Czego oczekujemy od LLM, a czego nie

LLM ma robić tylko to, czego nie policzą tańsze narzędzia:

| zadanie | kto | po co |
|---|---|---|
| kategorie treści zapytania i pasażu (dziedzina, forma, typ odpowiedzi, gatunek, kontekst) | LLM | skład i zróżnicowanie cięć; mapa składników na zadania PIRB |
| stopniowana trafność pary (0–3) i sposób, w jaki pasaż odpowiada | LLM | bramka pozytywów po polsku; akapity-mosty (hotpotqa); fałszywe negatywy (te same pola na parach z negatywem) |
| jakość polszczyzny zapytania i pasażu | LLM | jedyna wada, której roberta-v3 i gemma nie widzą |
| ogólność zapytania | LLM | ryzyko fałszywych negatywów w batchu; werdykt `zbyt_ogolne` z audytu v2 |
| pokrycie leksykalne, długości, ogon studencki, wyniki rerankerów | skrypty (lexical_difficulty, speakleash_pool_diag, pliki źródłowe) | nie pytać LLM o rzeczy policzalne |
| ocena negatywów bez flagi | nikt | 20k negatywów na plik; sędzi się tylko kandydatów (v3 neg ≥ najsłabszy pos) |

Jednostka pracy: **jedna para (zapytanie, pasaż) = jedno wywołanie**, ten sam prompt dla
pozytywów i dla kandydatów na fałszywe negatywy. Pola zapytania powtarzają się w każdej
parze tego samego wiersza; zgodność tych powtórzeń jest darmową miarą wiarygodności modelu.

## 2. Pola

Wartości to krótkie tokeny ASCII (stabilniejsze w JSON małego modelu), definicje po polsku
w prompcie. 12 pól, każde 3–9 wartości, bez pól wolnotekstowych i bez samooceny pewności.

### Zapytanie (7 pól)

1. **`domain`** — dziedzina potrzeby informacyjnej. Lista dopasowana do korpusów PIRB
   (fiqa = finance, legal-questions = law, nfcorpus/scifact/treccovid = medicine/science,
   allegro-faq/mfaq = shopping/everyday, czy-wiesz/wiki-def/1z10 = encyklopedia ogólna):
   `finance_economy`, `law_administration`, `medicine_health`, `science_nature`,
   `technology_it`, `history`, `geography_travel`, `politics_society`, `culture_entertainment`
   (film, muzyka, literatura, gry), `sport`, `business_work`, `everyday_shopping_home`
   (zakupy, dom, jedzenie, poradniki), `education_language`, `religion_philosophy`, `other`.
2. **`query_form`** — jak sformułowano zapytanie: `full_question` (pełne zdanie pytające),
   `short_question` (fragment pytający bez czasownika: "stolica Mongolii"), `keywords`
   (same rzeczowniki/nazwy), `claim` (zdanie oznajmujące do zweryfikowania), `instruction`
   ("wyjaśnij…", "podaj…"), `title_or_name` (tytuł, nazwa własna).
3. **`answer_type`** — czego szuka pytający: `entity` (osoba, miejsce, rzecz, organizacja),
   `number_or_date`, `yes_no`, `definition`, `explanation_or_cause`, `how_to_or_advice`,
   `opinion_or_comparison`, `list`, `claim_verification` (dla `claim`), `other`.
4. **`specificity`** — ile dobrych pasaży może istnieć: `specific` (jedna odpowiedź, jeden
   fakt), `broad` (szeroki temat, setki pasujących pasaży), `ambiguous` (brakuje kontekstu,
   odniesienia bez desygnatu: "kiedy to się stało"). Odpowiednik `zbyt_ogolne` z audytu v2,
   z rozdzieleniem ogólności od niejednoznaczności.
5. **`hops`** — ile faktów trzeba połączyć, żeby odpowiedzieć: `single` albo `multi` (np.
   "w którym roku urodził się reżyser filmu X"). Wykrywa wielokrokowość hotpotqa niezależnie
   od pasażu.
6. **`cultural_context`** — osadzenie: `polish`, `foreign` (konkretny kraj/kultura inna niż
   Polska), `universal` (nauka, zdrowie, definicje). Zbiory tłumaczone z angielskiego są
   amerykocentryczne; PIRB nie.
7. **`query_polish`** — jakość polszczyzny zapytania, ścisła definicja z audytu v2 (wadą są
   uszkodzone znaki, przekręcone nazwy, kalki dające bezsens, urwane zdania, słowa spoza
   polskiego i angielskiego; wadą nie są angielskie nazwy, mała litera, styl telegraficzny):
   `natural`, `awkward_but_clear` (kalka, nienaturalny szyk, ale zrozumiałe), `broken`
   (błędy zmieniające sens, urwane), `untranslated_parts` (fragmenty w innym języku),
   `nonsense`.

### Pasaż i para (5 pól)

8. **`relevance`** — stopniowana trafność, liczona wyłącznie na podstawie tego pasażu:
   `3` pasaż odpowiada wprost i w całości (dla `claim`: potwierdza lub obala wprost),
   `2` odpowiada częściowo lub wymaga połączenia z wiedzą spoza pasażu,
   `1` na temat, ale nie odpowiada,
   `0` nie na temat.
   `odpowiadalne` z audytu v2 odpowiada `relevance == 3`. Fałszywy negatyw: para
   (zapytanie, negatyw) z `relevance >= 2`.
9. **`answer_location`** — jak pasaż odpowiada (tylko gdy `relevance >= 2`, inaczej `none`):
   `explicit` (odpowiedź stoi w tekście), `inferred` (trzeba wywnioskować z treści pasażu),
   `needs_other_source` (pasaż daje tylko ogniwo, reszta jest gdzie indziej: akapit-most),
   `none`.
10. **`passage_genre`** — gatunek tekstu: `encyclopedia`, `news`, `scientific_or_technical`,
    `legal_or_official`, `forum_or_qa` (wpis użytkownika, odpowiedź na forum),
    `faq_or_manual` (FAQ, instrukcja, regulamin sklepu), `product_or_offer`,
    `literary_or_other`.
11. **`passage_polish`** — ta sama skala co `query_polish`.
12. **`passage_self_contained`** — czy pasaż jest zrozumiały sam w sobie: `yes`, `no`
    (urwany w pół zdania, zaimki bez odniesienia, tabela bez nagłówka).

## 3. Co z tego wynika dla cięć (mapowanie pola → decyzja)

| decyzja | pola | przepis |
|---|---|---|
| bramka pozytywów po polsku | `relevance`, `query_polish`, `passage_polish` | pozytyw ⇔ `relevance == 3` i żadna skala polszczyzny poniżej `awkward_but_clear` |
| hotpotqa bez mostów | `answer_location`, `hops` | jeden pozytyw na wiersz: `explicit` przed `inferred`; odrzuć `needs_other_source` |
| fałszywe negatywy | `relevance` na kandydatach | usuń negatyw z `relevance >= 2` albo przenieś go do pozytywów |
| zróżnicowanie składu | `domain`, `answer_type`, `query_form`, `passage_genre` | cięcie stratyfikowane: kwoty na dziedziny niedoreprezentowane względem PIRB (finance, law, medicine, everyday_shopping) |
| ryzyko FN w batchu | `specificity` | ogranicz udział `broad`/`ambiguous`, albo użyj ich tylko z sampler-em no_duplicates po temacie |
| kontekst | `cultural_context` | sprawdzić, czy `foreign` koreluje ze stratą na mkqa/allegro; ewentualna kwota na `polish`/`universal` |

Nadmiarowe pola (podejrzewam `passage_self_contained`, `cultural_context`) wypadają po
kalibracji, nie przed nią.

## 4. Prompt: zasady dla modelu 27B

- System: jedna strona, po polsku, definicje pól z jednym przykładem na wartość tam, gdzie
  granica jest nieoczywista (`short_question` vs `keywords`, `relevance` 2 vs 1,
  `awkward_but_clear` vs `broken`). Bez prośby o uzasadnienia.
- Wejście: zapytanie, pasaż obcięty do 2 000 znaków (do klasyfikacji wystarczy), źródło
  zbioru NIE jest podawane (żeby nie podpowiadać gatunku ani dziedziny).
- Wyjście: JSON o stałej kolejności kluczy, wymuszony schematem po stronie vLLM; parser nie
  musi naprawiać niczego. Ustalony porządek: najpierw pola zapytania, potem pasażu, na końcu
  `relevance` i `answer_location` (model najpierw "czyta", potem ocenia).
- Jeden lub dwa krótkie przykłady few-shot (zapytanie z `claim` i z `short_question`), bo
  bez nich małe modele mylą `query_form` z `answer_type`.
- Koszt: ok. 1 000 tokenów wejścia na parę, ok. 60 tokenów wyjścia.

## 5. Kalibracja przed produkcją

1. Zbiór kalibracyjny: 240 par, po 40 z każdego z sześciu plików (fiqa-2073, squadv2-2073,
   msmarco-clean-2073, nq, hotpotqa, fever), w tym 60 kandydatów na fałszywe negatywy.
   Etykiety odniesienia: model silny (Fable) na całości, użytkownik na 50 parach.
2. Miara per pole: zgodność Qwen vs odniesienie (dokładność i κ Cohena); pole zostaje,
   jeśli κ ≥ 0.6; pole z κ 0.4–0.6 zostaje tylko gdy decyduje o cięciu (tabela §3);
   poniżej 0.4 wypada albo zmienia definicję.
3. Spójność: dla wierszy z kilkoma pozytywami udział zgodnych pól zapytania między parami.
   Poniżej 90 % na którymś polu oznacza, że pole jest źle zdefiniowane.
4. Dopiero potem produkcja: trzy pule speakleash po 8 000 wierszy (cięcia 2 073 są ich
   podzbiorami) i trzy cięcia odniesienia, łącznie ok. 36 000 par, plus ok. 1 500 par
   kandydatów na fałszywe negatywy.

## 6. Format wyjściowy w kuchni danych

Plik boczny `annotations/<zbiór>/llm_v1.jsonl`, rekord na parę:

```json
{"query_id": "...", "side": "pos", "index": 0, "model": "qwen3-27b", "prompt": "llm_v1",
 "domain": "finance_economy", "query_form": "full_question", "answer_type": "explanation_or_cause",
 "specificity": "specific", "hops": "single", "cultural_context": "universal", "query_polish": "natural",
 "passage_genre": "forum_or_qa", "passage_polish": "awkward_but_clear", "passage_self_contained": "yes",
 "relevance": 3, "answer_location": "explicit"}
```

Źródłowe wiersze pozostają nietknięte; `cut.py` łączy je z plikami bocznymi po
(`query_id`, `side`, `index`). Proweniencja (model, wersja promptu) jest w każdym rekordzie,
żeby skale różnych narzędzi nigdy nie trafiły do jednej kolumny.
