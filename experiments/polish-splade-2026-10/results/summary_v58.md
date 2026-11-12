# v58 (12.11, quantized regime, nDCG@10): native Polish FAQ rows in the student-mined arm

faq20 = the rows of the student-mined arm unchanged plus 20000 native Polish FAQ rows (WebFAQ v2 + clips/mqa after the selection funnel of the data kitchen), negatives mined by r2v12 under the same rules from the answers of the funnel's pool, the same ensemble teacher with the fixed dense scale, the same recipe (from soup-v12, 2 epochs). New arms are measured in their first bias form with at most 410 dims/doc.

## Result

| model | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| arm-dense-kl-n5kj-ens400-r2v12 | 64.20 | 43.15 | 67.02 | 404.4 | - | - | - |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias010 | 64.63 | 43.45 | 70.15 | 405.3 | 413.6 | 130.8 | 16461 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias013 | 64.64 | 43.40 | 69.74 | 376.5 | 384.0 | 122.2 | 14479 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias015 | 64.67 | 43.35 | 69.86 | 358.0 | - | - | - |
| arm-dense-kl-n5kj-mined-ens400-r2v12-s43-bias015 | 64.61 | 43.45 | 69.59 | 366.7 | - | - | - |
| arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 | 64.96 | 43.63 | 72.82 | 374.7 | 381.5 | 121.9 | 14079 |
| **arm-dense-kl-n5kj-mined-faq20-ens400-r2v12-bias015** | 64.84 | 43.54 | 70.39 | 377.9 | 385.8 | 127.6 | 15289 |

## Reading

arm-dense-kl-n5kj-mined-faq20-ens400-r2v12-bias015 64.84 / 43.54 at 377.9 dims/doc against the mined arm arm-dense-kl-n5kj-mined-ens400-r2v12-bias013 64.64 / 43.40 at 376.5: delta +0.20 / +0.14, trec-covid 70.39 against 69.74 -> at the edge of the seed noise on the means, but the gain sits on the FAQ tasks and the other tasks are flat on average: a source with its own task profile, to be judged on top of the other sources and not by the means alone. Per task: Allegro FAQ A +1.44, Allegro FAQ B +1.04, mfaq +0.63 (mean of the three +1.04); MAUPQA tasks mean +0.02; fiqa-pl +0.31; all other tasks mean +0.00. Against the dose arm of v56 arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 (64.96 / 43.63 at 374.7): delta -0.12 / -0.09.

Corrected on 12.11: the first version of this summary said 'no clear gain on the FAQ tasks either', against its own per-task table. All three sets move up together (tiny10, heldout8, trec-covid). The gain of the FAQ rows sits on Allegro FAQ A/B and mfaq, the gain of the SQuAD-v2 dose of v56 on arguana-pl, scifact-pl, fiqa-pl and the legal questions, so the two sources are complementary rather than interchangeable. Per added row the effect is similar: FAQ +0.20 / +0.14 for 20 000 rows, SQuAD-v2 +0.32 / +0.23 for 40 093. The largest losses of the FAQ arm are scifact-pl -0.58 and maupqa-mkqa -0.44. Both arms also train for more steps than the 20 k arm, which these numbers cannot separate from the data. Queue v60 measures the two sources together.

Seed noise of this recipe is 0.13-0.2 (one seed here). Nothing further was started by this queue.

### tiny10 per task

| task | mined arm (bias013) | dose arm (v56) | FAQ arm | delta vs mined arm |
|---|---:|---:|---:|---:|
| maupqa-gpt3-cc | 18.81 | 18.88 | 18.79 | -0.02 |
| maupqa-mkqa | 65.62 | 65.60 | 65.19 | -0.44 |
| maupqa-poleval2021-pairs | 83.84 | 83.95 | 84.01 | +0.16 |
| maupqa-templates | 76.31 | 76.57 | 76.54 | +0.23 |
| poleval-2022-test-A-allegro-faq (FAQ) | 77.85 | 78.80 | 79.30 | +1.44 |
| poleval-2022-test-B-allegro-faq (FAQ) | 74.47 | 74.77 | 75.51 | +1.04 |
| arguana-pl | 49.38 | 50.24 | 49.64 | +0.26 |
| nfcorpus-pl | 32.93 | 32.84 | 32.85 | -0.08 |
| scifact-pl | 68.16 | 68.84 | 67.58 | -0.58 |
| gpt-exams | 99.04 | 99.10 | 99.05 | +0.02 |

### heldout8 per task

| task | mined arm (bias013) | dose arm (v56) | FAQ arm | delta vs mined arm |
|---|---:|---:|---:|---:|
| maupqa-gpt3.5-cc | 20.51 | 20.54 | 20.60 | +0.09 |
| maupqa-multilingual-NLI | 14.86 | 14.89 | 14.72 | -0.14 |
| mfaq (FAQ) | 63.54 | 63.56 | 64.18 | +0.63 |
| poleval-2022-test-A-legal-questions | 84.35 | 85.06 | 84.38 | +0.03 |
| maupqa-wiki-def | 67.86 | 67.87 | 67.98 | +0.12 |
| scidocs-pl | 14.76 | 14.81 | 14.69 | -0.06 |
| maupqa-1z10 | 49.18 | 49.46 | 49.32 | +0.14 |
| fiqa-pl | 32.17 | 32.88 | 32.48 | +0.31 |

## Selection funnel (pairs left after each step)

| step | removed | left |
|---|---:|---:|
| a: input |  | 5394188 |
| a: filters.equals | 3086074 | 2308114 |
| a: filters.null | 1942 | 2306172 |
| a: filters.not_true | 100057 | 2206115 |
| a: filters.query_words | 3396 | 2202719 |
| a: filters.passage_words | 69500 | 2133219 |
| a: filters.domains_blocklist | 4092 | 2129127 |
| a: decontaminate.query | 0 | 2129127 |
| a: decontaminate.passage | 16254 | 2112873 |
| a: decontaminate.template_family | 15675 | 2097198 |
| a: priority.missing_score | 0 | 2097198 |
| a: dedup.passage | 49200 | 2047998 |
| a: caps | 1648086 | 399912 |
| a: near_dup.query | 167037 | 232875 |
| b: gemma and roberta-v3 both at or above percentile 55 of their coverage band | 153930 | 78945 |
| c: generic site-dependent questions (clusters) | 10408 | 68537 |
| c: near-duplicate questions (cosine >= 0.90) | 4050 | 64487 |
| d: candidates, equally from the coverage bands | 0 | 64487 |
| rows drawn from the candidates (seed 42) | | 60000 |

Final leak check against 547072 benchmark queries (pirb-queries.jsonl): **0** selected queries match after normalisation (normalised text or sorted tokens), 7 more only by the loose key (letters and digits; left out of the rows). Pool for mining: 64487 answers. Rows by source: {'webfaq_v2_pol': 56419, 'clips_mqa_pl_all_question': 3581}; by coverage band: {'0': 8961, '1': 11525, '2': 12402, '3': 12896, '4': 14216}; domains: 5442; domains with 'allegro' in the funnel's output: {'sprzedaznaallegro.pl': 1} (allegro.pl itself is on the blocklist).

Decontamination on the whole input: {'query_hits': 22, 'passage_hits': 254042, 'template_family_rows': 146188, 'families_q': 190889, 'families_a': 11061}

### Step b: the reranker threshold

Chosen percentile: **55** (lowest percentile with precision >= 0.85 on at least 20 kept calibration pairs). On the calibration pairs (240 pairs with Qwen labels, 140 good = relevance >= 2, context none, answer_quality informative, language not broken): precision **0.855**, recall **0.379**.
The calibration sample is stratified (source x coverage band x gemma band), not a random sample of the pool, so the precision is a guide, not an estimate for the pool.

| percentile | calibration pairs kept | good among them | precision | recall | pool pairs kept |
|---:|---:|---:|---:|---:|---:|
| 0 | 240 | 140 | 0.583 | 1.000 | 232875 |
| 5 | 165 | 121 | 0.733 | 0.864 | 215851 |
| 10 | 149 | 112 | 0.752 | 0.800 | 200300 |
| 15 | 136 | 105 | 0.772 | 0.750 | 185570 |
| 20 | 115 | 90 | 0.783 | 0.643 | 171376 |
| 25 | 105 | 83 | 0.790 | 0.593 | 156467 |
| 30 | 95 | 75 | 0.789 | 0.536 | 142472 |
| 35 | 89 | 70 | 0.787 | 0.500 | 129422 |
| 40 | 78 | 63 | 0.808 | 0.450 | 116384 |
| 45 | 73 | 60 | 0.822 | 0.429 | 103069 |
| 50 | 66 | 55 | 0.833 | 0.393 | 90303 |
| 55 | 62 | 53 | 0.855 | 0.379 | 78945 |
| 60 | 54 | 48 | 0.889 | 0.343 | 65592 |
| 65 | 50 | 45 | 0.900 | 0.321 | 55000 |
| 70 | 45 | 41 | 0.911 | 0.293 | 44725 |
| 75 | 43 | 39 | 0.907 | 0.279 | 35050 |
| 80 | 32 | 29 | 0.906 | 0.207 | 25412 |
| 85 | 26 | 24 | 0.923 | 0.171 | 17437 |
| 90 | 15 | 14 | 0.933 | 0.100 | 10105 |
| 95 | 8 | 8 | 1.000 | 0.057 | 2836 |

Score cut-offs by coverage band (gemma / roberta-v3; pool -> kept): band 0: 21.50 / 8.25 (34787 -> 11667); band 1: 23.88 / 9.75 (43750 -> 15419); band 2: 25.75 / 10.94 (46895 -> 16340); band 3: 26.88 / 11.88 (51213 -> 17076); band 4: 28.38 / 13.12 (56230 -> 18443).

### Step c: question similarity (OPI-PIB/PolDense-400M)

Rules: >= 5 neighbours at cosine >= 0.85 from >= 3 domains; the question and those neighbours go; greedy in priority order, cosine >= 0.9 to a kept question.
Calibration check (Qwen `context: site`): among the 2 calibration questions the cluster rule would flag, 0 have context site (0 %); among the 238 it would not flag, 89 (37 %).

## Mining of the FAQ rows

| rows | teacher P(pos) median | share with P(pos) > 0.95 | rows with a negative above the positive | student ranks the positive first |
|---|---:|---:|---:|---:|
| FAQ (60000) | 0.982 | 0.655 | 3864 (6.4 %) | 0.868 |
| old rows of the mined arm (20292) | 0.852 | 0.303 | 2658 (13.1 %) | 0.844 |

In FAQ data the same general question has different answers on different sites, so more negatives above the positive are expected; no filter is applied (v57).

## LLM judge (not run by this queue)

Candidate file: `/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/kitchen-data/webfaq-kd-v1/faq_kd_candidates_v1.jsonl` (64487 pairs). The command for the user's inference server (prompt v3):

```bash
cd /home/jsowa/work/speakleash/retrieval-data-kitchen
python annotate/annotate_pairs.py --vendor vllm --base-url "$JUDGE_BASE_URL" --model "$JUDGE_MODEL" --concurrency 8 \
    --pairs /media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/kitchen-data/webfaq-kd-v1/faq_kd_candidates_v1.jsonl \
    --system-file annotate/prompts/SYS_FAQ_V3.txt --user-file annotate/prompts/DATA_FAQ_V1.txt \
    --schema-file annotate/prompts/SCHEMA_FAQ_V2.json --prompt-version faq-v3 --max-tokens 200 --passage-chars 1500 \
    --output /media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/kitchen-data/webfaq-kd-v1/faq_kd_candidates_v1.v3.qwen.llm.jsonl \
    --journal /media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/kitchen-data/webfaq-kd-v1/faq_kd_candidates_v1.v3.qwen.journal.jsonl
```

The verdicts can drop mined rows afterwards (pair_id = `faq:<row>`, the rows carry `faq_row`); no new mining is needed.

## mine-hard-negatives (step D of the kitchen): smoke only

Smoke on 200 queries: rc 0, complete. Linear estimate for 100 000 queries over 64487 passages: **12.1 h** (ingest corpus (PolDense-150M + SPLADE -> LanceDB) 0.12 h; score positives, candidate reranker 1.11 h; mine candidates (hybrid + candidate reranker) 6.39 h; score positives, roberta-v3 1.39 h; final rerank, roberta-v3 (adaptive) 3.06 h; export with mined positives 0.0 h). The full run was not started.

## Package for the 5070

`datasets/transfer/transfer-5070-faq-20261112.tar.gz` (rows with negatives and teacher scores, mining sidecars, the selection manifest and funnel reports, SHA256SUMS).

## Examples for judging the thresholds of step c


### Question pairs at cosine 0.80-0.85 (232385 pairs in the neighbour lists)

| cos | a | b | domain_a | domain_b |
|---|---|---|---|---|
| 0.828 | Jakie metody płatności są dostępne w SpinBetter Casino? | Jakie metody płatności mogę wykorzystać w kasynie mobilnym online? | kasynopolska10.com | slotsjudge-pl.com |
| 0.81 | Jak optycznie powiększyć mały przedpokój? | Jak powiększyć małe usta? | mebloteka24.pl | neonia.com.pl |
| 0.807 | Jakie materiały są używane w druku 3D? | Jakie materiały są wykorzystywane w otulaczach? | kde.edu.pl | rozmowyodzieciach.pl |
| 0.804 | Co to są dostępne w Homla kody rabatowe? | Jakie są formy płatności w Homla online? | bravokupony.pl | wydawajdobrze.com |
| 0.842 | Jak często trzeba odkamieniać czajnik elektryczny? | Jak często powinno się odpowietrzać grzejniki? | barokko.pl | technonicol.pl |
| 0.819 | Czy można wykonywać te ćwiczenia w biurze? | Jaka jest prawidłowa postawa przy pracy w biurze? | bcptoken.pl | mojcodziennik.pl |
| 0.808 | Czy brodzik można zamontować na poziomie podłogi? | Jak zamontować wannę wolnostojącą? | 123lazienka.pl | komfort-wanny.pl |
| 0.802 | Jakie są główne zabytki i atrakcje historyczne w Crotone? | Jakie są atrakcje turystyczne w L’Aquila? | wlochy-wiadomosci.pl | wlochy-wiadomosci.pl |
| 0.803 | Jakie szczepienia przed wyjazdem na Madagaskar? | Jakie szczepienia przed wyjazdem na Seszele wykonam w Krakowie? | tropicalmed.pl | tropicalmed.pl |
| 0.817 | Jak długo trwa leczenie magnetoterapią? | Jak długo zazwyczaj trwa terapia radzenia sobie z gniewem? | rehabilitacja-mazowsze.pl | rolecatcher.com |
| 0.823 | Jaka jest różnica między Dragon Home a Dragon Professional? | Jaka jest różnica między domową a Profesjonalna kostkarka do lodu? | blitzhandel24.pl | klarstein.pl |
| 0.81 | Jak często powinno się pielęgnować sierść psa? | Jak długo trwa linienie u psa? | cricksydog.pl | unizoo.pl |
| 0.823 | które mieszkania w ustroniu morskim cieszą się największą popularnością? | które domy gościnne w kopalinie cieszą się największą popularnością? | e-wczasy.pl | e-wczasy.pl |
| 0.82 | Jakie składniki zawierają naturalne kremy do rąk? | czy odżywczy balsam do ciała i rąk jest naturalny? | mandispa.pl | nutridome.pl |
| 0.801 | Dlaczego warto zdecydować się na badania satysfakcji i zaangażowania pracowników? | Dlaczego relacje z pracownikami są ważne? | badania-rynku.com.pl | ecommercebridge.pl |
| 0.8 | Jak długo trwa proces redukcji etatu? | Czy pracodawca musi uzasadnić redukcję etatu? | toptalents.io | toptalents.io |
| 0.834 | Ile czasu zajmuje przelot z Warszawy na Rodos? | Jak długo trwa lot z Phuket do Warszawa? | hoteleuropa.pl | opodo.pl |
| 0.814 | Dlaczego warto sadzić kłącza kanny do doniczek? | Dlaczego warto używać donic ze styropianu? | sensagri.eu | sklepogrodnik.pl |
| 0.836 | Ile kosztuje strona One Page? | Ile kosztuje strona internetowa / koszt wykonania strony www | jlprojekt.pl | vizim.pl |
| 0.807 | Dlaczego warto inwestować w ergonomiczne meble biurowe? | Dlaczego warto wybierać meble z recyklingu? | klonplyt.pl | kornig.pl |
| 0.803 | Czy Betsafe bonus bez depozytu wymaga kodowania promocyjnego? | Czy potrzebuję Billybets kod promocyjny do otrzymania bonusu? | kodybonusowe.net | kodybonusowe.net |
| 0.808 | Czy mogę używać AI HTML designerów do tworzenia sklepów internetowych? | Jak korzystać z generatora stron internetowych z AI? | mobiri.se | wowslider.com |
| 0.834 | Jak zrobić własny korek do wanny? | Jak zrobić oczko wodne samemu? | komfort-wanny.pl | ogrodwpigulce.pl |
| 0.806 | Jak długo trwa proces odświeżania ubrań za pomocą parownicy? | Jak długo trwa remont kawalerki? | ceneo.pl | pomyslnaremont.pl |
| 0.816 | Jakie informacje są potrzebne do przygotowania wyceny na produkty z nadrukiem indywidualnym? | Jakie informacje są potrzebne do wyceny znaku towarowego i marki? | t-pack.pl | alpha-capital.pl |
| 0.843 | Gdzie mogę najtaniej kupić opony Nokian WR A4? | Gdzie mogę najtaniej kupić opony Pirelli P Zero Corsa Asimmetrico 2? | ranking-opon.eu | ranking-opon.eu |
| 0.817 | Czy MOTOEXPERT oferuje usługi audytu pojazdów firmowych i flotowych? | Jakie korzyści dla firm leasingowych oferuje współpraca z MotoExpert? | rzeczoznawca-samochodowy.de | rzeczoznawca-samochodowy.de |
| 0.805 | Jak dbać o plecak, by nie stracił wodoodporności? | Jak dbać o wkładki do butów, aby zachowały swoje właściwości? | plecakownia.pl | bhp-shop.com |
| 0.803 | Od kiedy śliniaki dla niemowląt? | Kiedy można wprowadzić poduszkę dla dziecka? | bajum.pl | danex24.pl |
| 0.817 | Jak długo trwa kurs dla konserwatorów sprzętu gaśniczego? | Ile czasu ważna jest gaśnica samochodowa? | star.edu.pl | swiatmotoryzacji.com |

### Question pairs at cosine 0.85-0.90 (45493 pairs in the neighbour lists)

| cos | a | b | domain_a | domain_b |
|---|---|---|---|---|
| 0.899 | Jakie są korzyści współpracy z MotoExpert w procesie likwidacji szkód komunikacyjnych? | Dlaczego warto wybrać MOTOEXPERT do obsługi szkody? | wyceny-szkod-po-wypadku.pl | wypadeksamochodowywaustrii.pl |
| 0.883 | Jak wybrać odpowiedni miernik wilgotności? | Jak dobrać odpowiedni osuszacz? | bbosa.pl | dombezwirusow.pl |
| 0.864 | Jakie korzyści niesie ze sobą korzystanie z ekologicznych tekturowych pojemników na jedzenie? | Jakie są korzyści z używania torb ekologicznych? | eko-opakowania.pl | eko-opakowania.pl |
| 0.876 | czy projektujecie sklepy internetowe lub strony internetowe dla firm z wrocławia | czy projektujecie sklepy internetowe lub strony internetowe dla firm z torunia | m16.pl | m16.pl |
| 0.872 | 5- Ile kosztuje usługa Kraków Bieżanów-Prokocim Budowlane na HomeRun? | 5- Ile kosztuje usługa Kraków Prądnik Biały Sprzątanie na HomeRun? | homerun.pl | homerun.pl |
| 0.888 | Czy bonus bez depozytu można wykorzystać na dowolnej grze w kasynie? | Czy mogę użyć bonusów od depozytu na wszystkie gry? | kasynoplonline.com | polandbcgame.com |
| 0.883 | Jaki jest najprostszy sposób na pobranie wielu zdjęć lub filmów z Instagrama? | Jak pobierać filmy z Instagrama za darmo? | igram.world | vidownloader.cc |
| 0.862 | Czy kryptowaluty AI to dobra inwestycja? | Czy inwestowanie w AI jest ryzykowne? | coinjournal.net | beincrypto.com |
| 0.876 | Czy boostowanie zwycięstw w LoL jest bezpieczne? | Czy boostowanie w Overwatch 2 jest bezpieczne? | eloking.com | eloking.com |
| 0.894 | Co to jest AI Builder stron afiliacyjnych? | Czym jest builder sklepu dropshippingowego AI? | mobiri.se | mobiri.se |
| 0.859 | Jak czyścić płyty z betonu? | jak czyścić płytki ścienne? | weneve.com | salonyhoff.pl |
| 0.859 | O której godzinie można się zameldować i wymeldować z Holiday Inn Zocalo? | O której godzinie można się zameldować i wymeldować z Hotel Huerta Honda? | trivago.pl | trivago.pl |
| 0.873 | Jak długo trwa zazwyczaj sesja terapii energetycznej? | Jak długo trwa sesja refleksologii? | rolecatcher.com | rpklublin.pl |
| 0.854 | Jakie nowoczesne elementy mogę dodać do architektury mojego ogrodu? | Jakie elementy wodne najlepiej komponują się z nowoczesnym ogrodem? | buskozdroj24.pl | livendo.pl |
| 0.864 | Czy zapewnienie nagród za udział w badaniu satysfakcji klienta jest konieczne? | Jakie korzyści płyną z motywowania klientów do uczestnictwa w badaniu dotyczącym satysfakcji? | badania-satysfakcji-klientow.pl | badanie-opinii.pl |
| 0.872 | Kiedy będą pierwsze wyniki SEO pojawiła się promocja? sztuczne kryształy w Augustowie? | Kiedy będą pierwsze wyniki SEO pojawiła się promocja? skoki kangoo w Augustowie? | seo.computer | seo.computer |
| 0.85 | Jakie rodzaje napraw gwarancyjnych wykonuje motoexpert? | Jakie usługi oferuje MOTOEXPERT po kolizji? | rzeczoznawca-samochodowy.de | wypadek-samochodowy-w-niemczech.pl |
| 0.854 | Czy Polacy mogą grać w wypłacalnych kasynach internetowych? | Czy niemieckie kasyna są dostępne dla polskich graczy? | kasynoonline.io | polskiekasynaonline24.com |
| 0.898 | Czy MOTOEXPERT oferuje pomoc w likwidacji szkód komunikacyjnych w Niemczech? | Czy MOTOEXPERT zajmuje się tylko sprawami odszkodowań za wypadki komunikacyjne? | wypadeksamochodowywpolsce.pl | wypadeksamochodowywpolsce.pl |
| 0.852 | Jaka jest rola ergonomicznych krzeseł obrotowych w miejscu pracy? | Dlaczego meble do biura powinny być ergonomiczne? | uksslawa.pl | female.pl |
| 0.874 | Czy można pomalować meble kuchenne z okleiny? | Czy można wymienić okleinę na meblach? | meblowe-inspiracje.pl | parkmebli.pl |
| 0.889 | Jak długo trwa proces pozycjonowania i kiedy można oczekiwać pierwszych rezultatów? | po jakim czasie pojawią się efekty i jak sprawdzić wyniki pozycjonowania? | seosem24.pl | ks.pl |
| 0.899 | Jaka jest różnica między kulturą masową a kulturą wysoką? | Czym różni się kultura masowa od kultury popularnej? | kulturamasowa.pl | kulturamasowa.pl |
| 0.872 | Czy można kupić Bitcoin anonimowo? | Czy mogę zachować anonimowość grając Bitcoinem w kasynie online? | icobench.com | kasynaonlinepolskie.com |
| 0.863 | Jakie są koszty wywozu szamba w województwie lubelskim? | Jakie są koszty wywozu szamba we Wrocławiu? | slonik24.pl | slonik24.pl |
| 0.852 | Jak długo trwa sesja zdjęciowa? | Ile czasu trwa typowa sesja masażu? | michalkowalski.pl | polskie-spa.pl |
| 0.854 | Jak dobrać odpowiednią obrożę elektryczną dla psa? | Jaką obrożę wybrać dla szczeniaka? | bielskpodlaski24.pl | aquaelzoo.pl |
| 0.867 | Jak mierzyć okno balkonowe i duże przeszklenia (karnisz i szyna)? | Jak zmierzyć okna, aby dobrać odpowiednie firany i zasłony? | magdalena24.pl | magdalena24.pl |
| 0.852 | Czy istnieją jakieś zagrożenia związane z magnesami neodymowymi walcowymi? | Dlaczego trzeba uważać przy obchodzeniu się z neodymowymi magnesami? | dhit.pl | xinxin.pl |
| 0.886 | Gdzie w Adamczowicach można naprawić filtr FAP do aut spod znaku Aixam? | Jak w Chwaszczynie można wyremontować filtr FAP do aut sygnowanych logo Aixam? | filtry-dpf-fap.pl | filtry-dpf-fap.pl |

### Question pairs at cosine 0.90-0.95 (9411 pairs in the neighbour lists)

| cos | a | b | domain_a | domain_b |
|---|---|---|---|---|
| 0.919 | Który silnik jest najczęściej polecany dla Audi A4 B8? | Jakie są dostępne opcje silników do modelu Audi A4 B8? | profesjonalnikierowcy.pl | profesjonalnikierowcy.pl |
| 0.921 | Czy mogę zarobić na swojej muzyce za pomocą PIBEEP? | Czy mogę sprzedawać swoje usługi muzyczne na PIBEEP? | pibeep.com | pibeep.com |
| 0.902 | Czy YouTube może wykrywać filmy wyprodukowane przez AI? | Czy mogę używać sztucznej inteligencji do tworzenia filmów na YouTube? | rask.ai | rask.ai |
| 0.917 | Jaka jest rola pszczół w ekosystemie? | Jakie jest znaczenie dzikich pszczół w ekosystemie? | malopolskie24.pl | pszczoly24.pl |
| 0.932 | Czy mogę stracić cały swój majątek ogłaszając upadłość konsumencką? | Czy upadłość konsumencka spowoduje utratę mojego majątku? | upadlosckonsumenckapolska.pl | upadlosckonsumenckapolska.pl |
| 0.907 | Jakie usługi świadczy sieć MOTOEXPERT dla osób poszkodowanych w Niemczech? | Jak firma MOTOEXPERT może pomóc osobie poszkodowanej w wypadku? | rzeczoznawca-samochodowy.de | wypadeksamochodowywpolsce.pl |
| 0.903 | Jakie korzyści wynikają z korzystania z usług obsługi logistycznej sklepów internetowych? | Czym jest obsługa logistyczna sklepów internetowych? | enemer.pl | enemer.pl |
| 0.924 | Jak mogę zabezpieczyć swoją stronę WordPress przed atakami? | Jak mogę zabezspieczyć swoją witrynę, która oparta o WordPress przed włamaniem się? | betterize.pl | webpanda.pl |
| 0.901 | Jaka jest główna różnica między wełną mineralną a styropianem? | Jakie są różnice między pianką PUR a wełną mineralną? | fastm.pl | komisreda.pl |
| 0.92 | Co to jest tworzenie stron internetowych z pomocą sztucznej inteligencji? | Co to jest kreator stron AI? | wowslider.com | wowslider.com |
| 0.926 | Jakie warzywa i owoce można spożywać na diecie lekkostrawnej? | Co można jeść przy diecie lekkostrawnej? | allevo.pl | koon.pl |
| 0.917 | W jaki sposób MotoExpert pomaga w sprawach odszkodowawczych po wypadkach? | Jakie są korzyści współpracy z MotoExpert w procesie likwidacji szkód komunikacyjnych? | rzeczoznawca-samochodowy.de | wyceny-szkod-po-wypadku.pl |
| 0.933 | Czy potrzebuję własnej strony internetowej? | Czy własna strona jest konieczna? | aleksandraniedzielska.pl | rankit.pl |
| 0.941 | Czy AI może pomóc w SEO strony internetowej? | Czy AI może pomóc w pozycjonowaniu stron? | mobiri.se | wowslider.com |
| 0.901 | czy kupony xbox mogą łączyć się z innymi kuponami przy zamówieniu? | czy kupony hotels mogą łączyć się z innymi kuponami przy zamówieniu? | noclegi-maja.pl | noclegi-maja.pl |
| 0.912 | Jakie są różnice między TPE a silikonem w prawdziwych lalkach? | Czym różnią się modele TPE i silikonowe? | hydoll.de | hydoll.de |
| 0.923 | Gdzie kupić taśmę do pakowania? | Gdzie kupię taśmę spinającą? | kartony24.eu | folie-kakol.pl |
| 0.913 | Jakie materiały można ciąć laserem w HMM Steel? | Jakie rury i profile można poddać cięciu laserowemu w HMM Steel? | hmmsteel.pl | hmmsteel.pl |
| 0.901 | Dlaczego zrównoważone praktyki biznesowe są ważne dla polskich firm? | Jakie znaczenie ma wdrażanie zielonych praktyk biznesowych dla polskich firm? | biogaz-tech.pl | liczevat.pl |
| 0.906 | Czy poczta z kwiatami Gorzów Wielkopolski dostarcza wieńce pogrzebowe na cmentarz? | Czy florystyczna poczta Poznań dowozi wieńce pogrzebowe na cmentarz? | laflora.pl | laflora.pl |
| 0.939 | Czy mogę jeść spaghetti carbonara podczas ciąży? | Jak bezpiecznie przygotować spaghetti carbonara podczas ciąży? | rozmowykobiece.pl | rozmowykobiece.pl |
| 0.917 | Czy rumianek na pniu jest łatwy w uprawie? | Czy margaretki na pniu są łatwe w uprawie? | malyogrod.pl | sensagri.eu |
| 0.927 | Czy Jeton jest bezpieczną metodą płatności? | Czy dokonywanie wpłat w kasynie za pośrednictwem Jeton jest bezpieczne? | bestcasinos-pl.com | kasyno-polskie-online.com |
| 0.927 | Co lepsze: plecak czy torba na laptopa? | Co lepsze tornister czy plecak? | plecakownia.pl | novakid.pl |
| 0.916 | Ile czasu zajmuje wykonanie kuchnie na zamówienie w Szczecinie? | Ile czasu zajmuje wykonanie kuchnie na zamówienie w Zielonej Górze? | kuchniekamduo.pl | kuchniekamduo.pl |
| 0.949 | Jaką pojemność ma bateria w Xiaomi Redmi Note 9? | Jakiej pojemności baterię zastosowano w Xiaomi Redmi Note 9S? | instrukcja-pdf.pl | instrukcja-pdf.pl |
| 0.933 | Jak dokonać wypłaty pieniędzy z konta w kasynie 888Starz Casino? | Jak wypłacić pieniądze z 888Starz? | polskiesloty.com | probukmacher.pl |
| 0.903 | Ile trwa audyt energetyczny? | Ile czasu zajmuje wykonanie audytu? | e-swiadectwa.com | mindu.pl |
| 0.932 | Jak ocieplenie pianką PUR różni się od wełny mineralnej? | Jak pianka pur wypada w porównaniu z wełną mineralną i styropianem? | budujemydom.pl | dcsystem.pl |
| 0.904 | Czy możliwe jest dokonywanie wypłat z Paysafecard we wszystkich kasynach online? | Czy mogę wypłacić wygrane z kasyna online za pomocą paysafecard? | goracekasyna.com | kasynopaysafecard.pl |

### Removed generic clusters (centre question, its neighbours at cosine >= 0.85)

- **Czy markizy kempingowe są łatwe w montażu?** (elcamp.pl; 8 neighbours, 8 domains): Czy szafki kempingowe są łatwe w montażu? [elcamp.pl, 0.93]; Czy montaż rolet dzień noc jest trudny? [rolety24.net, 0.87]; Czy bramy uchylne są łatwe w montażu? [kreatywnabudowa.pl, 0.86]; Czy moskitiery na okna są łatwe w montażu? [knall.com.pl, 0.86]; Czy mobilne regały są łatwe w montażu? [klaudynaczyta.pl, 0.86]
- **Czy mogę grać w kasynie bez wpłacania pieniędzy?** (topbonuzy.com; 9 neighbours, 8 domains): Czy można grać w kasynie bez depozytu? [casinosinstructor.com, 0.93]; Czy wygram w kasynie za prawdziwe pieniądze bez wpłaty? [polskiekasynaonline.net, 0.91]; Czy można grać w kasynie online bez wydawania pieniędzy? [casinosgossip.com, 0.91]; Czy możesz grać w kasyna online bez prawdziwych pieniędzy? [icecasino.com, 0.91]; Czy mogę wygrać prawdziwe pieniądze bez dokonywania wpłaty? [onlinecasinopulse.com, 0.88]
- **Jakie są normy dotyczące czasu wiązania betonu B20?** (schoolofknockingondoors.com; 5 neighbours, 4 domains): Beton B20 – czas schnięcia [idealny-dom.pl, 0.89]; Jak długo schnie beton B20? [chemik24.pl, 0.87]; Ile czasu potrzeba, aby beton B20 całkowicie wyschnął? [idealny-dom.pl, 0.86]; Jaka jest klasa wytrzymałości betonu B20? [krajowo.pl, 0.86]; Jaki jest optymalny stosunek wody do cementu dla betonu B20? [nowacegla.pl, 0.85]
- **Ile kosztuje życie w Rzeszowie?** (turystycznyninja.pl; 9 neighbours, 8 domains): Ile kosztuje internet stacjonarny w Rzeszowie? [panwybierak.pl, 0.89]; Ile kosztuje taksówka w Rzeszowie [supertaxi19622.pl, 0.87]; Ile kosztuje życie w Tarnobrzegu? [turystycznyninja.pl, 0.87]; Jakie są ceny wynajmu sali weselnej w lokalizacji Rzeszów? [weselna24.pl, 0.86]; Ile kosztuje fotograf ślubny w Rzeszowie? [wedding.pl, 0.86]
- **Jak zbudować stół ze starych desek krok po kroku?** (eleganckie-stoly.pl; 9 neighbours, 3 domains): Jak krok po kroku wykonać stół z palet? [eleganckie-stoly.pl, 0.92]; Jak przebiega instrukcja krok po kroku tworzenia stołu z palet? [eleganckie-stoly.pl, 0.91]; Jak zrobić stół ogrodowy z drewna? [eleganckie-stoly.pl, 0.91]; Jak zrobić stół warsztatowy z drewna? [eleganckie-stoly.pl, 0.89]; Jakie są podstawowe kroki do wykonania stołu warsztatowego z drewna? [eleganckie-stoly.pl, 0.89]
- **Czy picie wody z cytryną może mieć jakieś negatywne skutki?** (krajowo.pl; 6 neighbours, 4 domains): Co się stanie, jeśli będzie Pan/Pani codziennie pić wodę z cytryną? [ecosoft.com, 0.89]; Jakie są główne korzyści zdrowotne z picia wody z cytryną? [twojeporadniki.pl, 0.88]; Czy wypicie szklanki wody z cytryną na czczo rzeczywiście przynosi korzyści zdrowotne? [krajowo.pl, 0.87]; Czy picie wody z cytryną sprzyja odchudzaniu? [twojeporadniki.pl, 0.87]; Czy picie wody z cytryną “odkwasza” organizm? [twojeporadniki.pl, 0.86]
- **Czy mogę samodzielnie zamontować fototapetę?** (jakurzadzac.pl; 16 neighbours, 12 domains): Czy mogę samodzielnie zamontować tapetę? [milejdi.pl, 0.95]; Czy mogę samodzielnie zamontować tapety 3D? [galeriemebli.pl, 0.92]; Czy mogę samodzielnie zamontować meble do ściany? [klonplyt.pl, 0.89]; Czy panele ścienne można samodzielnie zamontować? [galeriemebli.pl, 0.89]; Czy można samodzielnie usunąć tapetę? [albin.com.pl, 0.89]
- **Czy MOTOEXPERT pomaga również w uzyskaniu odszkodowania w euro?** (wyceny-szkod-po-wypadku.pl; 18 neighbours, 7 domains): Czy MOTOEXPERT Austria oferuje pomoc w uzyskaniu odszkodowania? [wypadeksamochodowywaustrii.pl, 0.89]; Czy MOTOEXPERT oferuje również wsparcie prawne w dochodzeniu odszkodowania? [wypadeksamochodowywczechach.pl, 0.89]; Czy MOTOEXPERT oferuje pomoc w likwidacji szkód komunikacyjnych w Niemczech? [wypadeksamochodowywpolsce.pl, 0.88]; Czy mogę skorzystać z pomocy firmy MOTOEXPERT przy zgłoszeniu szkody w Niemczech? [wypadeksamochodowy.eu, 0.88]; Czy firma MOTOEXPERT oferuje również pomoc prawną poszkodowanym w Niemczech? [wypadeksamochodowywpolsce.pl, 0.88]
- **Gdzie dokładnie leży Dubaj?** (egzotycznywyjazd.pl; 5 neighbours, 3 domains): Gdzie dokładnie znajduje się Dubaj? [podrozowacnajlepiej.pl, 0.99]; Jakie jest położenie Dubaju na mapie Zjednoczonych Emiratów Arabskich? [wroclaw24.pl, 0.91]; W jakim kraju znajduje się Dubaj? [hoteleuropa.pl, 0.90]; Jakie jest położenie geograficzne Dubaju? [hoteleuropa.pl, 0.90]; Nad jakim morzem leży Dubaj? [hoteleuropa.pl, 0.85]
- **Czy używanie Bitcoin w kasynach online jest bezpieczne?** (onlinekasynopl.com; 10 neighbours, 9 domains): Czy bezpiecznie jest używać Bitcoin Cash jako opcji bankowej dla hazardu online? [livecasinofinder.com, 0.90]; Dlaczego warto używać Bitcoin w kasynach online? [hiwave.network, 0.89]; Jak korzystać z Bitcoin w kasynach online? [casinos.cc, 0.89]; Czy casino online Bitcoin jest legalnym miejscem do gry? [jednoreki-bandyta.pl, 0.88]; Czy kasyna Bitcoin są legalne? [outkast-shop.com, 0.87]
- **Czy mogę wypłacić wygrane z bonusu bez depozytu w Vulkan Bet?** (kasynopolska10.com; 7 neighbours, 6 domains): Czy Vulkan Vegas Casino oferuje bonus bez depozytu? [kasynoplonline.com, 0.89]; Czy Kasyno Vulkan Vegas oferuje bonus bez depozytu? [casinoonline-pl.com, 0.89]; Czy mogę odebrać Vulkan Bet bonus bez depozytu dwa razy? [slotozilla-pl.com, 0.87]; Czy jest możliwość skorzystania z dwóch bonusów bez depozytu w VulkanSpiele? [gryonline2.pl, 0.86]; Czy jest potrzebny kod promocyjny Vulkan Vegas do uzyskania bonusu bez depozytu? [sizzlinghot-spot.com, 0.86]
- **Jak MOTOEXPERT Czechy może mi pomóc finansowo po wypadku drogowym?** (wypadeksamochodowywpolsce.pl; 22 neighbours, 6 domains): Jak MOTOEXPERT może pomóc w uzyskaniu odszkodowania za szkodę komunikacyjną w Czechach? [wypadeksamochodowywczechach.pl, 0.93]; Jak firma MOTOEXPERT może pomóc osobie poszkodowanej w wypadku? [wypadeksamochodowywpolsce.pl, 0.89]; Jakie korzyści daje skorzystanie z usług MOTOEXPERT przy likwidacji szkody w Czechach? [wypadeksamochodowywczechach.pl, 0.89]; Jaką pomoc prawną oferuje MOTOEXPERT po wypadku samochodowym? [wypadeksamochodowywpolsce.pl, 0.89]; W jaki sposób MotoExpert pomaga w sprawach odszkodowawczych po wypadkach? [rzeczoznawca-samochodowy.de, 0.88]
- **Czy mogę samodzielnie zamontować osłonę na grzejnik?** (energetycznecentrum.pl; 12 neighbours, 10 domains): Czy mogę samodzielnie zamontować kratę ochronną na mojej ładowarce? [agroglas.pl, 0.87]; Czy mogę samodzielnie zamontować baterię termostatyczną? [bbosa.pl, 0.87]; Czy mogę samodzielnie zainstalować kocioł grzewczy? [memex.pl, 0.87]; Czy można samodzielnie zamontować wieszak na ręczniki? [emajsterkowicz.pl, 0.86]; Czy mogę zamontować szynę zasłonową samodzielnie? [remonty-przebudowy-adaptacje.pl, 0.86]
- **Jaki jest koszt wykonania ogrodzenia kutego?** (muratordom.pl; 10 neighbours, 3 domains): Ile kosztuje metal na ogrodzenie? [shemoans.pl, 0.90]; Ile kosztuje ogrodzenie z siatki? [shemoans.pl, 0.89]; Ile kosztuje ogrodzenie metalowe? [shemoans.pl, 0.89]; Ile kosztuje brama i furtka do ogrodzenia? [shemoans.pl, 0.87]; Jakie są koszty robocizny przy montażu ogrodzenia? [shemoans.pl, 0.87]
- **Jak zamontować gałki do mebli?** (meblownia.pl; 7 neighbours, 6 domains): Jak zamontować uchwyty w meblach kuchennych? [marbelo.pl, 0.91]; Jak zamontować kryształowe gałki? [meblowyuchwyt.pl, 0.91]; Jak prawidłowo montować uchwyty meblowe? [deton.pl, 0.90]; Jak zamontować uchwyty do mebli łazienkowych? [marbelo.pl, 0.89]; Gdzie stosować gałki meblowe? [meblownia.pl, 0.86]
- **Czy sofa z funkcją relaks nadaje się do małego mieszkania?** (umebluje.pl; 9 neighbours, 9 domains): Czy duże kanapy są odpowiednie do małych pomieszczeń? [meblesid.pl, 0.88]; Czy szafy przesuwne nadają się do małych mieszkań? [vipmeble.com.pl, 0.88]; Czy styl loftowy sprawdzi się w małych mieszkaniach? [newstylemeble.pl, 0.87]; Czy fotel nowoczesny sprawdzi się w małym mieszkaniu? [kb-london.com, 0.87]; Czy rozkładane stoły jadalniane nadają się do małych przestrzeni? [boconcept.com, 0.86]
- **Czy mogę zagrać w dowolną grę z moim bonusem powitalnym?** (onlinecasinopulse.com; 6 neighbours, 6 domains): W jakie gry w kasynie można grać, korzystając z bonusu powitalnego? [kasynaonlinepolskie.com, 0.88]; Czy bonus bez depozytu można wykorzystać w dowolnej grze? [bonusbezdepozytu.com, 0.87]; Czy bonus bez depozytu można wykorzystać na dowolnej grze w kasynie? [kasynoplonline.com, 0.87]; Czy będę mógł grać w sloty za pieniądze z bonusu powitalnego? [kasynopl.net, 0.86]; Czy mogę wykorzystać bonus powitalny do zagrania w gry karciane online? [icecasino.com, 0.85]
- **Co lepiej ochroni markę – rejestracja nazwy czy logo firmy?** (znakitowarowe-blog.pl; 5 neighbours, 3 domains): Co lepiej zastrzec – logo czy nazwę Twojej firmy? Co daje silniejsza ochronę? [znakitowarowe-blog.pl, 0.94]; Lepiej zgłosić do rejestracji nazwę czy logo? [ohstudio.pl, 0.93]; Co lepiej zastrzec: nazwę czy logo? [kancelarialech.pl, 0.93]; Co lepiej zastrzegać – nazwę czy logo firmy? Co bardziej się opłaca? [znakitowarowe-blog.pl, 0.91]; Czy zastrzegając logo zabezpieczam również nazwę firmy? [znakitowarowe-blog.pl, 0.87]
- **Czy mogę samodzielnie zamontować obrzeża trawnikowe?** (ogrod.info.pl; 11 neighbours, 10 domains): Czy listwy przypodłogowe można montować samodzielnie? [remonty-przebudowy-adaptacje.pl, 0.89]; Czy mogę samodzielnie zamontować ogrodzenie panelowe? [shemoans.pl, 0.88]; Czy mogę samodzielnie ułożyć kostkę brukową w ogrodzie? [nowacegla.pl, 0.88]; Czy mogę samodzielnie zamontować ścianki działowe? [klonplyt.pl, 0.88]; Czy można samodzielnie ułożyć kostkę brukową? [komfort-wanny.pl, 0.87]
- **Jaki rozmiar narzuty na łóżko 180x200?** (darymex.pl; 5 neighbours, 3 domains): Jaki jest optymalny rozmiar narzuty dla łóżka 160×200 cm? [pieknesny.pl, 0.92]; Jaka narzuta pasuje na łóżko 140x200? [pieknesny.pl, 0.90]; Jaki rozmiar kołdry do pościeli 200/200? [hippo-sklep.pl, 0.89]; jaki rozmiar prześcieradła pasuje na materac 160 x 200 cm? [wspolczesnaalkowa.com, 0.87]; Dla kogo pościele 180x200? [hippo-sklep.pl, 0.87]
- **Jakie są korzyści z wyboru kompletów kół z felgami stalowymi?** (voida.pl; 5 neighbours, 4 domains): Jakie są wady felg stalowych? [turlamfele.pl, 0.89]; Czym różnią się alufelgi od felg stalowych? [voida.pl, 0.86]; Felgi aluminiowe vs stalowe – co wybrać? [chemik24.pl, 0.86]; Czy wybrać felgi stalowe czy aluminiowe? [chemik24.pl, 0.86]; Czy felgi stalowe są wytrzymalsze niż alufelgi? [oponeo.pl, 0.85]
- **Czy mogę wykorzystać moje darmowe spiny na dowolny slot?** (kasynopolska10.com; 7 neighbours, 6 domains): Czy darmowe spiny można wykorzystać w dowolnej grze slotowej? [onlinecasinopulse.com, 0.95]; Czy darmowe spiny mogą być wykorzystane na dowolnym automacie? [kodybonusowe.net, 0.95]; Czy darmowe spiny bez depozytu mogę wykorzystać na każdym slocie? [topbonusy.com, 0.90]; Czy mogę wybrać, na których automatach chcę wykorzystać moje darmowe spiny? [casinoble.pl, 0.90]; Czy mogę wykorzystać darmowe spiny w mobilnej wersji kasyna? [kodybonusowe.net, 0.87]
- **Jaką farbą najlepiej pomalować meble na biało?** (meblowe-inspiracje.pl; 9 neighbours, 5 domains): Jakie są techniki malowania mebli na biało? [meblowe-inspiracje.pl, 0.89]; Jaki kolor ścian będzie najlepiej pasować do białych mebli? [meblowe-inspiracje.pl, 0.88]; Jak pomalować lakierowane meble na biało? [parkmebli.pl, 0.88]; Czy malowanie mebli na biało jest trwałe? [meblowe-inspiracje.pl, 0.87]; Czy mogę przemalować meble na biało samodzielnie? [meblowe-inspiracje.pl, 0.87]
- **Czym jest generator szablonów internetowych AI?** (mobiri.se; 17 neighbours, 3 domains): Co to jest generator szablonów HTML AI? [mobiri.se, 0.94]; Czym jest projektant stron internetowych AI? [mobiri.se, 0.89]; Co to jest generator kodu stron AI? [wowslider.com, 0.89]; Co to jest AI kreator stron internetowych HTML? [wowslider.com, 0.89]; Co to jest kreator stron AI? [wowslider.com, 0.89]
- **Czy za korzystanie z Blik w kasynie pobierana jest prowizja?** (malarstwonaszkle.pl; 8 neighbours, 7 domains): Czy kasyna pobierają prowizję za wpłatę Blikiem? [znaki.fm, 0.95]; Czy kasyna z opcją BLIK pobierają prowizje i opłaty za przeprowadzone transakcje? [szczecin.pl, 0.92]; Czy muszę uiścić opłatę, aby korzystać z Blika w kasynie online? [zajazdwilczyapetyt.pl, 0.91]; Czy muszę płacić, by skorzystać z Blika w kasynie online? [nv-kasyno-poland.com, 0.89]; Czy kasyno płatność BLIK pobiera opłaty za transakcje wykonane daną metodą? [rankingcasino.pl, 0.88]
- **Czy wykładzina PCV jest łatwa w czyszczeniu?** (carpetfloor.eu; 7 neighbours, 3 domains): Czy wykładziny winylowe są łatwe do czyszczenia? [carpetfloor.eu, 0.92]; Czy wykładziny PVC są antybakteryjne i łatwe w czyszczeniu? [grenwykladziny.pl, 0.92]; Czy wykładziny PCV o właściwościach elektrostatycznych są trudne w utrzymaniu? [carpetfloor.eu, 0.89]; Czy płytki dywanowe są łatwe w utrzymaniu czystości? [sklepwykladziny.pl, 0.87]; Czy wykładziny hotelowe można łatwo czyścić? [carpetfloor.eu, 0.86]
- **Czy można pobrać aplikację 20 bet?** (kasynopl.net; 10 neighbours, 10 domains): Czy istnieje aplikacja mobilna 20Bet i jak ją pobrać? [20bets-online.com, 0.95]; CZY MOGĘ POBRAĆ 20BET APP NA SWÓJ TELEFON? [20betpl.com, 0.94]; Czy 20Bet oferuje aplikację mobilną? [kasynoplonline.com, 0.93]; Czy 20Bet posiada aplikację mobilną? [polskiekasyno.com, 0.93]; Czy Istnieje Dostępna Do Pobrania Aplikacja Mobilna Bet20 Kasyno? [20bet.online, 0.90]
- **Jakie powłoki są stosowane na magnesów neodymowych?** (magnesy-neodymowe.pl; 9 neighbours, 6 domains): Jakie są różne typy powłok dla magnesów neodymowych i jakie mają właściwości? [dhit.pl, 0.94]; Czy neodymowe magnesy mogą być pokrywane farbą dla ochrony przed korozją? [xlink.com.pl, 0.87]; W jaki metodą magnesy neodymowe są produkowane? [euro-magnesy.pl, 0.87]; Czy magnesy neodymowe mogą być lakierowane dla ochrony przed korozją? [neo-cube.pl, 0.87]; W jaki sposób neodymowe magnesy są produkowane? [neo-cube.pl, 0.86]
- **Jak ubiegać się o bonus powitalny w GGBet?** (ggbet.win; 6 neighbours, 4 domains): Jak się zalogować do GGBet? [ggbet-poland.com, 0.88]; Gdzie mogę znaleźć pełny opis warunków bonusowych dla GGBet? [gg-bet-casino.com, 0.86]; Czy bonus za rejestrację w kasynie GGBet wymaga wpisywania kodu promocyjnego? [gg-bet-pl.com, 0.86]; Czy bonus powitalny od GGBet można wykorzystać kilka razy? [ggbet-poland.com, 0.85]; Jaki jest obecnie bonus na start oferowany przez GGBet nowym graczom? [ggbet.world, 0.85]
- **Jakie korzyści płyną z przeprowadzania badań ankietowych online w firmach?** (badania-rynku.com.pl; 8 neighbours, 6 domains): Jakie korzyści płyną z przeprowadzania ankiet online dla małych firm? [analizy-danych.pl, 0.95]; Jakie korzyści niesie ze sobą przeprowadzanie badań ankietowych? [analizy-danych.pl, 0.89]; Dlaczego firmy potrzebują odpowiedzi z ankiet online? [elmaxi.com.pl, 0.89]; Jakie korzyści przynosi zastosowanie badań online w marketingu farmaceutycznym i medycznym? [rejestry-medyczne.pl, 0.88]; Dlaczego małe przedsiębiorstwa decydują się na przeprowadzanie ankiet online? [analizy-danych.pl, 0.88]

### Random pairs removed in step c

| query | domain | reason |
|---|---|---|
| W jakich miastach Niemiec można skorzystać z usług rzeczoznawców MOTOEXPERT? | rzeczoznawca-samochodowy.de | generic cluster |
| Czy muszę płacić, aby pobrać muzykę mp3 na SnapTik? | snaptik.biz | generic cluster |
| Jak prawidłowo ułożyć sztućce na stole? | eleganckie-stoly.pl | near-duplicate of a kept question |
| Jak długo trwa analiza wyników badań rynku? | badanie-opinii.pl | generic cluster |
| Jakie korzyści dają smart liczniki? | technonicol.pl | generic cluster |
| Jak łączyć czapkę z zimową garderobą? | tacomat.pl | generic cluster |
| Czy usługi VPN bez rejestrów są legalne? | surfshark.com | generic cluster |
| Jak długo e-wiza do Gwinei pozostaje ważna? | evisajet.com | generic cluster |
| Jak gotować ścierki kuchenne? | kuchenneprzyrzady.com | generic cluster |
| Czym są magnesy płytkowe neodymowe tj. MPL 12x10x4 / N38? | dhit.pl | generic cluster |
| Jak przedłużyć życie świnki morskiej? | bukowno24.pl | near-duplicate of a kept question |
| Dlaczego warto konwertować pliki AVI na format MP4 w dzisiejszym świecie wideo? | mp4-converter.com | generic cluster |
| Czy kasyno na żywo jest aktywne całą dobę? | bestcasinos-pl.com | generic cluster |
| Gdzie kupić dobre leżaki z technorattanu? | dekorio.pl | generic cluster |
| Czym umyć wannę? | komfort-wanny.pl | generic cluster |
| Jak długo buduje się dom szkieletowy? | ekspertbudowlany.pl | generic cluster |
| Czy mogę grać w hot slot 777 na urządzeniach mobilnych? | bookofra.com.pl | generic cluster |
| Jaka jest różnica między bonusami bez depozytu a bonusami powitalnymi? | slotzy.net | generic cluster |
| Czym jest UX na stronie internetowej? | noveo.pl | generic cluster |
| Na jakiej wysokości wieszak na ręcznik? | armatura24.pl | generic cluster |

## Random pairs removed in step b

| query | passage | coverage_band | gemma | roberta_v3 | pct_gemma | pct_roberta_v3 |
|---|---|---|---|---|---|---|
| Jak aktywować bonus Miss Joker? | Bonus uruchamia się przy wystąpieniu specjalnych symboli na bębnach. | 1 | 18.62 | 5.09 | 10.3 | 5.9 |
| Czy przedstawione ceny są ostateczne? | W zwiąku z tym, że każda usługa realizowana jest na indywidualne zamówienie i wymaga innego nakładu pracy - wycena jest również indywidualna. Ceny przedstawione na stronie są orientacyjne i nie są ostateczne. | 4 | 23.5 | 8.88 | 4.5 | 3.7 |
| Dlaczego zostałem obciążony, ale nie otrzymałem potwierdzenia zamówienia? | Nieudane transakcje: Gdy próba transakcji jest podejmowana w naszym kasie, następuje autoryzacja na użytej karcie, co natychmiast zabiera środki z dostępnego kredytu. Jeśli wystąpią jakiekolwiek z poniższych zapobiegawcz… | 1 | 17.75 | 5.44 | 7.0 | 7.0 |
| Jaki jest aktualny spis Twoich rękawic nitrylowych? | Nasze magazyny w Europie Środkowej przechowują 1 000 000 pudełek wysokiej klasy dwuręcznych bezpudrowych rękawic do badań nitrylowych, które są dostępne do natychmiastowej wysyłki. | 2 | 17.75 | 6.28 | 2.3 | 4.5 |
| Jaką lampę najlepiej wybrać do sypialni? | Najlepiej wybrać lampę o ciepłym świetle, które nie będzie razić w oczy oraz dostosować ją do stylu i kolorystyki wnętrza. | 3 | 24.75 | 9.62 | 24.4 | 18.8 |
| Jakie korzyści przynosi odkrycie horoskopu wedyjskiego według daty urodzenia? | Odkrycie horoskopu wedyjskiego według daty urodzenia pozwala na lepsze zrozumienie naszej osobowości, predyspozycji, a także wpływu planet na nasze życie. Umożliwia to również świadome kształtowanie przyszłości oraz leps… | 3 | 29.0 | 11.75 | 90.6 | 52.3 |
| Co sądzą właściciele psów o kastracji chemicznej? | Opinie właścicieli są podzielone. Niektórzy zauważają zmniejszenie problemów behawioralnych, inni nie widzą wyraźnych efektów po zabiegu. | 1 | 22.5 | 8.88 | 37.3 | 38.0 |
| jak zrealizować mój kod bad monday apparel rabatowy? | zeskanuj naszą listę kodów kuponów i kliknij przycisk „pobierz kod”. kod zostanie wyświetlony na ekranie, a bad monday apparel strona otworzy się w nowej zakładce. wprowadź kod kuponu przy bad monday apparel kasie, a rab… | 3 | 27.0 | 11.88 | 55.4 | 54.8 |
| Jak przygotować Longfill Angel Pussy Juice do użytku? | Wystarczy uzupełnić butelkę zawierającą 13 ml aromatu wybraną bazą, aż do osiągnięcia łącznej pojemności 60 ml. Następnie wstrząsnąć mieszanką, aby aromat równomiernie się połączył. | 0 | 21.88 | 5.59 | 59.2 | 19.7 |
| Jakie informację muszę podać, by pobierać płatności przez Stripe? | Do skonfigurowania integracji Stripe w formularzu wystarczą dane logowania Twojego konta Stripe. | 0 | 20.75 | 9.0 | 43.7 | 68.8 |
| Czy metoda BLW jest zdrowa? | Metoda BLW jest zdrowa, pod warunkiem że dieta dziecka jest odpowiednio zbilansowana i dostarcza wszystkich niezbędnych składników odżywczych, w tym żelaza. Ta metoda wspiera kształtowanie prawidłowych nawyków żywieniowy… | 4 | 29.0 | 11.0 | 66.6 | 17.6 |
| Wrocław - Pamplona - Jaki jest czas lotu? | Zazwyczaj lot na trasie Wrocław - Pamplona trwa od ∞g NaNm do -∞g NaNm. | 2 | 22.62 | 12.69 | 19.4 | 83.2 |
| Co wyróżnia ofertę Pani firmy? | Usługa kuchnie na wymiar obejmuje nie tylko przygotowanie projektu funkcjonalnego kuchni, ale także wykonanie mebli na zamówienie. W moim zespole dodatkowo znajdziesz doświadczonych stolarzy, którzy zrealizują naszą wizj… | 0 | 12.69 | 2.62 | 2.9 | 6.4 |
| Wirtualne kasyna rzadko oferują atrakcyjne bonusy za rejestrację. | Dla nowych graczy interesujące wydają się tylko dwie możliwości. Umożliwiają one rozpoczęcie zabawy bez potrzeby inwestowania. Dzięki temu ryzyko jest minimalne – nawet jeśli nie zdobędziesz żadnych prawdziwych pieniędzy… | 0 | 15.56 | 4.72 | 8.7 | 14.0 |
| Czy mogę później zmodernizować szlaban o system ANPR? | Tak, możesz zmodernizować Parklio Gate za pomocą dowolnego inteligentnego systemu, w tym ANPR. | 3 | 26.5 | 10.0 | 47.2 | 23.3 |
| Jaki jest kod EAN LEGO 40716 Środowisko naturalne kosmitów? | Kod EAN LEGO Środowisko naturalne kosmitów to 5702017597171. | 4 | 27.12 | 11.0 | 29.6 | 17.6 |
| Dlaczego Palo Santo szybko gaśnie? | Prawdopodobnie podpalasz je zbyt krótko. Prawidłowe podpalenie Palo Santo jest wtedy, gdy złapie ogień i przez dłuższą chwilę go podtrzymuje. Dopiero wtedy należy je zdmuchnąć, gasząc płomień. W ten sposób podpalone Palo… | 2 | 24.75 | 7.56 | 38.1 | 9.6 |
| wózki adamex - dlaczego warto kupić w naszym sklepie internetowym? | wózki firmy adamex umożliwiają transport dziecka w najlepszych możliwych warunkach. są solidne oraz wygodne. zapraszamy do zapoznania się z ofertą. kliknij i dowiedz się więcej! | 1 | 22.75 | 8.88 | 39.5 | 38.0 |
| Jak dbać o skórę po usuwaniu? | W dniu zabiegu trzeba schładzać miejsce z tatuażem co godzinę, na kolejne 2 tygodnie należy odstawić opalanie, saunę, solarium, baseny. Można stosować krem SPF50. | 0 | 16.75 | 4.53 | 13.0 | 13.0 |
| czy autorski cms jest dobrym wyborem? | stojąc przed zleceniem zewnętrznej firmie zadania wykonania strony internetowej, często możemy spotkać się z pytaniem o system zarządzania treścią. najczęściej wybieramy pomiędzy wordpressem a rozwiązaniem dedykowanym. c… | 4 | 27.25 | 14.06 | 31.7 | 73.7 |

## Queue state (v58)

```
chain:v58_sel_smoke	ok	-	96s
chain:v58_sel_full	ok	-	222s
chain:v58_rerank	ok	-	569s
chain:v58_rerank_calib	ok	-	6s
chain:v58_agree	ok	-	1s
chain:v58_similarity	ok	-	91s
chain:v58_candidates	ok	-	2s
chain:v58_mine_smoke	ok	-	49s
chain:v58_mine_part0	ok	-	447s
chain:v58_mine_part1	ok	-	232s
chain:v58_mine_part2	ok	-	213s
train:arm-dense-kl-n5kj-mined-faq20-ens400-r2v12	ok	-	9175s
sparsity:arm-dense-kl-n5kj-mined-faq20-ens400-r2v12	ok	551.4	23s
sparsity:arm-dense-kl-n5kj-mined-faq20-ens400-r2v12-bias005	ok	487.6	24s
sparsity:arm-dense-kl-n5kj-mined-faq20-ens400-r2v12-bias010	ok	429.8	24s
sparsity:arm-dense-kl-n5kj-mined-faq20-ens400-r2v12-bias015	ok	377.9	23s
regime:quantized:arm-dense-kl-n5kj-mined-faq20-ens400-r2v12-bias015:tiny10	ok	64.84 10	599s
regime:quantized:arm-dense-kl-n5kj-mined-faq20-ens400-r2v12-bias015:heldout8	ok	43.54 8	2104s
regime:quantized:arm-dense-kl-n5kj-mined-faq20-ens400-r2v12-bias015:heldout2_treccovid	ok	70.39 1	1333s
```
