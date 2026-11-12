# v59 (quantized regime, nDCG@10): MS MARCO queries where the dense model wins and SPLADE does not

Diagnostics on 200000 queries of the judged full MS MARCO-pl pairs (500732 pairs, 388327 queries; left out: {'benchmark query': 767}), then, when it is large enough, the hard-reachable set H with negatives mined by r2v12 from the pool of the sample's positives and the scores of the same ensemble teacher with the fixed dense scale. No arm is trained here: queue v60 (SQuAD-v2 dose + FAQ) runs first and queue v61 trains the rows of its winner plus H.

**|H| = 0; kept coverage bands: none; the arm: NOT built (|H| below 8000).**

## Reading

No arm was trained in this queue: |H| is below the threshold. The diagnostics below stand on their own.

## Diagnostics

Sample: 200000 queries, one positive each; 71618 pairs are relevant (>= 2) with natural Polish on both sides, and every table below counts only those. Ranks are positions of the positive among the 198026 positives of the sample (ranks above 64 are not resolved). base = base polish-splade, current = mined arm, bias 0.13, dense = PolDense-400M.

### Positive in the top 3: dense against SPLADE

| SPLADE model | both | only dense | only SPLADE | neither |
|---|---:|---:|---:|---:|
| base polish-splade | 59714 (83.4 %) | 7258 (10.1 %) | 1678 (2.3 %) | 2968 (4.1 %) |
| mined arm, bias 0.13 | 60985 (85.2 %) | 5987 (8.4 %) | 1769 (2.5 %) | 2877 (4.0 %) |

### Pairs the dense model gets right (top 3): coverage band x rank of the positive under SPLADE


base polish-splade:

| coverage | rank 1 | rank 2-10 | rank 11-50 | rank >50 | all |
|---|---:|---:|---:|---:|---:|
| < 0.2 | 873 | 887 | 307 | 687 | 2754 |
| 0.2-0.4 | 4772 | 2764 | 442 | 293 | 8271 |
| 0.4-0.6 | 9151 | 3930 | 420 | 284 | 13785 |
| 0.6-0.8 | 14472 | 4107 | 211 | 83 | 18873 |
| >= 0.8 | 18355 | 4413 | 336 | 185 | 23289 |

mined arm, bias 0.13:

| coverage | rank 1 | rank 2-10 | rank 11-50 | rank >50 | all |
|---|---:|---:|---:|---:|---:|
| < 0.2 | 937 | 918 | 281 | 618 | 2754 |
| 0.2-0.4 | 4900 | 2750 | 376 | 245 | 8271 |
| 0.4-0.6 | 9429 | 3839 | 300 | 217 | 13785 |
| 0.6-0.8 | 14776 | 3865 | 173 | 59 | 18873 |
| >= 0.8 | 18948 | 4057 | 190 | 94 | 23289 |

### Transitions base -> current model, pairs at rank 2-50 under the base model

| coverage | pairs (all) | to rank 1 | worse rank | pairs (dense top 3) | to rank 1 | worse rank |
|---|---:|---:|---:|---:|---:|---:|
| < 0.2 | 1497 | 192 (12.8 %) | 440 (29.4 %) | 1194 | 183 (15.3 %) | 328 (27.5 %) |
| 0.2-0.4 | 3969 | 611 (15.4 %) | 906 (22.8 %) | 3206 | 574 (17.9 %) | 676 (21.1 %) |
| 0.4-0.6 | 5212 | 1013 (19.4 %) | 1031 (19.8 %) | 4350 | 954 (21.9 %) | 770 (17.7 %) |
| 0.6-0.8 | 4996 | 1042 (20.9 %) | 788 (15.8 %) | 4318 | 998 (23.1 %) | 590 (13.7 %) |
| >= 0.8 | 5381 | 1427 (26.5 %) | 686 (12.7 %) | 4749 | 1367 (28.8 %) | 529 (11.1 %) |

### The hard-reachable set H

H = relevant pairs with natural Polish, dense top 3, current model at rank 2-50, coverage below 0.8, in a band where at least 25 % of the pairs at rank 2-50 under the base model reached rank 1 under the current model (the 'all' columns above).

- |H| = **0** (by band: {}); the arm needs at least 8000: **NO arm, diagnostics only**.
- Bands left out as not reachable: < 0.2: 12.8 % of 1497 pairs reached rank 1, 1199 candidates lost; 0.2-0.4: 15.4 % of 3969 pairs reached rank 1, 3126 candidates lost; 0.4-0.6: 19.4 % of 5212 pairs reached rank 1, 4139 candidates lost; 0.6-0.8: 20.9 % of 4996 pairs reached rank 1, 4038 candidates lost. The band >= 0.8 is left out by definition (purely lexical matches).

### Example pairs (dense top 3; rank under the current model; ranks above 64 shown as 65)


**low coverage (< 0.4), rank 2-10** (3668 pairs)

| query | passage | base | current | dense | coverage |
|---|---|---:|---:|---:|---:|
| jak długo żyją wiewiórki | O: Pospolita wiewiórka szara może żyć do 12,5 lat na wolności lub 20 lat w niewoli. Inne gatunki mają różną długość życia; na przykład wiewiórka latająca żyje od 5 do 6 lat na wolności i do 10 lat w n… | 4 | 2 | 1 | 0.33 |
| do czego używana jest jira | Służy do śledzenia błędów, śledzenia problemów i zarządzania projektami. Nazwa JIRA jest właściwie odziedziczona od japońskiego słowa Gojira, które oznacza Godzilla. Podstawowym zastosowaniem tego nar… | 2 | 2 | 2 | 0.33 |
| jaka jest nazwa zabiegu chirurgicznego polegającego na nacięciu błony bębenkowej w celu złagodzenia i odprowadzenia płyn… | Rurki uszne są wprowadzane podczas ambulatoryjnej procedury chirurgicznej zwanej myringotomią. Myringotomia to nacięcie (mały otwór) w bębenku lub błonie bębenkowej. Najczęściej odbywa się to pod mikr… | 2 | 2 | 2 | 0.27 |
| czy powinienem pozwolić, aby bateria mojego iPhone’a całkowicie się rozładowała przed ładowaniem | 1 Nie ładuj całkowicie ani nie rozładowuj całkowicie baterii urządzenia — naładuj ją do około 50%. 2 Jeśli przechowujesz urządzenie z całkowicie rozładowaną baterią, bateria może popaść w stan głęboki… | 4 | 5 | 1 | 0.33 |
| kto jest żoną gregga allmana | Gregg Allman, 64-letni współzałożyciel The Allman Brothers Band, zaręczył się z 24-letnią Shannon Williams, ogłosił 25 maja 2012 w odcinku „Piers Morgan Tonight”. Allman był żonaty sześć razy, z Shell… | 1 | 2 | 2 | 0.33 |
| ile uncji w dużym jajku | Tak więc duże jajka mają 2 uncje, a średnie 1,75 uncji. Powyższe odnosi się do wagi jajka ORAZ skorupki, ale jeśli chcesz poznać wagę jajka poza skorupką (dużej, ponieważ jest to najczęściej spotykane… | 3 | 2 | 3 | 0.33 |
| kto był aktorem, który grał Omara w serialu telewizyjnym The Wire | Grany przez Michaela K. Williamsa, Omar był legendą podziemia Baltimore, który nigdy nie odstępował od swojego własnego zestawu zasad, z których złotym jest to, że nigdy nie groził nikomu spoza gry. W… | 2 | 2 | 2 | 0.17 |
| definicja umowy dotyczącej ziemi | Umowa gruntowa to pisemna umowa prawna lub umowa używana do zakupu nieruchomości, takiej jak niezabudowana działka, dom, budynek mieszkalny, budynek handlowy lub inna nieruchomość. Umowa gruntowa jest… | 4 | 2 | 2 | 0.00 |
| miasta, w których bije się monety | Właściwie są 4 mennice. Monety obiegowe są produkowane w Denver (znak mennicy D) lub Filadelfii (P lub bez znaku mennicy, w zależności od daty monety i nominału) Monety próbne są produkowane w San Fra… | 31 | 8 | 2 | 0.25 |
| która część ucha jest dotknięta przez zespół Meniere'a | Choroba Ménière'a to schorzenie ucha wewnętrznego, które może powodować zawroty głowy, specyficzny rodzaj zawrotów głowy, w których czujesz, że się kręcisz. Może również powodować dzwonienie w uchu (s… | 3 | 2 | 2 | 0.20 |
| jak długo trwa lot z Las Vegas do Kalifornii | Czas lotu z Las Vegas, NV do Los Angeles, CA. Całkowity czas lotu z Las Vegas, NV do Los Angeles, CA to 41 minut. Jest to średni czas lotu w powietrzu (od kół do kół na pasie startowym) oparty na rzec… | 2 | 2 | 2 | 0.33 |
| do którego stanu dołączyło Hawaje | Powszechnie uważa się, że w 1959 r. odbył się plebiscyt, w którym mieszkańcy Hawajów głosowali za stanem Unii, a 18 sierpnia 1959 r. Hawaje zostały przyjęte do Związku. Co doprowadziło do tego wydarze… | 5 | 6 | 3 | 0.25 |
| jaka była ważna różnica między systemami politycznymi Rzymian i Greków | Rzymianie zwrócili się ku posiadłościom, produkując oliwę z oliwek i wino. W Grecji społeczeństwo zostało podzielone na niewolników, wyzwoleńców, metyków, obywateli i kobiety. Rzym miał niewolników, w… | 9 | 6 | 1 | 0.14 |
| jak dowiedzieć się, na jakiej sieci działa twój telefon | Wiem, że w przypadku niektórych telefonów Samsung możesz wybrać *#2263#, aby wymusić użycie pewnych pasm (pod warunkiem, że jest to obsługiwane przez sprzęt telefonu) i możesz wybrać *#0011#, aby dowi… | 16 | 7 | 1 | 0.33 |
| numer telefonu do zgłaszania roszczeń w esurance | 1-800-ESURANCE lub (800) 378-7262. Firma Esurance powstała w 1999 roku i była jedną z pierwszych firm oferujących polisę ubezpieczeniową online. Esurance oferuje ochronę motocykli, łodzi, skuterów wod… | 1 | 2 | 3 | 0.20 |
| jak długo fluconazol pozostaje w organizmie | Flukonazol: Maksymalne stężenia w osoczu u zdrowych ochotników na czczo występują pomiędzy 1 a 2 godzinami, a okres półtrwania w końcowej fazie eliminacji z osocza wynosi około 30 godzin (zakres: 20 d… | 2 | 2 | 1 | 0.00 |
| jaka jest populacja Guam | Ludność Guamu 2014. Według stanu na 1 stycznia 2015 r. ludność Guamu szacowano na 169 312 osób. Jest to wzrost o 1,31% (2 196 osób) w porównaniu do 167 116 mieszkańców rok wcześniej. W 2014 roku przyr… | 2 | 2 | 2 | 0.00 |
| czy szklanka whisky wpływa na leki | 1 Wiele leków przyjmowanych z alkoholem może powodować problemy – w tym leki przeciwlękowe, leki nasenne i leki przeciwbólowe na receptę. 2 Skutki uboczne mogą się nasilić, jeśli pijesz alkohol i zaży… | 15 | 7 | 3 | 0.25 |
| proporcjonalny retributywizm definicja | Wynagradzająca sprawiedliwość. Sprawiedliwość retrybucyjna to teoria sprawiedliwości, która utrzymuje, że najlepszą reakcją na przestępstwo jest proporcjonalna kara wymierzona ze względu na nią, a nie… | 3 | 5 | 1 | 0.33 |
| w jakim celu przyjmuje się lasix | Furosemid 20 mg tabletki. Furosemid (Lasix®) to lek na receptę często stosowany w leczeniu wysokiego ciśnienia krwi i retencji wody spowodowanej chorobą nerek lub niewydolnością nerek, marskością lub … | 3 | 4 | 2 | 0.25 |

**low coverage (< 0.4), rank > 50** (863 pairs)

| query | passage | base | current | dense | coverage |
|---|---|---:|---:|---:|---:|
| co oznacza striking | uderzający. 1 atrakcyjny; imponujące: scena o uderzającym pięknie. 2 zauważalne; rzuca się w oczy: uderzający brak entuzjazmu. 3 strajkują, jako robotnicy. 4 zdolny do zaatakowania wroga, zwłaszcza z … | 65 | 65 | 1 | 0.00 |
| co oznacza siren | Syrena (rzeczownik). nimfa morska (po części kobieta, po części ptak) mająca zwabić żeglarzy na zagładę na skałach, na których mieszkały nimfy. Odyseusz nakazał swojej załodze zatkać uszy, aby nie sły… | 65 | 65 | 3 | 0.00 |
| definicja słowa krip | rzeczownik, slang: uwłaczające i obraźliwe. 1. termin używany w odniesieniu do osoby, która jest częściowo lub całkowicie niezdolna do używania jednej lub więcej kończyn. Pochodzenie Crip. 1915-1920. … | 65 | 65 | 1 | 0.00 |
| dysmenorrhea to stan czego? | Bolesne miesiączkowanie to termin medyczny określający bolesne skurcze, które mogą wystąpić bezpośrednio przed lub w trakcie miesiączki. Istnieją dwa rodzaje bolesnego miesiączkowania: pierwotne boles… | 65 | 65 | 2 | 0.00 |
| gdzie jest tübingen | Położenie Tybingi na mapie. Tybinga to miasto położone w Badenii-Wirtembergii w Niemczech. Znajduje się na 48,52 szerokości geograficznej i 9,05 długości geograficznej i znajduje się na wysokości 333 … | 65 | 65 | 1 | 0.00 |
| co zastępuje Windows Movie Maker | 4. Avidemux. Avidemux to minimalistyczne rozwiązanie do edycji wideo i alternatywa dla twórców filmów, gdy potrzebujesz ukończyć zadania edycyjne w krótszym czasie. Zapewnia wszystkie imponujące funkc… | 49 | 62 | 1 | 0.00 |
| jaki kolor ma calico | Prawdziwy kot perkalowy jest pomarańczowy – czasami określany jako czerwony – czarno-biały. Kot zazwyczaj ma łaty tych kolorów w różnych rozmiarach i miejscach na całym ciele. | 65 | 65 | 3 | 0.00 |
| jaki to kolor sea blue | Morski / #006994 kolor szesnastkowy. W przestrzeni kolorów RGB szesnastkowy #006994 (znany również jako błękit morski) składa się z 0% czerwonego, 41,2% zielonego i 58% niebieskiego. Natomiast w przes… | 65 | 65 | 1 | 0.33 |
| latent definicja biologia | 1 Okres utajenia (również okres utajenia), czas między ekspozycją na patogen, substancję chemiczną lub promieniowanie, a pojawienie się pierwszych objawów. Opóźnienie wirusa, zdolność wirusa do pozost… | 65 | 65 | 2 | 0.00 |
| jaki rodzaj badania to RCT | Badanie z randomizacją jest jednym z najprostszych, ale najpotężniejszych narzędzi badawczych. W istocie randomizowane kontrolowane badanie jest badaniem, w którym ludzie są losowo przydzielani do jed… | 65 | 65 | 3 | 0.33 |
| z czego najbardziej znany jest senegal | 2. Lac Rose: Znane również jako Pink Lake, to płytkie, słone jezioro jest jednym z najpopularniejszych miejsc turystycznych w Senegalu. W słoneczne dni woda wydaje się jasnoróżowa dzięki dużej zawarto… | 15 | 52 | 2 | 0.25 |
| w jakim hrabstwie znajduje się OKC? | Miasta w tym hrabstwie to: Oklahoma City, Edmond, Midwest City, Del City, Bethany, The Village, Warr Acres, Choctaw, Harrah, Nichols Hills. Hrabstwo Oklahoma ma przewidywany średni poziom ochrony prze… | 65 | 59 | 1 | 0.25 |
| co to jest ramdisk | Nazwa mówi wszystko. Dysk RAM to po prostu kilka modułów pamięci, które zostały zgrupowane, a następnie przeznaczone do przechowywania długoterminowego zamiast krótkoterminowego. Wykorzystywana pamięć… | 65 | 65 | 1 | 0.00 |
| co to jest green computing | Zielona informatyka. Zielone przetwarzanie, badanie i praktyka wydajnych i przyjaznych dla środowiska zasobów obliczeniowych, jest obecnie przedmiotem zainteresowania nie tylko organizacji ekologiczny… | 65 | 65 | 1 | 0.00 |
| co to jest filly | Klaczka to koń żeński, który jest zbyt młody, aby można go było nazwać klaczą. W użyciu są dwie konkretne definicje: W większości przypadków klaczka to koń płci żeńskiej w wieku poniżej czterech lat. … | 65 | 65 | 1 | 0.00 |
| jakie były zasługi williama smitha | W ten sposób, wraz ze Steno, Lehmann pomógł zapoczątkować ideę kolumny stratygraficznej, omówioną w dalszej części tego eseju. Człowiekiem uważanym za ojca stratygrafii był jednak angielski inżynier i… | 65 | 65 | 2 | 0.33 |
| co to jest switch hub | Avaya ERS 2550T-PWR, 50-portowy przełącznik Ethernet. Przełącznik sieciowy (zwany także koncentratorem przełączającym, koncentratorem mostkującym, oficjalnie mostem MAC) to komputerowe urządzenie siec… | 65 | 65 | 2 | 0.00 |
| którego minerału wchłanianie jest znacznie zwiększane przez witaminę c? | Z drugiej strony, na wchłanianie żelaza niehemowego, głównej puli pokarmowej, duży wpływ ma skład posiłku. Kwas askorbinowy jest silnym wzmacniaczem wchłaniania żelaza niehemowego i może odwracać hamu… | 65 | 65 | 1 | 0.17 |
| definicja black pride | „Czarna duma to ruch zachęcający ludzi do bycia dumnymi z bycia czarnymi w różnym stopniu. Pokrewne ruchy obejmują czarny nacjonalizm, Czarne Pantery, afrocentryzm i czarną supremację” – stwierdza pie… | 59 | 65 | 1 | 0.00 |
| definicja impact | 1. uderzanie jednej rzeczy o drugą; kolizja. 2. wpływ; efekt: wpływ Einsteina na współczesną fizykę. 3. mocne uderzenie: olbrzymie uderzenie strzału. 4. siła wywierana przez nową ideę, koncepcję, tech… | 65 | 65 | 1 | 0.00 |

**medium coverage (0.4-0.6), rank 11-50** (300 pairs)

| query | passage | base | current | dense | coverage |
|---|---|---:|---:|---:|---:|
| co powinno znajdować się w pierwszym akapicie | Ustęp wprowadzający. Zobacz najpierw Pisanie akapitów wstępnych, aby poznać różne sposoby zaangażowania czytelnika w esej. W akapicie wstępnym powinno się również znaleźć sformułowanie tezy, rodzaj mi… | 65 | 43 | 1 | 0.50 |
| kod ICD ostra niewydolność oddechowa | Nie obejmuje to ostrej i przewlekłej niewydolności oddechowej (518,84), ostrej niewydolności oddechowej (518,82), ostrej niewydolności oddechowej po urazie i zabiegu chirurgicznym (518,51), przewlekłe… | 8 | 13 | 2 | 0.40 |
| jak nazywał się pan douglas w green acres | Seria / Zielone Akry. Od góry po lewej: pan Kimball, Eb, pan Haney. Drugi rząd: Lisa, Oliver, matka Olivera. Oliver Wendell Douglas (Eddie Albert), odnoszący sukcesy nowojorski prawnik, i jego eleganc… | 13 | 48 | 1 | 0.40 |
| definicja okoliczności | warunek, szczegół, część lub atrybut w odniesieniu do czasu, miejsca, sposobu, agenta itp., który towarzyszy, określa lub modyfikuje fakt lub wydarzenie; czynnik modyfikujący lub wpływający: Nie oceni… | 22 | 12 | 3 | 0.50 |
| co to jest zakręt fusiformny | Zakręt wrzecionowaty, znany również jako (nieciągły) zakręt potyliczno-skroniowy, jest częścią płata skroniowego i potylicznego w obszarze Brodmanna 37. Zakręt wrzecionowaty ma kontrowersyjną historię… | 17 | 11 | 1 | 0.50 |
| co oznacza profe | ProFe to forma żelaza mineralnego. Żelazo jest ważne dla wielu funkcji w organizmie, zwłaszcza dla transportu tlenu we krwi. ProFe stosuje się w zapobieganiu i leczeniu niedoborów żelaza oraz niedokrw… | 7 | 33 | 1 | 0.50 |
| stanley cup jest dla jakiego sportu | Puchar Stanleya jest również znany jako Święty Graal hokeja. Obecnie istnieją trzy wersje Pucharu Stanleya: oryginalne trofeum użyte w Dominion Hockey Challenge Cup, puchar prezentacyjny oraz replika … | 5 | 30 | 2 | 0.50 |
| w jakim stanie urodził się trump | Główny artykuł: Pochodzenie Donalda Trumpa. Trump urodził się 14 czerwca 1946 roku w Jamaica Estates w Queens, dzielnicy Nowego Jorku. Był drugim najmłodszym dzieckiem pięciorga dzieci Mary i Freda Tr… | 5 | 27 | 1 | 0.50 |
| hcf volume | Szybka odpowiedź. HCF to akronim oznaczający sto stóp sześciennych i jest standardową jednostką pomiaru objętości wody, zwykle do celów pomiaru i rozliczeń wody. Sto stóp sześciennych wody to 748 galo… | 28 | 16 | 1 | 0.50 |
| co to jest wyciskanie wojskowe | Prasa wojskowa. Prasa wojskowa to ćwiczenie, które nazywa się tak, ponieważ uważa się, że jest odzwierciedleniem siły w siłach zbrojnych. Jest to odmiana prasy barkowej, która oprócz mięśni naramienny… | 65 | 19 | 1 | 0.50 |
| std stringi | std::ciąg. Ciągi to obiekty reprezentujące sekwencje znaków. Standardowa klasa ciągów zapewnia obsługę takich obiektów za pomocą interfejsu podobnego do standardowego kontenera bajtów, ale dodaje funk… | 65 | 43 | 1 | 0.50 |
| jakie jest zadanie dyrektora operacyjnego w przemyśle wytwórczym? | Chief Operating Officer (COO) Chief Operating Officer (COO) to dyrektor wykonawczy firmy, który nadzoruje bieżące operacje biznesowe w firmie. COO podlega CEO (Chief Executive Officer) i zwykle jest z… | 5 | 11 | 2 | 0.40 |
| średnie wynagrodzenie dla ea | Ile płaci Electronic Arts w Stanach Zjednoczonych? Średnia pensja Electronic Arts waha się od około 33 280 USD rocznie w przypadku testera zapewniania jakości do 150 000 USD rocznie w przypadku Head S… | 20 | 32 | 1 | 0.50 |
| kim jest millie | Millie to prywatny francuski silnik wąskotorowy Sir Roberta Norramby'ego. Pracuje na kolei osiedlowej. Millie to pełna życia francuska lokomotywa, której właścicielem jest sir Robert Norramby, który j… | 3 | 13 | 1 | 0.50 |
| jaki jest minimalny wynik ASVAB dla armii | Minimalny wynik 31 jest wymagany, aby dostać się, jeśli masz maturę. Jeśli masz GED, będziesz potrzebować 50. Wiadomo, że armia podnosi swoje wyniki do 50 dla wszystkich w czasach wysokiego zaciągu. Z… | 9 | 13 | 3 | 0.50 |
| co to jest reakcja lotu? | Reakcja „walcz lub uciekaj” to reakcja fizjologiczna wyzwalana, gdy odczuwamy silną emocję, taką jak strach. Strach jest normalną emocją, którą odczuwa się w odpowiedzi na niebezpieczeństwo lub zagroż… | 65 | 43 | 1 | 0.50 |
| kim jest piosenkarz | Śpiew jest czynnością polegającą na wytwarzaniu dźwięków muzycznych za pomocą głosu i wzbogaca regularną mowę poprzez zastosowanie trwałej tonacji, rytmu i różnorodnych technik wokalnych. Osoba śpiewa… | 65 | 31 | 2 | 0.50 |
| do czego shar pei były pierwotnie używane | Historia. Chiński Shar-Pei pochodził z południowych prowincji Chin, gdzie był ceniony jako myśliwy, pasterz, strażnik i wojownik. Niektórzy historycy uważają, że Shar-Pei jest starożytną rasą, chociaż… | 47 | 19 | 1 | 0.40 |
| kto jest ceo adp | Carlos A Rodriguez był dyrektorem generalnym Automatic Data (ADP) od mniej niż sześciu miesięcy. Pan Rodriguez jest w firmie od 13 lat. 48-letni dyrektor zajmuje 12 miejsce w kategorii Usługi i zaopat… | 4 | 30 | 1 | 0.50 |
| dzisiejszy kurs złota cena w Ćennaju | Cena jednego kilograma sztabki czystego złota w Chennai wynosi 3 079 000 rupii, a 22-karatowe sztabki złota za kilogram sprzedawane są na bazarze w Chennai za 2 879 000 rupii. | 14 | 11 | 1 | 0.40 |


## Queue state (v59)

```
chain:v59_sample	ok	-	23s
chain:v59_ranks	ok	-	takeover
chain:v59_tables	ok	-	2s
```
