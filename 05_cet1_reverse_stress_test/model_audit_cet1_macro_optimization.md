# Model Audit: `cet1_macro_optimization.py`

## Översikt

Den största risken i modellen är inte nödvändigtvis optimeraren utan att flera centrala antaganden, kommentarer och ekvationer inte pekar åt samma håll. Det gör att designpunkten kan vara numeriskt korrekt enligt koden men ändå konceptuellt feltolkad.

Det här dokumentet delar upp observationerna i tre kategorier:

1. Dokumentationsfel
2. Verkliga logiska modellfel
3. Hårda men försvarbara designval

---

## 1. Dokumentationsfel

Det här är sådant som gör modellen svår att läsa och försvara, även om koden fortfarande kan köras.

### 1.1 Problemets dimension är fel beskriven

I toppen av filen beskrivs problemet som om
`Δ ∈ R^5`

Men implementationen använder:
- 6 basvariabler
- 2 laggar per variabel
- totalt 18 dimensioner

Kodreferenser:
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:11`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:11>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:80`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:80>)

Konsekvens:
- Läsaren tror att modellen optimerar i låg dimension.
- I praktiken optimeras ett mycket rikare laggat scenario.

### 1.2 Texten säger avstånd från historiskt medelvärde, koden minimerar från dagens läge

I den formella problembeskrivningen ser det ut som om Mahalanobisavståndet mäts från historiskt medelvärde.

Men i koden definieras målfunktionen som:
`D_M² = (δ - δ_baseline)^T Σ^-1 (δ - δ_baseline)`

Kodreferenser:
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:318`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:318>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:327`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:327>)

Konsekvens:
- Designpunkten är “närmsta scenario från idag”, inte “närmsta scenario från normalläget”.
- Det är en helt annan tolkning.

### 1.3 Kommentarer om depletion är inkonsistenta med faktiska parametrar

Koden har:
- `CET1_RATIO_0 = 0.17`
- `DEPLETION = 0.1`
- `R_OMEGA = 0.07`

Men kommentarer och sammanfattning nämner fortfarande 300 bps eller andra nivåer.

Kodreferenser:
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:75`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:75>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:77`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:77>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:656`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:656>)

Konsekvens:
- Det går inte att veta om designpunkten avser ett mildare eller mycket hårdare breakdown-scenario utan att läsa koden rad för rad.

### 1.4 Betakällan är dåligt dokumenterad

Kommentaren säger att modellen använder:
“OLS betas re-estimated on the features selected by Elastic Net”

Men filen som läses är:
`data/final/per_sector_elastic_net_betas.csv`

Kodreferens:
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:137`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:137>)

Konsekvens:
- Även om filen råkar innehålla rätt kolumner blir det svårt att försvara exakt vilken modell som faktiskt driver PD-responsen.

---

## 2. Verkliga logiska modellfel

Det här är viktigare. Här handlar det inte bara om förklaringar eller kommentarer, utan om att modellen faktiskt inte implementerar samma objekt som den säger sig implementera.

### 2.1 CET1-ekvationen i texten matchar inte implementationen

I texten står:
`R(Δ) = (CET1^0 - L_q(Δ)) / RWA(Δ)`

Men i koden används:
- `loss_base = portfolio_loss(delta_baseline)`
- `incr_loss = portfolio_loss(delta) - loss_base`
- `R(Δ) = (CET1_0 - incr_loss) / stressed_rwa(delta)`

Kodreferenser:
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:15`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:15>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:355`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:355>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:366`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:366>)

Varför detta är viktigt:
- Full stressed loss och incremental loss är inte samma objekt.
- Om `CET1_0` tolkas som dagens CET1-kapital, då är incremental loss logiskt rimlig.
- Om dokumentationen säger full loss, då är modellen fel beskriven.
- Om du själv tänker i ena termen men optimerar den andra, blir designpunkten feltolkad.

### 2.2 Målfunktionen och den formella problemdefinitionen är olika problem

Texten säger i praktiken:
- hitta den minsta plausibla avvikelsen från historisk norm som bryter CET1

Koden gör:
- hitta den minsta ytterligare förflyttningen från dagens makroläge som bryter CET1

Kodreferenser:
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:220`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:220>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:291`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:291>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:318`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:318>)

Varför detta är viktigt:
- Om idag redan är stressat, blir designpunkten mycket “närmare”.
- Det är inte fel i sig, men det är ett annat reverse stress test än det texten antyder.

### 2.3 Sektorer med noll känslighet ligger kvar i portföljen

Modellen filtrerar bort sektorer utan sensitivitetstabell, men inte sektorer vars koefficienter i praktiken är alla noll.

Kodreferenser:
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:161`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:161>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:176`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:176>)

Varför detta är viktigt:
- Exponeringar i sådana sektorer stressas inte via makrokanalen.
- De bidrar ändå till portfölj-EAD, förlustbas och kapitalstruktur.
- Det kan göra att reverse stress-svaret blir för trögt eller konstigt.

### 2.4 RWA-modellen för “hela banken” bygger på en residualpost som inte är modellerad

Koden sätter:
- corporate RWA från portföljen
- total bank RWA från ett antaget CET1-ratio
- resten blir `RWA_other`
- `RWA_other` hålls konstant i alla scenarier

Kodreferenser:
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:241`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:241>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:254`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:254>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:341`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:341>)

Varför detta är viktigt:
- Reverse stress-resultatet kallas bank-wide CET1.
- Men bara corporate RWA stressas explicit.
- Resten av banken är en fast residual.
- Det är ett starkt strukturellt antagande och kan ge en designpunkt som ser mer precis ut än den egentligen är.

---

## 3. Hårda men försvarbara designval

Det här är inte nödvändigtvis fel, men det är antaganden som bär mycket av resultatet. Om designpunkten känns “fel” kan det vara här problemet ligger.

### 3.1 Syntetisk kapitalstruktur

Du sätter:
- `CET1_0 = EAD_total / EAD_CET1_RATIO`
- `RWA_total_0 = CET1_0 / CET1_RATIO_0`

Kodreferenser:
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:247`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:247>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:250`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:250>)

Det är försvarbart om målet är en stylized bank.
Men det betyder också:
- designpunkten bestäms starkt av `17%` CET1-ratio och `6x` EAD/CET1
- inte bara av portföljdata och betas

Om det här antagandet är felkalibrerat blir hela reverse stress-scenariot felkalibrerat.

### 3.2 Permanent shock över alla laggar

I känslighetsanalysen och startpunkterna använder du “permanent shock”-logik:
samma avvikelse läggs på nutida variabel och alla laggar.

Kodreferenser:
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:398`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:398>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:430`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:430>)

Det är försvarbart om du vill modellera en persistent regimskiftesscenario.
Men det innebär också:
- stor strukturell bindning mellan samtid och historia
- mindre frihet för optimeraren att hitta ekonomiskt mer realistiska bana-liknande scenarier

### 3.3 Fast korrelation `rho`

`rho` beräknas från bas-PD en gång och hålls sedan konstant under stress.

Kodreferenser:
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:171`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:171>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:302`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:302>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:348`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:348>)

Det är försvarbart för stabilitet och enkelhet.
Men eftersom din korrelationsfunktion i grunden är PD-beroende är det samtidigt en medveten förenkling.

### 3.4 Samma ASRF-kvantil används för både förlust och kapital

I nuvarande fil står `ASRF_QUANTILE = 0.999`, vilket undviker negativa RWA och är matematiskt stabilt.

Kodreferenser:
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:68`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:68>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:241`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:241>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:348`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:348>)

Det är försvarbart om du vill ha en konsekvent “tail-state” definition.
Men det är fortfarande ett tungt antagande:
- samma kvantil driver både expected portfolio loss under systematic stress och regulatory-style capital charge
- det behöver inte vara det enda rimliga valet

---

## Bedömning av ekvationslogiken

## Det som faktiskt är logiskt i implementationen

Följande kedja är internkonsistent:

1. Makroscenario uttrycks relativt dagens läge via `delta - delta_baseline`
2. Detta översätts till logit-PD-justeringar via sektorbetas
3. Stressade PD används i både loss och RWA
4. CET1-ratio byggs som dagens CET1 minus inkrementell förlust dividerat med stressad total RWA
5. Optimeraren söker minsta ytterligare Mahalanobisförflyttning från idag som bryter tröskeln

Kodreferenser:
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:295`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:295>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:338`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:338>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:355`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:355>)
- [`05_cet1_reverse_stress_test/cet1_macro_optimization.py:422`](</Users/gustavjerresand/RST2.0-Modellen/05_cet1_reverse_stress_test/cet1_macro_optimization.py:422>)

## Det som inte stämmer mellan matematik och kod

Följande bör betraktas som verkliga mismatchar:

- Full loss i texten, incremental loss i koden
- Avstånd från historiskt mean i texten, avstånd från idag i koden
- 5-dimensionsproblem i texten, 18-dimensionsproblem i koden
- Dokumenterad tröskel och faktisk tröskel är inte samma
- Betakällan är semantiskt oklar

---

## Min sammanfattande bedömning

Om du inte gillar designpunkten skulle jag i första hand misstänka att problemet ligger i modellens definition, inte i solver eller optimeringsalgoritm.

De mest sannolika orsakerna är:

- Du optimerar inte exakt det problem du beskriver i text.
- Kapitalstrukturen är för hårt antagen och för lite datadriven.
- Breakdown-tröskeln verkar ha ändrats utan att resten av dokumentationen hängt med.
- Reverse stress-svaret gäller egentligen “corporate-stressad bank med fast residual-RWA”, inte en fullt modellerad bank.
- PD-kanalen och RWA-kanalen bygger på betakällor och residualantaganden som inte är helt spårbara.

Om du vill ha en trovärdig designpunkt behöver du först bestämma vilken av följande två modeller du faktiskt vill stå för:

1. “Närmsta scenario från idag som bryter CET1”
2. “Närmsta scenario från historisk norm som bryter CET1”

Just nu säger texten mest det ena och koden gör mest det andra.
