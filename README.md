# TTK4265 Assignment 3: Satellite Ocean Color Data (gruppe 2)

Analyse av HYPSO-2 (hyperspektral) og Sentinel-3 OLCI (multispektral) over Nord-Adriaterhavet ved Venezia, 14. mai 2025.
Begge bildene er atmosfærekorrigert med OC-SMART (Level-2).

Denne grenen (`anna_prosjekt3`) inneholder **oppgave 2–5**. Oppgave 1 er ikke med her.

## Innhold

| Fil | Hva |
|---|---|
| `felles.py` | Felles funksjoner: innlesing, maskering, beskjæring av Sentinel-3 og transekt |
| `oppgave2_chla.ipynb` | Oppgave 2: Chl-a-kart side om side |
| `oppgave3_chla_transekt.ipynb` | Oppgave 3: Chl-a langs transekt + Rrs-sammenligning i tre punkter |
| `oppgave4_tsm.ipynb` | Oppgave 4: TSM-kart side om side |
| `oppgave5_tsm_transekt.ipynb` | Oppgave 5: TSM langs samme transekt |
| `oppgave*_*.png` | Ferdige figurer til rapporten (200 dpi) |
| `environment.yml` | Conda-miljøet |

Nederst i hver notebook ligger en celle **«Answers for the report»** med tolkning på engelsk som kan brukes i rapporten.

## Kom i gang

### 1. Legg inn datafilene

`.h5`-filene er for store for GitHub og ligger derfor **ikke** i repoet (se `.gitignore`).
Last dem ned fra kursets side og legg dem i samme mappe som notebookene, med disse navnene:

```
HYPSO2_HSI_aeronetvenice_2025-05-14T10-45-06Z-l1d_L2_OCSMART.h5
S3B_OL_1_EFR____20250514T091545_20250514T091845_20250515T101142_0179_106_264_2160_ESA_O_NT_004_L2_OCSMART.h5
```

### 2. Lag conda-miljøet

```bash
conda env create -f environment.yml
conda activate optiskfjernmaling
```

### 3. Kjør notebookene

Åpne en notebook i VS Code eller Jupyter og velg kernelen **optiskfjernmaling**. Kjør så alle cellene.
Notebookene er uavhengige av hverandre og kan kjøres i hvilken som helst rekkefølge.

Første gang kartene tegnes, laster cartopy ned kystlinjer fra Natural Earth, så du trenger internett.

## Metode i korte trekk

- **Maskering:** bare piksler med `L2_flags == 0` og verdier mellom 0 og 100 brukes. Land og sky er allerede NaN i filene.
- **Beskjæring:** Sentinel-3 beskjæres til omrisset av HYPSO-2-bildet (polygon fra kantene i lat/lon-gridet), ikke bare en rektangulær boks.
- **Projeksjon:** alle kart er plottet i UTM sone 33N (EPSG:32633).
- **Fargeskala:** felles logaritmisk skala for begge sensorene: Chl-a 0,1–30 mg m⁻³, TSM 0,1–20 g m⁻³.
- **Transekt:** 89 km fra Po-plumen (44,60°N, 12,35°E) til klarere vann (44,45°N, 13,45°E), med punkter hver 150 m.
  For hvert punkt brukes nærmeste piksel. Maks avstand er 600 m for HYPSO-2 (piksler ca. 140 × 900 m) og 300 m for OLCI (ca. 300 × 300 m).
  Start- og sluttpunkt endres i `felles.py` (`TRANSEKT_START`, `TRANSEKT_SLUTT`) og gjelder da for både oppgave 3 og 5.

## Merk

- Variablene heter `chlor_a(oci)` og `tsm(yoc)` i filene, ikke `chlor_a_oci_` og `tsm_yoc_` som i oppgaveteksten.
- Filene har ingen enhetsattributter. Vi bruker standardenhetene: Rrs i sr⁻¹, Chl-a i mg m⁻³ og TSM i g m⁻³.
