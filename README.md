# NMR Prediction Plugin for MoleditPy

[![CI](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/actions/workflows/test.yml/badge.svg)](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/actions/workflows/test.yml)

Offline 1H/13C chemical-shift estimates from **nmrshiftdb2 experimental data**,
using an original **Python + RDKit** reference search. Version 4 needs no Java,
CDK, predictor JARs, network connection or first-run training.

## Install and use

Install the complete release ZIP through MoleditPy's Plugin Manager, or extract
its `nmr_predicator_nmrshiftdb2/` folder into your user plugin directory. Keep the
`data/` folder and notices alongside the code. MoleditPy supplies the Python,
RDKit, Qt and visualization environment.

Choose **Analysis → NMR Prediction (nmrshiftdb2)**, a nucleus, and a reference solvent. The result window
shows chemical shifts, reference ranges, independent reference counts, match
types, estimated coupling patterns and a
spectrum. Table/peak selections highlight the corresponding atoms in 3D.
Multiplets and Lorentzian broadening remain optional. CSV export includes
couplings and the data attribution. The result window and plots identify the
reference data and its license.

## What the estimates mean

- The GUI offers **CDCl3, DMSO-d6, CD3OD, D2O, acetone-d6, C6D6,
  pyridine-d5, THF-d8, CD3CN and CCl4**. Only the selected solvent contributes
  references. Unreported, ambiguous and mixed solvents are excluded; there is
  no fallback to another solvent. Availability varies substantially by solvent.
  Reported temperatures are restricted to 273–323 K; unreported temperatures
  are included. There is no temperature/concentration correction.
- A known stereostructure can use its experimental reference-group median. Other
  structures search rooted neighborhoods from radius 4 down to 2, requiring at
  least **three independent molecular connectivities** for a local estimate.
- Duplicate spectra and symmetric sites are aggregated within a molecular
  connectivity before combining independent references.
- **E/Z bonds and tetrahedral configurations are part of the search keys.**
  Unspecified configurations stay separate from specified ones within the
  matched environment. A 3D conformer can supply missing stereochemical tags;
  existing specified tags are retained. Enantiomers share references in the
  supported achiral solvents, while diastereomers remain distinct. Local matches
  only account for stereochemistry within their matched neighborhood.
- Every carbon-bound proton on a given carbon gets the same group estimate;
  stereospecific/diastereotopic H shifts and exchangeable OH/NH/SH protons are
  not modeled. Enhanced and non-tetrahedral stereochemistry are unsupported.
- Charged molecules, radicals, disconnected structures, unsupported elements,
  and isotope substitutions other than 1H/13C are rejected. The heavy-atom limit
  is 256. Missing references are reported as unsupported, not replaced by guesses.
- **Reference min/max show the contributing experimental site/group values,
  not a confidence interval.** `Ref. molecules` counts independent molecular
  connectivities; repeated measurements can widen the range but never inflate
  that count. A single reference with a narrow or zero range is not evidence of
  high accuracy. `Exact` identifies the same stereostructure; `Local (n)` shows
  the neighborhood radius in bonds. These fields and the selected solvent are
  included in CSV exports.
- Multiplicity and J remain separate rule-based approximations in `coupling.py`,
  not experimental fits or output of this shift estimator.

This is a reference-search replacement, not a claim of numerical equivalence to
the old Java predictor. It is intended for exploratory interpretation, not
unqualified peak assignments.

## Validation

The pinned export contains 58,187 records. The accepted subset and exact counts
are in `data/provenance.json`. Validation holds out entire nonstereochemical
molecular connectivities, including duplicate spectra and stereoisomers, so a
held-out molecule cannot also supply exact-structure or local training references.

| Solvent | Accepted spectra | 1H coverage | 1H MAE / ppm | 13C coverage | 13C MAE / ppm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acetone-d6 | 116 | 0.0% | — | 16.9% | 1.822 |
| C6D6 | 88 | 0.0% | — | 0.0% | — |
| CCl4 | 63 | 0.0% | — | 32.0% | 1.266 |
| CD3CN | 21 | 0.0% | — | 0.0% | — |
| CD3OD | 251 | 14.1% | 0.134 | 36.5% | 1.604 |
| CDCl3 | 2494 | 46.3% | 0.165 | 41.3% | 1.673 |
| D2O | 824 | 50.4% | 0.202 | 34.4% | 4.598 |
| DMSO-d6 | 407 | 29.9% | 0.361 | 42.0% | 1.843 |
| Pyridine-d5 | 62 | — | — | 21.9% | 1.097 |
| THF-d8 | 12 | — | — | — | — |

Errors refer only to supported targets. This is an internal held-out split of
one public export, not an independent experimental cohort or validation of
stereospecific H predictions. A dash means no supported held-out target or no
held-out data; it is not a successful accuracy estimate. RMSE, denominators and
per-radius results are in `data/provenance.json`. The shipped index then uses
the full accepted subset, including exact-stereostructure references. Coverage of an individual
molecule may be lower or higher than this table.

## Data and software licenses

Contains information from [nmrshiftdb2](https://nmrshiftdb.nmr.uni-koeln.de/), made
available under the **nmrshiftdb2 Database License**. The source data and derived
reference index keep that license; the plugin's first-party software is **GPLv3**.
See [LICENSING.md](LICENSING.md), the package's `NOTICE.txt`, and
`THIRD_PARTY_SOURCES.md`. Keep the attribution when publicly using output.

The public data export is transformed directly using the included generation script.

## Development and reference regeneration

```sh
python -m pip install pytest rdkit
python -m pytest tests/ -q
python scripts/build_reference.py --source /path/to/nmrshiftdb2withsignals.sd
python scripts/release_common.py package --version 4.0.0
```

Download the assigned SDF from the official data page linked in
`THIRD_PARTY_SOURCES.md`. The builder checks its pinned SHA-256 before reading
the data. The transformation, filtering and aggregation are fully included in
the repository and ZIP. Runtime prediction never downloads or rebuilds data.
