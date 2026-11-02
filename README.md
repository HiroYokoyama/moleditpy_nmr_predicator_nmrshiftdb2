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

Choose **Analysis → NMR Prediction (nmrshiftdb2)** and a nucleus. The result window
shows chemical shifts, reference ranges, estimated coupling patterns and a
spectrum. Table/peak selections highlight the corresponding atoms in 3D.
Multiplets and Lorentzian broadening remain optional. CSV export includes
couplings and the data attribution. The result window and plots identify the
reference data and its license.

## What the estimates mean

- The reference condition is **CDCl3**. These estimates do not predict solvent
  changes. Reported temperatures are restricted to 273–323 K; unreported
  temperatures are included. There is no temperature/concentration correction.
- A known connectivity can use its experimental reference-group median. Other
  structures search rooted neighborhoods from radius 4 down to 2, requiring at
  least **three independent molecular connectivities** for a local estimate.
- Duplicate spectra and symmetric sites are aggregated within a molecular
  connectivity before combining independent references.
- Stereochemistry is pooled. Every carbon-bound proton on a given carbon gets
  the same group estimate; stereospecific/diastereotopic H shifts and exchangeable
  OH/NH/SH protons are not modeled. This changes the old Java model's behavior.
- Charged molecules, radicals, disconnected structures, unsupported elements,
  and isotope substitutions other than 1H/13C are rejected. The heavy-atom limit
  is 256. Missing references are reported as unsupported, not replaced by guesses.
- Min/max are the observed reference molecule-median range, **not a confidence
  interval**. Multiplicity and J remain separate rule-based approximations in
  `coupling.py`, not experimental fits or output of this shift estimator.

This is a reference-search replacement, not a claim of numerical equivalence to
the old Java predictor. It is intended for exploratory interpretation, not
unqualified peak assignments.

## Validation

The pinned export contains 58,187 records. The accepted subset and exact counts
are in `data/provenance.json`. Validation holds out entire nonstereochemical
molecular connectivities, including duplicate spectra and stereoisomers, so a
held-out molecule cannot also supply exact-structure or local training references.

| Nucleus | Held-out molecule/environment targets | Predicted | Coverage | MAE (ppm) | RMSE (ppm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1H | 531 | 277 | 52.2% | 0.163 | 0.309 |
| 13C | 4,009 | 1,945 | 48.5% | 1.755 | 3.022 |

Errors refer only to supported targets. This is an internal held-out split of
one public export, not an independent experimental cohort or validation of
stereospecific H predictions. The shipped index then uses the full accepted
subset, including exact-connectivity references. Coverage of an individual
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
