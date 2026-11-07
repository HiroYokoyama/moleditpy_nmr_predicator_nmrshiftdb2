# NMR Prediction Plugin for MoleditPy

[![CI](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/actions/workflows/test.yml/badge.svg)](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/actions/workflows/test.yml)
[![GitHub tag](https://img.shields.io/github/v/tag/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2?label=version)](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/tags)
[![GitHub Downloads](https://img.shields.io/github/downloads/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/total)](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/releases)

Offline 1H/13C chemical-shift estimates from **nmrshiftdb2 experimental data**,
using a **Python + RDKit** reference search. The release ZIP includes prepared
reference data and is ready to use offline.

## Install and use

Requires **MoleditPy >=5.0.0, <6.0.0** with PySide6.

Install the complete release ZIP through MoleditPy's Plugin Installer, or extract
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

Reference-search estimates support exploratory interpretation. Treat unsupported
sites and the reported reference ranges/counts explicitly when assigning peaks.

## Validation and examples

[Validation and benchmarks](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/blob/v4.0.4/docs/VALIDATION.md) contains the held-out error and
coverage tables, representative-molecule regression checks, result screenshots,
and reproducible timing measurements. Errors are measured only for supported
targets; reference coverage depends on the molecule and selected solvent.

## Data and software licenses

Contains information from [nmrshiftdb2](https://nmrshiftdb.nmr.uni-koeln.de/), made
available under the **nmrshiftdb2 Database License**. The source data and derived
reference index keep that license; the plugin's first-party software is **GPLv3**.
See [LICENSING.md](LICENSING.md), [NOTICE.txt](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/blob/v4.0.4/nmr_predicator_nmrshiftdb2/NOTICE.txt), and
[THIRD_PARTY_SOURCES.md](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/blob/v4.0.4/nmr_predicator_nmrshiftdb2/THIRD_PARTY_SOURCES.md). Keep the attribution when publicly using output.

## Documentation and development

- [Architecture](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/blob/v4.0.4/ARCHITECTURE.md): search keys, aggregation and prediction flow.
- [Contributing](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/blob/v4.0.4/CONTRIBUTING.md): development setup, checks, regeneration and packaging.
- [Data sources](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/blob/v4.0.4/nmr_predicator_nmrshiftdb2/THIRD_PARTY_SOURCES.md): pinned export and rebuild instructions.
- [License scope](LICENSING.md): software, database and dependency terms.
