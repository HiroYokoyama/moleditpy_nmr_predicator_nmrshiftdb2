# Data sources and runtime dependencies

Version 4 uses Python/RDKit database search and includes an experimental
reference index generated from the public SDF export.

## Public source data

- Official data overview: https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/t1.html
- Assigned SDF: https://downloads.sourceforge.net/project/nmrshiftdb2/data/nmrshiftdb2withsignals.sd
- Pinned SHA-256: `0e86688360e23c88ccf0eb82a1251315fa57ec6f5b376dc8886f3311793a0afe`
- Size: 158,569,118 bytes.
- Data license: [data/DATABASE-LICENSE.txt](data/DATABASE-LICENSE.txt).

The source digest was checked against a direct download from the official data
distribution. `scripts/build_reference.py` reads its atom-assigned 1H/13C spectra
and generates rooted-fragment search keys with solvent partitions, tetrahedral
configuration and E/Z bond descriptors. Enantiomers share references in the
supported achiral solvents; unspecified stereo remains separate. This is not a CDK HOSE table.
`data/provenance.json` records filtering, aggregation and held-out validation.
The full machine-readable derivative index and its transformation source are
provided. No original full SDF is needed at runtime.

## Rebuild

From a repository checkout:

```sh
python scripts/build_reference.py --source /path/to/nmrshiftdb2withsignals.sd
```

The same command works inside an extracted plugin package. A source checksum
mismatch is rejected; a changed export requires an explicit data review and a
new provenance record. The builder imports only the backend module, without Qt.

## Separate terms

The derivative index remains under the nmrshiftdb2 Database License. Attribution
is displayed in the result window, saved plots, and CSV export. Preserve it when
publicly using outputs. The transformation source and derivative index are
available for alteration. The functional plugin remains GPLv3, satisfying the
data license's section 4.5 requirement for an OSI-approved software license.

RDKit, PySide6/Qt, NumPy, Matplotlib and PyVista retain their own terms. No
third-party library binaries are bundled. Older tags retain their original
contents and terms; their predictor JAR notices are not statements about v4.
