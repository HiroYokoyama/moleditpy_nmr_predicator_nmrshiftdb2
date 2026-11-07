# License scope

Copyright (C) 2026 Hiromichi Yokoyama / HiroYokoyama and the contributors
identified in the source history and retained notices.

## Plugin software

First-party plugin software is licensed under the GNU General Public License,
version 3 only (`GPL-3.0-only`). You may redistribute and/or modify it under those
terms. It is provided without warranty, including implied warranties of
merchantability or fitness for a particular purpose. The complete license is in
[LICENSE](LICENSE). Retain applicable copyright and license notices when conveying
source or binaries, and provide corresponding source as required by GPLv3.

## Reference database

The source snapshot in `data-source/` and the machine-readable derivative index
`nmr_predicator_nmrshiftdb2/data/reference.json.gz` are licensed under the
**nmrshiftdb2 Database License**, separately from the software. The unmodified full
text accompanies both data locations as `DATABASE-LICENSE.txt`. Copyright and
database rights remain with the nmrshiftdb2 contributors.

Contains information from nmrshiftdb2 (www.nmrshiftdb.org), which is made available
here under the nmrshiftdb2 Database License.

License URI: https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/nmrshiftdb2datalicense.txt

The index embeds its data-license URI, attribution and rights notice. Its source
export and SHA-256, filtering conditions and validation are recorded in
[provenance.json](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/blob/v4.0.3/nmr_predicator_nmrshiftdb2/data/provenance.json) and
[THIRD_PARTY_SOURCES.md](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/blob/v4.0.3/nmr_predicator_nmrshiftdb2/THIRD_PARTY_SOURCES.md). The complete transformation
source is included as `database.py` and `scripts/build_reference.py`; the complete
derivative index is included in the runtime ZIP in a machine-readable format.
The original SDF is available from the repository through Git LFS and is not a
runtime download or dependency.

The data license requires retained notices, attribution for publicly used outputs,
share-alike treatment of derivative databases, and access to the derivative data
or alteration method under its terms. Section 4.5 requires software relying on
this data to use an OSI-approved license; the functional plugin is GPLv3. The
software license does not replace the separate database license.

The result window links to the data license. About, exported CSV, saved plots and
the bundled notices identify the data source and applicable data license. Keep
appropriate attribution associated with publicly used outputs. The display of
reference ranges is not a grant of rights in third-party content.

## Separately installed dependencies

RDKit, PySide6/Qt, NumPy, Matplotlib and PyVista retain their own licenses. No
third-party library binaries are bundled in the plugin ZIP. If you separately
redistribute those dependencies, retain their required notices and satisfy their
applicable source and replacement/modification obligations.
