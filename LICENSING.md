# License scope

First-party plugin code is GPLv3 (`GPL-3.0-only`); see LICENSE. The independently
written Python/RDKit search backend is offered under the same license.

`nmr_predicator_nmrshiftdb2/data/reference.json.gz` is a derivative database under
the nmrshiftdb2 Database License, not GPLv3. Its full terms are in
`data/DATABASE-LICENSE.txt`. The data origin, exact source digest and validation
are in `data/provenance.json` and `THIRD_PARTY_SOURCES.md`. The release includes
the derivative index and the source code used to create it.

The data license requires retained notices, attribution for publicly used
outputs, share-alike treatment of derivative databases, availability of the
derivative data or alteration method, and an OSI-approved license for software
depending on the database. Our plugin code is GPLv3 and our index remains under
the data license. These are separate grants; neither replaces the other.

Version 4 uses Python and RDKit and does not bundle Java or CDK binaries.

Separately installed RDKit, PySide6/Qt, NumPy, Matplotlib and PyVista retain their
own licenses. Preserve their applicable notices and distribution obligations if
you separately bundle them. Earlier releases retain their original grants.
