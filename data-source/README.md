# Source database snapshot

`nmrshiftdb2withsignals.sd` is the original assigned SDF export, stored in Git LFS.
It is separately licensed under the **nmrshiftdb2 Database License**; see
[DATABASE-LICENSE.txt](DATABASE-LICENSE.txt). Copyright and database rights remain
with the nmrshiftdb2 contributors.

Contains information from nmrshiftdb2 (www.nmrshiftdb.org), which is made available
here under the nmrshiftdb2 Database License.

- Official source: https://downloads.sourceforge.net/project/nmrshiftdb2/data/nmrshiftdb2withsignals.sd
- SHA-256: `0e86688360e23c88ccf0eb82a1251315fa57ec6f5b376dc8886f3311793a0afe`
- Size: 158,569,118 bytes.
- License URI: https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/nmrshiftdb2datalicense.txt

From a repository checkout:

```sh
git lfs pull --include="data-source/nmrshiftdb2withsignals.sd"
python scripts/build_reference.py
```

The builder checks the source digest before generating the index. The raw source
snapshot is for regeneration and is excluded from the runtime plugin ZIP. Runtime
prediction uses the approximately 2 MB prepared index bundled inside the plugin.
