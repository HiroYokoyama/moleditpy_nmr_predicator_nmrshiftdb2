# Contributing

## Report a problem

Include the steps to reproduce, expected and actual behavior, traceback (if
available), OS, Python, MoleditPy and plugin versions. For reference-search
issues, include a shareable structure or SMILES, nucleus and selected solvent.

## Development setup

Use Python 3.13 to match the main test workflow. From the repository root:

```sh
python -m pip install pytest rdkit
python -m pytest tests/ -q
```

The main suite uses GUI stubs. For integration with the real application, keep
`python_molecular_editor` beside this checkout, or set `CI_MAIN_APP_SRC` to its
`moleditpy/src` directory. For interactive GUI work, use MoleditPy >=5.0.0,
<6.0.0 with its PySide6, RDKit and visualization dependencies, and copy or symlink
`nmr_predicator_nmrshiftdb2/` into the user plugin directory:

- Windows: `C:\Users\<YourName>\.moleditpy\plugins\`
- Linux / macOS: `~/.moleditpy/plugins/`

Retain the package's `data/` directory and notices. Prediction uses the bundled
index; the original SDF is needed only to regenerate reference data.

## Checks

```sh
python -m pytest tests/ -q
python -m pip install ruff
python -m ruff check .
```

Real Qt callbacks are checked separately, in a fresh process with PySide6:

```sh
python -m pip install PySide6-Essentials==6.11.2
python -m unittest discover -s tests -p test_qt_callbacks.py -v
```

The callback CI sets `REQUIRE_REAL_QT=1`, `REQUIRED_QT_VERSION=6.11.2` and
`QT_QPA_PLATFORM=offscreen`, and runs on Windows, macOS and Linux. The main test
workflow runs on Windows. See [validation and benchmarks](docs/VALIDATION.md)
for scientific validation, regression coverage and timing methodology.

## Reference regeneration

From a repository checkout with Git LFS installed:

```sh
git lfs pull --include="data-source/nmrshiftdb2withsignals.sd"
python scripts/build_reference.py
```

The builder verifies the source checksum before processing the SDF. Source
metadata and rebuild options are documented in
[THIRD_PARTY_SOURCES.md](https://github.com/HiroYokoyama/moleditpy_nmr_predicator_nmrshiftdb2/blob/v4.0.3/nmr_predicator_nmrshiftdb2/THIRD_PARTY_SOURCES.md).
Review provenance and validation changes whenever reference generation changes.

## Packaging and release

Set `PLUGIN_VERSION` in `nmr_predicator_nmrshiftdb2/__init__.py`, then run:

```sh
python scripts/release_common.py verify --version 4.0.4
python scripts/release_common.py package --version 4.0.4
```

Use the intended version in both commands. The release configuration includes
runtime data, required notices, the README and reference-generation/benchmark
scripts. Architecture, contribution and validation guides, maintenance-tool
documentation, the original SDF and screenshots remain in the repository and
are excluded from the ZIP. The README links to hosted guides. A `v*.*.*` tag triggers the release workflow, which publishes the
ZIP and notifies the plugin collection when `REGISTRY_PAT` is configured.
Confirm the collection's registry URL and SHA-256 after publishing.

Shared maintenance scripts are managed upstream; see
[scripts/COMMON_TOOLS.md](scripts/COMMON_TOOLS.md).

## Pull requests

Use a branch and describe the resulting behavior and verification. Add tests
for changed prediction or interaction behavior, and manually verify GUI
changes. Keep internal exceptions visible to callers; UI callbacks should log
tracebacks and show an actionable message. Avoid empty exception handlers.

Contributions to plugin software use GPL-3.0-only. Source and derived reference
data retain the separate database license; see [LICENSING.md](LICENSING.md).
