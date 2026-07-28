"""
Regression tests for the v2.4.0 chemistry fixes.

Three defects are covered here, all of which produced confidently wrong
output rather than an error:

1. ``RemoveStereochemistry`` ran before ``Compute2DCoords``. The backend reads
   double-bond geometry off those 2D coordinates, so wiping the E/Z flags
   first made every olefin come back as its *trans* isomer — Z-2-butene was
   predicted with the E-2-butene spectrum.
2. Hydrogens were never made explicit. ``predictorh.jar`` numbers heavy atoms
   first and reports one line per hydrogen, so on a molecule with implicit H
   every returned index landed past the end of the molecule and a 1H run
   yielded no peaks at all.
3. A failed sanitization was logged and then ignored, letting a chemically
   invalid structure through to the predictor.
"""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from nmr_predicator_nmrshiftdb2 import PredictorWorker

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "nmr_predicator_nmrshiftdb2"
LIB_DIR = PLUGIN_DIR / "lib"


class _Worker(PredictorWorker):
    """PredictorWorker with the Qt signals replaced by plain recorders."""

    def __init__(self, mol, nucleus):
        self.mol = mol
        self.nucleus = nucleus
        self.plugin_dir = PLUGIN_DIR
        self.results = []
        self.errors = []
        self.finished_signal = MagicMock(emit=self.results.append)
        self.error_signal = MagicMock(emit=self.errors.append)


def _run_capturing_molblock(mol, nucleus="13C", stdout="1: 1.0 2.0 3.0"):
    """Run the worker with java stubbed out; return (worker, molfile text)."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["molblock"] = Path(cmd[-1]).read_text(encoding="ascii")
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    worker = _Worker(mol, nucleus)
    with (
        patch.object(subprocess, "run", side_effect=fake_run),
        patch.object(shutil, "which", return_value="/usr/bin/java"),
    ):
        worker.run()
    return worker, captured.get("molblock", "")


def _molblock_atom_symbols(molblock):
    """Element symbols from a V2000 molfile's atom block, in file order."""
    lines = molblock.splitlines()
    counts = lines[3].split()
    n_atoms = int(counts[0])
    return [lines[4 + i].split()[3] for i in range(n_atoms)]


def _double_bond_is_cis(molblock):
    """True when the two methyls of 2-butene sit on the same side of C=C.

    Reads the geometry straight out of the molfile the backend receives,
    which is exactly the information the predictor uses.
    """
    mol = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=False)
    Chem.AssignStereochemistryFrom3D(mol) if mol.GetConformer().Is3D() else None
    Chem.SetBondStereoFromDirections(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    for bond in mol.GetBonds():
        if bond.GetStereo() in (
            Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOCIS,
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# 1. Stereochemistry must survive into the 2D layout
# ---------------------------------------------------------------------------


class TestStereochemistryPreserved:
    def test_cis_olefin_stays_cis_in_the_molfile(self):
        mol = Chem.MolFromSmiles(r"C/C=C\C")  # Z-2-butene
        _worker, molblock = _run_capturing_molblock(mol)
        assert molblock, "no molfile was handed to the backend"
        assert _double_bond_is_cis(molblock), (
            "Z-2-butene reached the predictor as its trans isomer — the E/Z "
            "flags were cleared before the 2D layout was generated"
        )

    def test_trans_olefin_stays_trans_in_the_molfile(self):
        mol = Chem.MolFromSmiles(r"C/C=C/C")  # E-2-butene
        _worker, molblock = _run_capturing_molblock(mol)
        assert not _double_bond_is_cis(molblock)

    def test_cis_and_trans_produce_different_geometry(self):
        """The two isomers must not collapse onto one another."""
        _w, cis = _run_capturing_molblock(Chem.MolFromSmiles(r"C/C=C\C"))
        _w, trans = _run_capturing_molblock(Chem.MolFromSmiles(r"C/C=C/C"))
        assert _double_bond_is_cis(cis) != _double_bond_is_cis(trans)


# ---------------------------------------------------------------------------
# 2. Explicit hydrogens
# ---------------------------------------------------------------------------


class TestExplicitHydrogens:
    def test_implicit_h_molecule_gains_explicit_h(self):
        mol = Chem.MolFromSmiles("CCO")  # ethanol, 3 heavy atoms, H implicit
        _worker, molblock = _run_capturing_molblock(mol, nucleus="1H")
        symbols = _molblock_atom_symbols(molblock)
        assert symbols.count("H") == 6, f"expected 6 explicit H, got {symbols}"

    def test_heavy_atoms_keep_their_indices(self):
        """AddHs must append, never renumber — 13C indices depend on it."""
        mol = Chem.MolFromSmiles("CCO")
        _worker, molblock = _run_capturing_molblock(mol)
        symbols = _molblock_atom_symbols(molblock)
        assert symbols[:3] == ["C", "C", "O"]

    def test_1h_predictions_are_returned_for_implicit_h_input(self):
        # Java numbers heavy atoms first: ethanol's H are java 4..9.
        stdout = "\n".join(f"{i}: 0.5 {1.0 + i / 10:.1f} 2.0" for i in range(4, 10))
        mol = Chem.MolFromSmiles("CCO")
        worker, _mb = _run_capturing_molblock(mol, nucleus="1H", stdout=stdout)
        assert not worker.errors, worker.errors
        assert worker.results, "1H run produced no peaks for an implicit-H molecule"
        data = worker.results[0]["data"]
        assert len(data) == 6
        assert {d["atom"] for d in data} == {"H"}

    def test_predictions_record_the_parent_heavy_atom(self):
        stdout = "\n".join(f"{i}: 0.5 1.0 2.0" for i in range(4, 10))
        mol = Chem.MolFromSmiles("CCO")
        worker, _mb = _run_capturing_molblock(mol, nucleus="1H", stdout=stdout)
        data = worker.results[0]["data"]
        # Every H must point at a heavy atom of ethanol (0=C, 1=C, 2=O).
        assert all(d["parent_idx"] in (0, 1, 2) for d in data)

    def test_13c_still_maps_carbons_by_index(self):
        mol = Chem.MolFromSmiles("CCO")
        worker, _mb = _run_capturing_molblock(
            mol, nucleus="13C", stdout="1: 10.0 15.0 20.0\n2: 50.0 57.0 60.0"
        )
        data = worker.results[0]["data"]
        assert [d["idx"] for d in data] == [0, 1]
        assert {d["atom"] for d in data} == {"C"}


# ---------------------------------------------------------------------------
# 3. Sanitization failure must abort
# ---------------------------------------------------------------------------


class TestSanitizationAborts:
    def test_invalid_valence_reports_an_error(self):
        mol = Chem.MolFromSmiles("C[N](C)(C)(C)C", sanitize=False)
        worker = _Worker(mol, "13C")
        with patch.object(shutil, "which", return_value="/usr/bin/java"):
            worker.run()
        assert worker.errors, "an unsanitizable structure was accepted"
        assert "sanitized" in worker.errors[0].lower()
        assert not worker.results

    def test_invalid_structure_never_reaches_java(self):
        mol = Chem.MolFromSmiles("C[N](C)(C)(C)C", sanitize=False)
        worker = _Worker(mol, "13C")
        with (
            patch.object(subprocess, "run") as run,
            patch.object(shutil, "which", return_value="/usr/bin/java"),
        ):
            worker.run()
        run.assert_not_called()

    def test_valid_structure_is_not_blocked(self):
        worker, molblock = _run_capturing_molblock(Chem.MolFromSmiles("CCO"))
        assert not worker.errors
        assert molblock


# ---------------------------------------------------------------------------
# End-to-end against the bundled JARs (skipped without a JRE)
# ---------------------------------------------------------------------------

_JARS_PRESENT = (LIB_DIR / "predictorc.jar").exists() and (
    LIB_DIR / "predictorh.jar"
).exists()

pytestmark_e2e = pytest.mark.skipif(
    not (_JARS_PRESENT and shutil.which("java")),
    reason="needs a JRE and the bundled nmrshiftdb2 JARs",
)


@pytestmark_e2e
class TestAgainstRealBackend:
    """The numbers below are the predictor's own, and match experiment:
    trans-2-butene CH3 ~16.8 ppm, cis-2-butene CH3 ~11.4 ppm."""

    def _predict(self, smiles, nucleus):
        worker = _Worker(Chem.MolFromSmiles(smiles), nucleus)
        worker.run()
        assert not worker.errors, worker.errors
        return worker.results[0]["data"]

    def test_cis_and_trans_butene_differ(self):
        trans = self._predict(r"C/C=C/C", "13C")
        cis = self._predict(r"C/C=C\C", "13C")
        trans_ch3 = min(d["ppm"] for d in trans)
        cis_ch3 = min(d["ppm"] for d in cis)
        assert trans_ch3 == pytest.approx(16.98, abs=0.5)
        assert cis_ch3 == pytest.approx(11.52, abs=0.5)
        assert trans_ch3 - cis_ch3 > 3.0

    def test_toluene_1h_separates_methyl_from_aromatic(self):
        data = self._predict("Cc1ccccc1", "1H")
        assert len(data) == 8  # 3 methyl + 5 aromatic
        shifts = sorted(d["ppm"] for d in data)
        assert shifts[0] == pytest.approx(2.30, abs=0.5)  # CH3
        assert shifts[-1] == pytest.approx(7.14, abs=0.5)  # aromatic
