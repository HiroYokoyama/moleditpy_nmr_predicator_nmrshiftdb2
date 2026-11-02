"""Native search, input integrity, source-assignment and data-distribution checks."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
from unittest.mock import MagicMock

import pytest
from rdkit import Chem

from nmr_predicator_nmrshiftdb2 import database as db, PredictorWorker

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reference_builder", ROOT / "scripts/build_reference.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def make_index(tmp_path, smiles="CCO", nucleus="1H", radius=2, count=3):
    heavy, _ = db.heavy_graph(db.prepare_molecule(Chem.MolFromSmiles(smiles)))
    entries = {}
    for atom in heavy.GetAtoms():
        if atom.GetAtomicNum() == 6:
            key = db.environment_keys(heavy, atom.GetIdx())[radius]
            entries[db.index_key(nucleus, radius, key)] = [1.2, 1.0, 1.4, count]
    path = tmp_path / "reference.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(dict(encoder=db.ENCODER, solvents=[db.SOLVENT], entries=entries), handle)
    return path


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1O", "C1CCCCC1", "CC(C)(C)OC(=O)C", "c1ncccc1"])
def test_keys_are_atom_order_and_map_independent(smiles):
    mol, _ = db.heavy_graph(db.prepare_molecule(Chem.MolFromSmiles(smiles)))
    original = [db.environment_keys(mol, i) for i in range(mol.GetNumAtoms())]
    rng = random.Random(941)
    for _ in range(8):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        permuted = Chem.RenumberAtoms(mol, order)
        for atom in permuted.GetAtoms():
            atom.SetAtomMapNum(987 + atom.GetIdx())
        graph, _ = db.heavy_graph(permuted)
        for new, old in enumerate(order):
            assert db.environment_keys(graph, new) == original[old]


def test_boundary_degree_and_hydrogen_counts_change_key():
    a, _ = db.heavy_graph(db.prepare_molecule(Chem.MolFromSmiles("CCCC")))
    b, _ = db.heavy_graph(db.prepare_molecule(Chem.MolFromSmiles("CCCCC")))
    assert db.environment_keys(a, 0)[2] == db.environment_keys(b, 0)[2]
    assert db.environment_keys(a, 0)[3] != db.environment_keys(b, 0)[3]
    assert db.environment_keys(a, 0)[0] != db.environment_keys(b, 0)[0]


def test_attached_h_group_estimates_and_exchangeable_sites(tmp_path):
    path = make_index(tmp_path)
    result = db.predict(Chem.MolFromSmiles("CCO"), "1H", path)
    assert len(result["data"]) == 5
    assert result["unsupported"] == [8]
    assert {row["parent_idx"] for row in result["data"]} == {0, 1}
    assert all(row["references"] == 3 for row in result["data"])


def test_sparse_local_reference_is_not_a_prediction(tmp_path):
    path = make_index(tmp_path, count=2)
    with pytest.raises(ValueError, match="No supported sites"):
        db.predict(Chem.MolFromSmiles("CCO"), "1H", path)


def test_exact_connectivity_can_use_one_independent_measured_structure(tmp_path):
    path = make_index(tmp_path, radius=0, count=1)
    result = db.predict(Chem.MolFromSmiles("CCO"), "1H", path)
    assert all(row["method"] == "exact_stereostructure" for row in result["data"])


def test_missing_database_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="Reinstall"):
        db.predict(Chem.MolFromSmiles("CCO"), "1H", tmp_path / "missing.gz")


def test_geometry_atom_order_and_properties_are_preserved(tmp_path):
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=4)
    mol.SetProp("_Name", "original")
    before = Chem.MolToMolBlock(mol)
    result = db.predict(mol, "1H", make_index(tmp_path))
    assert Chem.MolToMolBlock(mol) == before
    assert result["mol_with_h"].GetProp("_Name") == "original"
    assert result["mol_with_h"].GetConformer().Is3D()
    assert [a.GetSymbol() for a in result["mol_with_h"].GetAtoms()] == [a.GetSymbol() for a in mol.GetAtoms()]


@pytest.mark.parametrize("smiles", ["[Na+]", "CC.[Cl-]", "[CH3]", "[2H]C", "[15NH3]", "[NH4+]"])
def test_out_of_scope_chemistry_is_rejected(smiles):
    with pytest.raises(ValueError):
        db.prepare_molecule(Chem.MolFromSmiles(smiles))


def test_invalid_valence_is_rejected():
    mol = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
    with pytest.raises(ValueError, match="sanitized"):
        db.prepare_molecule(mol)


def test_cancellation_is_checked_before_io(tmp_path):
    with pytest.raises(db.PredictionCancelled):
        db.predict(Chem.MolFromSmiles("CCO"), "1H", tmp_path / "missing", cancelled=lambda: True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_index_rejects_nonfinite_values(bad):
    keys = {r: str(r) for r in (0, *db.RADII)}
    with pytest.raises(ValueError, match="invalid"):
        db.find_reference({db.index_key("13C", 2, "2"): [bad, 0, 1, 3]}, "13C", keys)


def test_assignment_parser_handles_explicit_and_attached_protons():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert builder.assigned_shifts("1.0;0;3|1.4;0;4|", mol, "1H") == {0: 1.2}
    assert builder.assigned_shifts("1.2;0;0|", Chem.MolFromSmiles("CCO"), "1H") == {0: 1.2}
    assert builder.assigned_shifts("2.0;0;8|", mol, "1H") == {}


@pytest.mark.parametrize("text", ["nan;0;0|", "1;0;999|", "garbage|", "700;0;0|"])
def test_bad_assignments_fail_instead_of_relabeling(text):
    with pytest.raises(ValueError):
        builder.assigned_shifts(text, Chem.MolFromSmiles("CC"), "13C")


def test_indexed_conditions_keep_calculated_and_measured_spectra_separate():
    assert builder.indexed_values("0:CDCl3 2:DMSO") == {"0": "CDCl3", "2": "DMSO"}
    assert builder.indexed_values("1: 2:Quantum ESPRESSO") == {"1": "", "2": "Quantum ESPRESSO"}


def test_duplicate_spectra_cannot_inflate_independent_support():
    keys = {r: str(r) for r in (0, *db.RADII)}
    index = builder.aggregate([("same-molecule", "CDCl3", "13C", keys, 20.0, 18.0, 22.0)] * 20)
    assert db.index_key("13C", 2, "2") not in index
    assert index[db.index_key("13C", 0, "0")][3] == 1


def test_bundled_index_matches_provenance_and_has_no_java():
    data = ROOT / "nmr_predicator_nmrshiftdb2/data"
    metadata = json.loads((data / "provenance.json").read_text())
    assert hashlib.sha256((data / "reference.json.gz").read_bytes()).hexdigest() == metadata["index_sha256"]
    assert metadata["source_sha256"] == builder.SOURCE_SHA256
    assert (data / "DATABASE-LICENSE.txt").is_file()
    assert not list((ROOT / "nmr_predicator_nmrshiftdb2").rglob("*.jar"))
    assert all(metadata["validation"]["CDCl3"][n]["predicted"] > 0 for n in ["1H", "13C"])


def test_real_index_predicts_without_any_external_process(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("External process was launched"))
    result = db.predict(Chem.MolFromSmiles("CC(=O)OCC"), "13C")
    assert len(result["data"]) >= 3
    assert all(-50 < row["ppm"] < 300 for row in result["data"])


def test_worker_emits_native_result_and_couplings(monkeypatch, tmp_path):
    path = make_index(tmp_path)
    (tmp_path / "data").mkdir()
    path.rename(tmp_path / "data/reference.json.gz")
    worker = PredictorWorker(Chem.MolFromSmiles("CCO"), "1H", tmp_path)
    worker.finished_signal, worker.error_signal = MagicMock(), MagicMock()
    worker.run()
    worker.error_signal.emit.assert_not_called()
    result = worker.finished_signal.emit.call_args.args[0]
    assert result["data"][0]["mult"] == "t"


def test_worker_reports_missing_data(tmp_path):
    worker = PredictorWorker(Chem.MolFromSmiles("CCO"), "1H", tmp_path)
    worker.finished_signal, worker.error_signal = MagicMock(), MagicMock()
    worker.run()
    worker.finished_signal.emit.assert_not_called()
    assert "Reinstall" in worker.error_signal.emit.call_args.args[0]


def test_reference_build_is_reproducible_from_assigned_source(tmp_path, monkeypatch):
    source = tmp_path / "source.sd"
    writer = Chem.SDWriter(str(source))
    for i, smiles in enumerate(["CCO", "CCCO", "CCCCO"]):
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("Solvent", "0:Chloroform-D1 (CDCl3)")
        mol.SetProp("Temperature [K]", "0:298")
        mol.SetProp("Spectrum 13C 0", f"{15 + i};0;0|")
        # This calculated spectrum must not enter the experimental references.
        mol.SetProp("Spectrum 13C 1", "250;0;0|")
        mol.SetProp("NMRProgram", "1:calculated")
        writer.write(mol)
    writer.close()
    monkeypatch.setattr(builder, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    builder.build(source, tmp_path / "one")
    builder.build(source, tmp_path / "two")
    assert (tmp_path / "one/reference.json.gz").read_bytes() == (tmp_path / "two/reference.json.gz").read_bytes()
    meta = json.loads((tmp_path / "one/provenance.json").read_text())
    assert meta["audit"]["accepted_spectra"] == 3
    assert meta["audit"]["rejected_calculated_spectra"] == 3
    with gzip.open(tmp_path / "one/reference.json.gz", "rt") as handle:
        entries = json.load(handle)["entries"]
    assert all(value[2] < 250 for value in entries.values())


def test_plugin_zip_includes_regeneration_source_and_no_binaries(tmp_path):
    import shutil
    from zipfile import ZipFile

    spec = importlib.util.spec_from_file_location("nmr_release", ROOT / "scripts/release_common.py")
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    config = json.loads((ROOT / "scripts/release-config.json").read_text())
    staged = tmp_path / "staged"
    shutil.copytree(ROOT / "nmr_predicator_nmrshiftdb2", staged / "nmr_predicator_nmrshiftdb2")
    for name in config["include"]:
        destination = staged / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    archive = release.package(staged, config, "4.0.0")
    with ZipFile(archive) as zipped:
        names = zipped.namelist()
        prefix = "nmr_predicator_nmrshiftdb2/"
        for name in ["LICENSE", "LICENSING.md", "NOTICE.txt", "data/DATABASE-LICENSE.txt",
                     "data/reference.json.gz", "data/provenance.json", "scripts/build_reference.py"]:
            assert prefix + name in names
        assert not any(name.endswith((".jar", ".class", ".pyc")) for name in names)
        zipped.extractall(tmp_path / "installed")
    # Verify the included regeneration script also resolves the standalone
    # backend when run from the installed package, without importing the GUI.
    import sys
    result = subprocess.run(
        [sys.executable, str(tmp_path / "installed" / prefix / "scripts/build_reference.py"), "--help"],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr
