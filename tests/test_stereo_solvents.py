"""Stereochemical identity, solvent isolation, and visible reference evidence."""

import csv
import gzip
import json
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import nmr_predicator_nmrshiftdb2 as plugin
from nmr_predicator_nmrshiftdb2 import database as db
from test_database import builder
from test_nmr_coverage import _make_dialog


def keys(smiles, center=0):
    mol, _ = db.heavy_graph(db.prepare_molecule(Chem.MolFromSmiles(smiles)))
    return db.environment_keys(mol, center)


@pytest.mark.parametrize("radius", [0, 2, 3, 4])
def test_e_z_and_unspecified_bonds_do_not_collapse(radius):
    assert len({keys(s)[radius] for s in ["C/C=C/C", "C/C=C\\C", "CC=CC"]}) == 3


@pytest.mark.parametrize("radius", [0, 2, 3, 4])
def test_enantiomers_share_achiral_solvent_reference(radius):
    assert keys("C[C@H](F)[C@H](Cl)O")[radius] == keys("C[C@@H](F)[C@@H](Cl)O")[radius]


@pytest.mark.parametrize("radius", [0, 2, 3, 4])
def test_diastereomers_and_unspecified_centers_are_distinct(radius):
    assert len({keys(s)[radius] for s in [
        "C[C@H](F)[C@H](Cl)O", "C[C@H](F)[C@@H](Cl)O", "CC(F)C(Cl)O"
    ]}) == 3


@pytest.mark.parametrize("smiles", ["C/C=C/C", "C[C@H](F)[C@H](Cl)O", "C[C@H]1CCC[C@H]1C"])
def test_stereo_keys_survive_atom_renumbering(smiles):
    mol, _ = db.heavy_graph(db.prepare_molecule(Chem.MolFromSmiles(smiles)))
    order = list(reversed(range(mol.GetNumAtoms())))
    other = Chem.RenumberAtoms(mol, order)
    for new, old in enumerate(order):
        assert db.environment_keys(other, new) == db.environment_keys(mol, old)


def test_explicit_hydrogens_do_not_change_stereostructure():
    mol = Chem.MolFromSmiles("C[C@H](F)[C@H](Cl)O")
    a, _ = db.heavy_graph(mol)
    b, _ = db.heavy_graph(Chem.AddHs(mol))
    assert db.environment_keys(a, 0) == db.environment_keys(b, 0)


def test_axial_stereo_is_rejected_instead_of_silently_pooled():
    mol = Chem.MolFromSmiles("c1ccccc1-c1ccccc1")
    single = next(b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE)
    single.SetStereo(Chem.BondStereo.STEREOATROPCW)
    with pytest.raises(ValueError, match="Axial"):
        db.prepare_molecule(mol)


def test_stereochemistry_is_perceived_from_3d_without_mutating_host():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](F)[C@H](Cl)O"))
    assert AllChem.EmbedMolecule(mol, randomSeed=751) == 0
    expected, _ = db.heavy_graph(db.prepare_molecule(mol))
    Chem.RemoveStereochemistry(mol)
    before = Chem.MolToMolBlock(mol)
    inferred, _ = db.heavy_graph(db.prepare_molecule(mol))
    assert db.environment_keys(inferred, 0) == db.environment_keys(expected, 0)
    assert Chem.MolToMolBlock(mol) == before


def write_index(path, entries, solvents):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"encoder": db.ENCODER, "solvents": solvents, "entries": entries}, handle)


def test_solvent_filter_never_borrows_other_solvents(tmp_path):
    code = keys("CCO")[0]
    entries = {db.index_key("13C", 0, code, "CDCl3"): [15, 14, 16, 1],
               db.index_key("13C", 0, code, "DMSO-d6"): [20, 19, 21, 1]}
    path = tmp_path / "reference.gz"
    write_index(path, entries, ["CDCl3", "DMSO-d6", "D2O"])
    for solvent, ppm in [("CDCl3", 15), ("DMSO-d6", 20)]:
        result = db.predict(Chem.MolFromSmiles("CCO"), "13C", path, solvent=solvent)
        assert result["solvent"] == solvent
        assert result["data"][0]["ppm"] == ppm
        assert solvent in result["model_note"]
    with pytest.raises(ValueError, match="No supported sites"):
        db.predict(Chem.MolFromSmiles("CCO"), "13C", path, solvent="D2O")


def test_stereoisomers_return_their_own_observed_values(tmp_path):
    trans, cis = keys("C/C=C/C"), keys("C/C=C\\C")
    entries = {db.index_key("13C", 0, trans[0]): [16.8, 16, 17, 1],
               db.index_key("13C", 0, cis[0]): [11.4, 11, 12, 1]}
    path = tmp_path / "reference.gz"
    write_index(path, entries, ["CDCl3"])
    for smiles, ppm in [("C/C=C/C", 16.8), ("C/C=C\\C", 11.4)]:
        result = db.predict(Chem.MolFromSmiles(smiles), "13C", path)
        assert result["data"][0]["ppm"] == ppm
    with pytest.raises(ValueError, match="No supported sites"):
        db.predict(Chem.MolFromSmiles("CC=CC"), "13C", path)


def test_source_aliases_are_explicit_and_mixed_solvents_are_excluded():
    reverse = {label: name for name, label in db.SOLVENTS.items()}
    assert reverse["Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)"] == "DMSO-d6"
    assert "DMSO" not in reverse
    assert "Chloroform-D1 (CDCl3) + Methanol-D4 (CD3OD)" not in reverse


def test_repeated_measurements_contribute_range_without_inflating_support():
    codes = keys("CCO")
    entries = builder.aggregate([("one", "CDCl3", "13C", codes, 15, 12, 18)])
    assert entries[db.index_key("13C", 0, codes[0])] == [15, 12, 18, 1]
    assert db.index_key("13C", 2, codes[2]) not in entries


def test_csv_identifies_range_support_match_and_solvent(tmp_path):
    dialog, _, _ = _make_dialog(data=[dict(idx=0, atom="C", ppm=15.0, min=12.0, max=18.0,
                                         references=1, radius=0, method="exact_stereostructure")], nucleus="13C")
    dialog.solvent = "DMSO-d6"
    path = tmp_path / "prediction.csv"
    messages = SimpleNamespace(information=MagicMock(), critical=MagicMock())
    with patch.object(plugin, "QFileDialog", SimpleNamespace(getSaveFileName=lambda *args: (str(path), ""))), \
         patch.object(plugin, "QMessageBox", messages):
        dialog.export_csv()
    messages.critical.assert_not_called()
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][3:7] == ["Ref. min (ppm)", "Ref. max (ppm)", "Ref. molecules", "Match (bonds)"]
    assert rows[1][3:7] == ["12.00", "18.00", "1", "Exact"]
    assert ["# Solvent", "DMSO-d6"] in rows


def test_gui_selection_is_forwarded_to_worker():
    context = MagicMock()
    context.current_molecule = Chem.MolFromSmiles("CCO")
    context.get_main_window.return_value.nmr_worker = None
    worker = MagicMock()
    with patch.object(plugin, "ask_conditions", return_value=("13C", "DMSO-d6", True)), \
         patch.object(plugin, "PredictorWorker", return_value=worker) as constructor, \
         patch.object(plugin, "QProgressDialog"):
        plugin.run_prediction(context)
    assert constructor.call_args.kwargs["solvent"] == "DMSO-d6"
    worker.start.assert_called_once()


def test_exact_matches_skip_local_fragment_work(tmp_path, monkeypatch):
    entries = {db.index_key("13C", 0, keys("CCO")[0]): [15, 12, 18, 1]}
    # Include both C sites so no unmatched site needs local fragments.
    entries[db.index_key("13C", 0, keys("CCO", 1)[0])] = [55, 54, 56, 1]
    path = tmp_path / "reference.gz"
    write_index(path, entries, ["CDCl3"])
    monkeypatch.setattr(db, "environment_keys", lambda *a, **kw: pytest.fail("Unneeded local fragment generation"))
    assert len(db.predict(Chem.MolFromSmiles("CCO"), "13C", path)["data"]) == 2
