"""The coupling estimates as wired into the nmrshiftdb2 predictor.

The shared module itself is tested in test_coupling.py; this file checks
that the worker attaches the multiplets and that the result window shows,
plots and exports them. Stubs come from conftest.py.
"""
import csv
from unittest.mock import MagicMock

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import nmr_predicator_nmrshiftdb2 as nmrmod
from nmr_predicator_nmrshiftdb2 import PredictorWorker, ResultDialog


def _run_worker(monkeypatch, tmp_path, mol, nucleus, stdout):
    """Feed deterministic reference estimates to the real annotation/UI path."""
    import re
    from nmr_predicator_nmrshiftdb2 import database

    def fake_predict(mol, nucleus, *args, **kwargs):
        expanded = database.prepare_molecule(mol)
        data = []
        for index, low, ppm, high in re.findall(r"(\d+):\s*(\S+)\s+(\S+)\s+(\S+)", stdout):
            index = int(index) - 1
            atom = expanded.GetAtomWithIdx(index)
            parent = atom.GetNeighbors()[0].GetIdx() if atom.GetAtomicNum() == 1 else index
            data.append(dict(idx=index, parent_idx=parent, atom=atom.GetSymbol(),
                             min=float(low), ppm=float(ppm), max=float(high)))
        return dict(nucleus=nucleus, data=data, mol_with_h=expanded)

    monkeypatch.setattr(database, "predict", fake_predict)
    worker = PredictorWorker(mol, nucleus, tmp_path)
    worker.finished_signal = MagicMock()
    worker.error_signal = MagicMock()
    worker.run()
    worker.error_signal.emit.assert_not_called()
    return worker.finished_signal.emit.call_args[0][0]


def _ethanol_3d():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


def test_worker_adds_1h_multiplets(monkeypatch, tmp_path):
    # CCO + H: atoms 0 C, 1 C, 2 O, 3-5 H on C0, 6-7 H on C1, 8 H on O
    stdout = "\n".join(
        [f"{i + 1}: 1.0 1.2 1.4" for i in (3, 4, 5)]
        + [f"{i + 1}: 3.4 3.6 3.8" for i in (6, 7)]
        + ["9: 1.5 2.5 3.5"]
    )
    payload = _run_worker(monkeypatch, tmp_path, _ethanol_3d(), "1H", stdout)
    by_idx = {d["idx"]: d for d in payload["data"]}
    assert by_idx[3]["mult"] == "t" and by_idx[3]["j_text"] == "7.0"
    assert by_idx[6]["mult"] == "q"
    assert by_idx[8]["mult"] == ""  # OH: exchangeable, no multiplet


def test_worker_adds_couplings_for_a_2d_molecule(monkeypatch, tmp_path):
    """No 3D on the host: coupling.ensure_3d embeds a copy for the dihedrals."""
    mol = Chem.MolFromSmiles("CCO")
    AllChem.Compute2DCoords(mol)
    stdout = "\n".join(f"{i + 1}: 1.0 1.2 1.4" for i in (3, 4, 5))
    payload = _run_worker(monkeypatch, tmp_path, mol, "1H", stdout)
    assert {d["mult"] for d in payload["data"]} == {"t"}


def test_worker_adds_13c_one_bond_couplings(monkeypatch, tmp_path):
    payload = _run_worker(monkeypatch, tmp_path, _ethanol_3d(), "13C", "1: 10 18 26\n2: 50 58 66")
    by_idx = {d["idx"]: d for d in payload["data"]}
    assert (by_idx[0]["mult"], by_idx[0]["j_ch"]) == ("q", 125.0)
    assert (by_idx[1]["mult"], by_idx[1]["j_ch"]) == ("t", 141.0)


def test_worker_keeps_input_geometry_for_coupling_estimates(monkeypatch, tmp_path):
    """A native reference lookup does not flatten or mutate host coordinates."""
    mol = _ethanol_3d()
    before = Chem.MolToMolBlock(mol)
    payload = _run_worker(monkeypatch, tmp_path, mol, "13C", "1: 10 18 26")
    assert Chem.MolToMolBlock(mol) == before
    assert payload["mol_with_h"].GetConformer().Is3D()


# -- result window -----------------------------------------------------------


def _dialog(nucleus="1H"):
    mol = _ethanol_3d()
    if nucleus == "1H":
        data = [{"idx": i, "atom": "H", "ppm": 1.2, "min": 1.0, "max": 1.4} for i in (3, 4, 5)]
    else:
        data = [{"idx": 0, "atom": "C", "ppm": 18.0, "min": 10.0, "max": 26.0}]
    nmrmod.coupling.annotate_predictions(mol, data, nucleus)
    mw = MagicMock()
    mw.current_mol = mol
    context = MagicMock()
    context.get_main_window.return_value = mw
    return ResultDialog(None, {"nucleus": nucleus, "data": data, "mol_with_h": mol}, context)


def test_table_has_coupling_columns():
    dlg = _dialog("1H")
    assert dlg.table.item(0, 7).text() == "t"
    assert dlg.table.item(0, 8).text() == "7.0"


def test_headers():
    assert nmrmod.table_headers("1H")[-2:] == ["Mult.", "J (Hz)"]
    assert nmrmod.table_headers("13C")[-2:] == ["Mult. (1H-coupled)", "1J(CH) (Hz)"]


def test_multiplets_on_by_default_for_1h_only():
    assert _dialog("1H").multiplet_chk.isChecked() is True
    assert _dialog("13C").multiplet_chk.isChecked() is False


def test_broadening_is_on_by_default():
    dlg = _dialog("1H")
    assert dlg.broadening_chk.isChecked() is True
    assert dlg.linewidth_spin.value() == 1.0
    assert _dialog("13C").linewidth_spin.value() == 2.0
    dlg.plot_spectrum()
    ax = dlg.figure.axes[0]
    ax.stem.assert_not_called()
    x, y = ax.plot.call_args[0][:2]
    assert len(x) > 100
    assert max(y) == pytest.approx(1.5, abs=0.01)  # centre of the 1:2:1 triplet, 3 H


def test_broadening_toggle_disables_the_width():
    dlg = _dialog("1H")
    dlg.broadening_chk.setChecked(False)
    dlg._on_broadening_toggled(False)
    assert dlg.linewidth_spin.isEnabled() is False


def test_plot_splits_the_signal_when_multiplets_are_shown():
    dlg = _dialog("1H")
    dlg.broadening_chk.setChecked(False)  # sticks, to count the lines
    dlg.plot_spectrum()
    shifts, heights = dlg.figure.axes[0].stem.call_args[0][:2]
    assert len(shifts) == 3  # triplet
    assert sum(heights) == pytest.approx(3.0)

    dlg.multiplet_chk.setChecked(False)
    dlg.plot_spectrum()
    shifts, heights = dlg.figure.axes[0].stem.call_args[0][:2]
    assert list(shifts) == [1.2] and list(heights) == [3.0]


@pytest.mark.parametrize("nucleus, margin", [("1H", 1.0), ("13C", 10.0)])
def test_auto_fit_leaves_room_beside_the_end_peaks(nucleus, margin):
    dlg = _dialog(nucleus)
    dlg.auto_scale_chk.setChecked(True)
    dlg.broadening_chk.setChecked(False)
    dlg.multiplet_chk.setChecked(False)  # fit to the signal centres
    dlg.plot_spectrum()
    shifts = [item["ppm"] for item in dlg.data]
    left, right = dlg.figure.axes[0].set_xlim.call_args[0]
    assert left == pytest.approx(max(shifts) + margin)
    assert right == pytest.approx(min(shifts) - margin)


def test_observe_frequency_for_13c():
    assert nmrmod.observe_mhz(400.0, "13C") == pytest.approx(100.58, abs=0.01)


def test_describe_multiplet():
    assert nmrmod.describe_multiplet({"mult": "dq", "j_text": "17.0, 7.0"}) == ", dq (J = 17.0, 7.0 Hz)"
    assert nmrmod.describe_multiplet({"mult": "s", "j_text": ""}) == ", s"
    assert nmrmod.describe_multiplet({}) == ""


def test_csv_export_includes_couplings(tmp_path, monkeypatch):
    dlg = _dialog("1H")
    path = tmp_path / "out.csv"
    monkeypatch.setattr(nmrmod.QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""), raising=False)
    monkeypatch.setattr(nmrmod.QMessageBox, "information", lambda *a, **k: None, raising=False)
    dlg.export_csv()
    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert rows[0][-2:] == ["Mult.", "J (Hz)"]
    assert rows[1][-2:] == ["t", "7.0"]
