"""Prediction validation and result lifecycle regressions."""

from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem
from test_nmr_coverage import _make_dialog

import nmr_predicator_nmrshiftdb2 as plugin
from nmr_predicator_nmrshiftdb2 import coupling


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e999"])
@pytest.mark.parametrize("column", [0, 1, 2])
def test_nonfinite_predictions_are_skipped(tmp_path, value, column):
    mol = Chem.MolFromSmiles("C")
    worker = plugin.PredictorWorker(mol, "13C", tmp_path)
    fields = ["1.0", "2.0", "3.0"]
    fields[column] = value
    assert worker._parse_output("1: " + " ".join(fields), mol) == []


def test_deuterium_is_not_a_predicted_1h_nucleus(tmp_path):
    mol = Chem.MolFromSmiles("[2H]C")
    worker = plugin.PredictorWorker(mol, "1H", tmp_path)
    assert worker._parse_output("1: 1 2 3", mol) == []


def test_deuterium_is_not_a_coupling_proton():
    mol = Chem.AddHs(Chem.MolFromSmiles("[2H]C([2H])([2H])C"))
    assert coupling.predict_ch_couplings(mol)[1]["n_h"] == 0
    assert all(
        not coupling._is_coupling_proton(a)
        for a in mol.GetAtoms()
        if a.GetIsotope() == 2
    )


def test_dialog_done_stops_timer_and_removes_highlights():
    dialog, window, _context = _make_dialog()
    dialog.sel_timer = MagicMock()
    dialog._highlight_actors[0] = object()
    dialog.done(0)
    dialog.sel_timer.stop.assert_called_once()
    assert dialog._highlight_actors == {}
    window.plotter.remove_actor.assert_called()


def test_structure_replacement_invalidates_highlights():
    dialog, window, _context = _make_dialog()
    window.current_mol = Chem.MolFromSmiles("CCC")
    dialog.highlight_atom(0)
    window.plotter.add_mesh.assert_not_called()
    assert "changed" in dialog.status_label.setText.call_args[0][0]


def test_cleanup_uses_original_plotter_and_clears_cache_on_errors():
    dialog, window, _context = _make_dialog()
    original = MagicMock()
    original.remove_actor.side_effect = RuntimeError("closed actor")
    original.render.side_effect = RuntimeError("closed renderer")
    dialog._highlight_plotter = original
    dialog._highlight_actors[1] = object()
    dialog._label_actors[1] = object()
    dialog.clear_3d_visuals()
    assert original.remove_actor.call_count == 3
    assert dialog._highlight_actors == dialog._label_actors == {}
    window.plotter.remove_actor.assert_not_called()


def test_running_prediction_cannot_be_replaced():
    context = MagicMock()
    context.current_molecule = Chem.MolFromSmiles("C")
    context.get_main_window.return_value.nmr_worker.isRunning.return_value = True
    with (
        patch.object(plugin, "ask_nucleus") as ask,
        patch.object(plugin, "QMessageBox"),
    ):
        plugin.run_prediction(context)
    ask.assert_not_called()
