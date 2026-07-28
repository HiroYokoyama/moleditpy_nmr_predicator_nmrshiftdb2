"""
Additional coverage tests for nmr_predicator_nmrshiftdb2:
  - PredictorWorker.run() success / no-prediction / CalledProcessError / generic-exception paths
  - ResultDialog construction and interaction (hover, click, highlight, sync, export, close)
  - run_prediction() orchestration (no molecule, cancelled nucleus dialog, success, error)
  - run() entry point (with and without a .host wrapper)

Qt / pyvista / matplotlib stubs are installed by conftest.py before this file is
imported; the QTableWidget / QTableWidgetItem / QDialog / QPushButton stubs were
extended there specifically so ResultDialog can be instantiated in this file.
"""
import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from rdkit import Chem
from rdkit.Chem import AllChem

import nmr_predicator_nmrshiftdb2 as nmrmod
from nmr_predicator_nmrshiftdb2 import (
    PredictorWorker,
    ResultDialog,
    run_prediction,
    run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embedded_mol(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


class FakeEdit3DManager:
    def __init__(self):
        self.selected_atoms_3d = []


class FakeMainWindow:
    """Minimal stand-in for the host main window used by ResultDialog."""
    def __init__(self, mol):
        self.plotter = MagicMock()
        self.current_mol = mol
        self.selected_atoms_3d = []  # presence used by hasattr() check
        self.edit_3d_manager = FakeEdit3DManager()
        self.nmr_result_dialog = None


def _make_dialog(data=None, nucleus="1H"):
    mol = _embedded_mol("CC")  # 2 C + 6 H = 8 atoms
    if data is None:
        data = [
            {"idx": 2, "atom": "H", "ppm": 1.2, "min": 1.0, "max": 1.4},
            {"idx": 3, "atom": "H", "ppm": 1.2, "min": 1.0, "max": 1.4},
            {"idx": 0, "atom": "C", "ppm": 15.0, "min": 14.0, "max": 16.0},
        ]
    mw = FakeMainWindow(mol)
    context = MagicMock()
    context.get_main_window.return_value = mw
    result_data = {"nucleus": nucleus, "data": data, "mol_with_h": mol}
    dialog = ResultDialog(None, result_data, context)
    return dialog, mw, context


# ---------------------------------------------------------------------------
# PredictorWorker.run() — success / empty / error branches
# ---------------------------------------------------------------------------

def test_run_success_emits_finished_signal(monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules["shutil"], "which", lambda _: "/usr/bin/java")

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "predictorh.jar").touch()

    mol = Chem.AddHs(Chem.MolFromSmiles("C"))  # C + 4H
    worker = PredictorWorker(mol, "1H", tmp_path)
    worker.finished_signal = MagicMock()
    worker.error_signal = MagicMock()

    fake_proc = MagicMock()
    fake_proc.stdout = "2: 0.9 1.1 1.3"  # java idx 2 -> 0-based 1 -> H
    fake_proc.stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_proc)

    worker.run()

    worker.finished_signal.emit.assert_called_once()
    worker.error_signal.emit.assert_not_called()
    payload = worker.finished_signal.emit.call_args[0][0]
    assert payload["nucleus"] == "1H"
    assert len(payload["data"]) == 1
    assert payload["data"][0]["atom"] == "H"


def test_run_emits_error_when_no_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules["shutil"], "which", lambda _: "/usr/bin/java")

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "predictorh.jar").touch()

    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    worker = PredictorWorker(mol, "1H", tmp_path)
    worker.finished_signal = MagicMock()
    worker.error_signal = MagicMock()

    fake_proc = MagicMock()
    fake_proc.stdout = "no peaks here"
    fake_proc.stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_proc)

    worker.run()

    worker.error_signal.emit.assert_called_once()
    worker.finished_signal.emit.assert_not_called()
    assert "No NMR peaks predicted" in worker.error_signal.emit.call_args[0][0]


def test_run_emits_error_on_called_process_error(monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules["shutil"], "which", lambda _: "/usr/bin/java")

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "predictorh.jar").touch()

    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    worker = PredictorWorker(mol, "1H", tmp_path)
    worker.finished_signal = MagicMock()
    worker.error_signal = MagicMock()

    def _raise(*a, **kw):
        raise subprocess.CalledProcessError(1, "java", output="out", stderr="boom")

    monkeypatch.setattr(subprocess, "run", _raise)

    worker.run()

    worker.error_signal.emit.assert_called_once()
    msg = worker.error_signal.emit.call_args[0][0]
    assert "Java Execution Failed" in msg
    assert "boom" in msg


def test_run_emits_error_on_unexpected_exception(monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules["shutil"], "which", lambda _: "/usr/bin/java")

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "predictorh.jar").touch()

    # Passing None as the molecule makes Chem.Mol(None) raise inside run(),
    # exercising the generic `except Exception` branch.
    worker = PredictorWorker(None, "1H", tmp_path)
    worker.finished_signal = MagicMock()
    worker.error_signal = MagicMock()

    worker.run()

    worker.error_signal.emit.assert_called_once()
    assert "Unexpected Error" in worker.error_signal.emit.call_args[0][0]


# ---------------------------------------------------------------------------
# ResultDialog construction + plot_spectrum
# ---------------------------------------------------------------------------

def test_dialog_construction_populates_table():
    dialog, mw, context = _make_dialog()
    assert dialog.table.rowCount() == 3
    assert dialog.nucleus == "1H"


def test_plot_spectrum_empty_data():
    dialog, mw, context = _make_dialog(data=[])
    dialog.plot_spectrum()  # should not raise on the "no peaks" branch


def test_plot_spectrum_auto_scale_and_manual():
    dialog, mw, context = _make_dialog()
    dialog.auto_scale_chk.isChecked.return_value = True
    dialog.plot_spectrum()
    dialog.auto_scale_chk.isChecked.return_value = False
    dialog.plot_spectrum()


# ---------------------------------------------------------------------------
# on_hover
# ---------------------------------------------------------------------------

def test_on_hover_leaves_axes_resets_hover():
    dialog, mw, context = _make_dialog()
    dialog._hover_idx = 1
    event = MagicMock()
    event.inaxes = None
    dialog.on_hover(event)
    assert dialog._hover_idx == -1


def test_on_hover_empty_data_returns_early():
    dialog, mw, context = _make_dialog(data=[])
    event = MagicMock()
    event.inaxes = MagicMock()
    event.xdata = 5.0
    dialog.on_hover(event)  # no shifts -> should just return


def test_on_hover_near_peak_highlights():
    dialog, mw, context = _make_dialog()
    event = MagicMock()
    event.inaxes.get_xlim.return_value = (0.0, 20.0)
    event.xdata = 1.2
    dialog.on_hover(event)
    assert dialog._hover_idx != -1


def test_on_hover_far_from_peak_resets():
    dialog, mw, context = _make_dialog()
    dialog._hover_idx = 0
    event = MagicMock()
    event.inaxes.get_xlim.return_value = (0.0, 20.0)
    event.xdata = 999.0
    dialog.on_hover(event)
    assert dialog._hover_idx == -1


def test_on_hover_matches_persistent_peak():
    dialog, mw, context = _make_dialog()
    dialog._persistent_ppm = 1.2
    event = MagicMock()
    event.inaxes.get_xlim.return_value = (0.0, 20.0)
    event.xdata = 1.2
    dialog.on_hover(event)  # should hit the "already persistent" early return path


# ---------------------------------------------------------------------------
# _restore_persistent_highlight
# ---------------------------------------------------------------------------

def test_restore_persistent_highlight_with_selection():
    dialog, mw, context = _make_dialog()
    dialog._persistent_ppm = 15.0
    dialog._restore_persistent_highlight()


def test_restore_persistent_highlight_without_selection():
    dialog, mw, context = _make_dialog()
    dialog._persistent_ppm = None
    dialog._restore_persistent_highlight()
    assert "Hover" in dialog.status_label.setText.call_args[0][0]


# ---------------------------------------------------------------------------
# on_graph_click / on_table_click
# ---------------------------------------------------------------------------

def test_on_graph_click_no_axes():
    dialog, mw, context = _make_dialog()
    event = MagicMock()
    event.inaxes = None
    dialog.on_graph_click(event)  # early return, no crash


def test_on_graph_click_empty_data():
    dialog, mw, context = _make_dialog(data=[])
    event = MagicMock()
    event.inaxes = MagicMock()
    event.xdata = 1.0
    dialog.on_graph_click(event)


def test_on_graph_click_near_peak_selects_rows():
    dialog, mw, context = _make_dialog()
    event = MagicMock()
    event.inaxes.get_xlim.return_value = (0.0, 20.0)
    event.xdata = 1.2
    dialog.on_graph_click(event)
    assert dialog._persistent_ppm == pytest.approx(1.2)


def test_on_graph_click_far_from_peak_clears_selection():
    dialog, mw, context = _make_dialog()
    dialog._persistent_ppm = 1.2
    event = MagicMock()
    event.inaxes.get_xlim.return_value = (0.0, 20.0)
    event.xdata = 999.0
    dialog.on_graph_click(event)
    assert dialog._persistent_ppm is None


def test_on_table_click_triggers_highlight():
    dialog, mw, context = _make_dialog()
    dialog.on_table_click(0, 0)
    assert dialog._persistent_ppm == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# highlight_atom / clear_3d_visuals
# ---------------------------------------------------------------------------

def test_highlight_atom_multi_peak():
    dialog, mw, context = _make_dialog()
    dialog.highlight_atom(0, persistent=True)  # idx 0 -> ppm 1.2 shared by 2 atoms
    assert len(dialog._highlight_actors) == 2
    assert len(dialog._label_actors) == 2


def test_highlight_atom_single_peak_not_persistent():
    dialog, mw, context = _make_dialog()
    dialog.highlight_atom(2, persistent=False)  # idx 2 -> ppm 15.0, single atom
    assert len(dialog._highlight_actors) == 1


def test_highlight_atom_handles_exception():
    dialog, mw, context = _make_dialog()
    context.get_main_window.side_effect = RuntimeError("boom")
    dialog.highlight_atom(0)  # should be caught internally, not raised


def test_clear_3d_visuals_clears_actor_maps():
    dialog, mw, context = _make_dialog()
    dialog.highlight_atom(2, persistent=True)
    assert dialog._highlight_actors
    dialog.clear_3d_visuals()
    assert dialog._highlight_actors == {}
    assert dialog._label_actors == {}


def test_clear_3d_visuals_handles_exception():
    dialog, mw, context = _make_dialog()
    context.get_main_window.side_effect = RuntimeError("boom")
    dialog.clear_3d_visuals()  # should be silenced


# ---------------------------------------------------------------------------
# _update_graph_highlight
# ---------------------------------------------------------------------------

def test_update_graph_highlight_hover_then_persistent():
    dialog, mw, context = _make_dialog()
    dialog._update_graph_highlight(1.2, is_hover=True)
    assert dialog._hover_line is not None
    dialog._update_graph_highlight(15.0, is_hover=False)
    assert dialog._graph_line is not None


def test_update_graph_highlight_none_ppm():
    dialog, mw, context = _make_dialog()
    dialog._update_graph_highlight(None, is_hover=False)


# ---------------------------------------------------------------------------
# _sync_from_3d
# ---------------------------------------------------------------------------

def test_sync_from_3d_no_selected_atoms_attr():
    dialog, mw, context = _make_dialog()
    bare_mw = object()
    context.get_main_window.return_value = bare_mw
    dialog._sync_from_3d()  # hasattr check fails -> early return, no crash


def test_sync_from_3d_matches_and_dedupes():
    dialog, mw, context = _make_dialog()
    mw.edit_3d_manager.selected_atoms_3d = [2]
    dialog._sync_from_3d()
    assert dialog._last_selected == {2}

    # Calling again with the same selection should hit the early-return branch.
    dialog._sync_from_3d()


def test_sync_from_3d_clears_on_empty_selection():
    dialog, mw, context = _make_dialog()
    mw.edit_3d_manager.selected_atoms_3d = [2]
    dialog._sync_from_3d()
    mw.edit_3d_manager.selected_atoms_3d = []
    dialog._sync_from_3d()
    assert dialog._last_selected == set()


def test_sync_from_3d_no_matching_atom_in_data():
    dialog, mw, context = _make_dialog()
    mw.edit_3d_manager.selected_atoms_3d = [99]  # not present in dialog.data
    dialog._sync_from_3d()
    assert dialog._last_selected == {99}


# ---------------------------------------------------------------------------
# show_about / clear_selection / export_csv / closeEvent
# ---------------------------------------------------------------------------

def test_show_about_builds_message_box():
    dialog, mw, context = _make_dialog()
    with patch("nmr_predicator_nmrshiftdb2.QMessageBox") as mock_msgbox:
        dialog.show_about()
        mock_msgbox.assert_called_once()
        mock_msgbox.return_value.exec.assert_called_once()


def test_clear_selection_resets_state():
    dialog, mw, context = _make_dialog()
    dialog._persistent_ppm = 1.2
    dialog.clear_selection()
    assert dialog._persistent_ppm is None


def test_export_csv_cancelled():
    dialog, mw, context = _make_dialog()
    with patch("nmr_predicator_nmrshiftdb2.QFileDialog") as mock_fd, \
         patch("nmr_predicator_nmrshiftdb2.QMessageBox") as mock_msgbox:
        mock_fd.getSaveFileName.return_value = ("", "")
        dialog.export_csv()
        mock_msgbox.information.assert_not_called()


def test_export_csv_success(tmp_path):
    dialog, mw, context = _make_dialog()
    out_path = tmp_path / "out.csv"
    with patch("nmr_predicator_nmrshiftdb2.QFileDialog") as mock_fd, \
         patch("nmr_predicator_nmrshiftdb2.QMessageBox") as mock_msgbox:
        mock_fd.getSaveFileName.return_value = (str(out_path), "CSV Files (*.csv)")
        dialog.export_csv()
        mock_msgbox.information.assert_called_once()
    content = out_path.read_text()
    assert "Atom ID" in content
    assert "15.00" in content


def test_export_csv_write_failure(tmp_path):
    dialog, mw, context = _make_dialog()
    with patch("nmr_predicator_nmrshiftdb2.QFileDialog") as mock_fd, \
         patch("nmr_predicator_nmrshiftdb2.QMessageBox") as mock_msgbox:
        # A directory path cannot be opened for writing -> triggers except branch.
        mock_fd.getSaveFileName.return_value = (str(tmp_path), "CSV Files (*.csv)")
        dialog.export_csv()
        mock_msgbox.critical.assert_called_once()


def test_close_event_clears_dialog_reference():
    dialog, mw, context = _make_dialog()
    mw.nmr_result_dialog = dialog
    dialog.closeEvent(MagicMock())
    assert mw.nmr_result_dialog is None


def test_close_event_handles_missing_main_window():
    dialog, mw, context = _make_dialog()
    context.get_main_window.side_effect = RuntimeError("boom")
    dialog.closeEvent(MagicMock())  # should be silenced, not raised


# ---------------------------------------------------------------------------
# run_prediction()
# ---------------------------------------------------------------------------

def test_run_prediction_no_molecule_warns():
    context = MagicMock()
    context.current_molecule = None
    with patch("nmr_predicator_nmrshiftdb2.QMessageBox") as mock_msgbox:
        run_prediction(context)
        mock_msgbox.warning.assert_called_once()


def test_run_prediction_zero_atom_molecule_warns():
    context = MagicMock()
    context.current_molecule = Chem.RWMol()  # 0 atoms
    with patch("nmr_predicator_nmrshiftdb2.QMessageBox") as mock_msgbox:
        run_prediction(context)
        mock_msgbox.warning.assert_called_once()


def test_run_prediction_cancelled_nucleus_dialog_no_worker():
    context = MagicMock()
    context.current_molecule = Chem.MolFromSmiles("C")
    with patch("nmr_predicator_nmrshiftdb2.ask_nucleus", return_value=(None, False)), \
         patch("nmr_predicator_nmrshiftdb2.PredictorWorker") as mock_worker_cls:
        run_prediction(context)
        mock_worker_cls.assert_not_called()


def test_run_prediction_success_shows_dialog():
    mw = MagicMock()
    context = MagicMock()
    context.get_main_window.return_value = mw
    context.current_molecule = Chem.MolFromSmiles("CO")

    mock_worker_instance = MagicMock()
    with patch("nmr_predicator_nmrshiftdb2.ask_nucleus", return_value=("13C", True)), \
         patch("nmr_predicator_nmrshiftdb2.PredictorWorker", return_value=mock_worker_instance), \
         patch("nmr_predicator_nmrshiftdb2.QProgressDialog") as mock_progress_cls, \
         patch("nmr_predicator_nmrshiftdb2.ResultDialog") as mock_dialog_cls:
        mock_progress_cls.return_value.wasCanceled.return_value = False
        run_prediction(context)

        on_success = mock_worker_instance.finished_signal.connect.call_args[0][0]
        result = {"nucleus": "13C", "data": [], "mol_with_h": context.current_molecule}
        on_success(result)

        mock_dialog_cls.assert_called_once()
        # The worker reference is released by QThread.finished, not by
        # on_success — dropping it earlier can destroy a running thread.
        release = mock_worker_instance.finished.connect.call_args[0][0]
        assert mw.nmr_worker is mock_worker_instance
        release()
        assert mw.nmr_worker is None
        mock_worker_instance.deleteLater.assert_called_once()


def test_run_prediction_error_shows_message():
    mw = MagicMock()
    context = MagicMock()
    context.get_main_window.return_value = mw
    context.current_molecule = Chem.MolFromSmiles("CO")

    mock_worker_instance = MagicMock()
    with patch("nmr_predicator_nmrshiftdb2.ask_nucleus", return_value=("1H", True)), \
         patch("nmr_predicator_nmrshiftdb2.PredictorWorker", return_value=mock_worker_instance), \
         patch("nmr_predicator_nmrshiftdb2.QProgressDialog") as mock_progress_cls, \
         patch("nmr_predicator_nmrshiftdb2.QMessageBox") as mock_msgbox:
        mock_progress_cls.return_value.wasCanceled.return_value = False
        run_prediction(context)

        on_error = mock_worker_instance.error_signal.connect.call_args[0][0]
        on_error("kaboom")

        mock_msgbox.critical.assert_called_once()


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------

def _install_fake_plugin_interface(monkeypatch):
    fake_root = types.ModuleType("moleditpy")
    fake_plugins = types.ModuleType("moleditpy.plugins")
    fake_iface = types.ModuleType("moleditpy.plugins.plugin_interface")

    class FakePluginContext:
        def __init__(self, plugin_manager, name):
            self.plugin_manager = plugin_manager
            self.name = name

    fake_iface.PluginContext = FakePluginContext
    monkeypatch.setitem(sys.modules, "moleditpy", fake_root)
    monkeypatch.setitem(sys.modules, "moleditpy.plugins", fake_plugins)
    monkeypatch.setitem(sys.modules, "moleditpy.plugins.plugin_interface", fake_iface)
    return FakePluginContext


def test_run_uses_host_attribute_when_present(monkeypatch):
    FakePluginContext = _install_fake_plugin_interface(monkeypatch)

    host_mw = MagicMock()
    wrapper = MagicMock()
    wrapper.host = host_mw

    with patch("nmr_predicator_nmrshiftdb2.run_prediction") as mock_run_pred:
        run(wrapper)
        mock_run_pred.assert_called_once()
        ctx_arg = mock_run_pred.call_args[0][0]
        assert isinstance(ctx_arg, FakePluginContext)
        assert ctx_arg.plugin_manager is host_mw.plugin_manager


def test_run_uses_mw_directly_when_no_host(monkeypatch):
    FakePluginContext = _install_fake_plugin_interface(monkeypatch)

    class PlainMw:
        def __init__(self):
            self.plugin_manager = MagicMock()

    mw = PlainMw()

    with patch("nmr_predicator_nmrshiftdb2.run_prediction") as mock_run_pred:
        run(mw)
        mock_run_pred.assert_called_once()
        ctx_arg = mock_run_pred.call_args[0][0]
        assert isinstance(ctx_arg, FakePluginContext)
        assert ctx_arg.plugin_manager is mw.plugin_manager


# ---------------------------------------------------------------------------
# Cancelling the progress dialog
# ---------------------------------------------------------------------------


def _run_with_progress(cancelled, extra_patch):
    mw = MagicMock()
    context = MagicMock()
    context.get_main_window.return_value = mw
    context.current_molecule = Chem.MolFromSmiles("CO")
    worker = MagicMock()
    with patch("nmr_predicator_nmrshiftdb2.ask_nucleus", return_value=("1H", True)), \
         patch("nmr_predicator_nmrshiftdb2.PredictorWorker", return_value=worker), \
         patch("nmr_predicator_nmrshiftdb2.QProgressDialog") as mock_progress_cls, \
         patch(f"nmr_predicator_nmrshiftdb2.{extra_patch}") as patched:
        mock_progress_cls.return_value.wasCanceled.return_value = cancelled
        run_prediction(context)
        yield worker, patched, mw


def test_cancelled_prediction_does_not_open_the_result_dialog():
    worker, dialog_cls, _mw = next(_run_with_progress(True, "ResultDialog"))
    on_success = worker.finished_signal.connect.call_args[0][0]
    on_success({"nucleus": "1H", "data": [], "mol_with_h": None})
    dialog_cls.assert_not_called()


def test_cancelled_prediction_does_not_show_the_error_box():
    worker, msgbox, _mw = next(_run_with_progress(True, "QMessageBox"))
    on_error = worker.error_signal.connect.call_args[0][0]
    on_error("java exploded")
    msgbox.critical.assert_not_called()


# ---------------------------------------------------------------------------
# Java timeout
# ---------------------------------------------------------------------------


def test_java_timeout_is_reported(monkeypatch, tmp_path):
    import subprocess as _sp

    mol = Chem.MolFromSmiles("CO")
    worker = PredictorWorker(mol, "13C", tmp_path)

    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "predictorc.jar").write_bytes(b"jar")

    monkeypatch.setattr(nmrmod.shutil, "which", lambda _name: "/usr/bin/java")
    monkeypatch.setattr(
        nmrmod.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(_sp.TimeoutExpired("java", 1)),
    )

    errors = []
    worker.error_signal = MagicMock()
    worker.error_signal.emit = errors.append
    worker.run()

    assert errors and "timed out" in errors[0]


def test_subprocess_run_passes_a_timeout(monkeypatch, tmp_path):
    mol = Chem.MolFromSmiles("CO")
    worker = PredictorWorker(mol, "13C", tmp_path)

    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "predictorc.jar").write_bytes(b"jar")

    captured = {}

    def _fake_run(*a, **kw):
        captured.update(kw)
        raise RuntimeError("stop here")

    monkeypatch.setattr(nmrmod.shutil, "which", lambda _name: "/usr/bin/java")
    monkeypatch.setattr(nmrmod.subprocess, "run", _fake_run)
    worker.error_signal = MagicMock()
    worker.run()

    assert captured.get("timeout") == nmrmod.JAVA_TIMEOUT_SEC
