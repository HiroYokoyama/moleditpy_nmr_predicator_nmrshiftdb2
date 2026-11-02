"""
Edge-case tests for nmr_predicator_nmrshiftdb2:
  - _parse_output corner cases
  - _build_classpath real logic
  - PredictorWorker.run() error paths (no Java, JAR missing)

Qt / pyvista / matplotlib stubs are installed by conftest.py before this file
is imported.
"""
from unittest.mock import MagicMock


from rdkit import Chem
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------------
# _parse_output edge cases
# ---------------------------------------------------------------------------

















# ---------------------------------------------------------------------------
# _build_classpath
# ---------------------------------------------------------------------------











# ---------------------------------------------------------------------------
# PredictorWorker.run() error paths
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# 3D highlight must not disturb the user's camera
# ---------------------------------------------------------------------------


def _dialog_with_plotter(monkeypatch):
    """A ResultDialog whose 3D calls land on a recording plotter."""

    import nmr_predicator_nmrshiftdb2 as nmrmod

    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.Compute2DCoords(mol)

    mw = MagicMock()
    mw.current_mol = mol
    context = MagicMock()
    context.get_main_window.return_value = mw

    data = [
        {"idx": 0, "atom": "C", "ppm": 50.0, "min": 49.0, "max": 51.0},
        {"idx": 1, "atom": "O", "ppm": 60.0, "min": 59.0, "max": 61.0},
    ]
    dialog = object.__new__(nmrmod.ResultDialog)
    dialog.context = context
    dialog._source_mol = mw.current_mol
    dialog.data = data
    dialog._highlight_actors = {}
    dialog._label_actors = {}
    dialog._persistent_ppm = None
    dialog.status_label = MagicMock()
    dialog.figure = MagicMock()
    dialog.canvas = MagicMock()
    dialog._update_graph_highlight = MagicMock()
    return dialog, mw.plotter


def test_highlight_never_resets_the_camera(monkeypatch):
    """PyVista refits the camera unless told not to, which threw away the
    zoom every time the user hovered a peak."""
    dialog, plotter = _dialog_with_plotter(monkeypatch)

    dialog.highlight_atom(0, persistent=False)

    assert plotter.add_mesh.call_args.kwargs["reset_camera"] is False
    assert plotter.add_point_labels.call_args.kwargs["reset_camera"] is False


def test_highlight_still_renders(monkeypatch):
    dialog, plotter = _dialog_with_plotter(monkeypatch)
    dialog.highlight_atom(0, persistent=True)
    plotter.render.assert_called()


def test_highlight_refuses_indices_past_the_current_structure(monkeypatch):
    """The prediction is a snapshot; after an edit the indices can dangle."""
    dialog, plotter = _dialog_with_plotter(monkeypatch)
    dialog.data = [{"idx": 999, "atom": "C", "ppm": 50.0, "min": 49.0, "max": 51.0}]

    dialog.highlight_atom(0, persistent=True)

    plotter.add_mesh.assert_not_called()
    message = dialog.status_label.setText.call_args[0][0]
    assert "Structure changed" in message


# ---------------------------------------------------------------------------
# 3D -> table sync reads the manager, not the window
# ---------------------------------------------------------------------------


def test_sync_reads_selection_from_the_edit_manager():
    """Regression: the guard checked mw.selected_atoms_3d, which never exists
    (it lives on Edit3DManager), so this sync never ran at all."""
    from types import SimpleNamespace

    import nmr_predicator_nmrshiftdb2 as nmrmod

    mw = SimpleNamespace(edit_3d_manager=SimpleNamespace(selected_atoms_3d={1}))
    context = MagicMock()
    context.get_main_window.return_value = mw

    dialog = object.__new__(nmrmod.ResultDialog)
    dialog.context = context
    dialog.data = [{"idx": 1, "atom": "C", "ppm": 50.0, "min": 49.0, "max": 51.0}]
    dialog._last_selected = set()
    dialog.table = MagicMock()
    dialog.highlight_atom = MagicMock()

    dialog._sync_from_3d()

    dialog.table.selectRow.assert_called_once_with(0)
    dialog.highlight_atom.assert_called_once_with(0)


def test_sync_without_an_edit_manager_is_a_noop():
    from types import SimpleNamespace

    import nmr_predicator_nmrshiftdb2 as nmrmod

    context = MagicMock()
    context.get_main_window.return_value = SimpleNamespace()

    dialog = object.__new__(nmrmod.ResultDialog)
    dialog.context = context
    dialog._last_selected = set()
    dialog.table = MagicMock()

    dialog._sync_from_3d()  # must not raise

    dialog.table.selectRow.assert_not_called()
