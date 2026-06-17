"""
Integration tests for nmr_predicator_nmrshiftdb2/__init__.py
Verifies the plugin contract without Qt/RDKit/PyVista/matplotlib.

Two execution modes
-------------------
1. Stub mode    Ealways runs (CI + local).
2. Real-context mode  Eruns when python_molecular_editor is present.

CI setup
--------
    - name: Clone main app (for real-context integration tests)
      run: git clone --depth 1 https://github.com/HiroYokoyama/python_molecular_editor.git
             ../python_molecular_editor || true
"""
import sys
import os
import types
import unittest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub heavy dependencies before importing the plugin
# ---------------------------------------------------------------------------

def _install_stubs():
    if "PyQt6" not in sys.modules or not hasattr(sys.modules["PyQt6"], "__file__"):
        pyqt6 = types.ModuleType("PyQt6")
        qt_core = types.ModuleType("PyQt6.QtCore")

        class _QThread:
            def __init__(self): pass

        qt_core.QThread = _QThread
        qt_core.pyqtSignal = lambda *a, **kw: MagicMock()
        qt_core.Qt = MagicMock()
        qt_core.QTimer = MagicMock()

        qt_widgets = types.ModuleType("PyQt6.QtWidgets")
        for cls_name in [
            "QDialog", "QVBoxLayout", "QHBoxLayout", "QPushButton",
            "QTableWidget", "QTableWidgetItem", "QLabel", "QComboBox",
            "QCheckBox", "QDoubleSpinBox", "QMessageBox", "QHeaderView",
            "QProgressDialog", "QFileDialog",
        ]:
            setattr(qt_widgets, cls_name, MagicMock())

        qt_gui = types.ModuleType("PyQt6.QtGui")

        sys.modules.setdefault("PyQt6", pyqt6)
        sys.modules.setdefault("PyQt6.QtCore", qt_core)
        sys.modules.setdefault("PyQt6.QtWidgets", qt_widgets)
        sys.modules.setdefault("PyQt6.QtGui", qt_gui)

    # RDKit
    rdkit_chem = types.ModuleType("rdkit.Chem")
    rdkit_chem.GetPeriodicTable = MagicMock(return_value=MagicMock())
    rdkit_chem.SanitizeMol = MagicMock()
    rdkit_chem.RemoveStereochemistry = MagicMock()
    rdkit_chem.AssignStereochemistry = MagicMock()
    rdkit_chem.Mol = MagicMock()
    rdkit_allchem = types.ModuleType("rdkit.Chem.AllChem")
    rdkit_allchem.Compute2DCoords = MagicMock()
    sys.modules.setdefault("rdkit", types.ModuleType("rdkit"))
    sys.modules.setdefault("rdkit.Chem", rdkit_chem)
    sys.modules.setdefault("rdkit.Chem.AllChem", rdkit_allchem)

    # PyVista
    sys.modules.setdefault("pyvista", types.ModuleType("pyvista"))

    # matplotlib
    mpl_backend_qtagg = types.ModuleType("matplotlib.backends.backend_qtagg")
    mpl_backend_qtagg.FigureCanvasQTAgg = MagicMock()
    mpl_backend_qtagg.NavigationToolbar2QT = MagicMock()
    mpl_figure = types.ModuleType("matplotlib.figure")
    mpl_figure.Figure = MagicMock()
    sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
    sys.modules.setdefault("matplotlib.backends", types.ModuleType("matplotlib.backends"))
    sys.modules.setdefault("matplotlib.backends.backend_qtagg", mpl_backend_qtagg)
    sys.modules.setdefault("matplotlib.figure", mpl_figure)


_install_stubs()

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from nmr_predicator_nmrshiftdb2 import initialize, PLUGIN_NAME, PLUGIN_VERSION


# ---------------------------------------------------------------------------
# Stub PluginContext
# ---------------------------------------------------------------------------

class _StubContext:
    def __init__(self):
        self._menu_actions = []
        self._status_messages = []

    def add_menu_action(self, path, callback, **kwargs):
        self._menu_actions.append((path, callback))

    def show_status_message(self, msg, duration=0):
        self._status_messages.append((msg, duration))

    # Full standard API stubs
    def get_main_window(self): return MagicMock()
    def register_save_handler(self, fn): pass
    def register_load_handler(self, fn): pass
    def register_document_reset_handler(self, fn): pass
    def register_file_opener(self, ext, fn, priority=0): pass
    def register_drop_handler(self, fn, priority=0): pass
    def add_export_action(self, label, fn): pass
    def add_analysis_tool(self, label, fn): pass
    def add_toolbar_action(self, fn, text, icon=None, tooltip=None): pass
    def register_window(self, key, win): pass
    def get_window(self, key): return None


# ---------------------------------------------------------------------------
# Tests: metadata
# ---------------------------------------------------------------------------

class TestMetadata(unittest.TestCase):
    def test_plugin_name_contains_nmr(self):
        self.assertIn("NMR", PLUGIN_NAME)

    def test_plugin_version_is_semver(self):
        parts = PLUGIN_VERSION.split(".")
        self.assertEqual(len(parts), 3)
        for p in parts:
            self.assertTrue(p.isdigit(), f"Non-numeric version part: {p!r}")


# ---------------------------------------------------------------------------
# Tests: initialize contract
# ---------------------------------------------------------------------------

class TestInitialize(unittest.TestCase):
    def setUp(self):
        self.ctx = _StubContext()
        initialize(self.ctx)

    def test_registers_menu_action(self):
        self.assertGreater(len(self.ctx._menu_actions), 0)

    def test_menu_action_path_contains_nmr(self):
        paths = [p for p, _ in self.ctx._menu_actions]
        self.assertTrue(
            any("NMR" in p or "nmr" in p.lower() for p in paths),
            f"Expected NMR in menu paths, got: {paths}",
        )

    def test_menu_action_is_callable(self):
        _, callback = self.ctx._menu_actions[0]
        self.assertTrue(callable(callback))

    def test_menu_path_is_namespaced(self):
        """Menu path should use a category prefix, e.g. 'Analysis/...'."""
        path, _ = self.ctx._menu_actions[0]
        self.assertIn("/", path)


# ---------------------------------------------------------------------------
# Real PluginContext tier
# ---------------------------------------------------------------------------

_MAIN_APP_CANDIDATES = [
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..",
                     "python_molecular_editor", "moleditpy", "src")
    ),
    os.environ.get("CI_MAIN_APP_SRC", ""),
]
_MAIN_APP_SRC = next(
    (p for p in _MAIN_APP_CANDIDATES if p and os.path.isdir(p)),
    None,
)
HAS_MAIN_APP = _MAIN_APP_SRC is not None

try:
    import pytest
    _skipif = pytest.mark.skipif(
        not HAS_MAIN_APP,
        reason="main app not found; clone python_molecular_editor or set CI_MAIN_APP_SRC",
    )
except ImportError:
    def _skipif(cls):
        return unittest.skip("pytest not available")(cls)



def _clear_qt_stubs():
    """Remove fake PyQt6 stub modules so real PyQt6 can be imported by moleditpy."""
    to_remove = [
        k for k in list(sys.modules)
        if k.startswith("PyQt6") and not hasattr(sys.modules[k], "__file__")
    ]
    for k in to_remove:
        del sys.modules[k]
    # Clear any moleditpy import that may have been attempted with stubs
    for k in [k for k in list(sys.modules) if k.startswith("moleditpy")]:
        del sys.modules[k]

@_skipif
class TestWithRealPluginContext(unittest.TestCase):
    """Verify initialize() works with the actual MoleditPy PluginContext."""

    @classmethod
    def setUpClass(cls):
        if not HAS_MAIN_APP:
            return
        # Load plugin_interface.py directly to avoid triggering moleditpy/__init__.py
        # which imports PyQt6 and conflicts with PySide6 loaded by pytest-qt on Windows.
        import importlib.util as _ilu
        _pi_path = os.path.join(_MAIN_APP_SRC, 'moleditpy', 'plugins', 'plugin_interface.py')
        _spec = _ilu.spec_from_file_location('moleditpy.plugins.plugin_interface', _pi_path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        cls.PluginContext = _mod.PluginContext
        mock_manager = MagicMock()
        mock_manager.get_main_window.return_value = MagicMock()
        cls.real_ctx = cls.PluginContext(mock_manager, PLUGIN_NAME)

    def test_real_initialize_does_not_raise(self):
        try:
            initialize(self.real_ctx)
        except Exception as e:
            self.fail(f"initialize(real_context) raised: {e}")

    def test_real_context_is_plugincontext_instance(self):
        self.assertIsInstance(self.real_ctx, self.PluginContext)

    def test_stub_interface_matches_real(self):
        for method in ["add_menu_action", "get_main_window"]:
            self.assertTrue(
                hasattr(self.PluginContext, method),
                f"Real PluginContext missing: {method}",
            )


if __name__ == "__main__":
    unittest.main()
