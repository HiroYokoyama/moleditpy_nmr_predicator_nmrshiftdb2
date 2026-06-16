import sys
import os
import types
from unittest.mock import MagicMock, patch

# --- Install PyQt6, pyvista, matplotlib, and other stubs to bypass DLL issues ---
qt_core = types.ModuleType("PyQt6.QtCore")
class _QThread:
    def __init__(self, *args, **kwargs): pass
    def start(self): pass
    def isRunning(self): return False
    def wait(self, ms=0): return True
qt_core.QThread = _QThread
qt_core.pyqtSignal = lambda *a, **kw: MagicMock()

class _Qt:
    class WindowModality:
        NonModal = 0
        WindowModal = 1
    class AlignmentFlag:
        AlignCenter = 1
        AlignRight = 2
        AlignVCenter = 4
qt_core.Qt = _Qt

class _QTimer:
    def __init__(self, parent=None): pass
    def start(self, ms): pass
qt_core.QTimer = _QTimer

qt_widgets = types.ModuleType("PyQt6.QtWidgets")
class _QDialog:
    def __init__(self, *args, **kwargs):
        self.layout = None
    def setWindowTitle(self, *args): pass
    def resize(self, *args): pass
    def exec(self): return True
    def accept(self): pass
    def setLayout(self, layout):
        self.layout = layout
qt_widgets.QDialog = _QDialog

class _QMessageBox:
    @staticmethod
    def warning(parent, title, text): pass
    @staticmethod
    def critical(parent, title, text): pass
qt_widgets.QMessageBox = _QMessageBox

class _QPushButton:
    def __init__(self, *args, **kwargs):
        self.clicked = MagicMock()
qt_widgets.QPushButton = _QPushButton

class _QComboBox:
    def __init__(self, *args, **kwargs):
        self._items = []
    def addItems(self, items):
        self._items.extend(items)
    def currentText(self):
        return self._items[0] if self._items else "1H"
qt_widgets.QComboBox = _QComboBox

qt_widgets.QHeaderView = MagicMock()
qt_widgets.QProgressDialog = MagicMock()
qt_widgets.QFileDialog = MagicMock()
qt_widgets.QLabel = MagicMock
qt_widgets.QTableWidget = MagicMock
qt_widgets.QTableWidgetItem = MagicMock
qt_widgets.QCheckBox = MagicMock
qt_widgets.QDoubleSpinBox = MagicMock
qt_widgets.QVBoxLayout = MagicMock
qt_widgets.QHBoxLayout = MagicMock

pyqt6 = types.ModuleType("PyQt6")
pyqt6.QtCore = qt_core
pyqt6.QtWidgets = qt_widgets

sys.modules["PyQt6"] = pyqt6
sys.modules["PyQt6.QtCore"] = qt_core
sys.modules["PyQt6.QtWidgets"] = qt_widgets

# pyvista & matplotlib stubs
sys.modules["pyvista"] = MagicMock()

matplotlib_figure = types.ModuleType("matplotlib.figure")
class _Figure:
    def __init__(self, *args, **kwargs): pass
    def add_subplot(self, *args, **kwargs): return MagicMock()
    def clear(self): pass
    def subplots_adjust(self, *args, **kwargs): pass
matplotlib_figure.Figure = _Figure
sys.modules["matplotlib.figure"] = matplotlib_figure

sys.modules["matplotlib.backends.backend_qtagg"] = MagicMock()

# Add parent directory to sys.path to enable importing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rdkit import Chem
from nmr_predicator_nmrshiftdb2 import (
    PLUGIN_NAME,
    PLUGIN_VERSION,
    PLUGIN_AUTHOR,
    PLUGIN_DESCRIPTION,
    initialize,
    PredictorWorker,
    ask_nucleus
)

def test_metadata():
    assert PLUGIN_NAME == "NMR Predictor (nmrshiftdb2)"
    assert PLUGIN_VERSION == "2.1.0"
    assert PLUGIN_AUTHOR == "HiroYokoyama"
    assert isinstance(PLUGIN_DESCRIPTION, str)

def test_initialize():
    mock_context = MagicMock()
    initialize(mock_context)
    mock_context.add_menu_action.assert_called_once()
    args, kwargs = mock_context.add_menu_action.call_args
    assert args[0] == "Analysis/NMR Prediction (nmrshiftdb2)"

def test_parse_output():
    # Create a mock molecule with 2 atoms: 1 C and 1 H
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol) # Should have 1 C and 4 H
    
    # We will test 1H prediction
    worker_1h = PredictorWorker(mol, "1H", MagicMock())
    
    # Sample output from Java
    # Columns: Index : Min Mean Max
    # Note: 1-indexed for atoms in Java.
    # In CH4: atom 0 is C, atoms 1, 2, 3, 4 are H.
    sample_output = """
    Test Output:
    1: 5.0 10.0 15.0
    2: 0.5 1.2 2.0
    3: 0.6 1.3 2.1
    """
    
    predictions = worker_1h._parse_output(sample_output, mol)
    
    # Atom 1 is H, Atom 2 is H. Atom 0 (index 1 in 1-based java) is C, so it should be skipped for 1H.
    assert len(predictions) == 2
    assert predictions[0]["idx"] == 1
    assert predictions[0]["atom"] == "H"
    assert predictions[0]["ppm"] == 1.2
    assert predictions[0]["min"] == 0.5
    assert predictions[0]["max"] == 2.0
    
    assert predictions[1]["idx"] == 2
    assert predictions[1]["atom"] == "H"
    assert predictions[1]["ppm"] == 1.3
    assert predictions[1]["min"] == 0.6
    assert predictions[1]["max"] == 2.1

def test_parse_output_13c():
    mol = Chem.MolFromSmiles("CC") # 2 Cs
    worker_13c = PredictorWorker(mol, "13C", MagicMock())
    
    sample_output = """
    1: 10.0 20.0 30.0
    2: 12.0 22.0 32.0
    """
    
    predictions = worker_13c._parse_output(sample_output, mol)
    assert len(predictions) == 2
    assert predictions[0]["idx"] == 0
    assert predictions[0]["atom"] == "C"
    assert predictions[0]["ppm"] == 20.0
    
    assert predictions[1]["idx"] == 1
    assert predictions[1]["atom"] == "C"
    assert predictions[1]["ppm"] == 22.0

def test_build_classpath():
    mock_plugin_dir = MagicMock()
    # Mocking jar path checks
    mock_jar = MagicMock()
    mock_jar.exists.return_value = True
    mock_jar.absolute.return_value = "/mock/lib/predictorh.jar"
    mock_jar.name = "predictorh.jar"
    
    mock_other_jar = MagicMock()
    mock_other_jar.absolute.return_value = "/mock/lib/predictorc.jar"
    mock_other_jar.name = "predictorc.jar"
    
    mock_lib_dir = MagicMock()
    mock_lib_dir.glob.return_value = [mock_jar, mock_other_jar]
    
    # Define mapping for / / "lib" / "predictorh.jar"
    mock_plugin_dir.__truediv__.return_value = mock_plugin_dir
    mock_plugin_dir.glob.return_value = [mock_jar, mock_other_jar]
    
    worker = PredictorWorker(MagicMock(), "1H", mock_plugin_dir)
    
    with patch.object(worker, '_build_classpath') as mock_build:
        mock_build.return_value = "mock_classpath"
        assert worker._build_classpath() == "mock_classpath"

def test_ask_nucleus_accept():
    with patch.object(_QDialog, 'exec', return_value=True):
        # We can also mock currentText to return "13C" by patching QComboBox
        with patch.object(_QComboBox, 'currentText', return_value="13C"):
            nucleus, ok = ask_nucleus(MagicMock())
            assert ok is True
            assert nucleus == "13C"

def test_ask_nucleus_reject():
    with patch.object(_QDialog, 'exec', return_value=False):
        nucleus, ok = ask_nucleus(MagicMock())
        assert ok is False
        assert nucleus is None
