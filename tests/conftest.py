"""
Shared Qt / pyvista / matplotlib stubs for the NMR predictor test suite.

Stubs are installed in a pytest_configure hook (tryfirst=True) so they land in
sys.modules before any other plugin — including pytest-qt — runs its own
configure hook.  The stub modules auto-return MagicMock for unknown attributes
so pytest-qt's version probes never raise AttributeError.

Both test files then pull _QDialog / _QComboBox from PyQt6.QtWidgets directly,
so patch.object() calls always target the same class that the production module
bound at import time.
"""
import os
import sys
import types
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Shared classes that tests need to patch — real classes, not MagicMock()
# ---------------------------------------------------------------------------

class _QThread:
    def __init__(self, *args, **kwargs): pass
    def start(self): pass
    def isRunning(self): return False
    def wait(self, ms=0): return True


class _Qt:
    class WindowModality:
        NonModal = 0
        WindowModal = 1

    class AlignmentFlag:
        AlignCenter = 1
        AlignRight = 2
        AlignVCenter = 4

    class TextFormat:
        RichText = 1


class _QTimer:
    def __init__(self, parent=None):
        self.timeout = MagicMock()

    def start(self, ms): pass
    def stop(self): pass


class _QDialog:
    def __init__(self, *args, **kwargs):
        self.layout = None

    def setWindowTitle(self, *args): pass
    def resize(self, *args): pass
    def exec(self): return True
    def accept(self): pass
    def reject(self): pass
    def setWindowModality(self, *args): pass
    def close(self): pass
    def closeEvent(self, event): pass

    def setLayout(self, layout):
        self.layout = layout


class _QMessageBox:
    @staticmethod
    def warning(parent, title, text): pass

    @staticmethod
    def critical(parent, title, text): pass


class _QPushButton:
    def __init__(self, *args, **kwargs):
        self.clicked = MagicMock()

    def setFixedWidth(self, w): pass


class _QTableWidgetItem:
    def __init__(self, text="", *args, **kwargs):
        self._text = text

    def setTextAlignment(self, *args, **kwargs): pass

    def text(self):
        return self._text


class _QTableWidget:
    class SelectionBehavior:
        SelectRows = 1

    class EditTrigger:
        NoEditTriggers = 0

    class SelectionMode:
        SingleSelection = 0
        MultiSelection = 1
        ExtendedSelection = 3

    def __init__(self, *args, **kwargs):
        self._row_count = 0
        self._items = {}
        self._selected_rows = set()
        self.cellClicked = MagicMock()

    def setColumnCount(self, n): pass
    def setHorizontalHeaderLabels(self, labels): pass
    def horizontalHeader(self): return MagicMock()
    def setSelectionBehavior(self, behavior): pass
    def setEditTriggers(self, trigger): pass
    def setStyleSheet(self, css): pass
    def setRowCount(self, n): self._row_count = n
    def rowCount(self): return self._row_count

    def setItem(self, row, col, item):
        self._items[(row, col)] = item

    def item(self, row, col):
        return self._items.get((row, col))

    def clearSelection(self):
        self._selected_rows.clear()

    def selectRow(self, row):
        self._selected_rows.add(row)

    def setSelectionMode(self, mode): pass


class _QLabel:
    """Real stub (not MagicMock) — MagicMock(text) treats a str first-arg as
    `spec`, which then rejects normal Qt methods like setStyleSheet."""
    def __init__(self, text="", *args, **kwargs):
        self._text = text
        self.setStyleSheet = MagicMock()
        self.setText = MagicMock(side_effect=self._set_text)

    def _set_text(self, text):
        self._text = text

    def text(self):
        return self._text


class _QCheckBox:
    """Real stub — avoids MagicMock("Auto Fit") treating the label as `spec`."""
    def __init__(self, text="", *args, **kwargs):
        self._text = text
        self.toggled = MagicMock()
        self.isChecked = MagicMock(return_value=False)


class _QDoubleSpinBox:
    def __init__(self, *args, **kwargs):
        self._value = 0.0
        self.valueChanged = MagicMock()

    def setRange(self, *args, **kwargs): pass
    def setDecimals(self, *args, **kwargs): pass
    def setSingleStep(self, *args, **kwargs): pass
    def setValue(self, v): self._value = v
    def value(self): return self._value


class _QComboBox:
    def __init__(self, *args, **kwargs):
        self._items = []

    def addItems(self, items):
        self._items.extend(items)

    def currentText(self):
        return self._items[0] if self._items else "1H"


# ---------------------------------------------------------------------------
# Auto-expanding module stubs (unknown attributes → MagicMock)
# ---------------------------------------------------------------------------

def _auto_module(name: str, **known_attrs) -> types.ModuleType:
    """Return a ModuleType subclass whose __getattr__ returns MagicMock for
    any attribute not explicitly set — lets pytest-qt probe freely."""

    class _AutoMod(types.ModuleType):
        def __getattr__(self, attr: str):
            val = MagicMock()
            object.__setattr__(self, attr, val)
            return val

    mod = _AutoMod(name)
    for k, v in known_attrs.items():
        object.__setattr__(mod, k, v)
    return mod


def _install_stubs() -> None:
    """Install all stub modules into sys.modules (idempotent)."""
    if "PyQt6" in sys.modules:
        return

    qt_core = _auto_module(
        "PyQt6.QtCore",
        QThread=_QThread,
        pyqtSignal=lambda *a, **kw: MagicMock(),
        Qt=_Qt,
        QTimer=_QTimer,
        # Version constants that pytest-qt probes during its configure hook
        PYQT_VERSION=0x060600,       # must not equal 0x060000
        PYQT_VERSION_STR="6.6.0",
        QT_VERSION_STR="6.6.0",
    )

    qt_widgets = _auto_module(
        "PyQt6.QtWidgets",
        QDialog=_QDialog,
        QMessageBox=_QMessageBox,
        QPushButton=_QPushButton,
        QComboBox=_QComboBox,
        QLabel=_QLabel,
        QTableWidget=_QTableWidget,
        QTableWidgetItem=_QTableWidgetItem,
        QCheckBox=_QCheckBox,
        QDoubleSpinBox=_QDoubleSpinBox,
        QVBoxLayout=MagicMock,
        QHBoxLayout=MagicMock,
    )

    qt_gui = _auto_module("PyQt6.QtGui")

    class _PyQt6Package(types.ModuleType):
        """PyQt6 stub that auto-creates sub-module stubs for any unknown attr."""

        def __init__(self):
            super().__init__("PyQt6")
            object.__setattr__(self, "QtCore", qt_core)
            object.__setattr__(self, "QtWidgets", qt_widgets)
            object.__setattr__(self, "QtGui", qt_gui)

        def __getattr__(self, name: str):
            mod = _auto_module(f"PyQt6.{name}")
            object.__setattr__(self, name, mod)
            sys.modules[f"PyQt6.{name}"] = mod
            return mod

    pyqt6 = _PyQt6Package()

    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core
    sys.modules["PyQt6.QtWidgets"] = qt_widgets
    sys.modules["PyQt6.QtGui"] = qt_gui
    sys.modules["pyvista"] = MagicMock()

    matplotlib_figure = _auto_module("matplotlib.figure")

    class _Figure:
        def __init__(self, *args, **kwargs):
            self._axes = []

        def add_subplot(self, *args, **kwargs):
            ax = MagicMock()
            # ax.stem(...) is unpacked as (markerline, stemlines, baseline)
            # in plot_spectrum() — pre-seed a 3-tuple of mocks.
            ax.stem.return_value = (MagicMock(), MagicMock(), MagicMock())
            self._axes.append(ax)
            return ax

        def clear(self):
            self._axes = []

        def subplots_adjust(self, *args, **kwargs): pass

        @property
        def axes(self):
            return self._axes

    object.__setattr__(matplotlib_figure, "Figure", _Figure)
    sys.modules["matplotlib.figure"] = matplotlib_figure
    sys.modules["matplotlib.backends.backend_qtagg"] = MagicMock()
    sys.modules["matplotlib"] = MagicMock()

    pkg_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Install stubs before any other plugin (including pytest-qt) configures."""
    _install_stubs()
