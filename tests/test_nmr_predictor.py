"""
Core tests for nmr_predicator_nmrshiftdb2.
Qt / pyvista / matplotlib stubs are installed by conftest.py before this file
is imported, so we just pull the classes we need to patch directly from the
already-installed stub modules.
"""
from unittest.mock import MagicMock, patch

# conftest.py already installed stubs and added the package root to sys.path.
# Import shared stub classes so our patch.object() calls target the same class
# that the production module bound at import time.
from PySide6.QtWidgets import QDialog as _QDialog, QComboBox as _QComboBox

from nmr_predicator_nmrshiftdb2 import (
    PLUGIN_NAME,
    PLUGIN_AUTHOR,
    PLUGIN_DESCRIPTION,
    initialize,
    ask_nucleus,
)


def test_metadata():
    assert PLUGIN_NAME == "NMR Predictor (nmrshiftdb2)"
    assert PLUGIN_AUTHOR == "HiroYokoyama"
    assert isinstance(PLUGIN_DESCRIPTION, str)


def test_initialize():
    mock_context = MagicMock()
    initialize(mock_context)
    mock_context.add_menu_action.assert_called_once()
    args, kwargs = mock_context.add_menu_action.call_args
    assert args[0] == "Analysis/NMR Prediction (nmrshiftdb2)"








def test_ask_nucleus_accept():
    with patch.object(_QDialog, 'exec', return_value=True):
        with patch.object(_QComboBox, 'currentText', return_value="13C"):
            nucleus, ok = ask_nucleus(MagicMock())
            assert ok is True
            assert nucleus == "13C"


def test_ask_nucleus_reject():
    with patch.object(_QDialog, 'exec', return_value=False):
        nucleus, ok = ask_nucleus(MagicMock())
        assert ok is False
        assert nucleus is None
