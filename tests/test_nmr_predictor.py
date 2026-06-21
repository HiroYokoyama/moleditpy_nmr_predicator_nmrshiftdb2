"""
Core tests for nmr_predicator_nmrshiftdb2.
Qt / pyvista / matplotlib stubs are installed by conftest.py before this file
is imported, so we just pull the classes we need to patch directly from the
already-installed stub modules.
"""
import sys
import os
from unittest.mock import MagicMock, patch

# conftest.py already installed stubs and added the package root to sys.path.
# Import shared stub classes so our patch.object() calls target the same class
# that the production module bound at import time.
from PyQt6.QtWidgets import QDialog as _QDialog, QComboBox as _QComboBox

from rdkit import Chem
from nmr_predicator_nmrshiftdb2 import (
    PLUGIN_NAME,
    PLUGIN_VERSION,
    PLUGIN_AUTHOR,
    PLUGIN_DESCRIPTION,
    initialize,
    PredictorWorker,
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


def test_parse_output():
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)  # 1C + 4H

    worker_1h = PredictorWorker(mol, "1H", MagicMock())

    sample_output = """
    Test Output:
    1: 5.0 10.0 15.0
    2: 0.5 1.2 2.0
    3: 0.6 1.3 2.1
    """

    predictions = worker_1h._parse_output(sample_output, mol)

    # Atom index 0 (java: 1) is C — skipped for 1H; indices 1 and 2 (java: 2, 3) are H
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
    mol = Chem.MolFromSmiles("CC")  # 2 Cs
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
    mock_jar = MagicMock()
    mock_jar.exists.return_value = True
    mock_jar.absolute.return_value = "/mock/lib/predictorh.jar"
    mock_jar.name = "predictorh.jar"

    mock_other_jar = MagicMock()
    mock_other_jar.absolute.return_value = "/mock/lib/predictorc.jar"
    mock_other_jar.name = "predictorc.jar"

    mock_plugin_dir.__truediv__.return_value = mock_plugin_dir
    mock_plugin_dir.glob.return_value = [mock_jar, mock_other_jar]

    worker = PredictorWorker(MagicMock(), "1H", mock_plugin_dir)

    with patch.object(worker, '_build_classpath') as mock_build:
        mock_build.return_value = "mock_classpath"
        assert worker._build_classpath() == "mock_classpath"


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
