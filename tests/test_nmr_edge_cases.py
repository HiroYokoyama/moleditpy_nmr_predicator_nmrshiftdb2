"""
Edge-case tests for nmr_predicator_nmrshiftdb2:
  - _parse_output corner cases
  - _build_classpath real logic
  - PredictorWorker.run() error paths (no Java, JAR missing)

Qt / pyvista / matplotlib stubs are installed by conftest.py before this file
is imported.
"""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rdkit import Chem
from nmr_predicator_nmrshiftdb2 import PredictorWorker


# ---------------------------------------------------------------------------
# _parse_output edge cases
# ---------------------------------------------------------------------------

def test_parse_output_empty_string():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    worker = PredictorWorker(mol, "1H", MagicMock())
    assert worker._parse_output("", mol) == []


def test_parse_output_no_matching_lines():
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    worker = PredictorWorker(mol, "1H", MagicMock())
    output = "No peaks found.\nError in processing."
    assert worker._parse_output(output, mol) == []


def test_parse_output_skips_out_of_range_index():
    mol = Chem.MolFromSmiles("CC")  # 2 atoms (no Hs)
    worker = PredictorWorker(mol, "13C", MagicMock())
    output = "100: 10.0 20.0 30.0"
    result = worker._parse_output(output, mol)
    assert result == []


def test_parse_output_skips_wrong_nucleus_1h_for_carbon():
    mol = Chem.MolFromSmiles("CC")
    worker = PredictorWorker(mol, "1H", MagicMock())
    output = "1: 10.0 20.0 30.0\n2: 12.0 22.0 32.0"
    result = worker._parse_output(output, mol)
    assert result == []


def test_parse_output_skips_wrong_nucleus_13c_for_hydrogen():
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))  # C + 4H
    worker = PredictorWorker(mol, "13C", MagicMock())
    output = "2: 1.0 2.0 3.0"  # atom index 1 (0-based) is H
    result = worker._parse_output(output, mol)
    assert result == []


def test_parse_output_handles_malformed_lines_gracefully():
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    worker = PredictorWorker(mol, "1H", MagicMock())
    malformed = "not a number: foo bar baz\ngarbage line\n   : 1 2 3"
    result = worker._parse_output(malformed, mol)
    assert isinstance(result, list)  # should not raise


def test_parse_output_min_mean_max_stored():
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))  # atom 0=C, 1-4=H
    worker = PredictorWorker(mol, "1H", MagicMock())
    output = "2: 0.5 1.2 2.0"  # atom index 1 (0-based) → H
    result = worker._parse_output(output, mol)
    assert len(result) == 1
    assert result[0]["min"] == pytest.approx(0.5)
    assert result[0]["ppm"] == pytest.approx(1.2)
    assert result[0]["max"] == pytest.approx(2.0)


def test_parse_output_multiple_h_atoms():
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))  # 1C + 4H (indices 0=C, 1-4=H)
    worker = PredictorWorker(mol, "1H", MagicMock())
    output = "2: 0.8 1.1 1.5\n3: 0.9 1.2 1.6"
    result = worker._parse_output(output, mol)
    assert len(result) == 2
    assert all(r["atom"] == "H" for r in result)


# ---------------------------------------------------------------------------
# _build_classpath
# ---------------------------------------------------------------------------

def test_build_classpath_primary_jar_is_first(tmp_path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "predictorh.jar").touch()
    (lib_dir / "predictorc.jar").touch()
    (lib_dir / "helper.jar").touch()

    mol = Chem.MolFromSmiles("C")
    worker = PredictorWorker(mol, "1H", tmp_path)
    classpath = worker._build_classpath()

    jars = classpath.split(os.pathsep)
    assert "predictorh.jar" in jars[0]


def test_build_classpath_excludes_other_nucleus_jar(tmp_path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "predictorh.jar").touch()
    (lib_dir / "predictorc.jar").touch()

    mol = Chem.MolFromSmiles("C")
    worker = PredictorWorker(mol, "1H", tmp_path)
    classpath = worker._build_classpath()

    assert "predictorc.jar" not in classpath


def test_build_classpath_13c_excludes_1h_jar(tmp_path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "predictorh.jar").touch()
    (lib_dir / "predictorc.jar").touch()

    mol = Chem.MolFromSmiles("C")
    worker = PredictorWorker(mol, "13C", tmp_path)
    classpath = worker._build_classpath()

    assert "predictorh.jar" not in classpath
    assert "predictorc.jar" in classpath


def test_build_classpath_raises_when_primary_jar_missing(tmp_path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    # predictorh.jar intentionally absent

    mol = Chem.MolFromSmiles("C")
    worker = PredictorWorker(mol, "1H", tmp_path)
    with pytest.raises(FileNotFoundError):
        worker._build_classpath()


def test_build_classpath_no_duplicates(tmp_path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "predictorh.jar").touch()

    mol = Chem.MolFromSmiles("C")
    worker = PredictorWorker(mol, "1H", tmp_path)
    classpath = worker._build_classpath()

    jars = classpath.split(os.pathsep)
    assert len(jars) == len(set(jars))


# ---------------------------------------------------------------------------
# PredictorWorker.run() error paths
# ---------------------------------------------------------------------------

def test_run_emits_error_when_java_not_found(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)

    mol = Chem.MolFromSmiles("CC")
    worker = PredictorWorker(mol, "1H", MagicMock())
    worker.error_signal = MagicMock()

    worker.run()

    worker.error_signal.emit.assert_called_once()
    msg = worker.error_signal.emit.call_args[0][0]
    assert "java" in msg.lower() or "Java" in msg


def test_run_emits_error_when_jar_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/java")

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    # No predictorh.jar created

    mol = Chem.MolFromSmiles("CC")
    worker = PredictorWorker(mol, "1H", tmp_path)
    worker.error_signal = MagicMock()
    worker.finished_signal = MagicMock()

    worker.run()

    worker.error_signal.emit.assert_called_once()
    msg = worker.error_signal.emit.call_args[0][0]
    assert "jar" in msg.lower() or "JAR" in msg or "not found" in msg.lower()
