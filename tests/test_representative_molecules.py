"""Release regression examples cover complete, partial and absent references."""

import json
from pathlib import Path

import pytest
from rdkit import Chem

from nmr_predicator_nmrshiftdb2 import database as db

CASES = [
    ("c1ccccc1", "CDCl3", 6, 6),
    ("Cc1ccccc1", "CDCl3", 8, 7),
    ("CC(=O)OCC", "CDCl3", 8, 4),
    ("C1CCCCC1", "CDCl3", 12, 6),
    ("CCOC(=O)/C=C/C", "CDCl3", 10, 6),
    ("CCOC(=O)/C=C\\C", "CDCl3", 5, 2),
    ("CCO", "CDCl3", 0, 0),
    ("CCO", "D2O", 5, 2),
    ("C[C@H](O)C(=O)O", "D2O", 0, 0),
    ("CC(=O)Oc1ccccc1C(=O)O", "CDCl3", 7, 9),
]


@pytest.mark.parametrize("smiles,solvent,h_count,c_count", CASES)
@pytest.mark.parametrize("nucleus", ["1H", "13C"])
def test_reference_examples_keep_assignments_and_unsupported_behavior(smiles, solvent, h_count, c_count, nucleus):
    mol = Chem.MolFromSmiles(smiles)
    before = mol.ToBinary()
    expected = h_count if nucleus == "1H" else c_count
    if expected == 0:
        with pytest.raises(ValueError, match="No supported sites"):
            db.predict(mol, nucleus, solvent=solvent)
    else:
        result = db.predict(mol, nucleus, solvent=solvent)
        assert len(result["data"]) == expected
        assert result["solvent"] == solvent
        indices = {row["idx"] for row in result["data"]}
        target = "H" if nucleus == "1H" else "C"
        total = sum(a.GetSymbol() == target for a in result["mol_with_h"].GetAtoms())
        assert len(indices) == expected
        assert expected + len(result["unsupported"]) == total
        for row in result["data"]:
            assert row["min"] <= row["ppm"] <= row["max"]
            assert result["mol_with_h"].GetAtomWithIdx(row["idx"]).GetSymbol() == target
            assert row["idx"] not in result["unsupported"]
    assert mol.ToBinary() == before


@pytest.mark.parametrize("case", json.loads(
    (Path(__file__).parent / "fixtures/representative_predictions.json").read_text(encoding="utf-8")
)["cases"], ids=lambda case: f"{case['name']}-{case['solvent']}-{case['nucleus']}")
def test_predictions_match_verified_baseline(case):
    mol = Chem.MolFromSmiles(case["smiles"])
    if "error" in case:
        with pytest.raises(ValueError, match=case["error"]):
            db.predict(mol, case["nucleus"], solvent=case["solvent"])
        return
    result = db.predict(mol, case["nucleus"], solvent=case["solvent"])
    assert result["unsupported"] == case["unsupported"]
    assert result["data"] == case["data"]
