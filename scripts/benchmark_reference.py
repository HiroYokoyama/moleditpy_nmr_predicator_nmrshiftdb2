"""Benchmark the README's ethyl (E)-crotonate without importing the GUI.

Timings exclude application/module startup and plotting. Cache-cold timings clear
the Python index cache, not the operating system's disk cache. The README v3
values are predictor output for comparison, not independent experimental truth.
"""

import argparse
import importlib.util
import json
import platform
from pathlib import Path
import statistics
import time

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "nmr_predicator_nmrshiftdb2" if (ROOT / "nmr_predicator_nmrshiftdb2").is_dir() else ROOT


def load_module(name):
    spec = importlib.util.spec_from_file_location("benchmark_" + name, PACKAGE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


db = load_module("database")
coupling = load_module("coupling")
SMILES = "CCOC(=O)/C=C/C"
OLD_README = {0: 1.24, 1: 4.20, 5: 5.88, 6: 7.00, 7: 1.89}
SITE_NAMES = {0: "ethyl CH3", 1: "OCH2", 3: "C=O", 5: "alpha vinyl CH",
              6: "beta vinyl CH", 7: "alkene CH3"}


def measure(function, repeats):
    elapsed = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        elapsed.append((time.perf_counter() - start) * 1000)
    ordered = sorted(elapsed)
    return {"median_ms": statistics.median(elapsed),
            "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "repeats": repeats}, result


def benchmark(repeats=100, coupling_repeats=20):
    mol2d = Chem.MolFromSmiles(SMILES)
    AllChem.Compute2DCoords(mol2d)
    mol3d = Chem.AddHs(Chem.MolFromSmiles(SMILES))
    if AllChem.EmbedMolecule(mol3d, randomSeed=961) != 0:
        raise RuntimeError("Could not generate benchmark conformer")
    report = {"molecule": "ethyl (E)-crotonate", "smiles": SMILES, "solvent": "CDCl3",
              "python": platform.python_version(), "rdkit": rdBase.rdkitVersion,
              "notes": __doc__.strip(), "nuclei": {}}
    for nucleus in ["1H", "13C"]:
        def predict():
            return db.predict(mol3d, nucleus)

        def cold():
            db._read_index.cache_clear()
            return predict()

        def annotated(mol):
            result = db.predict(mol, nucleus)
            coupling.annotate_predictions(result["mol_with_h"], result["data"], nucleus)
            return result

        cold_times, _ = measure(cold, 5)
        warm_times, _ = measure(predict, repeats)
        full_times, result = measure(lambda: annotated(mol3d), coupling_repeats)
        flat_times, _ = measure(lambda: annotated(mol2d), max(2, coupling_repeats // 2))
        groups = {}
        for row in result["data"]:
            parent = row["parent_idx"]
            if parent not in groups:
                groups[parent] = dict(row, site=SITE_NAMES.get(parent, str(parent)), equivalent_nuclei=0)
            groups[parent]["equivalent_nuclei"] += 1
        if nucleus == "1H":
            for parent, row in groups.items():
                row["old_readme_prediction_ppm"] = OLD_README[parent]
                row["difference_vs_old_prediction_ppm"] = row["ppm"] - OLD_README[parent]
        report["nuclei"][nucleus] = {
            "predicted_nuclei": len(result["data"]), "unsupported_ids": result["unsupported"],
            "cold_index_cache_prediction": cold_times, "warm_prediction": warm_times,
            "prediction_and_couplings_existing_3d": full_times,
            "prediction_and_couplings_2d_including_embedding": flat_times,
            "sites": list(groups.values()),
        }
    report["attribution"] = db.ATTRIBUTION
    report["data_license_url"] = db.DATA_LICENSE_URL
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--coupling-repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1 or args.coupling_repeats < 1:
        parser.error("Repeat counts must be positive")
    result = benchmark(args.repeats, args.coupling_repeats)
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    print(content)
