"""Generate our reference index directly from the official assigned SDF export.

Source and derived reference data retain the nmrshiftdb2 Database License.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

from rdkit import Chem, rdBase, RDLogger

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "nmr_predicator_nmrshiftdb2" if (ROOT / "nmr_predicator_nmrshiftdb2").is_dir() else ROOT
spec = importlib.util.spec_from_file_location("nmr_database_builder", PACKAGE / "database.py")
db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db)
SOURCE_URL = "https://downloads.sourceforge.net/project/nmrshiftdb2/data/nmrshiftdb2withsignals.sd"
SOURCE_SHA256 = "0e86688360e23c88ccf0eb82a1251315fa57ec6f5b376dc8886f3311793a0afe"
CALCULATION_FIELDS = ("Program", "NMRProgram", "NMRMethod", "NMRBasisSet", "GeomMethod", "GeomBasisSet")


def indexed_values(value):
    matches = list(re.finditer(r"(?:^|\s)(\d+):", value))
    return {m.group(1): value[m.end():matches[i + 1].start() if i + 1 < len(matches) else None].strip()
            for i, m in enumerate(matches)}


def assigned_shifts(text, mol, nucleus):
    """SDF assignments are zero-based; H assignments may name the attached atom."""
    result = defaultdict(list)
    for peak in text.split("|"):
        if not peak.strip():
            continue
        fields = peak.split(";")
        if len(fields) != 3:
            raise ValueError("Malformed assigned peak")
        ppm = float(fields[0])
        if not math.isfinite(ppm) or not (-10 <= ppm <= 30 if nucleus == "1H" else -50 <= ppm <= 300):
            raise ValueError("Nonfinite or out-of-range shift")
        # Ambiguous multi-atom assignments are not treated as confirmed labels.
        if not fields[2].strip().isdigit():
            continue
        index = int(fields[2])
        if not 0 <= index < mol.GetNumAtoms():
            raise ValueError("Assignment outside molecule")
        atom = mol.GetAtomWithIdx(index)
        if nucleus == "1H" and atom.GetAtomicNum() == 1:
            if atom.GetDegree() != 1:
                continue
            atom = atom.GetNeighbors()[0]
        if atom.GetAtomicNum() != 6:
            continue
        if nucleus == "1H" and atom.GetTotalNumHs(includeNeighbors=True) == 0:
            continue
        result[atom.GetIdx()].append(ppm)
    return {key: statistics.median(values) for key, values in result.items()}


def read_observations(source, audit):
    observations = defaultdict(list)
    supplier = Chem.SDMolSupplier(str(source), sanitize=False, removeHs=False)
    for row, mol in enumerate(supplier):
        audit["records"] += 1
        if mol is None:
            audit["rejected_parse"] += 1
            continue
        try:
            Chem.SanitizeMol(mol)
            expanded = db.prepare_molecule(mol)
            heavy, mapping = db.heavy_graph(expanded)
        except (ValueError, RuntimeError):
            audit["rejected_chemistry"] += 1
            continue
        # Recalculate identity from the actual graph. All stereoisomers and
        # duplicate spectra stay in one connectivity group, including holdout.
        identity = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=False)
        group = hashlib.sha256(identity.encode()).hexdigest()
        properties = mol.GetPropsAsDict(autoConvertStrings=False)
        solvents = indexed_values(properties.get("Solvent", ""))
        temperatures = indexed_values(properties.get("Temperature [K]", ""))
        calculated = set()
        for field in CALCULATION_FIELDS:
            calculated.update(indexed_values(properties.get(field, "")))
        codes = {}
        forms = None
        for name, value in properties.items():
            match = re.fullmatch(r"Spectrum (1H|13C) (\d+)", name)
            if not match:
                continue
            nucleus, spectrum = match.groups()
            if spectrum in calculated:
                audit["rejected_calculated_spectra"] += 1
                continue
            solvent = next((key for key, label in db.SOLVENTS.items() if solvents.get(spectrum) == label), None)
            if solvent is None:
                audit["rejected_other_or_missing_solvent"] += 1
                continue
            temperature = temperatures.get(spectrum)
            if temperature:
                try:
                    in_range = 273 <= float(temperature) <= 323
                except ValueError:
                    in_range = False
                if not in_range:
                    audit["rejected_temperature"] += 1
                    continue
            try:
                shifts = assigned_shifts(value, mol, nucleus)
            except (ValueError, IndexError):
                audit["rejected_assignments"] += 1
                continue
            audit["accepted_spectra"] += 1
            audit["accepted_solvent:" + solvent] += 1
            for parent, ppm in shifts.items():
                center = mapping[parent]
                if center not in codes:
                    forms = db.stereo_forms(heavy) if forms is None else forms
                    codes[center] = db.environment_keys(heavy, center, forms)
                keys = tuple((r, codes[center][r]) for r in (0, *db.RADII))
                observations[(group, solvent, nucleus, keys)].append(ppm)
        if (row + 1) % 10000 == 0:
            print(f"Read {row + 1} records; {len(observations)} grouped sites", file=sys.stderr, flush=True)
    return [(group, solvent, nucleus, dict(keys), statistics.median(values), min(values), max(values))
            for (group, solvent, nucleus, keys), values in sorted(observations.items())]


def aggregate(observations):
    groups = defaultdict(lambda: defaultdict(list))
    for group, solvent, nucleus, keys, ppm, low, high in observations:
        for radius, code in keys.items():
            groups[db.index_key(nucleus, radius, code, solvent)][group].append((ppm, low, high))
    index = {}
    for key, molecules in sorted(groups.items()):
        threshold = 1 if key.split(":")[1] == "0" else db.MIN_REFERENCES
        if len(molecules) < threshold:
            continue
        values = [statistics.median(v[0] for v in observations) for observations in molecules.values()]
        lows = [v[1] for observations in molecules.values() for v in observations]
        highs = [v[2] for observations in molecules.values() for v in observations]
        index[key] = [round(statistics.median(values), 6), round(min(lows), 6),
                      round(max(highs), 6), len(values)]
    return index


def validate(observations):
    train = [o for o in observations if int(o[0][:8], 16) % 5 != 0]
    heldout = [o for o in observations if int(o[0][:8], 16) % 5 == 0]
    index = aggregate(train)
    report = {}
    for solvent in sorted({o[1] for o in observations}):
        report[solvent] = {}
        for nucleus in ["1H", "13C"]:
            errors, counts = [], Counter()
            targets = [o for o in heldout if o[1] == solvent and o[2] == nucleus]
            for _, _, _, keys, ppm, _, _ in targets:
                reference = db.find_reference(index, nucleus, keys, solvent)
                if reference:
                    errors.append(reference["ppm"] - ppm)
                    counts[reference["radius"]] += 1
            report[solvent][nucleus] = {
                "targets": len(targets), "predicted": len(errors),
                "coverage": len(errors) / len(targets) if targets else 0,
                "mae_ppm": statistics.mean(abs(e) for e in errors) if errors else None,
                "rmse_ppm": math.sqrt(statistics.mean(e * e for e in errors)) if errors else None,
                "radius_counts": dict(sorted(counts.items())),
            }
    return report


def build(source, output):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Source checksum differs from the pinned public export; review before rebuilding.")
    RDLogger.DisableLog("rdApp.warning")
    audit = Counter()
    observations = read_observations(source, audit)
    metadata = {"encoder": db.ENCODER, "solvents": sorted({o[1] for o in observations}),
                "source_url": SOURCE_URL, "source_sha256": digest,
                "rdkit_version": rdBase.rdkitVersion,
                "minimum_independent_molecules": db.MIN_REFERENCES,
                "exact_connectivity_minimum_molecules": 1,
                "conditions": "solvent-specific; reported 273-323 K or unreported temperature; E/Z and tetrahedral stereo; enantiomers pooled in achiral solvents; carbon-bound H group medians",
                "audit": dict(sorted(audit.items())),
                "connectivity_groups": len({o[0] for o in observations}),
                "split": "SHA256(nonstereo canonical connectivity SMILES) first 8 hex digits modulo 5 = 0 held out",
                "validation": validate(observations)}
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata, entries=aggregate(observations))
    with (output / "reference.json.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            zipped.write(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    metadata["index_sha256"] = hashlib.sha256((output / "reference.json.gz").read_bytes()).hexdigest()
    metadata["entries"] = len(payload["entries"])
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=PACKAGE / "data")
    args = parser.parse_args()
    build(args.source, args.output)
