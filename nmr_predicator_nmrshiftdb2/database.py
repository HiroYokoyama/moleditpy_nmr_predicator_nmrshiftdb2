"""Original Python/RDKit local-environment search; no Java predictor code.

The generated reference index is separately licensed NMRShiftDB2 data. This
module is first-party GPLv3 code. Keys are our own encodings, not CDK HOSE codes.
"""

from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path

from rdkit import Chem

ENCODER = "moleditpy-rdkit-fragment-v1"
RADII = (4, 3, 2)
MIN_REFERENCES = 3
SOLVENT = "CDCl3"
ATTRIBUTION = (
    "Contains information from nmrshiftdb2 (www.nmrshiftdb.org), made available "
    "under the nmrshiftdb2 Database License."
)
DATA_LICENSE_URL = "https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/nmrshiftdb2datalicense.txt"


class PredictionCancelled(Exception):
    """The caller cancelled an in-progress prediction."""


def check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise PredictionCancelled()


def prepare_molecule(mol):
    """Copy/sanitize without renumbering host atoms; append explicit hydrogens."""
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Draw or load a molecule first.")
    result = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(result)
    except Exception as error:
        raise ValueError("The structure could not be sanitized; correct its valences.") from error
    if result.GetNumHeavyAtoms() > 256:
        raise ValueError("Local reference prediction supports at most 256 heavy atoms.")
    if len(Chem.GetMolFrags(result)) != 1:
        raise ValueError("Predict one connected molecule at a time.")
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in result.GetAtoms()):
        raise ValueError("Charged molecules and radicals are outside this reference model.")
    if any(a.GetAtomicNum() not in {1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
           for a in result.GetAtoms()):
        raise ValueError("The structure contains elements outside this reference model.")
    if any(a.GetIsotope() not in ({0, 1} if a.GetAtomicNum() == 1 else {0, 13} if a.GetAtomicNum() == 6 else {0})
           for a in result.GetAtoms()):
        raise ValueError("Isotope substitutions other than 1H/13C are unsupported.")
    return Chem.AddHs(result, addCoords=True)


def heavy_graph(mol):
    """Return a hydrogen-suppressed graph and original-index to graph-index map."""
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetIntProp("_nmr_original_index", atom.GetIdx())
        atom.SetAtomMapNum(0)
        atom.SetIsotope(0)
    heavy = Chem.RemoveHs(copy)
    mapping = {a.GetIntProp("_nmr_original_index"): a.GetIdx() for a in heavy.GetAtoms()}
    return heavy, mapping


def environment_keys(heavy, center):
    """Canonical rooted induced fragments with full atom degree/H annotations.

    Atom order, host atom-map numbers and drawing coordinates are irrelevant.
    Stereochemistry is deliberately pooled; this is a 2D reference estimator.
    Full degrees/H counts on boundary atoms distinguish truncated environments.
    """
    symbols = []
    for atom in heavy.GetAtoms():
        symbols.append(
            f"[{atom.GetAtomicNum()};d{atom.GetDegree()};h{atom.GetTotalNumHs()};"
            f"a{int(atom.GetIsAromatic())};q{atom.GetFormalCharge()}]"
        )
    shells = {center}
    frontier = {center}
    keys = {}
    full = Chem.MolFragmentToSmiles(
        heavy, atomsToUse=list(range(heavy.GetNumAtoms())), atomSymbols=symbols,
        rootedAtAtom=center, canonical=True, isomericSmiles=False, allBondsExplicit=True,
    )
    keys[0] = hashlib.sha256(full.encode("ascii")).hexdigest()
    for radius in range(1, max(RADII) + 1):
        frontier = {n.GetIdx() for i in frontier for n in heavy.GetAtomWithIdx(i).GetNeighbors()} - shells
        shells.update(frontier)
        if radius not in RADII:
            continue
        fragment = Chem.MolFragmentToSmiles(
            heavy, atomsToUse=sorted(shells), atomSymbols=symbols,
            rootedAtAtom=center, canonical=True, isomericSmiles=False,
            allBondsExplicit=True,
        )
        keys[radius] = hashlib.sha256(fragment.encode("ascii")).hexdigest()
    return keys


def index_key(nucleus, radius, code):
    return f"{nucleus}:{radius}:{code}"


def find_reference(index, nucleus, keys):
    for radius in (0, *RADII):
        value = index.get(index_key(nucleus, radius, keys[radius]))
        if value is not None and value[3] >= (1 if radius == 0 else MIN_REFERENCES):
            mean, low, high, count = value
            if not all(math.isfinite(v) for v in [mean, low, high]) or not low <= mean <= high:
                raise ValueError("Reference index contains invalid shift values.")
            return {"ppm": mean, "min": low, "max": high,
                    "references": count, "radius": radius,
                    "method": "exact_connectivity" if radius == 0 else "local_environment"}
    return None


@lru_cache(maxsize=2)
def _read_index(path, size, modified):
    del size, modified
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("encoder") != ENCODER or payload.get("solvent") != SOLVENT:
        raise ValueError("Unsupported NMR reference database version.")
    if not isinstance(payload.get("entries"), dict):
        raise ValueError("NMR reference database is missing its search index.")
    return payload


def load_index(path):
    path = Path(path).resolve()
    try:
        stat = path.stat()
    except FileNotFoundError as error:
        raise ValueError("The NMR reference database is missing. Reinstall the complete plugin ZIP.") from error
    return _read_index(str(path), stat.st_size, stat.st_mtime_ns)


def predict(mol, nucleus, index_path=None, cancelled=None):
    """Predict supported sites only; unsupported sites are reported, never guessed.

    Every attached C-H proton receives its carbon site's group-median estimate.
    Exchangeable protons and stereospecific proton differences are not modeled.
    min/max describe the observed molecule-median reference range, not a CI.
    """
    if nucleus not in {"1H", "13C"}:
        raise ValueError("Choose 1H or 13C.")
    check_cancelled(cancelled)
    expanded = prepare_molecule(mol)
    heavy, mapping = heavy_graph(expanded)
    payload = load_index(index_path or Path(__file__).with_name("data") / "reference.json.gz")
    predictions, unsupported = [], []
    cached = {}
    for atom in expanded.GetAtoms():
        check_cancelled(cancelled)
        if (nucleus == "1H" and atom.GetAtomicNum() != 1) or (nucleus == "13C" and atom.GetAtomicNum() != 6):
            continue
        parent = atom if nucleus == "13C" else atom.GetNeighbors()[0]
        if parent.GetAtomicNum() != 6:
            unsupported.append(atom.GetIdx())
            continue
        center = mapping[parent.GetIdx()]
        if center not in cached:
            cached[center] = find_reference(payload["entries"], nucleus, environment_keys(heavy, center))
        estimate = cached[center]
        if estimate is None:
            unsupported.append(atom.GetIdx())
            continue
        predictions.append(dict(estimate, idx=atom.GetIdx(), parent_idx=parent.GetIdx(),
                                atom=atom.GetSymbol()))
    if not predictions:
        raise ValueError("No supported sites were found in the CDCl3 reference database. "
                         "This structure needs more reference data; no shifts were guessed.")
    return {"nucleus": nucleus, "data": predictions, "mol_with_h": expanded,
            "unsupported": unsupported, "solvent": SOLVENT,
            "model_note": "CDCl3 reference estimates; 2D environments; attached H grouped; "
                          "stereochemistry pooled; min/max are reference ranges, not confidence intervals.",
            "attribution": ATTRIBUTION, "data_license_url": DATA_LICENSE_URL}
