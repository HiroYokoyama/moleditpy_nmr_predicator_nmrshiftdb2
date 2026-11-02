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
from rdkit.Chem import rdCIPLabeler

ENCODER = "moleditpy-rdkit-stereo-fragment-v2"
RADII = (4, 3, 2)
MIN_REFERENCES = 3
SOLVENT = "CDCl3"
SOLVENTS = {
    "CDCl3": "Chloroform-D1 (CDCl3)",
    "DMSO-d6": "Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)",
    "CD3OD": "Methanol-D4 (CD3OD)",
    "D2O": "Deuteriumoxide (D2O)",
    "Acetone-d6": "Acetone-D6 ((CD3)2CO)",
    "C6D6": "Benzene-D6 (C6D6)",
    "Pyridine-d5": "Pyridin-D5 (C5D5N)",
    "THF-d8": "Tetrahydrofuran-D8 (THF-D8, C4D4O)",
    "CD3CN": "Acetonitrile-D3 (CD3CN)",
    "CCl4": "TETRACHLORO-METHANE (CCl4)",
}
ATTRIBUTION = (
    "Contains information from nmrshiftdb2 (www.nmrshiftdb.org), which is made "
    "available here under the nmrshiftdb2 Database License."
)
DATA_LICENSE_URL = "https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/nmrshiftdb2datalicense.txt"
DATA_LICENSE_NAME = "nmrshiftdb2 Database License"


def license_metadata():
    """Notices embedded in the machine-readable derivative index itself."""
    return {"data_license": DATA_LICENSE_NAME, "data_license_url": DATA_LICENSE_URL,
            "attribution": ATTRIBUTION,
            "database_rights": "Database rights retained by the nmrshiftdb2 contributors."}


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
    if result.GetNumConformers() and result.GetConformer().Is3D():
        Chem.AssignStereochemistryFrom3D(result, replaceExistingTags=False)
    if result.GetStereoGroups() or any(
        a.GetChiralTag() not in {Chem.ChiralType.CHI_UNSPECIFIED,
                                Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
        for a in result.GetAtoms()
    ):
        raise ValueError("Enhanced or non-tetrahedral stereochemistry is outside this model.")
    if any(b.GetStereo() not in {Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY,
                                Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE,
                                Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}
           for b in result.GetBonds()):
        raise ValueError("Axial/atropisomeric stereochemistry is outside this model.")
    Chem.AssignStereochemistry(result, cleanIt=True, force=True)
    return Chem.AddHs(result, addCoords=True)


def heavy_graph(mol):
    """Return a hydrogen-suppressed graph and original-index to graph-index map."""
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetIntProp("_nmr_original_index", atom.GetIdx())
        atom.SetAtomMapNum(0)
        atom.SetIsotope(0)
    heavy = Chem.RemoveHs(copy)
    Chem.AssignStereochemistry(heavy, cleanIt=True, force=True)
    mapping = {a.GetIntProp("_nmr_original_index"): a.GetIdx() for a in heavy.GetAtoms()}
    return heavy, mapping


def _stereo_symbols(heavy):
    rdCIPLabeler.AssignCIPLabels(heavy)
    centers = dict(Chem.FindMolChiralCenters(heavy, includeUnassigned=True,
                                          useLegacyImplementation=False))
    symbols = []
    for atom in heavy.GetAtoms():
        configuration = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else centers.get(atom.GetIdx(), "-")
        symbols.append(
            f"[{atom.GetAtomicNum()};d{atom.GetDegree()};h{atom.GetTotalNumHs()};"
            f"a{int(atom.GetIsAromatic())};q{atom.GetFormalCharge()};s{configuration}]"
        )
    potential = {s.centeredOn for s in Chem.FindPotentialStereo(heavy)
                 if s.type == Chem.StereoType.Bond_Double}
    bonds = []
    for bond in heavy.GetBonds():
        order = {1.0: "-", 1.5: ":", 2.0: "=", 3.0: "#"}.get(bond.GetBondTypeAsDouble())
        if order is None:
            raise ValueError("Unsupported bond type in reference model.")
        stereo = bond.GetProp("_CIPCode") if bond.HasProp("_CIPCode") else ""
        if bond.GetIdx() in potential and not stereo:
            stereo = "?"
        bonds.append(order + ("{" + stereo + "}" if stereo else ""))
    return symbols, bonds


def stereo_forms(heavy):
    # Canonicalize a structure and its complete mirror together. All supported
    # solvents are achiral. Diastereomers and E/Z remain distinct; unknown
    # centers/bonds carry '?' and do not silently match specified configurations.
    forms = []
    has_assigned_center = any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in heavy.GetAtoms())
    for mirrored in ([False, True] if has_assigned_center else [False]):
        graph = Chem.Mol(heavy)
        if mirrored:
            for atom in graph.GetAtoms():
                atom.InvertChirality()
                if atom.HasProp("_CIPCode"):
                    atom.ClearProp("_CIPCode")
        Chem.AssignStereochemistry(graph, cleanIt=True, force=True)
        symbols, bonds = _stereo_symbols(graph)
        identity = Chem.MolToSmiles(graph, canonical=True, isomericSmiles=True)
        ranks = list(Chem.CanonicalRankAtoms(graph, breakTies=False, includeChirality=True))
        forms.append((graph, symbols, bonds, identity, ranks))
    return forms


def _fragment_code(center, atoms, forms):
    codes = []
    for graph, symbols, bonds, _, _ in forms:
        codes.append(Chem.MolFragmentToSmiles(
            graph, atomsToUse=atoms, atomSymbols=symbols, bondSymbols=bonds,
            rootedAtAtom=center, canonical=True, isomericSmiles=False,
            allBondsExplicit=True,
        ))
    return hashlib.sha256(min(codes).encode("ascii")).hexdigest()


def _full_code(center, forms):
    identity, rank = min((identity, ranks[center]) for _, _, _, identity, ranks in forms)
    return hashlib.sha256(f"{identity}|site:{rank}".encode("ascii")).hexdigest()


def environment_keys(heavy, center, forms=None):
    """Stereo-aware rooted fragments, invariant to ordering and global reflection.

    Preserve full boundary degree/H counts, CIP tetrahedral configurations and
    E/Z double-bond descriptors. Bond geometry is explicit even at a truncated
    fragment boundary. Attached-H assignments remain group medians.
    """
    shells = {center}
    frontier = {center}
    forms = stereo_forms(heavy) if forms is None else forms
    keys = {0: _full_code(center, forms)}
    for radius in range(1, max(RADII) + 1):
        frontier = {n.GetIdx() for i in frontier for n in heavy.GetAtomWithIdx(i).GetNeighbors()} - shells
        shells.update(frontier)
        if radius not in RADII:
            continue
        keys[radius] = _fragment_code(center, sorted(shells), forms)
    return keys


def index_key(nucleus, radius, code, solvent=SOLVENT):
    return f"{solvent}|{nucleus}:{radius}:{code}"


def find_reference(index, nucleus, keys, solvent=SOLVENT):
    for radius in (0, *RADII):
        if radius not in keys:
            continue
        value = index.get(index_key(nucleus, radius, keys[radius], solvent))
        if value is not None and value[3] >= (1 if radius == 0 else MIN_REFERENCES):
            mean, low, high, count = value
            if not all(math.isfinite(v) for v in [mean, low, high]) or not low <= mean <= high:
                raise ValueError("Reference index contains invalid shift values.")
            return {"ppm": mean, "min": low, "max": high,
                    "references": count, "radius": radius,
                    "method": "exact_stereostructure" if radius == 0 else "local_environment"}
    return None


@lru_cache(maxsize=2)
def _read_index(path, size, modified):
    del size, modified
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("encoder") != ENCODER or not isinstance(payload.get("solvents"), list):
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


def predict(mol, nucleus, index_path=None, cancelled=None, solvent=SOLVENT):
    """Predict supported sites only; unsupported sites are reported, never guessed.

    Every attached C-H proton receives its carbon site's group-median estimate.
    Exchangeable protons and stereospecific proton differences are not modeled.
    min/max describe the observed site/group measurement range, not a CI.
    """
    if nucleus not in {"1H", "13C"}:
        raise ValueError("Choose 1H or 13C.")
    if solvent not in SOLVENTS:
        raise ValueError("Choose a supported solvent from the prediction dialog.")
    check_cancelled(cancelled)
    expanded = prepare_molecule(mol)
    heavy, mapping = heavy_graph(expanded)
    payload = load_index(index_path or Path(__file__).with_name("data") / "reference.json.gz")
    if solvent not in payload["solvents"]:
        raise ValueError(f"No reference spectra are available for {solvent}.")
    predictions, unsupported = [], []
    cached = {}
    equivalent = {}
    forms = stereo_forms(heavy)
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
            full_code = _full_code(center, forms)
            if full_code not in equivalent:
                estimate = find_reference(payload["entries"], nucleus, {0: full_code}, solvent)
                if estimate is None:
                    estimate = find_reference(payload["entries"], nucleus,
                                              environment_keys(heavy, center, forms), solvent)
                equivalent[full_code] = estimate
            cached[center] = equivalent[full_code]
        estimate = cached[center]
        if estimate is None:
            unsupported.append(atom.GetIdx())
            continue
        predictions.append(dict(estimate, idx=atom.GetIdx(), parent_idx=parent.GetIdx(),
                                atom=atom.GetSymbol()))
    if not predictions:
        raise ValueError(f"No supported sites were found in the {solvent} reference database. "
                         "This structure needs more reference data; no shifts were guessed.")
    return {"nucleus": nucleus, "data": predictions, "mol_with_h": expanded,
            "unsupported": unsupported, "solvent": solvent,
            "model_note": f"{solvent} reference estimates; stereo within matched environment; "
                          "enantiomers pooled; attached H grouped; min/max are reference ranges, not confidence intervals.",
            "attribution": ATTRIBUTION, "data_license_url": DATA_LICENSE_URL}
