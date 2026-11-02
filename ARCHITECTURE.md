# Architecture

The plugin estimates 1H and 13C chemical shifts by searching a prepared index of
experimental nmrshiftdb2 references. Python coordinates the search, RDKit handles
molecular graphs and stereochemistry, and Qt presents the results. Prediction
runs entirely offline against the bundled compressed index. It does not scan
the original SDF, train a model, or contact a prediction service at runtime.

The GUI collects the nucleus and reference solvent, then starts a background
worker. The search prepares a copy of the input molecule, preserving its atom
indices for result selection and highlighting. It validates the supported
chemistry, retains specified stereochemistry, and can perceive missing tags
from an existing 3D conformer. A hydrogen-suppressed graph is used for reference
matching, while the result molecule contains explicit hydrogens. The original
input molecule is not modified.

Each local search key describes the neighborhood of a target carbon within
two, three or four bonds. The description includes atomic number, full-graph
heavy-atom degree, attached-hydrogen count, aromaticity, formal charge,
tetrahedral configuration, bond order and double-bond stereochemistry.
Full-graph degree and hydrogen counts are retained at the fragment boundary.
RDKit produces a rooted, canonical fragment representation with custom atom and
bond symbols; its SHA-256 digest becomes the structure part of the lookup key.
The complete lookup key also includes the selected solvent, nucleus and radius.
The digest is an identifier for matching environments, not a prediction model
or a measure of structural similarity.

A separate exact key combines the canonical isomeric representation of the
whole molecule with the target atom's symmetry class. Both exact and local
keys are normalized over a structure and its complete mirror because the
supported solvents are achiral. E/Z and diastereomeric distinctions are
preserved when represented in the matched environment; unspecified
stereochemistry remains distinct from specified stereochemistry. Local keys do
not account for geometric effects or stereochemical relationships outside
their encoded information. Individual diastereotopic proton shifts are not
resolved.

For each site, the search first tries the exact stereostructure key, which may
use one independent molecular reference. If no exact reference is available,
it tries local radius four, then three, then two. A local estimate requires at
least three independent molecular connectivities. The first qualified match is
used without mixing radii or borrowing data from another solvent. The index is
loaded into a cached Python dictionary, so lookup does not compare the query
against every reference molecule. RDKit performs native graph operations, and
symmetry-equivalent sites reuse computed results within a prediction.

Reference generation is a separate development step. The builder checks the
pinned SHA-256 of the original assigned SDF, then filters experimental spectra
by supported chemistry, exact source solvent labels and temperature conditions.
Reported temperatures must be between 273 and 323 K; missing temperatures are
accepted. Calculated spectra and unsupported or ambiguous assignments are
excluded. The builder uses the same key encoder as runtime prediction, so the
index and query representations agree.

Measurements are aggregated within molecular connectivity before independent
references are combined. The estimated shift is a median of the contributing
connectivity-level medians. The index also stores the observed minimum,
maximum and independent connectivity count. Repeated measurements can widen
the reference range without increasing that count. The range describes
contributing observations; it is not a calibrated confidence interval. A single
reference can therefore have equal minimum and maximum without establishing
zero prediction error.

For 13C, each supported carbon receives an atom-linked estimate. For 1H, every
hydrogen attached to a given carbon receives that carbon site's proton-group
estimate. Exchangeable OH, NH and SH protons remain unsupported. Partial matches
produce results with explicit unsupported-site indices; if no sites are
supported, the worker reports a reference-unavailable error. Multiplicity and
J-coupling annotations are calculated separately by the rule-based coupling
module after the shift search. The result dialog plots these annotations and
exposes shifts, ranges, reference counts and exact or local match types.

The original SDF is stored through Git LFS for regeneration and is excluded
from the runtime ZIP. The ZIP contains the prepared index, provenance,
generation source and required notices. Plugin software is GPL-3.0-only;
source and derivative reference data retain the separate nmrshiftdb2 Database
License. Data attribution accompanies exported results.

Validation separates complete molecular connectivities, including duplicate
spectra and stereoisomers, between training and held-out subsets. This measures
supported-target error and coverage rather than agreement with an earlier
predictor. Regression tests separately compare representative predictions with
stored baseline outputs, including shifts, reference ranges, reference counts,
atom assignments and unsupported sites. Other tests cover input integrity,
stereochemistry, solvent isolation, packaging and Qt callbacks. CI runs on
pull requests targeting main and on pushes to main; native Qt callback checks
cover Windows, macOS and Linux.
