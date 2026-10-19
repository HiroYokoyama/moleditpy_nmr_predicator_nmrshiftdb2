# Bundled Java components and source availability

The Python plugin remains GPLv3. The separate Java predictor runs through a
command-line process with MOL-file input and textual output. Its AGPL code,
CDK's LGPL code, and the prediction database retain their own license terms.
See `NOTICE.txt` and the license files in `lib/`.

## Verified binary provenance

Both predictor JARs match [upstream commit r2333](https://sourceforge.net/p/nmrshiftdb2/code/2333/)
(2023-09-30, “updated predictor jars”) byte for byte. Each JAR's ten Java class
files also match [r2097](https://sourceforge.net/p/nmrshiftdb2/code/2097/)
(2020-11-05); the database contents differ. The plugin repository introduced
these JARs in `b01b532365711bedc5cef0f900abcd2b0f1c5e75` and has not modified them.
`THIRD_PARTY_PROVENANCE.json` records hashes, immutable upstream binary URLs,
per-class comparisons and database hashes.

CDK 2.11 matches the [official release checksum](https://github.com/cdk/cdk/releases/download/cdk-2.11/cdk-2.11-checksums-sha256.txt).
Its versioned [source tree and build instructions](https://github.com/cdk/cdk/tree/cdk-2.11)
are available under tag `cdk-2.11`. The JAR includes additional dependencies and
their embedded notices; retain those notices when redistributing it.

## Predictor sources: verified location and remaining distinction

The existing README and NOTICE point to the [official NMRShiftDB2 project](https://sourceforge.net/projects/nmrshiftdb2/).
Its versioned source is accessible through SVN:

- [Source tree at r2097](https://svn.code.sf.net/p/nmrshiftdb2/code/!svn/rvr/2097/trunk/nmrshiftdb2/)
- [Source tree at r2333](https://svn.code.sf.net/p/nmrshiftdb2/code/!svn/rvr/2333/trunk/nmrshiftdb2/)
- [Official standalone predictor documentation](https://sourceforge.net/p/nmrshiftdb2/wiki/PredictorJars/)

The official documentation distinguishes the JAR's stripped-down
`PredictionTool` from the full application's class of the same name. The
published full-application source and `build.xml` have been located, but this
audit has not established a complete source/build recipe for regenerating all
standalone JAR classes. Binary provenance is verified; complete corresponding
source availability for the standalone variant remains a release compliance
item. A project homepage or a matching revision alone is not recorded as proof
that the standalone variant can be rebuilt. Do not describe this audit as a
complete AGPL/LGPL binary-distribution clearance until that item is resolved.

## Database terms

The unmodified `lib/nmrshiftdb2-database-license.txt` is the license published
alongside the predictors at r2333. It covers the embedded prediction tables,
separately from the Java code. Preserve its notices and attribution, and follow
its output attribution and derivative-database requirements. The GPLv3 plugin
uses an OSI-approved license as required by section 4.5; this does not relicense
the underlying database.
