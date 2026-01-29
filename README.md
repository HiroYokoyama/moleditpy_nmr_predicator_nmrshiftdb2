# NMR Prediction Plugin for Moleditpy

A plugin for **Moleditpy** that predicts **1H** and **13C** NMR chemical shifts using the **nmrshiftdb2** machine learning models. It provides an interactive spectrum viewer and 3D atom highlighting.

## Features

* **Prediction:** Predict 1H and 13C NMR shifts using the robust `nmrshiftdb2` Java library.
* **Interactive Spectrum:** Visualize the result as a stick spectrum using Matplotlib.
* **3D Visualization:** Hovering over peaks or table rows highlights the corresponding atoms in the Moleditpy 3D view (PyVista).
* **Data Table:** Detailed list of chemical shifts and atom assignments.

## Requirements

* **Moleditpy**: The host application.
* **Java Runtime Environment (JRE)**: Java 8 or later must be installed and added to your system's PATH (required to run the prediction engine).

## Installation

1.  Download the latest `plugin.zip` from the [Releases](../../releases) page.
2.  Extract the zip file into the `plugins` directory of your Moleditpy installation.
    * Structure should look like: `.../plugins/nmr_predictor/__init__.py`
3.  Ensure the `lib/` folder inside the plugin directory contains the required JAR files (`predictorh.jar`, `cdk-*.jar`, etc.).

## Usage

1.  Launch Moleditpy and draw or load a molecule.
2.  Go to the menu: **Analysis** > **NMR Prediction (nmrshiftdb2)**.
3.  Select the nucleus (`1H` or `13C`) and click **Predict**.
4.  The result dialog will appear. You can:
    * **Hover** over the graph peaks to see the assignment.
    * **Click** on the table rows to highlight atoms in 3D.
    * Use **Graph Controls** to zoom or auto-scale the spectrum.

## Licenses & Credits

This plugin bundles the following third-party libraries. Please refer to the `lib/` directory for full license texts.

### nmrshiftdb2 Predictor
* **Description:** Machine learning based NMR shift prediction.
* **License:** **AGPL v3** (GNU Affero General Public License)
* **Source Code:** [https://sourceforge.net/projects/nmrshiftdb2/](https://sourceforge.net/projects/nmrshiftdb2/)
* **Copyright:** The nmrshiftdb2 Project

### The Chemistry Development Kit (CDK)
* **Description:** Java library for structural chemoinformatics.
* **License:** **LGPL v2.1** (GNU Lesser General Public License)
* **Source Code:** [https://github.com/cdk/cdk](https://github.com/cdk/cdk)
* **Copyright:** The CDK Development Team

### Plugin License
This plugin itself is released under the **GNU General Public License v3 (GPL v3)**.
