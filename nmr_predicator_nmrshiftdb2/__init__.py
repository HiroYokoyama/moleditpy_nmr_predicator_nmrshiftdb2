
from ._qt_callbacks import connect_queued
import math
import csv
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QMessageBox,
    QHeaderView,
    QProgressDialog,
    QFileDialog,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from rdkit import Chem
import pyvista as pv
import numpy as np
import logging

from . import coupling, database

# Periodic table for VdW radii
PTABLE = Chem.GetPeriodicTable()

# --- Metadata (Plugin Development Manual Section 2) ---
PLUGIN_NAME = "NMR Predictor (nmrshiftdb2)"
PLUGIN_VERSION = "4.0.1"
PLUGIN_LICENSE = "GPL-3.0-only"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = "Estimate 1H and 13C NMR shifts from local solvent-filtered reference data using Python and RDKit."
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=5.0.0, <6.0.0"

#: 13C observe frequency relative to 1H (gyromagnetic ratio).
C13_TO_H1 = 0.25145

#: Room left beside the outermost peaks by "Auto Fit". 1 ppm is a lot of a
#: 1H axis but nothing on a 13C one, where it put the end peaks on the frame.
AUTO_FIT_MARGIN_PPM = {"1H": 1.0, "13C": 10.0}


def table_headers(nucleus):
    """Result table / CSV columns."""
    coupling_cols = (
        ["Mult.", "J (Hz)"] if nucleus == "1H" else ["Mult. (1H-coupled)", "1J(CH) (Hz)"]
    )
    return ["Atom ID", "Type", "Shift (ppm)", "Ref. min (ppm)", "Ref. max (ppm)",
            "Ref. molecules", "Match (bonds)"] + coupling_cols


def reference_match(item):
    if item.get("method") == "exact_stereostructure":
        return "Exact"
    if item.get("radius"):
        return f"Local ({item['radius']})"
    return ""


def observe_mhz(h1_mhz, nucleus):
    """Observe frequency of ``nucleus`` on a magnet with the given 1H frequency."""
    return h1_mhz if nucleus == "1H" else h1_mhz * C13_TO_H1


def describe_multiplet(item):
    """', dq (J = 17.0, 7.0 Hz)' for status lines; empty without coupling data."""
    if not item.get("mult"):
        return ""
    if item.get("j_text"):
        return f", {item['mult']} (J = {item['j_text']} Hz)"
    return f", {item['mult']}"


# --- 1. Background Worker (local Python/RDKit search) ---
class PredictorWorker(QThread):
    """Search the bundled reference data without Java or network access."""

    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, mol, nucleus, plugin_dir, solvent=database.SOLVENT):
        super().__init__()
        self.mol = mol
        self.nucleus = nucleus
        self.plugin_dir = plugin_dir
        self.solvent = solvent

    def run(self):
        try:
            result = database.predict(
                self.mol, self.nucleus,
                Path(self.plugin_dir) / "data" / "reference.json.gz",
                cancelled=getattr(self, "isInterruptionRequested", None),
                solvent=self.solvent,
            )
            coupling.annotate_predictions(result["mol_with_h"], result["data"], self.nucleus)
            self.finished_signal.emit(result)
        except database.PredictionCancelled:
            return
        except Exception as error:
            logging.getLogger(__name__).exception("NMR reference prediction failed")
            self.error_signal.emit(str(error))


# --- 2. Result Dialog (GUI) ---
class ResultDialog(QDialog):
    def __init__(self, parent, result_data, context):
        super().__init__(parent)
        self.setWindowTitle(f"NMR Prediction Result ({result_data['nucleus']})")
        self.resize(940, 820)
        self.setWindowModality(Qt.WindowModality.NonModal)

        self.context = context
        self._source_mol = getattr(context.get_main_window(), "current_mol", None)
        self._highlight_plotter = None
        self.data = result_data["data"]
        self.mol_with_h = result_data["mol_with_h"]
        self.nucleus = result_data["nucleus"]
        self.solvent = result_data.get("solvent", database.SOLVENT)
        self.model_note = result_data.get("model_note", "")
        self.unsupported = result_data.get("unsupported", [])

        # Track 3D actors for cleanup
        self._highlight_actors = {}  # atom_idx -> actor
        self._label_actors = {}  # atom_idx -> actor

        layout = QVBoxLayout()
        self.setLayout(layout)

        supported = len(self.data)
        total = supported + len(self.unsupported)
        note = QLabel(f"{self.solvent} · {supported}/{total} nuclei supported")
        layout.addWidget(note)

        # 1. NMR Graph (Matplotlib). Imported here rather than at module level:
        # matplotlib costs ~0.6 s, which every MoleditPy start would otherwise pay.
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as FigureCanvas,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(5, 3.5))
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Graph Controls
        range_card = QVBoxLayout()
        range_card.setContentsMargins(5, 5, 5, 5)

        range_title = QLabel("Graph Controls")
        range_title.setStyleSheet("font-weight: bold; color: #555;")
        range_card.addWidget(range_title)

        ctrl_row = QHBoxLayout()

        ctrl_row.addWidget(QLabel("Range (ppm):"))
        self.min_ppm_spin = QDoubleSpinBox()
        self.min_ppm_spin.setRange(-50, 500)
        self.min_ppm_spin.setDecimals(1)
        self.min_ppm_spin.setSingleStep(1.0)
        self.min_ppm_spin.setValue(-1.0 if self.nucleus == "1H" else -10.0)
        self.min_ppm_spin.valueChanged.connect(self.plot_spectrum)
        ctrl_row.addWidget(self.min_ppm_spin)

        ctrl_row.addWidget(QLabel("to"))
        self.max_ppm_spin = QDoubleSpinBox()
        self.max_ppm_spin.setRange(-50, 500)
        self.max_ppm_spin.setDecimals(1)
        self.max_ppm_spin.setSingleStep(1.0)
        self.max_ppm_spin.setValue(12.0 if self.nucleus == "1H" else 220.0)
        self.max_ppm_spin.valueChanged.connect(self.plot_spectrum)
        ctrl_row.addWidget(self.max_ppm_spin)

        self.auto_scale_chk = QCheckBox("Auto Fit")
        self.auto_scale_chk.toggled.connect(self.plot_spectrum)
        ctrl_row.addWidget(self.auto_scale_chk)

        ctrl_row.addStretch()
        range_card.addLayout(ctrl_row)

        mult_row = QHBoxLayout()
        self.multiplet_chk = QCheckBox(
            "Show multiplets" if self.nucleus == "1H" else "Show 1H-coupled multiplets"
        )
        # Default: split the 1H spectrum, keep 13C decoupled as usual.
        self.multiplet_chk.setChecked(self.nucleus == "1H")
        self.multiplet_chk.toggled.connect(self.plot_spectrum)
        mult_row.addWidget(self.multiplet_chk)
        mult_row.addWidget(QLabel("Spectrometer:"))
        self.mhz_spin = QDoubleSpinBox()
        self.mhz_spin.setRange(40.0, 1200.0)
        self.mhz_spin.setDecimals(0)
        self.mhz_spin.setSingleStep(100.0)
        self.mhz_spin.setSuffix(" MHz (1H)")
        self.mhz_spin.setValue(400.0)  # before connecting: no early redraw
        self.mhz_spin.valueChanged.connect(self.plot_spectrum)
        mult_row.addWidget(self.mhz_spin)
        mult_row.addStretch()
        range_card.addLayout(mult_row)

        broad_row = QHBoxLayout()
        self.broadening_chk = QCheckBox("Line broadening")
        self.broadening_chk.setChecked(True)
        self.broadening_chk.toggled.connect(self._on_broadening_toggled)
        broad_row.addWidget(self.broadening_chk)
        broad_row.addWidget(QLabel("Line width:"))
        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.1, 50.0)
        self.linewidth_spin.setDecimals(1)
        self.linewidth_spin.setSingleStep(0.5)
        self.linewidth_spin.setSuffix(" Hz")
        self.linewidth_spin.setValue(coupling.DEFAULT_LINEWIDTH_HZ.get(self.nucleus, 1.0))
        self.linewidth_spin.valueChanged.connect(self.plot_spectrum)
        broad_row.addWidget(self.linewidth_spin)
        broad_row.addStretch()
        range_card.addLayout(broad_row)

        layout.addWidget(self.toolbar)
        layout.addLayout(range_card)
        layout.addWidget(self.canvas)

        # 2. Table Result
        self.table = QTableWidget()
        headers = table_headers(self.nucleus)
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setStyleSheet("""
            QTableWidget::item:selected {
                background-color: pink;
                color: black;
            }
        """)

        # Populate Table
        self.table.setRowCount(len(self.data))
        for row, item in enumerate(self.data):
            id_item = QTableWidgetItem(str(item["idx"]))
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, id_item)

            type_item = QTableWidgetItem(item["atom"])
            type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, type_item)

            ppm_item = QTableWidgetItem(f"{item['ppm']:.2f}")
            ppm_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, 2, ppm_item)

            min_item = QTableWidgetItem(f"{item.get('min', 0.0):.2f}")
            min_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, 3, min_item)

            max_item = QTableWidgetItem(f"{item.get('max', 0.0):.2f}")
            max_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, 4, max_item)

            for col, text in ((5, str(item.get("references", ""))), (6, reference_match(item)),
                              (7, item.get("mult", "")), (8, item.get("j_text", ""))):
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, cell)

        layout.addWidget(self.table)
        reference_note = QLabel("Range is not a confidence interval. References count independent molecules.")
        reference_note.setWordWrap(True)
        layout.addWidget(reference_note)

        # 3. Credits and Status
        bottom_layout = QVBoxLayout()
        layout.addLayout(bottom_layout)

        self.status_label = QLabel("Hover or click peaks to see details.")
        self.status_label.setStyleSheet("color: #444; font-weight: bold;")
        bottom_layout.addWidget(self.status_label)

        # Bottom Buttons
        btn_row = QHBoxLayout()

        self.unselect_btn = QPushButton("Unselect All")
        self.unselect_btn.setFixedWidth(100)
        self.unselect_btn.clicked.connect(self.clear_selection)
        btn_row.addWidget(self.unselect_btn)

        export_btn = QPushButton("Export CSV")
        export_btn.setFixedWidth(100)
        export_btn.clicked.connect(self.export_csv)
        btn_row.addWidget(export_btn)

        btn_row.addStretch()

        about_btn = QPushButton("About")
        about_btn.setFixedWidth(80)
        about_btn.clicked.connect(self.show_about)
        btn_row.addWidget(about_btn)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

        # Powered by label - positioned at the very bottom right
        credit_row = QHBoxLayout()
        credit_row.addStretch()
        credit_label = QLabel(
            'Reference data: nmrshiftdb2 · <a href="' + database.DATA_LICENSE_URL + '">Database License</a>'
        )
        credit_label.setOpenExternalLinks(True)
        credit_label.setStyleSheet(
            "color: gray; font-size: 9px; font-style: italic; margin-top: 5px;"
        )
        credit_row.addWidget(credit_label)
        layout.addLayout(credit_row)

        # Plot the spectrum
        self.plot_spectrum()

        # Connect Signals
        self.table.cellClicked.connect(self.on_table_click)
        self.canvas.mpl_connect("button_press_event", self.on_graph_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # 3. Synchronize 3D -> UI Polling Timer
        self.sel_timer = QTimer(self)
        self.sel_timer.timeout.connect(self._sync_from_3d)
        self.sel_timer.start(300)
        self._last_selected = set()
        self._hover_idx = -1
        self._persistent_ppm = None
        self._graph_line = None
        self._hover_line = None

    def _on_broadening_toggled(self, checked):
        self.linewidth_spin.setEnabled(bool(checked))
        self.plot_spectrum()

    def plot_spectrum(self):
        """Plot NMR stick spectrum with multiplicity (proportional intensity)."""
        self.figure.clear()
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.24)
        self.figure.text(
            0.01, 0.005,
            "Contains information from nmrshiftdb2 (www.nmrshiftdb.org).\n"
            "Available under the nmrshiftdb2 Database License:\n" + database.DATA_LICENSE_URL,
            fontsize=5, va="bottom",
        )
        # figure.clear() destroyed both highlight artists; drop the stale
        # references so _update_graph_highlight does not try to remove them
        # from an axes they no longer belong to.
        self._graph_line = None
        self._hover_line = None
        ax = self.figure.add_subplot(111)

        if not self.data:
            ax.text(0.5, 0.5, "No peaks predicted", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return

        # One stick per nucleus; equal shifts stack up, and with multiplets on
        # each signal is split by its estimated couplings.
        sticks = coupling.spectrum_sticks(
            self.data,
            observe_mhz(self.mhz_spin.value(), self.nucleus),
            self.multiplet_chk.isChecked(),
        )
        shifts = [ppm for ppm, _height in sticks]
        intensities = [height for _ppm, height in sticks]

        # NMR Convention: X-axis descending
        if self.auto_scale_chk.isChecked():
            margin = AUTO_FIT_MARGIN_PPM.get(self.nucleus, 1.0)
            left, right = max(shifts) + margin, min(shifts) - margin
        else:
            left, right = self.max_ppm_spin.value(), self.min_ppm_spin.value()

        curve_x, curve_y = [], []
        if self.broadening_chk.isChecked():
            curve_x, curve_y = coupling.lorentzian_curve(
                sticks,
                self.linewidth_spin.value(),
                observe_mhz(self.mhz_spin.value(), self.nucleus),
                right,
                left,
            )
        if curve_x:
            # Lorentzian lines: easier to read than sticks once split.
            ax.plot(curve_x, curve_y, color="b", linewidth=1.0)
            ax.axhline(0, color="k", alpha=0.3, linewidth=1)
            max_int = max(curve_y)
        else:
            # Stem plot for sticks
            markerline, stemlines, baseline = ax.stem(
                shifts, intensities, linefmt="b-", markerfmt="None", basefmt="k-"
            )
            stemlines.set_linewidth(1.5)
            baseline.set_alpha(0.3)
            max_int = max(intensities) if intensities else 1.0

        ax.set_ylim(0, (max_int or 1.0) * 1.2)
        ax.set_xlim(left, right)

        ax.set_xlabel("Chemical Shift (ppm)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"{self.nucleus} NMR Reference Estimates ({self.solvent})")
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)

        # Re-draw the selection marker: changing the range must not look
        # like the selection was lost.
        if getattr(self, "_persistent_ppm", None) is not None:
            self._graph_line = ax.axvline(
                self._persistent_ppm,
                color="red",
                linestyle="-",
                alpha=0.8,
                linewidth=2,
            )

        self.canvas.draw()

    def on_hover(self, event):
        """Handle pointer movement while preserving the persistent selection."""
        if event.inaxes is None or event.xdata is None:
            if self._hover_idx != -1:
                self._hover_idx = -1
                self._update_graph_highlight(None, is_hover=True)
                # Restore the persistent selection, or clear temporary highlights.
                self._restore_persistent_highlight()
            return

        shifts = [item["ppm"] for item in self.data]
        if not shifts:
            return

        xlim = event.inaxes.get_xlim()
        tolerance = abs(xlim[1] - xlim[0]) * 0.02

        distances = [abs(s - event.xdata) for s in shifts]
        min_idx = np.argmin(distances)

        if distances[min_idx] < tolerance:
            if self._hover_idx != min_idx:
                self._hover_idx = min_idx

                target_ppm = shifts[min_idx]

                # Keep the selected peak red instead of adding a hover highlight.
                if (
                    self._persistent_ppm is not None
                    and abs(target_ppm - self._persistent_ppm) < 1e-4
                ):
                    self._update_graph_highlight(
                        None, is_hover=True
                    )  # Remove the orange hover line.
                    return

                # Highlight an unselected peak on hover.
                self._update_graph_highlight(target_ppm, is_hover=True)

                # Temporarily replace the persistent 3D highlight. Restore it
                # when the pointer leaves the peak.
                self.highlight_atom(min_idx, persistent=False)

                item = self.data[min_idx]
                self.status_label.setText(
                    f"Peak: {item['atom']}{item['idx']} at {item['ppm']:.2f} ppm"
                    f"{describe_multiplet(item)}"
                )
                self.status_label.setStyleSheet("color: #e67e22; font-weight: bold;")
        else:
            # Restore the selection when the pointer leaves all peaks.
            if self._hover_idx != -1:
                self._hover_idx = -1
                self._update_graph_highlight(None, is_hover=True)
                self._restore_persistent_highlight()

    def _restore_persistent_highlight(self):
        """Restore the persistent selection, or clear temporary highlights."""
        if self._persistent_ppm is not None:
            # Restore the selection that was active before hovering.
            for i, item in enumerate(self.data):
                if abs(item["ppm"] - self._persistent_ppm) < 1e-4:
                    # A persistent highlight restores the red selection color.
                    self.highlight_atom(i, persistent=True)
                    break
        else:
            self.clear_3d_visuals()
            self.status_label.setText("Hover over peaks to see in 3D.")
            self.status_label.setStyleSheet("color: #444; font-weight: bold;")

    def on_graph_click(self, event):
        """Handle click on plot - Sync to Table and 3D Highlight."""
        if event.inaxes is None or event.xdata is None:
            return

        # Find nearest peak
        shifts = [item["ppm"] for item in self.data]
        if not shifts:
            return

        click_x = event.xdata

        # Calculate tolerance as 2% of current x-axis range width
        xlim = event.inaxes.get_xlim()
        width = abs(xlim[1] - xlim[0])
        tolerance = width * 0.02  # 2% range

        distances = [abs(s - click_x) for s in shifts]
        min_idx = np.argmin(distances)

        if distances[min_idx] < tolerance:
            ppm = shifts[min_idx]

            # Select the table rows corresponding to the clicked peak.
            self.table.clearSelection()
            self.table.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
            for i, item in enumerate(self.data):
                # Select every row at the same chemical shift.
                if abs(item["ppm"] - ppm) < 1e-4:
                    self.table.selectRow(i)
            self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)

            # Show a red plot marker and red spheres on the corresponding 3D atoms.
            self.highlight_atom(min_idx, persistent=True)

        else:
            # Clear the selection when the click is far from every peak.
            self.clear_selection()

    def on_table_click(self, row, col):
        self.highlight_atom(row)

    def highlight_atom(self, row_idx, persistent=True):
        """Highlight atoms in 3D with VDW sphere and label. Handles multi-atom peaks."""
        if not 0 <= row_idx < len(self.data):
            return
        if not self._source_is_current():
            self.clear_3d_visuals()
            self.status_label.setText(
                "Structure changed since prediction - re-run the prediction."
            )
            return
        target_item = self.data[row_idx]
        target_ppm = target_item["ppm"]

        # Find all atoms with the same PPM (multiplicity)
        # Using a small tolerance for floating point comparison if needed,
        # but reference group values are identical for equivalent atoms.
        matching_atoms = [
            item for item in self.data if abs(item["ppm"] - target_ppm) < 1e-4
        ]
        is_multi = len(matching_atoms) > 1

        try:
            mw = self.context.get_main_window()
            plotter = mw.plotter

            # Remove all old highlights
            self.clear_3d_visuals()

            mol = mw.current_mol
            conf = mol.GetConformer()

            # Prediction indices come from a copy with explicit hydrogens; the
            # host molecule may keep its H implicit, so an H index can sit past
            # its end. Fall back to the heavy atom the H hangs off, and only
            # give up when even that is out of range (a real structure edit).
            self._highlight_plotter = plotter
            n_atoms = mol.GetNumAtoms()
            draw_targets = []
            for item in matching_atoms:
                idx = item["idx"]
                if not 0 <= idx < n_atoms:
                    idx = item.get("parent_idx", idx)
                if not 0 <= idx < n_atoms:
                    self.status_label.setText(
                        "Structure changed since prediction — re-run the prediction."
                    )
                    return
                draw_targets.append((idx, item))

            # Visual settings
            color = "red" if persistent else "orange"
            opacity = 0.5 if persistent else 0.4
            scale_factor = 1.4

            for atom_idx, item in draw_targets:
                ppm = item["ppm"]
                symbol = item["atom"]

                pos = conf.GetAtomPosition(atom_idx)
                point = (pos.x, pos.y, pos.z)
                if not all(math.isfinite(value) for value in point):
                    raise ValueError("Highlight coordinates must be finite.")

                # 1. Sphere Highlight
                # Use RDKit VdW radius scaled by 0.3 as in moledit core/reference
                radius = PTABLE.GetRvdw(symbol) * 0.3 * scale_factor
                sphere = pv.Sphere(radius=radius, center=point)

                actor = plotter.add_mesh(
                    sphere,
                    color=color,
                    opacity=opacity,
                    name=f"nmr_highlight_{atom_idx}",
                    pickable=False,
                    # PyVista defaults this to `not _first_time and not camera_set`,
                    # so on a shown viewer that the user only zoomed with the mouse
                    # every highlight would refit the camera and throw the zoom away.
                    reset_camera=False,
                )
                self._highlight_actors[atom_idx] = actor

                # 2. Add Label (Centered on atom)
                label_name = f"nmr_label_{atom_idx}"
                label_actor = plotter.add_point_labels(
                    [point],
                    [f"{symbol}{item['idx']}\n{ppm:.2f} {item.get('mult', '')}".rstrip()],
                    font_size=12 if not is_multi else 14,
                    text_color="white" if persistent else "yellow",
                    point_size=0,
                    always_visible=True,
                    bold=True,
                    name=label_name,
                    reset_camera=False,
                )
                self._label_actors[atom_idx] = label_actor

            if persistent:
                if is_multi:
                    self.status_label.setText(
                        f"Selected Equiv Peak: {len(matching_atoms)} atoms at {target_ppm:.2f} ppm"
                        f"{describe_multiplet(target_item)}"
                    )
                else:
                    self.status_label.setText(
                        f"Selected {target_item['atom']}{target_item['idx']}: {target_ppm:.2f} ppm"
                        f"{describe_multiplet(target_item)}"
                    )

            plotter.render()

            # Update graph highlight (persistent red line)
            if persistent:
                self._persistent_ppm = target_ppm
                self._update_graph_highlight(target_ppm)

        except Exception as e:
            logging.warning("NMR Predictor: highlight failed: %s", e)

    def _update_graph_highlight(self, ppm, is_hover=False):
        """Draw highlight on the graph."""
        if not self.figure.axes:
            return
        ax = self.figure.axes[0]

        # Cleanup
        if getattr(self, "_hover_line", None) is not None:
            try:
                self._hover_line.remove()
            except Exception as _e:
                logging.warning("NMR Predictor: hover line cleanup: %s", _e)
            self._hover_line = None

        if not is_hover:
            # Persistent highlight update
            if getattr(self, "_graph_line", None) is not None:
                try:
                    self._graph_line.remove()
                except Exception:
                    pass  # line may already be removed from axes

            if ppm is not None:
                self._graph_line = ax.axvline(
                    ppm, color="red", linestyle="-", alpha=0.8, linewidth=2
                )
            self.canvas.draw()
        else:
            # Hover highlight
            if ppm is not None:
                self._hover_line = ax.axvline(
                    ppm, color="orange", linestyle="--", alpha=0.5, linewidth=2
                )
            self.canvas.draw()

    def _sync_from_3d(self):
        """Sync from 3D selection to table."""
        if not self._source_is_current():
            self.clear_3d_visuals()
            self.table.clearSelection()
            self.sel_timer.stop()
            self.status_label.setText(
                "Structure changed since prediction - re-run the prediction."
            )
            return
        mw = self.context.get_main_window()
        # selected_atoms_3d lives on Edit3DManager, never on the main window, so
        # the old `hasattr(mw, 'selected_atoms_3d')` guard always bailed out and
        # this sync never ran.
        manager = getattr(mw, "edit_3d_manager", None)
        selected = getattr(manager, "selected_atoms_3d", None)
        if selected is None:
            return

        current_sel = set(selected)
        if current_sel == self._last_selected:
            return
        self._last_selected = current_sel

        if not current_sel:
            self.clear_3d_visuals()
            self.table.clearSelection()
            return

        # Find if any selected atom matches our prediction data
        # Takes the first one found in our data
        atom_to_select = None
        for atom_idx in current_sel:
            for i, item in enumerate(self.data):
                if item["idx"] == atom_idx:
                    atom_to_select = i
                    break
            if atom_to_select is not None:
                break

        if atom_to_select is not None:
            self.table.selectRow(atom_to_select)
            # Update visuals but don't loop back?
            # highlight_atom already cleans and redraws
            self.highlight_atom(atom_to_select)

    def show_about(self):
        """Show information about used libraries."""
        about_text = f"""
        <h3>NMR Predictor (nmrshiftdb2)</h3>
        <p>This plugin predicts NMR chemical shifts using the following libraries:</p>
        <ul>
            <li><b>NMRShiftDB2</b>: measured reference data; original Python search.<br>
            <a href="{database.DATA_LICENSE_URL}">nmrshiftdb2 Database License</a></li>

            <li><b>RDKit</b>: Molecule handling and 3D coordinate generation.<br>
            <a href="https://www.rdkit.org/">https://www.rdkit.org/</a></li>
            
            <li><b>Matplotlib</b>: NMR spectrum visualization.<br>
            <a href="https://matplotlib.org/">https://matplotlib.org/</a></li>
            
            <li><b>PyVista</b>: 3D highlighting and visualization.<br>
            <a href="https://www.pyvista.org/">https://www.pyvista.org/</a></li>
        </ul>
        <p>Mult. / J are rule-based estimates made by this plugin, not part of
        the nmrshiftdb2 prediction. Only 1H coupling is modeled; isotope
        couplings are omitted.</p>
        <p>{database.ATTRIBUTION}<br>
        <a href="{database.DATA_LICENSE_URL}">{database.DATA_LICENSE_NAME}</a></p>
        <p>{self.model_note}</p>
        <p>Exact matches use the same stereostructure. Local matches use the stated
        neighborhood radius in bonds. Reference ranges cover experimental site/group
        values; reference counts are independent connectivities, not repeated measurements.
        A single reference or zero range does not establish high accuracy.</p>
        <p>{self.solvent} estimates; E/Z and tetrahedral configurations matched;
        enantiomers pooled; attached H grouped.
        Reference ranges are not confidence intervals.</p>
        <p>Author: {PLUGIN_AUTHOR}<br>Version: {PLUGIN_VERSION}<br>
        Shared coupling module: {coupling.COUPLING_MODULE_VERSION}</p>
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("About NMR Predictor")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(about_text)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def clear_selection(self):
        """Clear table selection and 3D highlights."""
        self.table.clearSelection()
        self._persistent_ppm = None
        self._update_graph_highlight(None)
        self.clear_3d_visuals()
        self.status_label.setText("Selection cleared.")

    def clear_3d_visuals(self):
        """Remove highlights from the plotter that owns them."""
        highlights = getattr(self, "_highlight_actors", {})
        labels = getattr(self, "_label_actors", {})
        names = [f"nmr_highlight_{i}" for i in highlights]
        names += [f"nmr_label_{i}" for i in labels]
        highlights.clear()
        labels.clear()
        owner, self._highlight_plotter = getattr(self, "_highlight_plotter", None), None
        if not names:
            return
        try:
            plotter = (
                owner if owner is not None else self.context.get_main_window().plotter
            )
            for name in names + ["nmr_highlight"]:
                try:
                    plotter.remove_actor(name)
                except (RuntimeError, AttributeError, KeyError, ValueError):
                    logging.getLogger(__name__).exception(
                        "NMR Predictor: could not remove highlight %s", name
                    )
            plotter.render()
        except (RuntimeError, AttributeError, KeyError, ValueError):
            logging.getLogger(__name__).exception(
                "NMR Predictor: could not redraw after highlight cleanup"
            )

    def export_csv(self):
        """Export table data to CSV."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "nmr_prediction.csv", "CSV Files (*.csv)"
        )

        if not filename:
            return

        try:
            with open(filename, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(table_headers(self.nucleus))

                # Data
                for item in self.data:
                    writer.writerow(
                        [
                            item["idx"],
                            item["atom"],
                            f"{item['ppm']:.2f}",
                            f"{item.get('min', 0.0):.2f}",
                            f"{item.get('max', 0.0):.2f}",
                            item.get("references", ""),
                            reference_match(item),
                            item.get("mult", ""),
                            item.get("j_text", ""),
                        ]
                    )

                writer.writerow([])
                writer.writerow(["# Attribution", database.ATTRIBUTION])
                writer.writerow(["# Data license", database.DATA_LICENSE_URL])
                writer.writerow(["# Solvent", self.solvent])
                writer.writerow(["# Model", getattr(self, "model_note", f"{self.solvent} local reference estimates")])

            QMessageBox.information(
                self, "Success", f"Exported successfully to:\n{filename}"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Failed to save file:\n{str(e)}"
            )

    def _source_is_current(self):
        try:
            return getattr(
                self.context.get_main_window(), "current_mol", None
            ) is getattr(self, "_source_mol", None)
        except (AttributeError, RuntimeError):
            return False

    def done(self, result):
        self.sel_timer.stop()
        self.clear_3d_visuals()
        super().done(result)

    def closeEvent(self, event):
        """Cleanup on close."""
        self.clear_3d_visuals()
        self.sel_timer.stop()

        # Clear reference on main window
        try:
            mw = self.context.get_main_window()
            if hasattr(mw, "nmr_result_dialog") and mw.nmr_result_dialog is self:
                mw.nmr_result_dialog = None
        except Exception as _e:
            logging.warning("[__init__.py:786] silenced: %s", _e)

        super().closeEvent(event)


# --- 3. Main Logic ---


def run_prediction(context):
    """Main function triggered from the menu."""
    mw = context.get_main_window()
    mol = context.current_molecule  # Using modern property access

    if not mol or mol.GetNumAtoms() == 0:
        QMessageBox.warning(mw, "Warning", "Please draw or load a molecule first.")
        return

    current_worker = getattr(mw, "nmr_worker", None)
    if current_worker is not None and current_worker.isRunning():
        QMessageBox.information(
            mw, PLUGIN_NAME, "An NMR prediction is already running."
        )
        return

    # 1. Select Nucleus
    nucleus, solvent, ok = ask_conditions(mw)
    if not ok:
        return

    # 2. Setup Progress Dialog
    plugin_dir = Path(__file__).parent
    progress = QProgressDialog(
        f"Searching {nucleus} {solvent} reference data...", "Cancel", 0, 0, mw
    )
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)  # Show immediately
    progress.show()

    # 3. Start Worker Thread
    # We need to keep a reference to the worker to prevent garbage collection
    worker = PredictorWorker(mol, nucleus, plugin_dir, solvent=solvent)
    mw.nmr_worker = worker

    def _release():
        # Only drop the reference once the thread has actually finished.
        if getattr(mw, "nmr_worker", None) is worker:
            mw.nmr_worker = None
        worker.deleteLater()

    connect_queued(worker.finished, _release, cleanup=(worker.finished,))

    def on_success(result):
        if progress.wasCanceled() or context.current_molecule is not mol:
            progress.cancel()
            return
        progress.cancel()

        # Singleton behavior: check if dialog already exists
        if hasattr(mw, "nmr_result_dialog") and mw.nmr_result_dialog:
            try:
                mw.nmr_result_dialog.close()
            except Exception as _e:
                logging.warning("[__init__.py:825] silenced: %s", _e)

        # Show Result Dialog (Modeless)
        mw.nmr_result_dialog = ResultDialog(mw, result, context)
        mw.nmr_result_dialog.show()
        mw.nmr_result_dialog.raise_()
        mw.nmr_result_dialog.activateWindow()

    def on_error(msg):
        was_cancelled = progress.wasCanceled()
        progress.cancel()
        if was_cancelled:
            return
        QMessageBox.critical(mw, "Prediction Error", msg)

    connect_queued(worker.finished_signal, on_success, mw, (dict,), cleanup=(worker.finished,))
    connect_queued(worker.error_signal, on_error, mw, (str,), cleanup=(worker.finished,))
    request_interruption = getattr(worker, "requestInterruption", None)
    if request_interruption is not None:
        progress.canceled.connect(request_interruption)
    worker.start()


def ask_conditions(parent):
    """Choose nucleus and an explicitly filtered reference solvent."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("NMR Prediction Conditions")
    dialog.resize(360, 210)
    layout = QVBoxLayout()

    layout.addWidget(QLabel("Nucleus:"))

    combo = QComboBox()
    combo.addItems(["1H", "13C"])
    layout.addWidget(combo)

    layout.addWidget(QLabel("Reference solvent:"))
    solvent_combo = QComboBox()
    solvent_combo.addItems(list(database.SOLVENTS))
    layout.addWidget(solvent_combo)
    note = QLabel("Only the selected solvent is searched. Reference coverage varies by solvent.")
    note.setWordWrap(True)
    layout.addWidget(note)

    btn = QPushButton("Predict")
    btn.clicked.connect(dialog.accept)
    layout.addWidget(btn)

    dialog.setLayout(layout)

    if dialog.exec():
        return combo.currentText(), solvent_combo.currentText(), True
    return None, None, False


# --- 4. Initialization Hook (Plugin Development Manual Section 2) ---
def initialize(context):
    """Entry point called by MoleditPy on startup."""

    # Register directly in the Analysis menu (no sub-menu since it's a single item)
    context.add_menu_action(
        "Analysis/NMR Prediction (nmrshiftdb2)", lambda: run_prediction(context)
    )


def run(mw):
    """Automatic entry point for MoleditPy."""
    if hasattr(mw, "host"):
        mw = mw.host
    from moleditpy.plugins.plugin_interface import PluginContext

    context = PluginContext(mw.plugin_manager, PLUGIN_NAME)
    run_prediction(context)
