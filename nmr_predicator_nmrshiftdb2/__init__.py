import os
import shutil
import subprocess
import re
import csv
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QTableWidget, QTableWidgetItem, QLabel, QComboBox, 
    QCheckBox, QDoubleSpinBox, QMessageBox, QHeaderView, QProgressDialog, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from rdkit import Chem
import pyvista as pv
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
import tempfile
from rdkit.Chem import AllChem

# Periodic table for VdW radii
PTABLE = Chem.GetPeriodicTable()

# --- Metadata (Plugin Development Manual Section 2) ---
PLUGIN_NAME = "NMR Predictor (nmrshiftdb2)"
PLUGIN_VERSION = "1.1.2"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = "Predict 1H and 13C NMR shifts using nmrshiftdb2 (Java)."

# --- 1. Background Worker (Java Execution) ---
class PredictorWorker(QThread):
    """Worker thread to prevent GUI freeze during calculation"""
    finished_signal = pyqtSignal(dict) # On success
    error_signal = pyqtSignal(str)     # On error

    def __init__(self, mol, nucleus, plugin_dir):
        super().__init__()
        self.mol = mol
        self.nucleus = nucleus
        self.plugin_dir = plugin_dir

    def run(self):
        temp_mol_path = None
        try:
            # Check Java
            if not shutil.which("java"):
                self.error_signal.emit("Java Runtime (java command) not found.\nPlease install Java.")
                return

            jar_name = "predictorh.jar" if self.nucleus == "1H" else "predictorc.jar"
            jar_path = self.plugin_dir / "lib" / jar_name

            if not jar_path.exists():
                self.error_signal.emit(f"JAR file not found:\n{jar_path}\nPlease download it from SourceForge.")
                return


            # 1. Create calculation copy
            mol_calc = Chem.Mol(self.mol)

            # 2. Sanitize (Check structure integrity)
            try:
                Chem.SanitizeMol(mol_calc)
            except Exception as e:
                print(f"Warning: Sanitization failed: {e}")

            # 3. Clear existing stereochemistry flags
            # (Removes contradictory flags that cause "Unable to determine" errors)
            Chem.RemoveStereochemistry(mol_calc)

            # 5. Force coordinate regeneration (Critical)
            # Ignore existing coords and let RDKit generate a clean 2D layout.
            # This geometrically determines E/Z for ambiguous olefins.
            AllChem.Compute2DCoords(mol_calc)

            # 6. Re-assign stereochemistry
            # Re-flag based on the clean coordinates.
            Chem.AssignStereochemistry(mol_calc, force=True, cleanIt=True)
            
            # Set a generic name if missing (some readers crash on empty name)
            if not mol_calc.HasProp("_Name"):
                mol_calc.SetProp("_Name", "NMR_Calculation")
            
            # Ensure it has explicit Hydrogens and stereochemistry is perceived
            Chem.AssignStereochemistry(mol_calc, force=True, cleanIt=True)
            
            # Create temp file in system temp directory
            fd, temp_mol_path_str = tempfile.mkstemp(
                suffix=".mol", 
                prefix=f"nmrp_{self.nucleus}_"
            )
            os.close(fd)
            temp_mol_path = Path(temp_mol_path_str)
            
            try:
                # Write a clean V2000 Molfile.
                # includeStereo=False is CRITICAL to avoid parsing errors for alkenes in this backend.
                mol_block = Chem.MolToMolBlock(mol_calc, forceV3000=False, includeStereo=False)
                
                # Use CRLF and preserve raw RDKit formatting (except for line endings)
                # Manual padding was found to be detrimental for some readers.
                refined_block = mol_block.replace("\n", "\r\n")
                if not refined_block.endswith("\r\n"):
                    refined_block += "\r\n"
                
                with open(temp_mol_path, "wb") as f:
                    f.write(refined_block.encode("ascii", "ignore"))
                
                # Log for internal debugging (visible in Moledit log)
                #print(f"NMR Predictor: Running {self.nucleus} calculation. Temp file: {temp_mol_path}")

                classpath = self._build_classpath()

                # Build Java command. Documentation says: java Test myfile.mol [solvent [no3d]]
                # We omit optional flags to avoid environment-specific "solvent" validation errors.
                cmd = ["java", "-Xmx512m", "-cp", classpath, "Test", str(temp_mol_path)]
                
                startupinfo = None
                if os.name == 'nt':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                # Run Java from the lib directory so it can find its internal resources
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True, # Raise CalledProcessError on failure
                    startupinfo=startupinfo,
                    cwd=self.plugin_dir / "lib"
                )

                # Parse output
                predictions = self._parse_output(process.stdout, mol_calc)

                if not predictions:
                    err_msg = (
                        f"No NMR peaks predicted.\n"
                        f"Structure might be missing from the database or formatting is incompatible.\n"
                        f"Raw Output:\n{output[:1000]}\n"
                        f"Stderr:\n{process.stderr[:500]}"
                    )
                    self.error_signal.emit(err_msg)
                    return

                # Emit success
                self.finished_signal.emit({
                    "nucleus": self.nucleus,
                    "data": predictions,
                    "mol_with_h": mol_calc
                })

            finally:
                # Auto-delete temp file as requested by user
                if temp_mol_path and temp_mol_path.exists():
                    try: os.remove(temp_mol_path)
                    except: pass

        except subprocess.CalledProcessError as e:
            err_out = e.stderr if e.stderr else e.stdout
            self.error_signal.emit(f"Java Execution Failed (Code {e.returncode}):\n{err_out}")
        except Exception as e:
            self.error_signal.emit(f"Unexpected Error: {str(e)}")
        finally:
            pass

    def _build_classpath(self):
        """Constructs the Java classpath, ensuring the correct predictor JAR is first."""
        jar_name = "predictorh.jar" if self.nucleus == "1H" else "predictorc.jar"
        jar_path = self.plugin_dir / "lib" / jar_name
        other_jar_name = "predictorc.jar" if self.nucleus == "1H" else "predictorh.jar"

        if not jar_path.exists():
            raise FileNotFoundError(f"Required JAR not found: {jar_name}")

        jar_files = [jar_path]
        search_root = self.plugin_dir / "lib"
        for j in list(search_root.glob("*.jar")):
            if j.name != other_jar_name and j.name != jar_name:
                jar_files.append(j)

        abs_jars = []
        seen = set()
        for j in jar_files:
            p = str(j.absolute())
            if p not in seen:
                abs_jars.append(p)
                seen.add(p)
        
        return os.pathsep.join(abs_jars)

    def _parse_output(self, output, mol):
        """Parses the standard output from the Java prediction process."""
        predictions = []
        pattern = r"(\d+)\s*:\s*(\S+)\s+(\S+)"
        matches = list(re.finditer(pattern, output))
        
        num_atoms = mol.GetNumAtoms()
        target_symbol = "H" if self.nucleus == "1H" else "C"
        
        for match in matches:
            try:
                idx_java = int(match.group(1)) - 1
                val_str = match.group(3)
                ppm = float(val_str)
                
                if 0 <= idx_java < num_atoms:
                    atom = mol.GetAtomWithIdx(idx_java)
                    atom_symbol = atom.GetSymbol()
                    
                    if atom_symbol == target_symbol:
                        predictions.append({
                            "idx": idx_java,
                            "atom": atom_symbol,
                            "ppm": ppm
                        })
            except (ValueError, IndexError):
                continue
        return predictions

# --- 2. Result Dialog (GUI) ---
class ResultDialog(QDialog):
    def __init__(self, parent, result_data, context):
        super().__init__(parent)
        self.setWindowTitle(f"NMR Prediction Result ({result_data['nucleus']})")
        self.resize(600, 800)
        self.setWindowModality(Qt.WindowModality.NonModal)
        
        self.context = context
        self.data = result_data["data"]
        self.mol_with_h = result_data["mol_with_h"]
        self.nucleus = result_data["nucleus"]
        
        # Track 3D actors for cleanup
        self._highlight_actors = {} # atom_idx -> actor
        self._label_actors = {}     # atom_idx -> actor
        
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 1. NMR Graph (Matplotlib)
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
        
        layout.addWidget(self.toolbar)
        layout.addLayout(range_card)
        layout.addWidget(self.canvas)
        
        # 2. Table Result
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Atom ID", "Type", "Shift (ppm)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
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
            ppm_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 2, ppm_item)

        layout.addWidget(self.table)

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
        credit_label = QLabel("POWERED BY NMRShiftDB2")
        credit_label.setStyleSheet("color: gray; font-size: 9px; font-style: italic; margin-top: 5px;")
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

    def plot_spectrum(self):
        """Plot NMR stick spectrum with multiplicity (proportional intensity)."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if not self.data:
            ax.text(0.5, 0.5, "No peaks predicted", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return

        # Group peaks by PPM to handle multiplicity
        peak_map = {} # ppm -> count
        for item in self.data:
            ppm = round(item["ppm"], 4) # Group by rounded value
            peak_map[ppm] = peak_map.get(ppm, 0) + 1
            
        shifts = list(peak_map.keys())
        intensities = [float(peak_map[s]) for s in shifts]
        
        # Stem plot for sticks
        markerline, stemlines, baseline = ax.stem(shifts, intensities, 
                                                   linefmt='b-', markerfmt='None', 
                                                   basefmt='k-')
        stemlines.set_linewidth(1.5)
        baseline.set_alpha(0.3)
        
        # NMR Convention: X-axis descending
        is_auto = self.auto_scale_chk.isChecked()
        
        max_int = max(intensities) if intensities else 1.0
        ax.set_ylim(0, max_int * 1.2)
        
        if is_auto:
            # Auto scale
            ax.set_xlim(max(shifts) + 1.0, min(shifts) - 1.0)
            # Update spin boxes to reflect auto values (optional, but might be confusing)
            # self.min_ppm_spin.setValue(min(shifts) - 1.0)
            # self.max_ppm_spin.setValue(max(shifts) + 1.0)
        else:
            # Use manual range from spin boxes
            ax.set_xlim(self.max_ppm_spin.value(), self.min_ppm_spin.value())
            
        ax.set_xlabel("Chemical Shift (ppm)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"{self.nucleus} NMR Predicted Spectrum")
        ax.grid(True, axis='x', linestyle=':', alpha=0.5)
        
        self.canvas.draw()

    def on_hover(self, event):
        """マウス移動時の処理 - 選択状態を優先するよう修正"""
        if event.inaxes is None:
            if self._hover_idx != -1:
                self._hover_idx = -1
                self._update_graph_highlight(None, is_hover=True)
                # 修正: 選択状態があれば復元、なければクリア
                self._restore_persistent_highlight()
            return
            
        shifts = [item["ppm"] for item in self.data]
        if not shifts: return
            
        xlim = event.inaxes.get_xlim()
        tolerance = abs(xlim[1] - xlim[0]) * 0.02 
        
        distances = [abs(s - event.xdata) for s in shifts]
        min_idx = np.argmin(distances)
        
        if distances[min_idx] < tolerance: 
            if self._hover_idx != min_idx:
                self._hover_idx = min_idx
                
                target_ppm = shifts[min_idx]
                
                # 修正: 現在選択中のバー（Persistent）にホバーした場合は何もしない（赤のまま維持）
                # 「ハイライトされたバーは無効に」への対応
                if self._persistent_ppm is not None and abs(target_ppm - self._persistent_ppm) < 1e-4:
                    self._update_graph_highlight(None, is_hover=True) # オレンジ線は消す
                    return

                # 通常のホバー処理（未選択のバー）
                self._update_graph_highlight(target_ppm, is_hover=True)
                
                # 3Dハイライト（一時的）
                # highlight_atomを呼ぶと一時的にPersistent（赤）が消えるが、
                # 下記のelseブロックでの復元処理により、離れると赤に戻るようになる。
                self.highlight_atom(min_idx, persistent=False)
                
                item = self.data[min_idx]
                self.status_label.setText(f"Peak: {item['atom']}{item['idx']} at {item['ppm']:.2f} ppm")
                self.status_label.setStyleSheet("color: #e67e22; font-weight: bold;")
        else:
            # ピークから離れた場合
            if self._hover_idx != -1:
                self._hover_idx = -1
                self._update_graph_highlight(None, is_hover=True)
                self._restore_persistent_highlight()

    def _restore_persistent_highlight(self):
        """選択状態（Persistent）があれば復元し、なければクリアする"""
        if self._persistent_ppm is not None:
             # 選択状態を復元（ホバー前の状態に戻す）
             for i, item in enumerate(self.data):
                 if abs(item["ppm"] - self._persistent_ppm) < 1e-4:
                     # persistent=Trueで呼び直すことで赤色に戻す
                     self.highlight_atom(i, persistent=True)
                     break
        else:
            self.clear_3d_visuals()
            self.status_label.setText("Hover over peaks to see in 3D.")
            self.status_label.setStyleSheet("color: #444; font-weight: bold;")

    def on_graph_click(self, event):
        """Handle click on plot - Sync to Table and 3D Highlight."""
        if event.inaxes is None: return
        
        # Find nearest peak
        shifts = [item["ppm"] for item in self.data]
        if not shifts:
            return
            
        click_x = event.xdata
        
        # Calculate tolerance as 2% of current x-axis range width
        xlim = event.inaxes.get_xlim()
        width = abs(xlim[1] - xlim[0])
        tolerance = width * 0.02 # 2% range
        
        distances = [abs(s - click_x) for s in shifts]
        min_idx = np.argmin(distances)
        
        if distances[min_idx] < tolerance: 
            ppm = shifts[min_idx]
            
            # 1. Update Table Selection (既存の処理: テーブルの行を選択状態にする)
            self.table.clearSelection()
            self.table.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
            target_row = -1
            for i, item in enumerate(self.data):
                # 同じPPMを持つ行をすべて選択
                if abs(item["ppm"] - ppm) < 1e-4:
                    self.table.selectRow(i)
                    if i == min_idx: target_row = i
            self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)

            # 2. Trigger Highlight (追加: ここで可視化メソッドを呼ぶ)
            # これにより、グラフに赤い線が引かれ、3Dモデルに赤い球が表示されます
            self.highlight_atom(min_idx, persistent=True)
            
        else:
            # クリックがピークから遠い場合は選択解除
            self.clear_selection()

    def on_table_click(self, row, col):
        self.highlight_atom(row)

    def highlight_atom(self, row_idx, persistent=True):
        """Highlight atoms in 3D with VDW sphere and label. Handles multi-atom peaks."""
        target_item = self.data[row_idx]
        target_ppm = target_item["ppm"]
        
        # Find all atoms with the same PPM (multiplicity)
        # Using a small tolerance for floating point comparison if needed, 
        # but Java center values are usually identical for equivalent atoms.
        matching_atoms = [item for item in self.data if abs(item["ppm"] - target_ppm) < 1e-4]
        is_multi = len(matching_atoms) > 1

        try:
            mw = self.context.get_main_window()
            plotter = mw.plotter
            
            # Remove all old highlights
            self.clear_3d_visuals()

            conf = mw.current_mol.GetConformer()
            
            # Visual settings
            color = "red" if persistent else "orange"
            opacity = 0.5 if persistent else 0.4
            scale_factor = 1.4
            
            for item in matching_atoms:
                atom_idx = item["idx"]
                ppm = item["ppm"]
                symbol = item["atom"]
                
                pos = conf.GetAtomPosition(atom_idx)
                point = (pos.x, pos.y, pos.z)
                
                # 1. Sphere Highlight
                # Use RDKit VdW radius scaled by 0.3 as in moledit core/reference
                radius = PTABLE.GetRvdw(symbol) * 0.3 * scale_factor
                sphere = pv.Sphere(radius=radius, center=point)
                
                actor = plotter.add_mesh(
                    sphere,
                    color=color,
                    opacity=opacity,
                    name=f"nmr_highlight_{atom_idx}",
                    pickable=False
                )
                self._highlight_actors[atom_idx] = actor
                
                # 2. Add Label (Centered on atom)
                label_name = f"nmr_label_{atom_idx}"
                label_actor = plotter.add_point_labels(
                    [point],
                    [f"{symbol}{atom_idx}\n{ppm:.2f}"],
                    font_size=12 if not is_multi else 14,
                    text_color="white" if persistent else "yellow",
                    point_size=0,
                    always_visible=True,
                    bold=True,
                    name=label_name
                )
                self._label_actors[atom_idx] = label_actor
            
            if persistent:
                if is_multi:
                    self.status_label.setText(f"Selected Equiv Peak: {len(matching_atoms)} atoms at {target_ppm:.2f} ppm")
                else:
                    self.status_label.setText(f"Selected {target_item['atom']}{target_item['idx']}: {target_ppm:.2f} ppm")
            
            plotter.render()
            
            # Update graph highlight (persistent red line)
            if persistent:
                self._persistent_ppm = target_ppm
                self._update_graph_highlight(target_ppm)
            
        except Exception as e:
            print(f"Highlight failed: {e}")

    def _update_graph_highlight(self, ppm, is_hover=False):
        """Draw highlight on the graph."""
        ax = self.figure.axes[0]
        
        # Cleanup
        if hasattr(self, "_hover_line"):
            try: self._hover_line.remove()
            except: pass
            del self._hover_line
            
        if not is_hover:
            # Persistent highlight update
            if hasattr(self, "_graph_line"):
                try: self._graph_line.remove()
                except: pass
            
            if ppm is not None:
                self._graph_line = ax.axvline(ppm, color='red', linestyle='-', alpha=0.8, linewidth=2)
            self.canvas.draw()
        else:
            # Hover highlight
            if ppm is not None:
                self._hover_line = ax.axvline(ppm, color='orange', linestyle='--', alpha=0.5, linewidth=2)
            self.canvas.draw()

    def _sync_from_3d(self):
        """Sync from 3D selection to table."""
        mw = self.context.get_main_window()
        if not hasattr(mw, 'selected_atoms_3d'): return
        
        current_sel = set(mw.selected_atoms_3d)
        if current_sel == self._last_selected: return
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
            if atom_to_select is not None: break
        
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
            <li><b>NMRShiftDB2</b>: Java backend for shifts prediction.<br>
            <a href="https://nmrshiftdb.nmr.uni-koeln.de/">https://nmrshiftdb.nmr.uni-koeln.de/</a></li>
            
            <li><b>CDK (Chemistry Development Kit)</b>: Molecular processing in Java.<br>
            <a href="https://cdk.github.io/">https://cdk.github.io/</a></li>
            
            <li><b>RDKit</b>: Molecule handling and 3D coordinate generation.<br>
            <a href="https://www.rdkit.org/">https://www.rdkit.org/</a></li>
            
            <li><b>Matplotlib</b>: NMR spectrum visualization.<br>
            <a href="https://matplotlib.org/">https://matplotlib.org/</a></li>
            
            <li><b>PyVista</b>: 3D highlighting and visualization.<br>
            <a href="https://www.pyvista.org/">https://www.pyvista.org/</a></li>
        </ul>
        <p>Author: {PLUGIN_AUTHOR}<br>Version: {PLUGIN_VERSION}</p>
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
        """Remove all NMR related actors from plotter."""
        try:
            mw = self.context.get_main_window()
            plotter = mw.plotter
            
            # Remove spheres
            for atom_idx in list(self._highlight_actors.keys()):
                plotter.remove_actor(f"nmr_highlight_{atom_idx}")
            
            # Remove labels
            for atom_idx in list(self._label_actors.keys()):
                plotter.remove_actor(f"nmr_label_{atom_idx}")
            
            # Legacy cleanup
            plotter.remove_actor("nmr_highlight")
            
            self._highlight_actors.clear()
            self._label_actors.clear()
            plotter.render()
        except:
            pass

    def export_csv(self):
        """Export table data to CSV."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "nmr_prediction.csv", "CSV Files (*.csv)"
        )
        
        if not filename:
            return

        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # ヘッダー書き込み
                writer.writerow(["Atom ID", "Type", "Shift (ppm)"])
                
                # データ書き込み (self.data は計算結果のリスト)
                for item in self.data:
                    writer.writerow([
                        item["idx"], 
                        item["atom"], 
                        f"{item['ppm']:.2f}"
                    ])
            
            QMessageBox.information(self, "Success", f"Exported successfully to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save file:\n{str(e)}")

    def closeEvent(self, event):
        """Cleanup on close."""
        self.clear_3d_visuals()
        self.sel_timer.stop()
        
        # Clear reference on main window
        try:
            mw = self.context.get_main_window()
            if hasattr(mw, "nmr_result_dialog") and mw.nmr_result_dialog is self:
                mw.nmr_result_dialog = None
        except:
            pass
            
        super().closeEvent(event)

# --- 3. Main Logic ---

def run_prediction(context):
    """Main function triggered from the menu."""
    mw = context.get_main_window()
    mol = context.current_molecule # Using modern property access
    
    if not mol or mol.GetNumAtoms() == 0:
        QMessageBox.warning(mw, "Warning", "Please draw or load a molecule first.")
        return

    # 1. Select Nucleus
    nucleus, ok = ask_nucleus(mw)
    if not ok:
        return

    # 2. Setup Progress Dialog
    plugin_dir = Path(__file__).parent
    progress = QProgressDialog(f"Running {nucleus} NMR Prediction (nmrshiftdb2)...", "Cancel", 0, 0, mw)
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0) # Show immediately
    progress.show()

    # 3. Start Worker Thread
    # We need to keep a reference to the worker to prevent garbage collection
    mw.nmr_worker = PredictorWorker(mol, nucleus, plugin_dir)
    
    def on_success(result):
        progress.cancel()
        
        # Singleton behavior: check if dialog already exists
        if hasattr(mw, "nmr_result_dialog") and mw.nmr_result_dialog:
            try:
                mw.nmr_result_dialog.close()
            except:
                pass
        
        # Show Result Dialog (Modeless)
        mw.nmr_result_dialog = ResultDialog(mw, result, context)
        mw.nmr_result_dialog.show()
        mw.nmr_result_dialog.raise_()
        mw.nmr_result_dialog.activateWindow()
        
        # Cleanup worker reference
        mw.nmr_worker = None

    def on_error(msg):
        progress.cancel()
        QMessageBox.critical(mw, "Prediction Error", msg)
        mw.nmr_worker = None

    mw.nmr_worker.finished_signal.connect(on_success)
    mw.nmr_worker.error_signal.connect(on_error)
    mw.nmr_worker.start()

def ask_nucleus(parent):
    """Simple dialog to choose nucleus."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("Select Nucleus")
    dialog.resize(250, 120)
    layout = QVBoxLayout()
    
    layout.addWidget(QLabel("Select Nucleus type:"))
    
    combo = QComboBox()
    combo.addItems(["1H", "13C"])
    layout.addWidget(combo)
    
    btn = QPushButton("Predict")
    btn.clicked.connect(dialog.accept)
    layout.addWidget(btn)
    
    dialog.setLayout(layout)
    
    if dialog.exec():
        return combo.currentText(), True
    return None, False

# --- 4. Initialization Hook (Plugin Development Manual Section 2) ---
def initialize(context):
    """Entry point called by MoleditPy on startup."""
    
    # Register directly in the Analysis menu (no sub-menu since it's a single item)
    context.add_menu_action(
        "Analysis/NMR Prediction (nmrshiftdb2)", 
        lambda: run_prediction(context)
    )
    