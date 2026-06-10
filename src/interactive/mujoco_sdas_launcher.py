"""PyQt launcher for no-damper MuJoCo SDAS simulations."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from src.simulation.mujoco_sdas import MuJoCoSDASSimulator, VideoConfig, load_ohd_simulation
from src.simulation.mujoco_sdas_presets import OBJECT_PRESETS, apply_object_override


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SimulationWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)
    status = QtCore.pyqtSignal(str)

    def __init__(self, options: dict[str, Any]):
        super().__init__()
        self.options = options

    def run(self) -> None:
        try:
            self.status.emit("Loading .ohd simulation definition...")
            config = load_ohd_simulation(self.options["ohd_file"])
            if self.options.get("output_dir"):
                config.output_dir = Path(self.options["output_dir"]).resolve()
            config.duration = float(self.options["duration"])
            steps = int(self.options["steps"])
            if steps <= 0:
                raise ValueError("Analysis steps must be positive.")
            config.dt = config.duration / float(steps)

            preset = self.options["object_preset"]
            position = tuple(float(v) for v in self.options["object_position"])
            apply_object_override(
                config,
                preset,
                position=position,  # type: ignore[arg-type]
                size=self.options.get("object_size"),
                scale=self.options.get("object_scale"),
                mass=self.options.get("object_mass"),
                fixed=bool(self.options.get("fixed_object")),
                force_contact=bool(self.options.get("force_contact")),
            )
            if self.options.get("video_enabled"):
                config.video = VideoConfig(
                    enabled=True,
                    fps=float(self.options["video_fps"]),
                    keep_frames=bool(self.options.get("keep_video_frames")),
                    format="gif",
                )

            self.status.emit("Running MuJoCo numerical simulation...")
            result = MuJoCoSDASSimulator(config).run()
            self.finished_ok.emit(result.summary)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MuJoCoSDASLauncher(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker: Optional[SimulationWorker] = None
        self.current_summary: Optional[dict[str, Any]] = None
        self.movie: Optional[QtGui.QMovie] = None
        self._init_ui()

    def _init_ui(self) -> None:
        self.setWindowTitle("MuJoCo SDAS Numerical Simulation Launcher")
        self.setMinimumSize(1180, 760)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter)

        controls = QtWidgets.QScrollArea()
        controls.setWidgetResizable(True)
        controls_widget = QtWidgets.QWidget()
        controls.setWidget(controls_widget)
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(10)

        controls_layout.addWidget(self._build_file_group())
        controls_layout.addWidget(self._build_time_group())
        controls_layout.addWidget(self._build_object_group())
        controls_layout.addWidget(self._build_video_group())
        controls_layout.addStretch()

        self.run_button = QtWidgets.QPushButton("Run MuJoCo Simulation")
        self.run_button.setMinimumHeight(36)
        self.run_button.clicked.connect(self._run_simulation)
        controls_layout.addWidget(self.run_button)

        splitter.addWidget(controls)
        splitter.addWidget(self._build_result_panel())
        splitter.setSizes([420, 760])

        self.statusBar().showMessage("Ready")
        self._update_dt_label()

    def _build_file_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Simulation File")
        layout = QtWidgets.QGridLayout(group)

        self.ohd_edit = QtWidgets.QLineEdit(str(PROJECT_ROOT / "models" / "ohd test" / "mujoco_sdas_grasp_scanned_mug.ohd"))
        browse_ohd = QtWidgets.QPushButton("Browse...")
        browse_ohd.clicked.connect(self._browse_ohd)
        layout.addWidget(QtWidgets.QLabel(".ohd file"), 0, 0)
        layout.addWidget(self.ohd_edit, 0, 1)
        layout.addWidget(browse_ohd, 0, 2)

        self.output_edit = QtWidgets.QLineEdit(str(PROJECT_ROOT / "outputs" / "mujoco_sdas" / "gui_run"))
        browse_out = QtWidgets.QPushButton("Browse...")
        browse_out.clicked.connect(self._browse_output)
        layout.addWidget(QtWidgets.QLabel("Output dir"), 1, 0)
        layout.addWidget(self.output_edit, 1, 1)
        layout.addWidget(browse_out, 1, 2)
        return group

    def _build_time_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Analysis Steps")
        layout = QtWidgets.QGridLayout(group)

        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(0.01, 120.0)
        self.duration_spin.setDecimals(3)
        self.duration_spin.setSingleStep(0.1)
        self.duration_spin.setValue(1.4)
        self.duration_spin.valueChanged.connect(self._update_dt_label)

        self.steps_spin = QtWidgets.QSpinBox()
        self.steps_spin.setRange(1, 2_000_000)
        self.steps_spin.setSingleStep(100)
        self.steps_spin.setValue(700)
        self.steps_spin.valueChanged.connect(self._update_dt_label)

        self.dt_label = QtWidgets.QLabel()
        self.dt_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        layout.addWidget(QtWidgets.QLabel("Duration [s]"), 0, 0)
        layout.addWidget(self.duration_spin, 0, 1)
        layout.addWidget(QtWidgets.QLabel("Steps"), 1, 0)
        layout.addWidget(self.steps_spin, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Computed dt"), 2, 0)
        layout.addWidget(self.dt_label, 2, 1)
        return group

    def _build_object_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Grasp Object and Contact")
        layout = QtWidgets.QGridLayout(group)

        self.object_combo = QtWidgets.QComboBox()
        labels = {
            "keep": "Keep .ohd object",
            "none": "No object",
            "box": "Box",
            "cylinder": "Cylinder",
            "sphere": "Sphere",
            "scanned_mug": "Scanned mug",
        }
        for value in OBJECT_PRESETS:
            self.object_combo.addItem(labels[value], value)
        self.object_combo.setCurrentIndex(self.object_combo.findData("keep"))

        self.contact_check = QtWidgets.QCheckBox("Enable contact")
        self.contact_check.setChecked(True)
        self.fixed_check = QtWidgets.QCheckBox("Fixed object")

        self.pos_x = self._position_spin(0.045)
        self.pos_y = self._position_spin(0.125)
        self.pos_z = self._position_spin(0.0785)
        self.size_edit = QtWidgets.QLineEdit()
        self.size_edit.setPlaceholderText("optional, e.g. 0.018 0.03")
        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(0.001, 100.0)
        self.scale_spin.setDecimals(4)
        self.scale_spin.setValue(0.45)
        self.mass_spin = QtWidgets.QDoubleSpinBox()
        self.mass_spin.setRange(0.0, 1000.0)
        self.mass_spin.setDecimals(4)
        self.mass_spin.setValue(0.5)

        layout.addWidget(QtWidgets.QLabel("Object"), 0, 0)
        layout.addWidget(self.object_combo, 0, 1, 1, 3)
        layout.addWidget(self.contact_check, 1, 1)
        layout.addWidget(self.fixed_check, 1, 2)
        layout.addWidget(QtWidgets.QLabel("Position"), 2, 0)
        layout.addWidget(self.pos_x, 2, 1)
        layout.addWidget(self.pos_y, 2, 2)
        layout.addWidget(self.pos_z, 2, 3)
        layout.addWidget(QtWidgets.QLabel("Size override"), 3, 0)
        layout.addWidget(self.size_edit, 3, 1, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Scale"), 4, 0)
        layout.addWidget(self.scale_spin, 4, 1)
        layout.addWidget(QtWidgets.QLabel("Mass"), 4, 2)
        layout.addWidget(self.mass_spin, 4, 3)
        return group

    def _build_video_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Output")
        layout = QtWidgets.QGridLayout(group)

        self.video_check = QtWidgets.QCheckBox("Export process GIF")
        self.video_check.setChecked(True)
        self.video_fps_spin = QtWidgets.QDoubleSpinBox()
        self.video_fps_spin.setRange(1.0, 60.0)
        self.video_fps_spin.setDecimals(1)
        self.video_fps_spin.setValue(12.0)
        self.keep_frames_check = QtWidgets.QCheckBox("Keep PNG frames")

        layout.addWidget(self.video_check, 0, 0, 1, 2)
        layout.addWidget(QtWidgets.QLabel("GIF fps"), 1, 0)
        layout.addWidget(self.video_fps_spin, 1, 1)
        layout.addWidget(self.keep_frames_check, 2, 0, 1, 2)
        return group

    def _build_result_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)

        actions = QtWidgets.QHBoxLayout()
        self.open_output_button = QtWidgets.QPushButton("Open Output Folder")
        self.open_output_button.clicked.connect(self._open_output_folder)
        self.open_output_button.setEnabled(False)
        actions.addWidget(self.open_output_button)
        actions.addStretch()
        layout.addLayout(actions)

        self.preview_label = QtWidgets.QLabel("Run a simulation to preview screenshots or GIF output.")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumHeight(380)
        self.preview_label.setStyleSheet("background: #202426; color: #d8dee9; border: 1px solid #3b4045;")
        self.preview_label.setScaledContents(True)
        layout.addWidget(self.preview_label, stretch=3)

        self.summary_text = QtWidgets.QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(self.summary_text, stretch=2)
        return panel

    def _position_spin(self, value: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-10.0, 10.0)
        spin.setDecimals(5)
        spin.setSingleStep(0.005)
        spin.setValue(value)
        return spin

    def _browse_ohd(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select OHD simulation file",
            str(PROJECT_ROOT / "models"),
            "OHD files (*.ohd);;All files (*.*)",
        )
        if path:
            self.ohd_edit.setText(path)

    def _browse_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            str(PROJECT_ROOT / "outputs" / "mujoco_sdas"),
        )
        if path:
            self.output_edit.setText(path)

    def _update_dt_label(self) -> None:
        dt = self.duration_spin.value() / max(1, self.steps_spin.value())
        self.dt_label.setText(f"{dt:.6f} s")

    def _parse_size(self) -> Optional[list[float]]:
        raw = self.size_edit.text().replace(",", " ").strip()
        if not raw:
            return None
        return [float(part) for part in raw.split()]

    def _collect_options(self) -> dict[str, Any]:
        ohd = Path(self.ohd_edit.text()).expanduser()
        if not ohd.exists():
            raise FileNotFoundError(f"OHD file not found: {ohd}")
        return {
            "ohd_file": str(ohd),
            "output_dir": self.output_edit.text().strip(),
            "duration": self.duration_spin.value(),
            "steps": self.steps_spin.value(),
            "object_preset": self.object_combo.currentData(),
            "object_position": [self.pos_x.value(), self.pos_y.value(), self.pos_z.value()],
            "object_size": self._parse_size(),
            "object_scale": self.scale_spin.value(),
            "object_mass": self.mass_spin.value(),
            "fixed_object": self.fixed_check.isChecked(),
            "force_contact": self.contact_check.isChecked(),
            "video_enabled": self.video_check.isChecked(),
            "video_fps": self.video_fps_spin.value(),
            "keep_video_frames": self.keep_frames_check.isChecked(),
        }

    def _run_simulation(self) -> None:
        try:
            options = self._collect_options()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Simulation Settings", str(exc))
            return
        self.run_button.setEnabled(False)
        self.open_output_button.setEnabled(False)
        self.summary_text.setPlainText("Starting simulation...\n")
        self.statusBar().showMessage("Running...")
        self.worker = SimulationWorker(options)
        self.worker.status.connect(self._append_status)
        self.worker.finished_ok.connect(self._simulation_finished)
        self.worker.failed.connect(self._simulation_failed)
        self.worker.start()

    def _append_status(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self.summary_text.append(message)

    def _simulation_finished(self, summary: dict[str, Any]) -> None:
        self.current_summary = summary
        self.run_button.setEnabled(True)
        self.open_output_button.setEnabled(True)
        self.statusBar().showMessage("Simulation complete")
        self.summary_text.setPlainText(json.dumps(summary, indent=2, ensure_ascii=False))
        self._load_preview(summary)

    def _simulation_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.statusBar().showMessage("Simulation failed")
        self.summary_text.setPlainText(message)
        QtWidgets.QMessageBox.critical(self, "Simulation Failed", message)

    def _load_preview(self, summary: dict[str, Any]) -> None:
        video = summary.get("video")
        if video and Path(video).exists():
            self.movie = QtGui.QMovie(video)
            self.preview_label.setMovie(self.movie)
            self.movie.start()
            return
        screenshots = summary.get("screenshots") or []
        if screenshots:
            path = Path(screenshots[-1])
            if path.exists():
                pix = QtGui.QPixmap(str(path))
                self.preview_label.setPixmap(pix)
                return
        self.preview_label.setText("No preview image was generated.")

    def _open_output_folder(self) -> None:
        if not self.current_summary:
            return
        log_csv = self.current_summary.get("log_csv")
        folder = Path(log_csv).parent if log_csv else Path(self.output_edit.text())
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))


def run_gui() -> None:
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MuJoCoSDASLauncher()
    win.show()
    sys.exit(app.exec_())
