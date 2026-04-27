"""Main window for the Audio Analyser desktop GUI."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.gui.commands import (
    AnalysisCommandOptions,
    RenderCommandOptions,
    analysis_output_dir,
    build_analysis_command,
    build_render_command,
    render_output_dir,
    resolve_render_results_path,
)


class MainWindow(QMainWindow):
    """Small desktop wrapper around the existing analysis/render CLIs."""

    def __init__(self) -> None:
        """Initialize the GUI."""
        super().__init__()
        self.setWindowTitle("Audio Analyser")
        self.resize(900, 650)

        self.process: Optional[QProcess] = None
        self.current_stage = ""

        self.input_path = QLineEdit()
        self.project_dir = QLineEdit("AudioAnalyserProject")

        self.batch_workers = QSpinBox()
        self.batch_workers.setRange(1, 64)
        self.batch_workers.setValue(2)

        self.max_memory_gb = QDoubleSpinBox()
        self.max_memory_gb.setRange(0.1, 1024.0)
        self.max_memory_gb.setDecimals(1)
        self.max_memory_gb.setSingleStep(0.5)
        self.max_memory_gb.setValue(8.0)
        self.max_memory_gb.setSuffix(" GB")

        self.render_after_analysis = QCheckBox("Render graphs after analysis")
        self.render_after_analysis.setChecked(True)
        self.generate_report = QCheckBox("Generate Markdown report")
        self.generate_report.setChecked(True)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignLeft)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QTextEdit.NoWrap)

        self.start_button = QPushButton("Start")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)

        self._build_layout()
        self._connect_signals()

    def _build_layout(self) -> None:
        form = QFormLayout()
        form.addRow("Input", self._path_row(self.input_path, "File...", "Folder..."))
        form.addRow("Project folder", self._directory_row(self.project_dir))
        form.addRow("Batch workers", self.batch_workers)
        form.addRow("Max memory", self.max_memory_gb)
        form.addRow("", self.render_after_analysis)
        form.addRow("", self.generate_report)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.cancel_button)
        button_row.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(button_row)
        layout.addWidget(self.status_label)
        layout.addWidget(self.log_output, stretch=1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _path_row(
        self,
        line_edit: QLineEdit,
        file_label: str,
        folder_label: str,
    ) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        file_button = QPushButton(file_label)
        folder_button = QPushButton(folder_label)
        file_button.clicked.connect(self._choose_input_file)
        folder_button.clicked.connect(self._choose_input_folder)
        layout.addWidget(line_edit, stretch=1)
        layout.addWidget(file_button)
        layout.addWidget(folder_button)
        return row

    def _directory_row(self, line_edit: QLineEdit) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse...")
        button.clicked.connect(lambda: self._choose_directory(line_edit))
        layout.addWidget(line_edit, stretch=1)
        layout.addWidget(button)
        return row

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self.start_analysis)
        self.cancel_button.clicked.connect(self.cancel_process)
        self.render_after_analysis.toggled.connect(self.generate_report.setEnabled)

    def _choose_input_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            "",
            "Audio/Video Files (*.wav *.flac *.mp3 *.m4a *.aiff *.mkv);;All Files (*)",
        )
        if path:
            self.input_path.setText(path)

    def _choose_input_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select input folder")
        if path:
            self.input_path.setText(path)

    def _choose_directory(self, line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            line_edit.setText(path)

    def start_analysis(self) -> None:
        """Start the analysis subprocess."""
        if not self._validate_inputs():
            return

        options = AnalysisCommandOptions(
            input_path=Path(self.input_path.text()),
            project_dir=Path(self.project_dir.text()),
            batch_workers=int(self.batch_workers.value()),
            max_memory_gb=float(self.max_memory_gb.value()),
        )
        self._start_command(build_analysis_command(options), "analysis")

    def cancel_process(self) -> None:
        """Cancel the currently running subprocess."""
        if self.process is None:
            return
        self._append_log("Cancelling current process...")
        self.process.terminate()
        self.cancel_button.setEnabled(False)

    def _validate_inputs(self) -> bool:
        input_path = Path(self.input_path.text())
        project_dir = self.project_dir.text().strip()

        if not input_path.exists():
            QMessageBox.warning(
                self, "Missing input", "Select an existing file or folder."
            )
            return False
        if not project_dir:
            QMessageBox.warning(
                self, "Missing project folder", "Select a project folder."
            )
            return False
        return True

    def _start_command(self, command: List[str], stage: str) -> None:
        self.current_stage = stage
        self.status_label.setText(f"Running {stage}...")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self._append_log("")
        if stage == "analysis":
            project_dir = Path(self.project_dir.text())
            self._append_log(f"Project folder: {project_dir}")
            self._append_log(f"Analysis output: {analysis_output_dir(project_dir)}")
            if self.render_after_analysis.isChecked():
                self._append_log(f"Render output: {render_output_dir(project_dir)}")
                self._append_log(f"Markdown report: {self.generate_report.isChecked()}")
        self._append_log(f"$ {' '.join(command)}")

        self.process = QProcess(self)
        self.process.setProgram(command[0])
        self.process.setArguments(command[1:])
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.start()

    def _read_process_output(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput())
        self._append_log(data.decode(errors="replace").rstrip())

    def _process_finished(
        self,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        stage = self.current_stage
        success = exit_code == 0 and exit_status == QProcess.NormalExit
        self._append_log(f"{stage.capitalize()} finished with exit code {exit_code}.")

        if success and stage == "analysis" and self.render_after_analysis.isChecked():
            project_dir = Path(self.project_dir.text())
            analysis_dir = analysis_output_dir(project_dir)
            render_options = RenderCommandOptions(
                results_dir=resolve_render_results_path(
                    input_path=Path(self.input_path.text()),
                    analysis_output_dir=analysis_dir,
                ),
                output_dir=render_output_dir(project_dir),
                reports=self.generate_report.isChecked(),
            )
            self._start_command(build_render_command(render_options), "render")
            return

        self.process = None
        self.current_stage = ""
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Finished" if success else "Failed")

    def _append_log(self, text: str) -> None:
        if text:
            self.log_output.append(text)
        self.log_output.moveCursor(QTextCursor.End)
