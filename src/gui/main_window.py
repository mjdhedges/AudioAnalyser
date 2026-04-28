"""Main window for the Audio Analyser desktop GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QAction, QIcon, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.gui.about import APP_NAME, about_html
from src.gui.commands import (
    AnalysisCommandOptions,
    RenderCommandOptions,
    analysis_output_dir,
    build_analysis_command,
    build_render_command,
    render_output_dir,
    resolve_render_results_path,
)
from src.gui.progress import FileProgress, ProgressTracker

PROGRESS_PREFIX = "AA_PROGRESS "


class MainWindow(QMainWindow):
    """Small desktop wrapper around the existing analysis/render CLIs."""

    def __init__(self, icon: Optional[QIcon] = None) -> None:
        """Initialize the GUI."""
        super().__init__()
        self.setWindowTitle("Audio Analyser")
        if icon is not None:
            self.setWindowIcon(icon)
        self.resize(900, 650)
        self._build_menu()

        self.process: Optional[QProcess] = None
        self.current_stage = ""
        self.completed_files = 0
        self.total_files = 0
        self.process_text_buffer = ""
        self.progress_tracker = ProgressTracker()
        self.file_rows: Dict[str, int] = {}

        self.input_path = QLineEdit()
        self.project_dir = QLineEdit("AudioAnalyserProject")

        self.batch_workers = QSpinBox()
        self.batch_workers.setRange(1, 64)
        self.batch_workers.setValue(2)
        self.batch_workers.setToolTip(
            "Number of tracks to analyze at the same time. Use 1 for safest "
            "memory use; increase only when the machine has enough RAM."
        )
        self.batch_workers_help = QLabel(
            "Concurrent tracks. 1 = safest; higher values are faster but use more RAM."
        )
        self.batch_workers_help.setWordWrap(True)

        self.max_memory_gb = QDoubleSpinBox()
        self.max_memory_gb.setRange(0.1, 1024.0)
        self.max_memory_gb.setDecimals(1)
        self.max_memory_gb.setSingleStep(0.5)
        self.max_memory_gb.setValue(8.0)
        self.max_memory_gb.setSuffix(" GB")
        self.max_memory_gb.setToolTip(
            "Per-track memory estimate used to choose octave processing mode and "
            "schedule batch work. Actual RAM can be higher for multi-channel decode, "
            "resampling, and export."
        )
        self.max_memory_help = QLabel(
            "Per-track memory estimate, not a hard cap. Higher can be faster; lower "
            "switches large tracks to lower-memory processing sooner."
        )
        self.max_memory_help.setWordWrap(True)

        self.render_after_analysis = QCheckBox("Render graphs after analysis")
        self.render_after_analysis.setChecked(True)
        self.generate_report = QCheckBox("Generate Markdown report")
        self.generate_report.setChecked(True)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignLeft)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0/0 files")

        self.files_table = QTableWidget(0, 5)
        self.files_table.setHorizontalHeaderLabels(
            ["#", "File", "Status", "Time", "Details"]
        )
        self.files_table.verticalHeader().setVisible(False)
        self.files_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.files_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.files_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QTextEdit.NoWrap)

        self.start_button = QPushButton("Start")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)

        self._build_layout()
        self._connect_signals()

    def _build_menu(self) -> None:
        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("&About Audio Analyser", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_layout(self) -> None:
        form = QFormLayout()
        form.addRow("Input", self._path_row(self.input_path, "File...", "Folder..."))
        form.addRow("Project folder", self._directory_row(self.project_dir))
        form.addRow("Batch workers", self.batch_workers)
        form.addRow("", self.batch_workers_help)
        form.addRow("Memory estimate per track", self.max_memory_gb)
        form.addRow("", self.max_memory_help)
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
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.files_table, stretch=1)
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

    def _show_about(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"About {APP_NAME}")
        dialog.resize(520, 360)

        text = QTextBrowser(dialog)
        text.setOpenExternalLinks(True)
        text.setHtml(about_html())

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=dialog)
        buttons.rejected.connect(dialog.reject)

        layout = QVBoxLayout(dialog)
        layout.addWidget(text)
        layout.addWidget(buttons)
        dialog.exec()

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
            progress_json=True,
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
        self.completed_files = 0
        self.total_files = 0
        self.process_text_buffer = ""
        if stage == "analysis":
            self.progress_tracker.reset()
            self.file_rows.clear()
            self.files_table.setRowCount(0)
            self._update_progress_bar()
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
        self._handle_process_text(data.decode(errors="replace"))

    def _handle_process_text(self, text: str) -> None:
        self.process_text_buffer += text
        lines = self.process_text_buffer.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self.process_text_buffer = lines.pop()
        else:
            self.process_text_buffer = ""

        for raw_line in lines:
            line = raw_line.rstrip("\r\n")
            if line.startswith(PROGRESS_PREFIX):
                self._handle_progress_event(line[len(PROGRESS_PREFIX) :])
            else:
                self._append_log(line)

    def _handle_progress_event(self, payload: str) -> None:
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            self._append_log(f"Invalid progress event: {payload}")
            return

        event_name = event.get("event")
        file_progress = self.progress_tracker.handle_event(event)
        if file_progress is not None:
            self._upsert_file_row(file_progress)
        self._update_progress_bar()

        if event_name == "analysis_started":
            self.status_label.setText(
                f"Analysis started: {self.progress_tracker.total_files} file(s)"
            )
        elif event_name == "file_started":
            self.status_label.setText(
                f"Running {event.get('index')}/{event.get('total')}: {event.get('name')}"
            )
        elif event_name == "file_finished":
            status = "finished" if event.get("success") else "failed"
            self.status_label.setText(
                f"{self.progress_tracker.completed_files}/"
                f"{self.progress_tracker.total_files} {status}: {event.get('name')}"
            )
            if event.get("error"):
                self._append_log(f"Details for {event.get('name')}: {event.get('error')}")
        elif event_name == "analysis_finished":
            self.status_label.setText(
                "Analysis complete: "
                f"{event.get('successful', 0)} OK, {event.get('failed', 0)} failed"
            )

    def _upsert_file_row(self, file_progress: FileProgress) -> None:
        row = self.file_rows.get(file_progress.path)
        if row is None:
            row = self.files_table.rowCount()
            self.files_table.insertRow(row)
            self.file_rows[file_progress.path] = row

        elapsed = (
            f"{file_progress.elapsed_seconds:.2f}s"
            if file_progress.elapsed_seconds is not None
            else ""
        )
        details = file_progress.error or ""
        values = [
            str(file_progress.index),
            file_progress.name,
            file_progress.status,
            elapsed,
            details,
        ]
        for column, value in enumerate(values):
            self.files_table.setItem(row, column, QTableWidgetItem(value))

    def _update_progress_bar(self) -> None:
        total = max(self.progress_tracker.total_files, 0)
        completed = max(self.progress_tracker.completed_files, 0)
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(min(completed, total))
        self.progress_bar.setFormat(f"{completed}/{total} files")

    def _process_finished(
        self,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        stage = self.current_stage
        success = exit_code == 0 and exit_status == QProcess.NormalExit
        if self.process_text_buffer:
            self._handle_process_text("\n")
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
