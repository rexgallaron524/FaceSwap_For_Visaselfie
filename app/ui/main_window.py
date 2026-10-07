"""Milestone 0 desktop shell. Controls stay inactive until their stages exist."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.setWindowTitle("FaceLive")
        self.resize(1120, 740)
        self.setMinimumSize(900, 620)
        self.setStyleSheet("""
            QMainWindow, QWidget#shell { background: #101820; color: #edf3f7; }
            QLabel { color: #edf3f7; }
            QLabel#title { font-size: 28px; font-weight: 600; }
            QLabel#subtitle, QLabel#hint { color: #aabac7; }
            QGroupBox { color: #edf3f7; border: 1px solid #354550;
                        border-radius: 8px; margin-top: 16px; padding: 16px; }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 5px; }
            QPushButton, QComboBox { padding: 9px; border: 1px solid #4a5c69;
                                    border-radius: 5px; background: #23313c; }
            QPushButton:disabled, QComboBox:disabled, QCheckBox:disabled { color: #aabac7; }
            QFrame#preview { background: #17232d; border: 1px solid #354550;
                             border-radius: 10px; }
            QLabel#previewTitle { font-size: 22px; font-weight: 600; }
            QLabel#badge { color: #85dbc9; font-weight: 600; }
            QStatusBar { background: #17232d; color: #aabac7; }
            QMenuBar, QMenu { background: #17232d; color: #edf3f7; }
            QMenuBar::item:selected, QMenu::item:selected { background: #354550; }
        """)
        shell = QWidget()
        shell.setObjectName("shell")
        self.setCentralWidget(shell)
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(18)

        title = QLabel("FaceLive")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel("Your reference appearance. Your live surroundings.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(22)
        controls = QVBoxLayout()
        inputs = QGroupBox("Sources")
        inputs.setFixedWidth(270)
        source_layout = QVBoxLayout(inputs)
        source_layout.addWidget(QLabel("Physical camera"))
        self.camera_selector = QComboBox()
        self.camera_selector.addItem("Camera input unavailable")
        self.camera_selector.setEnabled(False)
        self.camera_selector.setToolTip("Camera capture will be added in a later milestone.")
        source_layout.addWidget(self.camera_selector)
        source_layout.addSpacing(12)
        source_layout.addWidget(QLabel("Reference library"))
        self.load_references = QPushButton("Load references…")
        self.load_references.setEnabled(False)
        self.load_references.setToolTip("Reference enrollment is not implemented yet.")
        source_layout.addWidget(self.load_references)
        source_layout.addWidget(QLabel("No reference library loaded"))
        controls.addWidget(inputs)

        output = QGroupBox("Output")
        output_layout = QVBoxLayout(output)
        self.replacement_toggle = QCheckBox("Enable face replacement")
        self.replacement_toggle.setEnabled(False)
        self.replacement_toggle.setToolTip("Face processing is not implemented yet.")
        output_layout.addWidget(self.replacement_toggle)
        self.virtual_camera_button = QPushButton("Start virtual camera")
        self.virtual_camera_button.setEnabled(False)
        self.virtual_camera_button.setToolTip("The native virtual camera is a later milestone.")
        output_layout.addWidget(self.virtual_camera_button)
        output_layout.addWidget(
            QLabel(f"Target: {config.video.width} × {config.video.height} · {config.video.fps} FPS")
        )
        controls.addWidget(output)
        controls.addStretch()
        content.addLayout(controls)

        preview_column = QVBoxLayout()
        preview_heading = QLabel("OUTPUT PREVIEW")
        preview_heading.setObjectName("badge")
        preview_column.addWidget(preview_heading)
        preview = QFrame()
        preview.setObjectName("preview")
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(28, 28, 28, 28)
        preview_layout.addStretch()
        self.preview_title = QLabel("Preview is idle")
        self.preview_title.setObjectName("previewTitle")
        self.preview_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_title)
        description = QLabel(
            "Camera capture and face processing will be available\nin later milestones."
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        description.setObjectName("hint")
        preview_layout.addWidget(description)
        preview_layout.addStretch()
        preview_column.addWidget(preview, 1)
        metrics = QGroupBox("Session")
        metrics_layout = QFormLayout(metrics)
        metrics_layout.addRow("Processing", QLabel("Idle"))
        metrics_layout.addRow("FPS / latency / dropped frames", QLabel("— / — / —"))
        metrics_layout.addRow("Virtual camera", QLabel("Unavailable"))
        if config.debug:
            metrics_layout.addRow("Diagnostics", QLabel("Enabled · waiting for pipeline"))
        preview_column.addWidget(metrics)
        content.addLayout(preview_column, 1)
        layout.addLayout(content, 1)

        footer = QLabel("Milestone 0 · Application shell")
        footer.setObjectName("hint")
        layout.addWidget(footer)
        self.statusBar().showMessage("Idle — no camera is open and no video is being sent.")
        quit_action = QAction("Exit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        self.menuBar().addMenu("File").addAction(quit_action)
