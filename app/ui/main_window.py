"""FaceLive desktop shell with physical-camera preview."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from math import hypot

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QTransform,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.camera import OpenCVCameraSource
from app.camera.metrics import CaptureMetrics
from app.camera.protocol import CameraSource
from app.compositing import AlphaFaceCompositor
from app.compositing.protocol import Compositor
from app.config import AppConfig
from app.pipeline.types import (
    CameraDevice,
    FaceState,
    FrameFormat,
    ReferenceWeight,
    StageError,
    TrackingResult,
    TrackingStatus,
    VideoFrame,
)
from app.reference import (
    PoseSpaceReferenceSelector,
    ReferenceLibrarySession,
    ReferenceLibraryStore,
)
from app.reference.protocol import ReferenceSelector
from app.rendering import GeometricFaceRenderer
from app.rendering.protocol import FaceRenderer
from app.stabilization import TemporalStabilizer
from app.stabilization.protocol import Stabilizer
from app.tracking import MediaPipeFaceTracker
from app.tracking.protocol import FaceTracker
from app.ui.reference_dialog import ReferenceEnrollmentDialog
from app.ui.reference_weights import ReferenceWeightsWidget


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: AppConfig,
        camera_source_factory: Callable[[], CameraSource] = OpenCVCameraSource,
        face_tracker_factory: Callable[[], FaceTracker] = MediaPipeFaceTracker,
        reference_selector: ReferenceSelector | None = None,
        face_renderer_factory: Callable[[], FaceRenderer] | None = None,
        compositor: Compositor | None = None,
        stabilizer: Stabilizer | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._source = camera_source_factory()
        self._tracker = face_tracker_factory()
        self._reference_selector = reference_selector or PoseSpaceReferenceSelector()
        self._renderer = (face_renderer_factory or GeometricFaceRenderer)()
        self._compositor = compositor or AlphaFaceCompositor()
        self._stabilizer = stabilizer or TemporalStabilizer()
        self._devices: tuple[CameraDevice, ...] = ()
        self._capturing = False
        self._opening = False
        self._last_image: QImage | None = None
        self._last_processed_image: QImage | None = None
        self._last_face_state: FaceState | None = None
        self._last_raw_face_state: FaceState | None = None
        self._last_reference_weights: tuple[ReferenceWeight, ...] = ()
        self._current_smoothing_ms = 0.0
        self._pending_tracking_frame: tuple[VideoFrame, QImage, int] | None = None
        self._tracking_drops = 0
        self._tracking_failed = False
        self._rendering_failed = False
        self._layout_mode: str | None = None
        self._metrics = CaptureMetrics()
        self._logger = logging.getLogger("facelive.ui")
        self._reference_library = ReferenceLibraryStore()
        self._reference_session = ReferenceLibrarySession(self._reference_library)
        restored = self._reference_session.restore()
        if restored:
            self._logger.info(
                "Restored reference library from %s", self._reference_session.active_path
            )
        elif self._reference_session.last_error:
            self._logger.warning(self._reference_session.last_error)
        self._camera_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-open")
        self._open_future: Future[tuple[FrameFormat, str | None, str | None]] | None = None

        self.setWindowTitle("FaceLive")
        self.resize(1440, 900)
        self.setMinimumSize(720, 620)
        window_font = QFont("Segoe UI Variable")
        window_font.setPointSizeF(10.5)
        self.setFont(window_font)
        self.setStyleSheet("""
            * {
                font-family: "Segoe UI Variable", "Segoe UI";
            }
            QMainWindow, QDialog, QMessageBox, QWidget#shell, QWidget#referenceGrid {
                background: #0b1017;
                color: #f4f7fb;
            }
            QLabel { color: #f4f7fb; background: transparent; }
            QLabel#title {
                color: #ffffff;
                font-size: 22pt;
                font-weight: 700;
            }
            QLabel#subtitle {
                color: #c4ced9;
                font-size: 11pt;
            }
            QLabel#hint { color: #b5c0cd; }
            QLabel#fieldLabel, QLabel#metricLabel {
                color: #c9d3de;
                font-weight: 500;
            }
            QLabel#weightLabel { color: #c8d5e5; font-size: 9pt; }
            QLabel#weightValue {
                color: #f7fbff;
                font-size: 9pt;
                font-weight: 600;
            }
            QProgressBar#referenceWeight {
                background: #111a25;
                border: 1px solid #34465b;
                border-radius: 5px;
            }
            QProgressBar#referenceWeight::chunk {
                background: #2688e8;
                border-radius: 4px;
            }
            QLabel#metricValue {
                color: #ffffff;
                font-size: 11pt;
                font-weight: 700;
            }
            QLabel#stateValue {
                color: #d6dee8;
                font-size: 11pt;
                font-weight: 700;
            }
            QLabel#stateValue[state="running"] { color: #5ee6a8; }
            QLabel#stateValue[state="opening"] { color: #7cc7ff; }
            QLabel#stateValue[state="no-face"] { color: #f2bb60; }
            QLabel#stateValue[state="error"] { color: #ff858d; }
            QGroupBox {
                color: #f4f7fb;
                background: #151c25;
                border: 1px solid #344252;
                border-radius: 12px;
                margin-top: 14px;
                padding: 0;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 7px;
                color: #eaf0f6;
                background: #151c25;
            }
            QPushButton {
                min-height: 40px;
                padding: 0 15px;
                color: #f7faff;
                background: #222d39;
                border: 1px solid #536477;
                border-radius: 8px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #2c3947;
                border-color: #8294a8;
            }
            QPushButton:pressed { background: #19232e; }
            QPushButton:focus { border: 2px solid #79b8ff; }
            QPushButton#primaryButton {
                color: #ffffff;
                background: #1677e8;
                border-color: #4398f4;
            }
            QPushButton#primaryButton:hover {
                background: #2588f5;
                border-color: #76b6ff;
            }
            QPushButton#primaryButton:pressed { background: #0f63c6; }
            QPushButton#primaryButton:disabled {
                color: #8996a5;
                background: #1a222c;
                border-color: #313d4b;
            }
            QPushButton:disabled {
                color: #8996a5;
                background: #1a222c;
                border-color: #313d4b;
            }
            QComboBox {
                min-height: 40px;
                padding: 0 38px 0 12px;
                color: #ffffff;
                background: #202a35;
                border: 1px solid #56687b;
                border-radius: 8px;
                selection-background-color: #1677e8;
                selection-color: #ffffff;
            }
            QComboBox:hover { border-color: #8194aa; }
            QComboBox:focus { border: 2px solid #79b8ff; }
            QComboBox:disabled {
                color: #8996a5;
                background: #1a222c;
                border-color: #313d4b;
            }
            QComboBox::drop-down { border: 0; width: 34px; }
            QComboBox QAbstractItemView {
                color: #f4f7fb;
                background: #202a35;
                border: 1px solid #536477;
                selection-background-color: #1677e8;
                selection-color: #ffffff;
                outline: 0;
            }
            QCheckBox {
                color: #edf2f7;
                spacing: 9px;
                min-height: 28px;
            }
            QCheckBox:disabled { color: #8996a5; }
            QCheckBox:focus { color: #ffffff; }
            QFrame#preview {
                background: #090d12;
                border: 1px solid #3c4b5d;
                border-radius: 12px;
            }
            QLabel#previewImage {
                color: #d3dce6;
                font-size: 12pt;
                font-weight: 500;
            }
            QLabel#badge {
                color: #79c7ff;
                font-size: 9pt;
                font-weight: 700;
            }
            QStatusBar {
                color: #c5d0dc;
                background: #111821;
                border-top: 1px solid #2f3b49;
            }
            QMenuBar, QMenu { color: #f4f7fb; background: #111821; }
            QMenuBar::item { padding: 7px 10px; }
            QMenuBar::item:selected, QMenu::item:selected {
                color: #ffffff;
                background: #263443;
            }
            QToolTip {
                color: #ffffff;
                background: #263443;
                border: 1px solid #64778c;
                padding: 6px;
            }
            QScrollArea, QScrollArea > QWidget > QWidget {
                background: #0b1017;
                border: 0;
            }
            QScrollArea#weightScroll,
            QScrollArea#weightScroll > QWidget > QWidget {
                background: #151c25;
            }
            QFrame#referenceCard {
                background: #151c25;
                border: 1px solid #344252;
                border-radius: 10px;
            }
            QLabel#dialogTitle {
                color: #ffffff;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#dialogDescription, QLabel#referenceGuidance { color: #b5c0cd; }
            QLabel#referenceCardTitle {
                color: #ffffff;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#referenceThumbnail {
                color: #9fabb8;
                background: #090d12;
                border: 1px solid #3c4b5d;
                border-radius: 7px;
            }
            QLabel#referenceStatus { color: #f2bb60; font-weight: 700; }
            QLabel#referenceStatus[valid="true"] { color: #5ee6a8; }
            QLabel#referenceSummary { color: #f2bb60; font-weight: 700; }
            QLabel#referenceSummary[complete="true"] { color: #5ee6a8; }
            QLabel#librarySaveState { color: #f2bb60; }
            QLabel#librarySaveState[saved="true"] { color: #5ee6a8; }
        """)
        self._build_ui()

        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(10)
        self._preview_timer.timeout.connect(self._poll_camera)
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(250)
        self._metrics_timer.timeout.connect(self._update_metrics)
        self._open_timer = QTimer(self)
        self._open_timer.setInterval(25)
        self._open_timer.timeout.connect(self._finish_camera_open)
        QTimer.singleShot(0, self.refresh_cameras)

    def _build_ui(self) -> None:
        shell = QWidget()
        shell.setObjectName("shell")
        self.setCentralWidget(shell)
        self._shell_layout = QVBoxLayout(shell)
        self._shell_layout.setContentsMargins(24, 20, 24, 14)
        self._shell_layout.setSpacing(8)

        title = QLabel("FaceLive")
        title.setObjectName("title")
        self._shell_layout.addWidget(title)
        subtitle = QLabel("Live physical-camera preview")
        subtitle.setObjectName("subtitle")
        self._shell_layout.addWidget(subtitle)

        self._content_scroll = QScrollArea()
        self._content_scroll.setObjectName("workspaceScroll")
        self._content_scroll.setWidgetResizable(True)
        self._content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._content_host = QWidget()
        self._content_host.setObjectName("contentHost")
        self._content_layout = QGridLayout(self._content_host)
        self._content_layout.setContentsMargins(0, 12, 0, 0)
        self._content_layout.setHorizontalSpacing(18)
        self._content_layout.setVerticalSpacing(16)
        self._content_scroll.setWidget(self._content_host)
        self._shell_layout.addWidget(self._content_scroll, 1)

        self.camera_card = QGroupBox("Camera input")
        self.camera_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        source_layout = QVBoxLayout(self.camera_card)
        source_layout.setContentsMargins(18, 22, 18, 18)
        source_layout.setSpacing(11)
        camera_label = QLabel("Physical camera")
        camera_label.setObjectName("fieldLabel")
        camera_label.setMinimumHeight(24)
        source_layout.addWidget(camera_label)
        self.camera_selector = QComboBox()
        self.camera_selector.setEnabled(False)
        source_layout.addWidget(self.camera_selector)
        camera_buttons = QHBoxLayout()
        camera_buttons.setSpacing(10)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("secondaryButton")
        self.refresh_button.clicked.connect(self.refresh_cameras)
        camera_buttons.addWidget(self.refresh_button)
        self.capture_button = QPushButton("Start preview")
        self.capture_button.setObjectName("primaryButton")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.toggle_capture)
        camera_buttons.addWidget(self.capture_button)
        source_layout.addLayout(camera_buttons)
        self.camera_format_label = QLabel(
            f"Requested: {self._config.video.width} × {self._config.video.height} · "
            f"{self._config.video.fps} FPS"
        )
        self.camera_format_label.setWordWrap(True)
        self.camera_format_label.setObjectName("hint")
        source_layout.addWidget(self.camera_format_label)
        self.mirror_preview_toggle = QCheckBox("Mirror local preview")
        self.mirror_preview_toggle.setChecked(True)
        self.mirror_preview_toggle.setToolTip(
            "Mirrors only this window. Captured and outgoing frames remain unmirrored."
        )
        self.mirror_preview_toggle.toggled.connect(self._render_last_image)
        source_layout.addWidget(self.mirror_preview_toggle)
        preview_mode_label = QLabel("Preview mode")
        preview_mode_label.setObjectName("fieldLabel")
        source_layout.addWidget(preview_mode_label)
        self.preview_mode_selector = QComboBox()
        self.preview_mode_selector.addItem("Original camera", "original")
        self.preview_mode_selector.addItem("Diagnostic tracking", "diagnostic")
        self.preview_mode_selector.addItem("Processed output", "processed")
        self.preview_mode_selector.setCurrentIndex(1)
        self.preview_mode_selector.setToolTip(
            "Choose the untouched camera, landmark diagnostics, or geometric replacement."
        )
        self.preview_mode_selector.currentIndexChanged.connect(self._preview_mode_changed)
        source_layout.addWidget(self.preview_mode_selector)
        self.smoothing_toggle = QCheckBox("Temporal smoothing")
        self.smoothing_toggle.setChecked(True)
        self.smoothing_toggle.setToolTip(
            "Smooths motion, expressions, and reference weights. Disable for comparison."
        )
        self.smoothing_toggle.toggled.connect(self._smoothing_toggled)
        source_layout.addWidget(self.smoothing_toggle)
        self.reference_library_card = QGroupBox("Reference library")
        self.reference_library_card.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        reference_layout = QVBoxLayout(self.reference_library_card)
        reference_layout.setContentsMargins(18, 22, 18, 18)
        reference_layout.setSpacing(9)
        self.load_references = QPushButton("Manage references…")
        self.load_references.clicked.connect(self.open_reference_enrollment)
        reference_layout.addWidget(self.load_references)
        self.reference_summary_label = QLabel()
        self.reference_summary_label.setObjectName("hint")
        reference_layout.addWidget(self.reference_summary_label)
        reference_hint = QLabel("Images are validated and preprocessed once during enrollment.")
        reference_hint.setObjectName("hint")
        reference_hint.setWordWrap(True)
        reference_layout.addWidget(reference_hint)
        # These controls remain part of the stable shell API but are not shown until
        # their implementation milestones.
        self.replacement_toggle = QCheckBox("Enable face replacement", self)
        self.replacement_toggle.setEnabled(False)
        self.replacement_toggle.hide()
        self.virtual_camera_button = QPushButton("Start virtual camera", self)
        self.virtual_camera_button.setEnabled(False)
        self.virtual_camera_button.hide()
        self.preview_card = QWidget()
        self.preview_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_column = QVBoxLayout(self.preview_card)
        preview_column.setContentsMargins(0, 0, 0, 0)
        preview_column.setSpacing(8)
        preview_heading = QLabel("LIVE PREVIEW")
        preview_heading.setObjectName("badge")
        preview_heading.setMinimumHeight(22)
        preview_column.addWidget(preview_heading)
        self.preview_frame = QFrame()
        self.preview_frame.setObjectName("preview")
        self.preview_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_layout = QVBoxLayout(self.preview_frame)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        self.preview_image = QLabel("Searching for physical cameras…")
        self.preview_image.setObjectName("previewImage")
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setMinimumSize(320, 180)
        self.preview_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_image.installEventFilter(self)
        preview_layout.addWidget(self.preview_image)
        preview_column.addWidget(self.preview_frame, 1)

        self.diagnostics_card = QGroupBox("Live diagnostics")
        self.diagnostics_card.setObjectName("metricsCard")
        self.diagnostics_card.setMinimumHeight(560)
        metrics_layout = QGridLayout(self.diagnostics_card)
        metrics_layout.setContentsMargins(18, 22, 18, 18)
        metrics_layout.setHorizontalSpacing(18)
        metrics_layout.setVerticalSpacing(5)
        self.capture_state_label = QLabel("Idle")
        self.capture_state_label.setObjectName("stateValue")
        self.capture_state_label.setProperty("state", "idle")
        self.capture_fps_label = QLabel("—")
        self.capture_fps_label.setObjectName("metricValue")
        self.tracking_state_label = QLabel("Idle")
        self.tracking_state_label.setObjectName("stateValue")
        self.tracking_state_label.setProperty("state", "idle")
        self.tracking_confidence_label = QLabel("—")
        self.tracking_confidence_label.setObjectName("metricValue")
        self.head_pose_label = QLabel("—")
        self.head_pose_label.setObjectName("metricValue")
        self.smile_label = QLabel("—")
        self.smile_label.setObjectName("metricValue")
        self.eye_closure_label = QLabel("—")
        self.eye_closure_label.setObjectName("metricValue")
        self.mouth_activity_label = QLabel("—")
        self.mouth_activity_label.setObjectName("metricValue")
        self.tracking_latency_label = QLabel("—")
        self.tracking_latency_label.setObjectName("metricValue")
        self.smoothing_state_label = QLabel("On")
        self.smoothing_state_label.setObjectName("stateValue")
        self.smoothing_state_label.setProperty("state", "running")
        self.smoothing_latency_label = QLabel("—")
        self.smoothing_latency_label.setObjectName("metricValue")
        self.smoothing_correction_label = QLabel("—")
        self.smoothing_correction_label.setObjectName("metricValue")
        self.rendering_state_label = QLabel("Idle")
        self.rendering_state_label.setObjectName("stateValue")
        self.rendering_state_label.setProperty("state", "idle")
        self.rendering_latency_label = QLabel("—")
        self.rendering_latency_label.setObjectName("metricValue")
        self.compositing_latency_label = QLabel("—")
        self.compositing_latency_label.setObjectName("metricValue")
        self.complete_frame_latency_label = QLabel("—")
        self.complete_frame_latency_label.setObjectName("metricValue")
        self.tracking_drops_label = QLabel("0")
        self.tracking_drops_label.setObjectName("metricValue")
        self.preview_drops_label = QLabel("0")
        self.preview_drops_label.setObjectName("metricValue")
        rows = (
            ("Capture state", self.capture_state_label),
            ("Capture FPS", self.capture_fps_label),
            ("Face tracking", self.tracking_state_label),
            ("Confidence", self.tracking_confidence_label),
            ("Yaw · pitch · roll", self.head_pose_label),
            ("Smile", self.smile_label),
            ("Eye closure L · R", self.eye_closure_label),
            ("Mouth activity", self.mouth_activity_label),
            ("Tracking latency", self.tracking_latency_label),
            ("Temporal smoothing", self.smoothing_state_label),
            ("Smoothing latency", self.smoothing_latency_label),
            ("Motion correction", self.smoothing_correction_label),
            ("Face rendering", self.rendering_state_label),
            ("Rendering latency", self.rendering_latency_label),
            ("Compositing latency", self.compositing_latency_label),
            ("Complete frame", self.complete_frame_latency_label),
            ("Preview skips", self.preview_drops_label),
            ("Tracker drops", self.tracking_drops_label),
        )
        for row, (label_text, value_widget) in enumerate(rows):
            label = QLabel(label_text)
            label.setObjectName("metricLabel")
            label.setMinimumHeight(25)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            value_widget.setMinimumHeight(25)
            value_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            metrics_layout.addWidget(label, row, 0)
            metrics_layout.addWidget(value_widget, row, 1)
            metrics_layout.setRowMinimumHeight(row, 25)
        metrics_layout.setColumnStretch(1, 1)
        metrics_layout.setColumnMinimumWidth(0, 135)

        self.reference_selection_card = QGroupBox("Reference selection")
        self.reference_selection_card.setObjectName("metricsCard")
        self.reference_selection_card.setMinimumHeight(290)
        selection_layout = QVBoxLayout(self.reference_selection_card)
        selection_layout.setContentsMargins(18, 22, 18, 14)
        selection_layout.setSpacing(8)
        self.reference_selection_status = QLabel("No enrolled references")
        self.reference_selection_status.setObjectName("hint")
        self.reference_selection_status.setMinimumHeight(24)
        self.reference_selection_status.setWordWrap(True)
        self.reference_weight_view = ReferenceWeightsWidget()
        selection_layout.addWidget(self.reference_selection_status)
        weight_scroll = QScrollArea()
        weight_scroll.setObjectName("weightScroll")
        weight_scroll.setWidgetResizable(True)
        weight_scroll.setFrameShape(QFrame.Shape.NoFrame)
        weight_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        weight_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        weight_scroll.setWidget(self.reference_weight_view)
        selection_layout.addWidget(weight_scroll, 1)

        footer = QLabel("Milestone 6 · Temporal stability and expression handling")
        footer.setObjectName("hint")
        self._shell_layout.addWidget(footer)
        self._apply_responsive_layout(self.width())
        self._update_reference_summary()
        self.statusBar().showMessage("Idle — no camera is open.")

        quit_action = QAction("Exit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        self.menuBar().addMenu("File").addAction(quit_action)

    def _layout_for_width(self, width: int) -> str:
        if self._layout_mode == "wide":
            return "wide" if width >= 1340 else ("medium" if width >= 860 else "compact")
        if self._layout_mode == "compact":
            return "compact" if width <= 940 else ("wide" if width >= 1420 else "medium")
        if self._layout_mode == "medium":
            if width >= 1420:
                return "wide"
            return "compact" if width < 860 else "medium"
        if width >= 1400:
            return "wide"
        return "medium" if width >= 900 else "compact"

    def _apply_responsive_layout(self, width: int) -> None:
        mode = self._layout_for_width(width)
        if mode == self._layout_mode:
            return
        while self._content_layout.count():
            self._content_layout.takeAt(0)
        for index in range(5):
            self._content_layout.setColumnMinimumWidth(index, 0)
            self._content_layout.setColumnStretch(index, 0)
            self._content_layout.setRowMinimumHeight(index, 0)
            self._content_layout.setRowStretch(index, 0)

        if mode == "wide":
            self._shell_layout.setContentsMargins(24, 20, 24, 14)
            self.preview_card.setMinimumHeight(520)
            self._content_layout.addWidget(
                self.camera_card, 0, 0, alignment=Qt.AlignmentFlag.AlignTop
            )
            self._content_layout.addWidget(
                self.reference_library_card, 1, 0, alignment=Qt.AlignmentFlag.AlignTop
            )
            self._content_layout.addWidget(self.preview_card, 0, 1, 2, 1)
            self._content_layout.addWidget(self.diagnostics_card, 0, 2)
            self._content_layout.addWidget(self.reference_selection_card, 1, 2)
            self._content_layout.setColumnMinimumWidth(0, 280)
            self._content_layout.setColumnMinimumWidth(1, 520)
            self._content_layout.setColumnMinimumWidth(2, 360)
            self._content_layout.setColumnStretch(1, 1)
            self._content_layout.setRowStretch(0, 3)
            self._content_layout.setRowStretch(1, 2)
        elif mode == "medium":
            self._shell_layout.setContentsMargins(22, 18, 22, 14)
            self.preview_card.setMinimumHeight(340)
            self._content_layout.addWidget(
                self.camera_card, 0, 0, alignment=Qt.AlignmentFlag.AlignTop
            )
            self._content_layout.addWidget(
                self.reference_library_card, 1, 0, alignment=Qt.AlignmentFlag.AlignTop
            )
            self._content_layout.addWidget(self.preview_card, 0, 1)
            self._content_layout.addWidget(self.diagnostics_card, 1, 1)
            self._content_layout.addWidget(self.reference_selection_card, 2, 1)
            self._content_layout.setColumnMinimumWidth(0, 270)
            self._content_layout.setColumnMinimumWidth(1, 480)
            self._content_layout.setColumnStretch(1, 1)
        else:
            self._shell_layout.setContentsMargins(16, 14, 16, 12)
            self.preview_card.setMinimumHeight(280)
            self._content_layout.addWidget(self.camera_card, 0, 0)
            self._content_layout.addWidget(self.preview_card, 1, 0)
            self._content_layout.addWidget(self.reference_library_card, 2, 0)
            self._content_layout.addWidget(self.diagnostics_card, 3, 0)
            self._content_layout.addWidget(self.reference_selection_card, 4, 0)
            self._content_layout.setColumnStretch(0, 1)

        self._layout_mode = mode
        QTimer.singleShot(0, self._render_last_image)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "_content_layout"):
            self._apply_responsive_layout(event.size().width())

    def refresh_cameras(self) -> None:
        """Refresh device names while preserving the selected device when possible."""
        if self._capturing:
            return
        selected_id = self.camera_selector.currentData()
        self.refresh_button.setEnabled(False)
        try:
            all_devices = self._source.enumerate_devices()
            self._devices = tuple(device for device in all_devices if not device.is_virtual)
        except StageError as exc:
            self._devices = ()
            self._show_camera_error(str(exc))
        finally:
            self.refresh_button.setEnabled(True)

        self.camera_selector.clear()
        for device in self._devices:
            self.camera_selector.addItem(device.display_name, device.device_id)
        if selected_id is not None:
            index = self.camera_selector.findData(selected_id)
            if index >= 0:
                self.camera_selector.setCurrentIndex(index)
        available = bool(self._devices)
        self.camera_selector.setEnabled(available)
        self.capture_button.setEnabled(available)
        if available:
            self.preview_image.setText("Select a camera and start the preview.")
            self.statusBar().showMessage(f"Found {len(self._devices)} physical camera(s).")
        else:
            self.preview_image.setText("No physical camera found.\nConnect one and choose Refresh.")
            self.statusBar().showMessage("No physical camera found.")

    def toggle_capture(self) -> None:
        if self._capturing:
            self.stop_capture()
        else:
            self.start_capture()

    def open_reference_enrollment(self) -> None:
        dialog = ReferenceEnrollmentDialog(self._reference_session, self)
        dialog.library_changed.connect(self._update_reference_summary)
        dialog.exec()
        self._update_reference_summary()

    def _update_reference_summary(self) -> None:
        self._stabilizer.reset()
        self._last_processed_image = None
        complete = self._reference_library.completed_required_count
        required = self._reference_library.required_count
        state = "Complete" if self._reference_library.is_complete else "Incomplete"
        self.reference_summary_label.setText(f"{state} · {complete} of {required} required")
        references = self._reference_library.references()
        self.reference_weight_view.set_references(
            self._reference_library.slots(),
            {reference.reference_id for reference in references},
        )
        if self._last_face_state is not None:
            self._update_reference_selection(self._last_face_state)
        else:
            self._clear_reference_selection(
                "Waiting for a tracked face" if references else "No enrolled references"
            )
        self._render_last_image()

    def start_capture(self) -> None:
        device_id = self.camera_selector.currentData()
        if not device_id:
            return
        requested = FrameFormat(
            self._config.video.width, self._config.video.height, self._config.video.fps
        )
        self._opening = True
        self._metrics.reset()
        self._reset_tracking_diagnostics("Opening", "opening")
        self._tracking_drops = 0
        self._tracking_failed = False
        self._rendering_failed = False
        self._pending_tracking_frame = None
        self._last_face_state = None
        self._last_raw_face_state = None
        self._last_processed_image = None
        self._last_reference_weights = ()
        self._stabilizer.reset()
        self.camera_selector.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.capture_button.setText("Opening…")
        self.capture_button.setEnabled(False)
        self._set_capture_state("Opening", "opening")
        self.preview_image.setText("Opening camera…")
        self.statusBar().showMessage("Opening the selected physical camera and face tracker…")
        self._open_future = self._camera_executor.submit(
            self._open_capture_pipeline, str(device_id), requested
        )
        self._open_timer.start()

    def _open_capture_pipeline(
        self, device_id: str, requested: FrameFormat
    ) -> tuple[FrameFormat, str | None, str | None]:
        negotiated = self._source.open(device_id, requested)
        tracking_error = None
        rendering_error = None
        try:
            self._tracker.open()
        except StageError as exc:
            tracking_error = str(exc)
        try:
            self._renderer.open(negotiated)
        except StageError as exc:
            rendering_error = str(exc)
        return negotiated, tracking_error, rendering_error

    def _finish_camera_open(self) -> None:
        future = self._open_future
        if future is None or not future.done():
            return
        self._open_timer.stop()
        self._open_future = None
        self._opening = False
        try:
            negotiated, tracking_error, rendering_error = future.result()
        except StageError as exc:
            self._source.close()
            try:
                self._tracker.close()
            except StageError as close_exc:
                self._logger.warning("Face tracker cleanup failed: %s", close_exc)
            self._renderer.close()
            self._show_camera_error(str(exc))
            self.camera_selector.setEnabled(bool(self._devices))
            self.refresh_button.setEnabled(True)
            self.capture_button.setText("Start preview")
            self.capture_button.setEnabled(bool(self._devices))
            return
        except Exception as exc:
            self._source.close()
            try:
                self._tracker.close()
            except StageError as close_exc:
                self._logger.warning("Face tracker cleanup failed: %s", close_exc)
            self._renderer.close()
            self._logger.exception("Unexpected camera startup failure")
            self._show_camera_error(f"Unexpected camera startup failure: {exc}")
            self.camera_selector.setEnabled(bool(self._devices))
            self.refresh_button.setEnabled(True)
            self.capture_button.setText("Start preview")
            self.capture_button.setEnabled(bool(self._devices))
            return

        self._capturing = True
        self.capture_button.setText("Stop preview")
        self.capture_button.setEnabled(True)
        self._set_capture_state("Running", "running")
        self.camera_format_label.setText(
            f"Active: {negotiated.width} × {negotiated.height} · "
            f"device reports {negotiated.fps:.1f} FPS"
        )
        if tracking_error is None:
            self._set_tracking_state("Waiting for face", "opening")
            if rendering_error is None:
                self._set_rendering_state("Ready", "running")
                self.statusBar().showMessage(
                    "Camera, face tracking, and configured face rendering are running."
                )
            else:
                self._rendering_failed = True
                self._set_rendering_state("Unavailable", "error")
                self.statusBar().showMessage(
                    f"Camera and tracking running; rendering unavailable: {rendering_error}"
                )
                self._logger.error("Face renderer startup failed: %s", rendering_error)
        else:
            self._tracking_failed = True
            self._set_tracking_state("Unavailable", "error")
            self.statusBar().showMessage(
                f"Camera running; face tracking unavailable: {tracking_error}"
            )
            self._logger.error("Face tracker startup failed: %s", tracking_error)
        self._preview_timer.start()
        self._metrics_timer.start()

    def stop_capture(self, *, message: str = "Camera preview stopped.") -> None:
        self._preview_timer.stop()
        self._metrics_timer.stop()
        self._source.close()
        try:
            self._tracker.close()
        except StageError as exc:
            self._logger.warning("Face tracker shutdown failed: %s", exc)
        self._renderer.close()
        self._capturing = False
        self._last_image = None
        self._last_processed_image = None
        self._last_face_state = None
        self._last_raw_face_state = None
        self._last_reference_weights = ()
        self._pending_tracking_frame = None
        self._tracking_failed = False
        self._rendering_failed = False
        self._stabilizer.reset()
        self._metrics.reset()
        self.camera_selector.setEnabled(bool(self._devices))
        self.refresh_button.setEnabled(True)
        self.capture_button.setText("Start preview")
        self.capture_button.setEnabled(bool(self._devices))
        self._set_capture_state("Idle", "idle")
        self.capture_fps_label.setText("—")
        self.preview_drops_label.setText("0")
        self._reset_tracking_diagnostics("Idle", "idle")
        self.preview_image.setPixmap(QPixmap())
        self.preview_image.setText("Preview is idle.")
        self.camera_format_label.setText(
            f"Requested: {self._config.video.width} × {self._config.video.height} · "
            f"{self._config.video.fps} FPS"
        )
        self.statusBar().showMessage(message)

    def _poll_camera(self) -> None:
        self._poll_tracking()
        try:
            frame = self._source.read_latest()
        except StageError as exc:
            detail = str(exc)
            self._logger.warning("Camera capture stopped: %s", detail)
            self.stop_capture(message=f"Camera error: {detail}")
            self._set_capture_state("Disconnected", "error")
            self.preview_image.setText(
                "Camera disconnected or stopped responding.\nReconnect it, then choose Refresh."
            )
            return
        if frame is not None:
            self._metrics.observe(frame)
            self._submit_tracking(frame)
        self._poll_tracking()

    def _submit_tracking(self, frame: VideoFrame) -> None:
        if self._tracking_failed:
            self._display_frame(frame)
            return
        image = self._frame_image(frame)
        try:
            accepted = self._tracker.submit(frame)
        except StageError as exc:
            self._tracking_failed = True
            self._stabilizer.reset()
            self._last_face_state = None
            self._last_image = image
            self._last_processed_image = None
            self._render_last_image()
            self._set_tracking_state("Error", "error")
            self.statusBar().showMessage(f"Face tracking stopped: {exc}")
            self._logger.error("Face tracking submission failed: %s", exc)
            return
        if accepted:
            self._pending_tracking_frame = (frame, image, time.perf_counter_ns())
        else:
            self._tracking_drops += 1
            self.tracking_drops_label.setText(str(self._tracking_drops))

    def _poll_tracking(self) -> None:
        if self._tracking_failed:
            return
        try:
            result = self._tracker.poll_latest()
        except StageError as exc:
            self._tracking_failed = True
            self._set_tracking_state("Error", "error")
            self.statusBar().showMessage(f"Face tracking stopped: {exc}")
            self._logger.error("Face tracking polling failed: %s", exc)
            return
        if result is None:
            return
        retained = self._pending_tracking_frame
        self._pending_tracking_frame = None
        if retained is None or (retained[0].frame_id, retained[0].timestamp_ns) != (
            result.frame_id,
            result.timestamp_ns,
        ):
            self._logger.warning(
                "Discarding tracking result without its matching frame: %s/%s",
                result.frame_id,
                result.timestamp_ns,
            )
            return
        frame, image, processing_started_ns = retained
        self._last_image = image
        self._last_processed_image = None
        raw_face = result.face
        face = raw_face
        held = False
        self._current_smoothing_ms = 0.0
        if self.smoothing_toggle.isChecked():
            smoothing_started = time.perf_counter_ns()
            try:
                if result.status is TrackingStatus.TRACKED:
                    face = self._stabilizer.update(result.face)
                elif result.status is TrackingStatus.NO_FACE:
                    face = self._stabilizer.coast(result.frame_id, result.timestamp_ns)
                    held = face is not None
                    if face is None:
                        self._stabilizer.reset()
            except StageError as exc:
                self._logger.warning("Temporal smoothing reset after invalid input: %s", exc)
                self._stabilizer.reset()
                face = raw_face
            self._current_smoothing_ms = (time.perf_counter_ns() - smoothing_started) / 1_000_000
        self._last_raw_face_state = raw_face
        self._last_face_state = face
        self._update_tracking_diagnostics(result, face, held=held)
        if face is not None:
            if self._preview_mode() == "processed":
                self._process_frame(frame, face, processing_started_ns)
            elif not self._rendering_failed:
                self._set_rendering_state("Ready", "running")
                self.rendering_latency_label.setText("—")
                self.compositing_latency_label.setText("—")
                self.complete_frame_latency_label.setText("—")
        elif not self._rendering_failed:
            self._set_rendering_state("Waiting for face", "no-face")
            self.rendering_latency_label.setText("—")
            self.compositing_latency_label.setText("—")
            self.complete_frame_latency_label.setText("—")
        self._render_last_image()

    @staticmethod
    def _frame_image(frame: VideoFrame) -> QImage:
        height, width = frame.rgb.shape[:2]
        return QImage(
            frame.rgb.data,
            width,
            height,
            int(frame.rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()

    def _display_frame(self, frame: VideoFrame) -> None:
        self._last_image = self._frame_image(frame)
        self._last_processed_image = None
        self._last_face_state = None
        self._render_last_image()

    def _process_frame(
        self, frame: VideoFrame, face: FaceState, processing_started_ns: int
    ) -> None:
        references = self._reference_library.references()
        weights = self._last_reference_weights
        if self._rendering_failed:
            return
        if not references or not weights:
            self._set_rendering_state("Needs references", "no-face")
            self.rendering_latency_label.setText("—")
            self.compositing_latency_label.setText("—")
            self.complete_frame_latency_label.setText("—")
            return
        try:
            render_started = time.perf_counter_ns()
            rendered = self._renderer.render(face, references, weights)
            render_finished = time.perf_counter_ns()
            output = self._compositor.composite(frame, rendered)
            composite_finished = time.perf_counter_ns()
        except (StageError, ValueError) as exc:
            self._set_rendering_state("Error", "error")
            self.rendering_latency_label.setText("—")
            self.compositing_latency_label.setText("—")
            self.complete_frame_latency_label.setText("—")
            self.statusBar().showMessage(f"Face rendering skipped: {exc}")
            self._logger.warning("Face rendering failed: %s", exc)
            return
        self._last_processed_image = self._frame_image(output)
        self._set_rendering_state("Processed", "running")
        self.rendering_latency_label.setText(
            f"{(render_finished - render_started) / 1_000_000:.1f} ms"
        )
        self.compositing_latency_label.setText(
            f"{(composite_finished - render_finished) / 1_000_000:.1f} ms"
        )
        self.complete_frame_latency_label.setText(
            f"{(composite_finished - processing_started_ns) / 1_000_000:.1f} ms"
        )

    def _render_last_image(self) -> None:
        display_image = self._oriented_preview_image()
        if display_image is None:
            if self._preview_mode() == "processed":
                self.preview_image.setPixmap(QPixmap())
                if not self._last_reference_weights:
                    self.preview_image.setText(
                        "Processed output unavailable.\nEnroll a reference face and track a face."
                    )
                else:
                    self.preview_image.setText("Waiting for the next processed frame…")
            return
        if self._preview_mode() == "diagnostic" and self._last_face_state is not None:
            display_image = display_image.copy()
            self._draw_tracking_overlay(display_image, self._last_face_state)
        pixmap = QPixmap.fromImage(display_image).scaled(
            self.preview_image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_image.setPixmap(pixmap)

    def _oriented_preview_image(self) -> QImage | None:
        """Return display orientation without changing the canonical captured image."""
        base_image = (
            self._last_processed_image if self._preview_mode() == "processed" else self._last_image
        )
        if base_image is None:
            return None
        if not self.mirror_preview_toggle.isChecked():
            return base_image
        return base_image.transformed(
            QTransform().scale(-1.0, 1.0), Qt.TransformationMode.FastTransformation
        )

    def _preview_mode(self) -> str:
        return str(self.preview_mode_selector.currentData())

    def _preview_mode_changed(self, _index: int) -> None:
        if self._preview_mode() == "processed" and self._capturing:
            self._last_processed_image = None
            if not self._rendering_failed:
                self._set_rendering_state("Waiting for frame", "opening")
        self._render_last_image()

    def _smoothing_toggled(self, enabled: bool) -> None:
        self._stabilizer.reset()
        self._last_reference_weights = ()
        self._last_processed_image = None
        self._set_smoothing_state("On" if enabled else "Off", "running" if enabled else "idle")
        self.smoothing_latency_label.setText("—")
        self.smoothing_correction_label.setText("—")
        if self._capturing:
            self.statusBar().showMessage(
                "Temporal smoothing enabled."
                if enabled
                else "Temporal smoothing disabled for comparison."
            )
        self._render_last_image()

    def _draw_tracking_overlay(self, image: QImage, face: FaceState) -> None:
        mirrored = self.mirror_preview_toggle.isChecked()

        def oriented_x(x: float) -> float:
            return image.width() - 1 - x if mirrored else x

        bounds_x = image.width() - face.bounds.x - face.bounds.width if mirrored else face.bounds.x
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        line_width = max(2.0, image.width() / 640)
        painter.setPen(QPen(QColor("#55e6a5"), line_width))
        painter.drawRoundedRect(
            QRectF(bounds_x, face.bounds.y, face.bounds.width, face.bounds.height), 8, 8
        )
        painter.setPen(QPen(QColor("#65baff"), line_width))
        radius = max(1.5, image.width() / 900)
        for point in face.landmarks[::6]:
            painter.drawEllipse(QPointF(oriented_x(point.x), point.y), radius, radius)
        center_x = oriented_x(face.center.x)
        center_y = face.center.y
        painter.setPen(QPen(QColor("#ffca65"), line_width))
        painter.drawLine(QPointF(center_x - 8, center_y), QPointF(center_x + 8, center_y))
        painter.drawLine(QPointF(center_x, center_y - 8), QPointF(center_x, center_y + 8))
        painter.end()

    def _update_metrics(self) -> None:
        fps = self._metrics.fps
        self.capture_fps_label.setText("—" if fps is None else f"{fps:.1f}")
        self.preview_drops_label.setText(str(self._metrics.preview_drops))

    def _update_tracking_diagnostics(
        self, result: TrackingResult, face: FaceState | None, *, held: bool = False
    ) -> None:
        self.tracking_latency_label.setText(
            "—" if result.latency_ms is None else f"{result.latency_ms:.1f} ms"
        )
        if result.status is TrackingStatus.ERROR:
            self._tracking_failed = True
            self._set_tracking_state("Error", "error")
            self._clear_face_metrics()
            self._clear_reference_selection("Tracking unavailable")
            self._stabilizer.reset()
            self.statusBar().showMessage(f"Face tracking stopped: {result.error}")
            return
        if face is None:
            self._set_tracking_state("No face", "no-face")
            self._clear_face_metrics()
            self._clear_reference_selection("No face tracked")
            self.smoothing_correction_label.setText("—")
            self.smoothing_latency_label.setText(
                f"{self._current_smoothing_ms:.2f} ms" if self.smoothing_toggle.isChecked() else "—"
            )
            return
        self._set_tracking_state(
            "Held briefly" if held else "Tracked", "opening" if held else "running"
        )
        self.tracking_confidence_label.setText(f"{face.tracking_confidence * 100:.0f}% derived")
        self.head_pose_label.setText(
            f"{face.pose.yaw:+.1f}° · {face.pose.pitch:+.1f}° · {face.pose.roll:+.1f}°"
        )
        shapes = face.blendshapes
        smile_values = [
            shapes[name] for name in ("mouth_smile_left", "mouth_smile_right") if name in shapes
        ]
        self.smile_label.setText(
            "Unavailable"
            if not smile_values
            else f"{sum(smile_values) / len(smile_values) * 100:.0f}%"
        )
        left_eye = shapes.get("eye_blink_left")
        right_eye = shapes.get("eye_blink_right")
        self.eye_closure_label.setText(
            "Unavailable"
            if left_eye is None or right_eye is None
            else f"{left_eye * 100:.0f}% · {right_eye * 100:.0f}%"
        )
        jaw_open = shapes.get("jaw_open")
        self.mouth_activity_label.setText(
            "Unavailable" if jaw_open is None else f"{jaw_open * 100:.0f}% open"
        )
        if held:
            self.smoothing_correction_label.setText("Holding last face")
        elif self.smoothing_toggle.isChecked() and result.face is not None:
            translation = hypot(
                result.face.center.x - face.center.x,
                result.face.center.y - face.center.y,
            )
            pose_delta = max(
                abs(result.face.pose.yaw - face.pose.yaw),
                abs(result.face.pose.pitch - face.pose.pitch),
                abs(result.face.pose.roll - face.pose.roll),
            )
            self.smoothing_correction_label.setText(f"{translation:.1f} px · {pose_delta:.1f}°")
        else:
            self.smoothing_correction_label.setText("—")
        self._update_reference_selection(face)

    def _update_reference_selection(self, face: FaceState) -> None:
        references = self._reference_library.references()
        if not references:
            self.smoothing_latency_label.setText(
                f"{self._current_smoothing_ms:.2f} ms" if self.smoothing_toggle.isChecked() else "—"
            )
            self._clear_reference_selection("No enrolled references")
            return
        try:
            weights = self._reference_selector.select(face, references)
            if self.smoothing_toggle.isChecked():
                smoothing_started = time.perf_counter_ns()
                weights = self._stabilizer.smooth_weights(weights, face.timestamp_ns)
                self._current_smoothing_ms += (
                    time.perf_counter_ns() - smoothing_started
                ) / 1_000_000
        except ValueError as exc:
            self._logger.error("Reference selection failed: %s", exc)
            self._clear_reference_selection("Selection error")
            return
        except StageError as exc:
            self._logger.warning("Reference-weight smoothing reset: %s", exc)
            self._stabilizer.reset()
            self._clear_reference_selection("Smoothing reset")
            return
        self.smoothing_latency_label.setText(
            f"{self._current_smoothing_ms:.2f} ms" if self.smoothing_toggle.isChecked() else "—"
        )
        self._last_reference_weights = weights
        self.reference_weight_view.set_weights(weights)
        if not weights:
            self.reference_selection_status.setText("Pose unsupported by current references")
            return
        smile_values = [
            face.blendshapes[name]
            for name in ("mouth_smile_left", "mouth_smile_right")
            if name in face.blendshapes
        ]
        smile = sum(smile_values) / len(smile_values) if smile_values else 0.0
        self.reference_selection_status.setText(
            f"Active · yaw {face.pose.yaw:+.1f}° · pitch {face.pose.pitch:+.1f}° · "
            f"smile {smile * 100:.0f}%"
        )

    def _clear_reference_selection(self, status: str) -> None:
        self._last_reference_weights = ()
        self.reference_weight_view.set_weights(())
        self.reference_selection_status.setText(status)

    def _clear_face_metrics(self) -> None:
        self.tracking_confidence_label.setText("—")
        self.head_pose_label.setText("—")
        self.smile_label.setText("—")
        self.eye_closure_label.setText("—")
        self.mouth_activity_label.setText("—")

    def _reset_tracking_diagnostics(self, state_text: str, state: str) -> None:
        self._set_tracking_state(state_text, state)
        self._clear_face_metrics()
        self.tracking_latency_label.setText("—")
        self.tracking_drops_label.setText("0")
        self._set_smoothing_state(
            "On" if self.smoothing_toggle.isChecked() else "Off",
            "running" if self.smoothing_toggle.isChecked() else "idle",
        )
        self.smoothing_latency_label.setText("—")
        self.smoothing_correction_label.setText("—")
        self._set_rendering_state("Idle", "idle")
        self.rendering_latency_label.setText("—")
        self.compositing_latency_label.setText("—")
        self.complete_frame_latency_label.setText("—")
        self._clear_reference_selection(
            "Waiting for a tracked face"
            if self._reference_library.references()
            else "No enrolled references"
        )

    def _set_tracking_state(self, text: str, state: str) -> None:
        self.tracking_state_label.setText(text)
        self.tracking_state_label.setProperty("state", state)
        self.tracking_state_label.style().unpolish(self.tracking_state_label)
        self.tracking_state_label.style().polish(self.tracking_state_label)

    def _set_rendering_state(self, text: str, state: str) -> None:
        self.rendering_state_label.setText(text)
        self.rendering_state_label.setProperty("state", state)
        self.rendering_state_label.style().unpolish(self.rendering_state_label)
        self.rendering_state_label.style().polish(self.rendering_state_label)

    def _set_smoothing_state(self, text: str, state: str) -> None:
        self.smoothing_state_label.setText(text)
        self.smoothing_state_label.setProperty("state", state)
        self.smoothing_state_label.style().unpolish(self.smoothing_state_label)
        self.smoothing_state_label.style().polish(self.smoothing_state_label)

    def _set_capture_state(self, text: str, state: str) -> None:
        self.capture_state_label.setText(text)
        self.capture_state_label.setProperty("state", state)
        self.capture_state_label.style().unpolish(self.capture_state_label)
        self.capture_state_label.style().polish(self.capture_state_label)

    def _show_camera_error(self, detail: str) -> None:
        self._logger.warning("Camera operation failed: %s", detail)
        self._set_capture_state("Error", "error")
        self.preview_image.setText(f"Camera unavailable.\n{detail}")
        self.statusBar().showMessage(f"Camera error: {detail}")

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa: N802
        if watched is self.preview_image and event.type() == QEvent.Type.Resize:
            self._render_last_image()
        return super().eventFilter(watched, event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._open_timer.stop()
        self._preview_timer.stop()
        self._metrics_timer.stop()
        self._camera_executor.shutdown(wait=True, cancel_futures=True)
        self._source.close()
        try:
            self._tracker.close()
        except StageError as exc:
            self._logger.warning("Face tracker shutdown failed: %s", exc)
        self._renderer.close()
        try:
            self._reference_session.flush()
        except StageError as exc:
            self._logger.error("Could not flush reference library at shutdown: %s", exc)
        self._reference_session.close()
        event.accept()
