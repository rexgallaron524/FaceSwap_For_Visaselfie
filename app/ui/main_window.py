"""FaceLive desktop shell with physical-camera preview."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QColor, QImage, QPainter, QPen, QPixmap, QTransform
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
    QVBoxLayout,
    QWidget,
)

from app.camera import OpenCVCameraSource
from app.camera.metrics import CaptureMetrics
from app.camera.protocol import CameraSource
from app.config import AppConfig
from app.pipeline.types import (
    CameraDevice,
    FaceState,
    FrameFormat,
    StageError,
    TrackingResult,
    TrackingStatus,
    VideoFrame,
)
from app.reference import ReferenceLibrarySession, ReferenceLibraryStore
from app.tracking import MediaPipeFaceTracker
from app.tracking.protocol import FaceTracker
from app.ui.reference_dialog import ReferenceEnrollmentDialog


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: AppConfig,
        camera_source_factory: Callable[[], CameraSource] = OpenCVCameraSource,
        face_tracker_factory: Callable[[], FaceTracker] = MediaPipeFaceTracker,
    ) -> None:
        super().__init__()
        self._config = config
        self._source = camera_source_factory()
        self._tracker = face_tracker_factory()
        self._devices: tuple[CameraDevice, ...] = ()
        self._capturing = False
        self._opening = False
        self._last_image: QImage | None = None
        self._last_face_state: FaceState | None = None
        self._pending_tracking_image: tuple[int, int, QImage] | None = None
        self._tracking_drops = 0
        self._tracking_failed = False
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
        self._open_future: Future[tuple[FrameFormat, str | None]] | None = None

        self.setWindowTitle("FaceLive")
        self.resize(1120, 740)
        self.setMinimumSize(900, 700)
        self.setStyleSheet("""
            * {
                font-family: "Segoe UI Variable", "Segoe UI";
                font-size: 14px;
            }
            QMainWindow, QDialog, QMessageBox, QWidget#shell, QWidget#referenceGrid {
                background: #0b1017;
                color: #f4f7fb;
            }
            QLabel { color: #f4f7fb; background: transparent; }
            QLabel#title {
                color: #ffffff;
                font-size: 30px;
                font-weight: 700;
            }
            QLabel#subtitle {
                color: #c4ced9;
                font-size: 15px;
            }
            QLabel#hint { color: #b5c0cd; }
            QLabel#fieldLabel, QLabel#metricLabel {
                color: #c9d3de;
                font-weight: 500;
            }
            QLabel#metricValue {
                color: #ffffff;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#stateValue {
                color: #d6dee8;
                font-size: 16px;
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
                margin-top: 17px;
                padding: 18px;
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
                font-size: 18px;
                font-weight: 500;
            }
            QLabel#badge {
                color: #79c7ff;
                font-size: 13px;
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
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(30, 24, 30, 18)
        layout.setSpacing(8)

        title = QLabel("FaceLive")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel("Live physical-camera preview")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(24)
        content.setContentsMargins(0, 16, 0, 0)
        controls = QVBoxLayout()
        controls.setSpacing(16)
        sources = QGroupBox("Camera input")
        sources.setFixedWidth(300)
        sources.setMinimumHeight(250)
        source_layout = QVBoxLayout(sources)
        source_layout.setContentsMargins(18, 22, 18, 18)
        source_layout.setSpacing(11)
        camera_label = QLabel("Physical camera")
        camera_label.setObjectName("fieldLabel")
        source_layout.addWidget(camera_label)
        self.camera_selector = QComboBox()
        self.camera_selector.setEnabled(False)
        source_layout.addWidget(self.camera_selector)
        camera_buttons = QHBoxLayout()
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
        self.diagnostic_overlay_toggle = QCheckBox("Show tracking overlay")
        self.diagnostic_overlay_toggle.setChecked(True)
        self.diagnostic_overlay_toggle.setToolTip(
            "Draws landmarks and bounds only in this local preview."
        )
        self.diagnostic_overlay_toggle.toggled.connect(self._render_last_image)
        source_layout.addWidget(self.diagnostic_overlay_toggle)
        controls.addWidget(sources)

        references = QGroupBox("Reference library")
        references.setMinimumHeight(155)
        reference_layout = QVBoxLayout(references)
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
        controls.addWidget(references)

        # These controls remain part of the stable shell API but are not shown until
        # their implementation milestones.
        self.replacement_toggle = QCheckBox("Enable face replacement", self)
        self.replacement_toggle.setEnabled(False)
        self.replacement_toggle.hide()
        self.virtual_camera_button = QPushButton("Start virtual camera", self)
        self.virtual_camera_button.setEnabled(False)
        self.virtual_camera_button.hide()
        self._update_reference_summary()
        controls.addStretch()
        content.addLayout(controls)

        preview_column = QVBoxLayout()
        preview_column.setSpacing(12)
        preview_heading = QLabel("LIVE PREVIEW")
        preview_heading.setObjectName("badge")
        preview_column.addWidget(preview_heading)
        preview = QFrame()
        preview.setObjectName("preview")
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        self.preview_image = QLabel("Searching for physical cameras…")
        self.preview_image.setObjectName("previewImage")
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setMinimumSize(320, 180)
        self.preview_image.installEventFilter(self)
        preview_layout.addWidget(self.preview_image)
        preview_column.addWidget(preview, 1)

        metrics = QGroupBox("Live diagnostics")
        metrics.setObjectName("metricsCard")
        metrics_layout = QGridLayout(metrics)
        metrics_layout.setContentsMargins(18, 22, 18, 18)
        metrics_layout.setHorizontalSpacing(16)
        metrics_layout.setVerticalSpacing(7)
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
        self.tracking_drops_label = QLabel("0")
        self.tracking_drops_label.setObjectName("metricValue")
        self.preview_drops_label = QLabel("0")
        self.preview_drops_label.setObjectName("metricValue")
        state_label = QLabel("State")
        state_label.setObjectName("metricLabel")
        fps_label = QLabel("Capture FPS")
        fps_label.setObjectName("metricLabel")
        tracking_label = QLabel("Face tracking")
        tracking_label.setObjectName("metricLabel")
        confidence_label = QLabel("Confidence")
        confidence_label.setObjectName("metricLabel")
        pose_label = QLabel("Yaw · pitch · roll")
        pose_label.setObjectName("metricLabel")
        smile_label = QLabel("Smile")
        smile_label.setObjectName("metricLabel")
        eyes_label = QLabel("Eye closure L · R")
        eyes_label.setObjectName("metricLabel")
        mouth_label = QLabel("Mouth activity")
        mouth_label.setObjectName("metricLabel")
        latency_label = QLabel("Tracking latency")
        latency_label.setObjectName("metricLabel")
        tracker_drops_label = QLabel("Tracker drops")
        tracker_drops_label.setObjectName("metricLabel")
        drops_label = QLabel("Preview skips")
        drops_label.setObjectName("metricLabel")
        metrics_layout.addWidget(state_label, 0, 0)
        metrics_layout.addWidget(self.capture_state_label, 0, 1)
        metrics_layout.addWidget(tracking_label, 0, 2)
        metrics_layout.addWidget(self.tracking_state_label, 0, 3)
        metrics_layout.addWidget(fps_label, 1, 0)
        metrics_layout.addWidget(self.capture_fps_label, 1, 1)
        metrics_layout.addWidget(confidence_label, 1, 2)
        metrics_layout.addWidget(self.tracking_confidence_label, 1, 3)
        metrics_layout.addWidget(pose_label, 2, 0)
        metrics_layout.addWidget(self.head_pose_label, 2, 1, 1, 3)
        metrics_layout.addWidget(smile_label, 3, 0)
        metrics_layout.addWidget(self.smile_label, 3, 1)
        metrics_layout.addWidget(eyes_label, 3, 2)
        metrics_layout.addWidget(self.eye_closure_label, 3, 3)
        metrics_layout.addWidget(mouth_label, 4, 0)
        metrics_layout.addWidget(self.mouth_activity_label, 4, 1)
        metrics_layout.addWidget(latency_label, 4, 2)
        metrics_layout.addWidget(self.tracking_latency_label, 4, 3)
        metrics_layout.addWidget(drops_label, 5, 0)
        metrics_layout.addWidget(self.preview_drops_label, 5, 1)
        metrics_layout.addWidget(tracker_drops_label, 5, 2)
        metrics_layout.addWidget(self.tracking_drops_label, 5, 3)
        metrics_layout.setColumnStretch(1, 1)
        metrics_layout.setColumnStretch(3, 1)
        preview_column.addWidget(metrics)
        content.addLayout(preview_column, 1)
        layout.addLayout(content, 1)

        footer = QLabel("Milestone 2 · Live face tracking and diagnostics")
        footer.setObjectName("hint")
        layout.addWidget(footer)
        self.statusBar().showMessage("Idle — no camera is open.")

        quit_action = QAction("Exit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        self.menuBar().addMenu("File").addAction(quit_action)

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
        complete = self._reference_library.completed_required_count
        required = self._reference_library.required_count
        state = "Complete" if self._reference_library.is_complete else "Incomplete"
        self.reference_summary_label.setText(f"{state} · {complete} of {required} required")

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
        self._pending_tracking_image = None
        self._last_face_state = None
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
    ) -> tuple[FrameFormat, str | None]:
        negotiated = self._source.open(device_id, requested)
        try:
            self._tracker.open()
        except StageError as exc:
            return negotiated, str(exc)
        return negotiated, None

    def _finish_camera_open(self) -> None:
        future = self._open_future
        if future is None or not future.done():
            return
        self._open_timer.stop()
        self._open_future = None
        self._opening = False
        try:
            negotiated, tracking_error = future.result()
        except StageError as exc:
            self._source.close()
            try:
                self._tracker.close()
            except StageError as close_exc:
                self._logger.warning("Face tracker cleanup failed: %s", close_exc)
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
            self.statusBar().showMessage("Camera preview and live face tracking are running.")
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
        self._capturing = False
        self._last_image = None
        self._last_face_state = None
        self._pending_tracking_image = None
        self._tracking_failed = False
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
            self._last_face_state = None
            self._last_image = image
            self._render_last_image()
            self._set_tracking_state("Error", "error")
            self.statusBar().showMessage(f"Face tracking stopped: {exc}")
            self._logger.error("Face tracking submission failed: %s", exc)
            return
        if accepted:
            self._pending_tracking_image = (frame.frame_id, frame.timestamp_ns, image)
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
        retained = self._pending_tracking_image
        self._pending_tracking_image = None
        if retained is None or retained[:2] != (result.frame_id, result.timestamp_ns):
            self._logger.warning(
                "Discarding tracking result without its matching frame: %s/%s",
                result.frame_id,
                result.timestamp_ns,
            )
            return
        self._last_image = retained[2]
        self._last_face_state = result.face
        self._update_tracking_diagnostics(result)
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
        self._last_face_state = None
        self._render_last_image()

    def _render_last_image(self) -> None:
        display_image = self._oriented_preview_image()
        if display_image is None:
            return
        if self.diagnostic_overlay_toggle.isChecked() and self._last_face_state is not None:
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
        if self._last_image is None:
            return None
        if not self.mirror_preview_toggle.isChecked():
            return self._last_image
        return self._last_image.transformed(
            QTransform().scale(-1.0, 1.0), Qt.TransformationMode.FastTransformation
        )

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

    def _update_tracking_diagnostics(self, result: TrackingResult) -> None:
        self.tracking_latency_label.setText(
            "—" if result.latency_ms is None else f"{result.latency_ms:.1f} ms"
        )
        if result.status is TrackingStatus.ERROR:
            self._tracking_failed = True
            self._set_tracking_state("Error", "error")
            self._clear_face_metrics()
            self.statusBar().showMessage(f"Face tracking stopped: {result.error}")
            return
        if result.status is TrackingStatus.NO_FACE:
            self._set_tracking_state("No face", "no-face")
            self._clear_face_metrics()
            return
        face = result.face
        self._set_tracking_state("Tracked", "running")
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

    def _set_tracking_state(self, text: str, state: str) -> None:
        self.tracking_state_label.setText(text)
        self.tracking_state_label.setProperty("state", state)
        self.tracking_state_label.style().unpolish(self.tracking_state_label)
        self.tracking_state_label.style().polish(self.tracking_state_label)

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
        try:
            self._reference_session.flush()
        except StageError as exc:
            self._logger.error("Could not flush reference library at shutdown: %s", exc)
        self._reference_session.close()
        event.accept()
