"""FaceLive desktop shell with physical-camera preview."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QImage, QPixmap, QTransform
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

from app.camera import OpenCVCameraSource
from app.camera.metrics import CaptureMetrics
from app.camera.protocol import CameraSource
from app.config import AppConfig
from app.pipeline.types import CameraDevice, FrameFormat, StageError, VideoFrame


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: AppConfig,
        camera_source_factory: Callable[[], CameraSource] = OpenCVCameraSource,
    ) -> None:
        super().__init__()
        self._config = config
        self._source = camera_source_factory()
        self._devices: tuple[CameraDevice, ...] = ()
        self._capturing = False
        self._opening = False
        self._last_image: QImage | None = None
        self._metrics = CaptureMetrics()
        self._logger = logging.getLogger("facelive.ui")
        self._camera_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-open")
        self._open_future: Future[FrameFormat] | None = None

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
            QLabel#previewImage { color: #aabac7; font-size: 18px; }
            QLabel#badge { color: #85dbc9; font-weight: 600; }
            QStatusBar { background: #17232d; color: #aabac7; }
            QMenuBar, QMenu { background: #17232d; color: #edf3f7; }
            QMenuBar::item:selected, QMenu::item:selected { background: #354550; }
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
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(18)

        title = QLabel("FaceLive")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel("Live physical-camera preview")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(22)
        controls = QVBoxLayout()
        sources = QGroupBox("Camera input")
        sources.setFixedWidth(285)
        source_layout = QVBoxLayout(sources)
        source_layout.addWidget(QLabel("Physical camera"))
        self.camera_selector = QComboBox()
        self.camera_selector.setEnabled(False)
        source_layout.addWidget(self.camera_selector)
        camera_buttons = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_cameras)
        camera_buttons.addWidget(self.refresh_button)
        self.capture_button = QPushButton("Start preview")
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
        controls.addWidget(sources)

        future = QGroupBox("Later milestones")
        future_layout = QVBoxLayout(future)
        self.load_references = QPushButton("Load references…")
        self.load_references.setEnabled(False)
        future_layout.addWidget(self.load_references)
        self.replacement_toggle = QCheckBox("Enable face replacement")
        self.replacement_toggle.setEnabled(False)
        future_layout.addWidget(self.replacement_toggle)
        self.virtual_camera_button = QPushButton("Start virtual camera")
        self.virtual_camera_button.setEnabled(False)
        future_layout.addWidget(self.virtual_camera_button)
        controls.addWidget(future)
        controls.addStretch()
        content.addLayout(controls)

        preview_column = QVBoxLayout()
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

        metrics = QGroupBox("Capture session")
        metrics_layout = QFormLayout(metrics)
        self.capture_state_label = QLabel("Idle")
        self.capture_fps_label = QLabel("—")
        self.preview_drops_label = QLabel("0")
        metrics_layout.addRow("State", self.capture_state_label)
        metrics_layout.addRow("Measured capture FPS", self.capture_fps_label)
        metrics_layout.addRow("Frames skipped by preview", self.preview_drops_label)
        preview_column.addWidget(metrics)
        content.addLayout(preview_column, 1)
        layout.addLayout(content, 1)

        footer = QLabel("Milestone 1 · Webcam capture and live preview")
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

    def start_capture(self) -> None:
        device_id = self.camera_selector.currentData()
        if not device_id:
            return
        requested = FrameFormat(
            self._config.video.width, self._config.video.height, self._config.video.fps
        )
        self._opening = True
        self._metrics.reset()
        self.camera_selector.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.capture_button.setText("Opening…")
        self.capture_button.setEnabled(False)
        self.capture_state_label.setText("Opening")
        self.preview_image.setText("Opening camera…")
        self.statusBar().showMessage("Opening the selected physical camera…")
        self._open_future = self._camera_executor.submit(
            self._source.open, str(device_id), requested
        )
        self._open_timer.start()

    def _finish_camera_open(self) -> None:
        future = self._open_future
        if future is None or not future.done():
            return
        self._open_timer.stop()
        self._open_future = None
        self._opening = False
        try:
            negotiated = future.result()
        except StageError as exc:
            self._show_camera_error(str(exc))
            self.camera_selector.setEnabled(bool(self._devices))
            self.refresh_button.setEnabled(True)
            self.capture_button.setText("Start preview")
            self.capture_button.setEnabled(bool(self._devices))
            return
        except Exception as exc:
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
        self.capture_state_label.setText("Running")
        self.camera_format_label.setText(
            f"Active: {negotiated.width} × {negotiated.height} · "
            f"device reports {negotiated.fps:.1f} FPS"
        )
        self.statusBar().showMessage("Camera preview is running. No face processing is active.")
        self._preview_timer.start()
        self._metrics_timer.start()

    def stop_capture(self, *, message: str = "Camera preview stopped.") -> None:
        self._preview_timer.stop()
        self._metrics_timer.stop()
        self._source.close()
        self._capturing = False
        self._last_image = None
        self._metrics.reset()
        self.camera_selector.setEnabled(bool(self._devices))
        self.refresh_button.setEnabled(True)
        self.capture_button.setText("Start preview")
        self.capture_button.setEnabled(bool(self._devices))
        self.capture_state_label.setText("Idle")
        self.capture_fps_label.setText("—")
        self.preview_drops_label.setText("0")
        self.preview_image.setPixmap(QPixmap())
        self.preview_image.setText("Preview is idle.")
        self.camera_format_label.setText(
            f"Requested: {self._config.video.width} × {self._config.video.height} · "
            f"{self._config.video.fps} FPS"
        )
        self.statusBar().showMessage(message)

    def _poll_camera(self) -> None:
        try:
            frame = self._source.read_latest()
        except StageError as exc:
            detail = str(exc)
            self._logger.warning("Camera capture stopped: %s", detail)
            self.stop_capture(message=f"Camera error: {detail}")
            self.capture_state_label.setText("Disconnected")
            self.preview_image.setText(
                "Camera disconnected or stopped responding.\nReconnect it, then choose Refresh."
            )
            return
        if frame is None:
            return
        self._metrics.observe(frame)
        self._display_frame(frame)

    def _display_frame(self, frame: VideoFrame) -> None:
        height, width = frame.rgb.shape[:2]
        bytes_per_line = int(frame.rgb.strides[0])
        self._last_image = QImage(
            frame.rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        self._render_last_image()

    def _render_last_image(self) -> None:
        display_image = self._oriented_preview_image()
        if display_image is None:
            return
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

    def _update_metrics(self) -> None:
        fps = self._metrics.fps
        self.capture_fps_label.setText("—" if fps is None else f"{fps:.1f}")
        self.preview_drops_label.setText(str(self._metrics.preview_drops))

    def _show_camera_error(self, detail: str) -> None:
        self._logger.warning("Camera operation failed: %s", detail)
        self.capture_state_label.setText("Error")
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
        event.accept()
