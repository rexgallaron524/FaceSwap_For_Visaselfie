"""Reference enrollment dialog with per-slot thumbnails and validation status."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.pipeline.types import StageError
from app.reference.model import ProcessedReference, ReferenceSlot
from app.reference.session import ReferenceLibrarySession, ReferencePersistenceError


class ReferenceCard(QFrame):
    def __init__(
        self,
        slot: ReferenceSlot,
        choose: Callable[[str], None],
        remove: Callable[[str], None],
    ) -> None:
        super().__init__()
        self.slot = slot
        self.setObjectName("referenceCard")
        self.setMinimumWidth(205)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(7)

        title = QLabel(slot.label)
        title.setObjectName("referenceCardTitle")
        layout.addWidget(title)
        guidance = QLabel(slot.guidance)
        guidance.setObjectName("referenceGuidance")
        guidance.setWordWrap(True)
        guidance.setFixedHeight(38)
        layout.addWidget(guidance)
        self.thumbnail = QLabel("No image")
        self.thumbnail.setObjectName("referenceThumbnail")
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail.setMinimumWidth(170)
        self.thumbnail.setFixedHeight(150)
        self.thumbnail.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.thumbnail)
        self.status = QLabel("Missing")
        self.status.setObjectName("referenceStatus")
        self.status.setProperty("valid", False)
        layout.addWidget(self.status)
        self.source = QLabel("Choose a clear image with one face.")
        self.source.setObjectName("referenceGuidance")
        self.source.setWordWrap(True)
        layout.addWidget(self.source)
        buttons = QHBoxLayout()
        self.choose_button = QPushButton("Choose image…")
        self.choose_button.clicked.connect(lambda: choose(slot.slot_id))
        buttons.addWidget(self.choose_button)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(lambda: remove(slot.slot_id))
        self.remove_button.setEnabled(False)
        buttons.addWidget(self.remove_button)
        layout.addLayout(buttons)

    def show_entry(self, entry: ProcessedReference | None) -> None:
        if entry is None:
            self.thumbnail.setPixmap(QPixmap())
            self.thumbnail.setText("No image")
            self._set_status("Missing", False)
            self.source.setText("Choose a clear image with one face.")
            self.remove_button.setEnabled(False)
            return
        height, width = entry.rgb.shape[:2]
        image = QImage(
            entry.rgb.data,
            width,
            height,
            int(entry.rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.thumbnail.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.thumbnail.setText("")
        self.thumbnail.setPixmap(pixmap)
        self._set_status("Valid", True)
        self.source.setText(entry.metadata.source_name)
        self.remove_button.setEnabled(True)

    def show_processing(self) -> None:
        self._set_status("Validating and aligning…", False)
        self.choose_button.setEnabled(False)

    def finish_processing(self) -> None:
        self.choose_button.setEnabled(True)

    def _set_status(self, text: str, valid: bool) -> None:
        self.status.setText(text)
        self.status.setProperty("valid", valid)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)


class ReferenceEnrollmentDialog(QDialog):
    library_changed = Signal()

    def __init__(self, session: ReferenceLibrarySession, parent=None) -> None:
        super().__init__(parent)
        self._session = session
        self._library = session.library
        self._cards: dict[str, ReferenceCard] = {}
        self.setWindowTitle("Reference Library — FaceLive")
        self.resize(1120, 760)
        self.setMinimumSize(900, 680)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)
        title = QLabel("Reference face library")
        title.setObjectName("dialogTitle")
        root.addWidget(title)
        guidance = QLabel(
            "Use recent images of the same consenting person. Keep lighting, camera distance, "
            "and framing consistent. Each image must contain one clear, unobstructed face."
        )
        guidance.setObjectName("dialogDescription")
        guidance.setWordWrap(True)
        root.addWidget(guidance)

        toolbar = QHBoxLayout()
        self.load_button = QPushButton("Load library…")
        self.load_button.clicked.connect(self._load_library)
        toolbar.addWidget(self.load_button)
        self.save_button = QPushButton("Save as…")
        self.save_button.setObjectName("primaryButton")
        self.save_button.clicked.connect(self._save_library)
        toolbar.addWidget(self.save_button)
        toolbar.addStretch()
        self.summary = QLabel()
        self.summary.setObjectName("referenceSummary")
        toolbar.addWidget(self.summary)
        root.addLayout(toolbar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(self.scroll, 1)
        self._rebuild_cards()

        footer = QHBoxLayout()
        self.location = QLabel()
        self.location.setObjectName("librarySaveState")
        self.location.setWordWrap(True)
        footer.addWidget(self.location, 1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        footer.addWidget(close_button)
        root.addLayout(footer)
        self.refresh()

    def _rebuild_cards(self) -> None:
        container = QWidget()
        container.setObjectName("referenceGrid")
        grid = QGridLayout(container)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        self._cards = {}
        for index, slot in enumerate(self._library.slots()):
            card = ReferenceCard(slot, self._choose_image, self._remove_reference)
            self._cards[slot.slot_id] = card
            grid.addWidget(card, index // 4, index % 4)
        grid.setRowStretch((len(self._cards) - 1) // 4 + 1, 1)
        self.scroll.setWidget(container)

    def refresh(self) -> None:
        for slot_id, card in self._cards.items():
            card.show_entry(self._library.processed(slot_id))
        complete = self._library.completed_required_count
        required = self._library.required_count
        self.summary.setText(f"{complete} of {required} required references valid")
        self.summary.setProperty("complete", complete == required)
        self.summary.style().unpolish(self.summary)
        self.summary.style().polish(self.summary)
        self.save_button.setEnabled(self._library.valid_count > 0)
        if self._session.last_error:
            self.location.setText(f"Storage error · {self._session.last_error}")
            self.location.setProperty("saved", False)
        elif self._session.dirty:
            self.location.setText(f"Unsaved changes · {self._session.active_path}")
            self.location.setProperty("saved", False)
        elif self._session.is_saved:
            self.location.setText(f"Saved automatically · {self._session.active_path}")
            self.location.setProperty("saved", True)
        else:
            self.location.setText(f"Will save automatically to {self._session.active_path}")
            self.location.setProperty("saved", False)
        self.location.style().unpolish(self.location)
        self.location.style().polish(self.location)

    def _choose_image(self, slot_id: str) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            f"Choose {self._library.slot(slot_id).label} reference",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*)",
        )
        if not filename:
            return
        card = self._cards[slot_id]
        card.show_processing()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            self._session.add_reference(slot_id, Path(filename))
        except ReferencePersistenceError as exc:
            QMessageBox.warning(self, "Reference added but not saved", str(exc))
        except StageError as exc:
            QMessageBox.warning(self, "Reference rejected", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
            card.finish_processing()
        self.refresh()
        self.library_changed.emit()

    def _remove_reference(self, slot_id: str) -> None:
        try:
            self._session.remove_reference(slot_id)
        except ReferencePersistenceError as exc:
            QMessageBox.warning(self, "Reference removed but not saved", str(exc))
        self.refresh()
        self.library_changed.emit()

    def _save_library(self) -> None:
        initial = str(self._session.active_path)
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save reference library", initial, "FaceLive library (*.json)"
        )
        if not filename:
            return
        try:
            self._session.save_as(Path(filename))
        except StageError as exc:
            QMessageBox.critical(self, "Could not save library", str(exc))
            return
        self.refresh()
        self.library_changed.emit()

    def _load_library(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load reference library", "", "FaceLive library (*.json)"
        )
        if not filename:
            return
        try:
            self._session.load(Path(filename))
        except ReferencePersistenceError as exc:
            QMessageBox.warning(self, "Library loaded but not remembered", str(exc))
            self._rebuild_cards()
            self.refresh()
            self.library_changed.emit()
            return
        except StageError as exc:
            QMessageBox.critical(self, "Could not load library", str(exc))
            self._rebuild_cards()
            self.refresh()
            self.library_changed.emit()
            return
        self._rebuild_cards()
        self.refresh()
        self.library_changed.emit()
