"""Compact developer visualization for pose-space reference weights."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QProgressBar, QSizePolicy, QWidget

from app.pipeline.types import ReferenceWeight
from app.reference.model import ReferenceSlot


class ReferenceWeightsWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(9)
        self._layout.setVerticalSpacing(6)
        self._layout.setColumnMinimumWidth(0, 112)
        self._layout.setColumnMinimumWidth(2, 48)
        self._layout.setColumnStretch(1, 1)
        self._bars: dict[str, QProgressBar] = {}
        self._value_labels: dict[str, QLabel] = {}
        self._values: dict[str, float] = {}
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def set_references(
        self, slots: tuple[ReferenceSlot, ...], available_reference_ids: set[str]
    ) -> None:
        while item := self._layout.takeAt(0):
            if widget := item.widget():
                widget.deleteLater()
        self._bars.clear()
        self._value_labels.clear()
        self._values = {slot.slot_id: 0.0 for slot in slots}
        for row, slot in enumerate(slots):
            label = QLabel(slot.label)
            label.setObjectName("weightLabel")
            label.setMinimumHeight(24)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            bar = QProgressBar()
            bar.setObjectName("referenceWeight")
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setMinimumHeight(22)
            value_label = QLabel("0.0%")
            value_label.setObjectName("weightValue")
            value_label.setMinimumHeight(24)
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            available = slot.slot_id in available_reference_ids
            label.setEnabled(available)
            bar.setEnabled(available)
            value_label.setEnabled(available)
            bar.setAccessibleName(f"{slot.label} reference weight")
            self._layout.addWidget(label, row, 0)
            self._layout.addWidget(bar, row, 1)
            self._layout.addWidget(value_label, row, 2)
            self._layout.setRowMinimumHeight(row, 24)
            self._bars[slot.slot_id] = bar
            self._value_labels[slot.slot_id] = value_label

    def set_weights(self, weights: tuple[ReferenceWeight, ...]) -> None:
        values = {weight.reference_id: weight.weight for weight in weights}
        for reference_id, bar in self._bars.items():
            value = values.get(reference_id, 0.0)
            self._values[reference_id] = value
            bar.setValue(round(value * 1000))
            self._value_labels[reference_id].setText(f"{value * 100:.1f}%")

    def displayed_weights(self) -> dict[str, float]:
        return dict(self._values)
