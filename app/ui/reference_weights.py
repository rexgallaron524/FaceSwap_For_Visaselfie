"""Compact developer visualization for pose-space reference weights."""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QProgressBar, QWidget

from app.pipeline.types import ReferenceWeight
from app.reference.model import ReferenceSlot


class ReferenceWeightsWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(10)
        self._layout.setVerticalSpacing(5)
        self._bars: dict[str, QProgressBar] = {}

    def set_references(
        self, slots: tuple[ReferenceSlot, ...], available_reference_ids: set[str]
    ) -> None:
        while item := self._layout.takeAt(0):
            if widget := item.widget():
                widget.deleteLater()
        self._bars.clear()
        row_count = max(1, (len(slots) + 1) // 2)
        for index, slot in enumerate(slots):
            column_group = index // row_count
            row = index % row_count
            label = QLabel(slot.label)
            label.setObjectName("weightLabel")
            bar = QProgressBar()
            bar.setObjectName("referenceWeight")
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setFormat("0.0%")
            bar.setTextVisible(True)
            bar.setFixedHeight(18)
            available = slot.slot_id in available_reference_ids
            label.setEnabled(available)
            bar.setEnabled(available)
            bar.setAccessibleName(f"{slot.label} reference weight")
            self._layout.addWidget(label, row, column_group * 2)
            self._layout.addWidget(bar, row, column_group * 2 + 1)
            self._layout.setColumnStretch(column_group * 2 + 1, 1)
            self._bars[slot.slot_id] = bar

    def set_weights(self, weights: tuple[ReferenceWeight, ...]) -> None:
        values = {weight.reference_id: weight.weight for weight in weights}
        for reference_id, bar in self._bars.items():
            value = values.get(reference_id, 0.0)
            bar.setValue(round(value * 1000))
            bar.setFormat(f"{value * 100:.1f}%")

    def displayed_weights(self) -> dict[str, float]:
        return {reference_id: bar.value() / 1000 for reference_id, bar in self._bars.items()}
