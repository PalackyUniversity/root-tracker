"""
Workflow step indicator bar.

Shows the 4 pipeline steps with minimal Lightroom-style buttons.
"""

from enum import IntEnum
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QPalette, QPainter, QPen, QPainterPath
from .theme import blend


class WorkflowStep(IntEnum):
    """Pipeline workflow steps."""
    LOAD = 0
    PREPROCESS = 1
    TRACK = 2


# Step names without numbers
STEP_NAMES = {
    WorkflowStep.LOAD: "Load",
    WorkflowStep.PREPROCESS: "Preprocess",
    WorkflowStep.TRACK: "Track",
}


class WorkflowChevron(QWidget):
    """A soft, round-ended directional separator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 20)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = blend(self.palette().color(QPalette.ColorRole.Window),
                      self.palette().color(QPalette.ColorRole.WindowText), .35)
        painter.setPen(QPen(color, 1.5, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        path = QPainterPath()
        path.moveTo(5, 7)
        path.lineTo(8, 10)
        path.lineTo(5, 13)
        painter.drawPath(path)


class WorkflowBar(QWidget):
    """
    Horizontal bar showing workflow steps.
    
    Each step is a minimal clickable button (Lightroom-style).
    Steps can be enabled/disabled based on progress.
    
    Signals:
        step_changed: Emitted when user clicks a different step.
    """
    
    step_changed = Signal(WorkflowStep)
    
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        self._current_step = WorkflowStep.LOAD
        self._step_buttons: dict[WorkflowStep, QPushButton] = {}
        self._separators: list[WorkflowChevron] = []
        self._completed_steps: set[WorkflowStep] = set()
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        # Minimal fixed height
        self.setFixedHeight(42)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(2)
        
        for step in WorkflowStep:
            btn = QPushButton(STEP_NAMES[step])
            btn.setCheckable(True)
            btn.setMinimumWidth(80)
            btn.clicked.connect(lambda checked, s=step: self._on_step_clicked(s))
            
            self._step_buttons[step] = btn
            layout.addWidget(btn)
            
            # Add separator between steps (except last)
            if step != WorkflowStep.TRACK:
                sep = WorkflowChevron()
                self._separators.append(sep)
                layout.addWidget(sep)
        
        # Set initial state
        self._update_button_states()
    
    def _on_step_clicked(self, step: WorkflowStep) -> None:
        """Handle step button click."""
        if step != self._current_step:
            self.set_current_step(step)
            self.step_changed.emit(step)
    
    def set_current_step(self, step: WorkflowStep) -> None:
        """
        Set the current workflow step.
        
        Args:
            step: The step to make current.
        """
        self._current_step = step
        self._update_button_states()
    
    def mark_step_completed(self, step: WorkflowStep) -> None:
        """
        Mark a step as completed.
        
        Args:
            step: The step to mark as done.
        """
        self._completed_steps.add(step)
        self._update_button_states()
    
    def is_step_completed(self, step: WorkflowStep) -> bool:
        """Check if a step is completed."""
        return step in self._completed_steps
    
    def get_current_step(self) -> WorkflowStep:
        """Get the current step."""
        return self._current_step
    
    def _update_button_states(self) -> None:
        """Update button checked states and styling (Lightroom-style)."""
        palette = self.palette()
        foreground = palette.color(QPalette.ColorRole.WindowText)
        background = palette.color(QPalette.ColorRole.Window)
        muted = QColor.fromRgb(*(
            round(fg * 0.72 + bg * 0.28)
            for fg, bg in zip(foreground.getRgb()[:3], background.getRgb()[:3])
        )).name()
        accent = palette.color(QPalette.ColorRole.Highlight).name()
        for step, btn in self._step_buttons.items():
            active = step == self._current_step
            btn.setChecked(active)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {accent if active else muted};
                    font-weight: normal;
                    border: 1px solid {accent if active else "transparent"};
                    border-radius: 5px;
                    padding: 6px 10px;
                }}
                QPushButton:hover {{ background-color: palette(midlight); }}
                QPushButton:focus {{ border: 1px solid palette(highlight); }}
            """)
