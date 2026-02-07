"""
Workflow step indicator bar.

Shows the 4 pipeline steps and highlights the current one.
"""

from enum import IntEnum
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QFrame, QLabel
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


class WorkflowStep(IntEnum):
    """Pipeline workflow steps."""
    LOAD = 0
    PREPROCESS = 1
    TRACK = 2
    EXPORT = 3


STEP_NAMES = {
    WorkflowStep.LOAD: "1. Load Images",
    WorkflowStep.PREPROCESS: "2. Preprocess",
    WorkflowStep.TRACK: "3. Track Roots",
    WorkflowStep.EXPORT: "4. Export",
}


class WorkflowBar(QWidget):
    """
    Horizontal bar showing workflow steps.
    
    Each step is a clickable button. The current step is highlighted.
    Steps can be enabled/disabled based on progress.
    
    Signals:
        step_changed: Emitted when user clicks a different step.
    """
    
    step_changed = Signal(WorkflowStep)
    
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        self._current_step = WorkflowStep.LOAD
        self._step_buttons: dict[WorkflowStep, QPushButton] = {}
        self._completed_steps: set[WorkflowStep] = set()
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(5)
        
        for step in WorkflowStep:
            btn = QPushButton(STEP_NAMES[step])
            btn.setCheckable(True)
            btn.setMinimumWidth(120)
            btn.clicked.connect(lambda checked, s=step: self._on_step_clicked(s))
            
            self._step_buttons[step] = btn
            layout.addWidget(btn)
            
            # Add arrow between steps (except last)
            if step != WorkflowStep.EXPORT:
                arrow = QLabel("→")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                font = QFont()
                font.setPointSize(14)
                arrow.setFont(font)
                layout.addWidget(arrow)
        
        layout.addStretch()
        
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
        """Update button checked states and styling."""
        for step, btn in self._step_buttons.items():
            btn.setChecked(step == self._current_step)
            
            # Style based on state
            if step == self._current_step:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4a90d9;
                        color: white;
                        font-weight: bold;
                        border: 2px solid #2a70b9;
                        border-radius: 5px;
                        padding: 8px;
                    }
                """)
            elif step in self._completed_steps:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #5cb85c;
                        color: white;
                        border: 2px solid #4cae4c;
                        border-radius: 5px;
                        padding: 8px;
                    }
                    QPushButton:hover {
                        background-color: #449d44;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f0f0f0;
                        color: #666;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                        padding: 8px;
                    }
                    QPushButton:hover {
                        background-color: #e0e0e0;
                    }
                """)
