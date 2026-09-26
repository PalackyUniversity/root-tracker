"""Palette-aware popup menus used by the GUI."""

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QMenu, QMenuBar, QWidget
from pathlib import Path

from .theme import menu_border, blend


_MENU_STYLE = """
    QMenu {
        background-color: palette(button);
        border: 1px solid MENU_BORDER;
        border-radius: 7px;
        padding: 3px;
    }
    QMenu::item {
        padding: 5px 10px 5px 12px;
        border-radius: 4px;
    }
    QMenu::item:selected {
        background-color: palette(highlight);
        color: palette(highlighted-text);
    }
    QMenu::indicator {
        width: 16px;
        height: 16px;
        subcontrol-position: center left;
        left: 4px;
        border: 1px solid MENU_BORDER;
        border-radius: 4px;
        background-color: palette(base);
    }
    QMenu::indicator:checked {
        background-color: palette(highlight);
        border-color: palette(highlight);
        image: url("CHECK_ICON");
    }
    QMenu::indicator:checked:selected { border-color: palette(highlighted-text); }
    QMenu::separator {
        height: 1px;
        background-color: SEPARATOR_COLOR;
        margin: 6px 10px;
        border: none;
    }
"""


class RoundedMenu(QMenu):
    """A menu with rounded corners that follows the active Qt palette."""

    def __init__(self, parent: QWidget, title: str = "") -> None:
        super().__init__(title, parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        check_icon = (Path(__file__).parent / "icons" / "check.svg").as_posix()
        separator = blend(self.palette().color(QPalette.ColorRole.Button),
                          self.palette().color(QPalette.ColorRole.WindowText), .18).name()
        self.setStyleSheet(_MENU_STYLE.replace("MENU_BORDER", menu_border(self.palette()))
                          .replace("CHECK_ICON", check_icon)
                          .replace("SEPARATOR_COLOR", separator))


class MenuBarPopup(RoundedMenu):
    """Place a top-level dropdown directly below its centered menu item."""

    def showEvent(self, event) -> None:
        super().showEvent(event)
        bar = self.parentWidget()
        if isinstance(bar, QMenuBar):
            # The menu bar items have a 4 px bottom margin for centering.
            action_rect = bar.actionGeometry(self.menuAction())
            popup_y = bar.mapToGlobal(QPoint(0, action_rect.bottom() - 3)).y()
            self.move(self.x(), popup_y)
