"""Palette-aware popup menus used by the GUI."""

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QMenu, QMenuBar, QWidget
from .theme import menu_border


_MENU_STYLE = """
    QMenu {
        background-color: palette(button);
        border: 1px solid MENU_BORDER;
        border-radius: 7px;
        padding: 3px;
    }
    QMenu::item {
        padding: 4px 22px;
        border-radius: 4px;
    }
    QMenu::item:selected {
        background-color: palette(highlight);
        color: palette(highlighted-text);
    }
    QMenu::indicator {
        width: 13px;
        height: 13px;
        subcontrol-position: center left;
        left: 12px;
    }
    QMenu::separator {
        height: 1px;
        background-color: palette(mid);
        margin: 3px 5px;
    }
"""


class RoundedMenu(QMenu):
    """A menu with rounded corners that follows the active Qt palette."""

    def __init__(self, parent: QWidget, title: str = "") -> None:
        super().__init__(title, parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(_MENU_STYLE.replace("MENU_BORDER", menu_border(self.palette())))


class MenuBarPopup(RoundedMenu):
    """Place a top-level dropdown directly below its centered menu item."""

    def showEvent(self, event) -> None:
        super().showEvent(event)
        bar = self.parentWidget()
        if isinstance(bar, QMenuBar):
            # The menu bar items have an 8 px bottom margin for centering.
            action_rect = bar.actionGeometry(self.menuAction())
            popup_y = bar.mapToGlobal(QPoint(0, action_rect.bottom() - 7)).y()
            self.move(self.x(), popup_y)
