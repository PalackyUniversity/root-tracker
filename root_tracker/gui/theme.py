"""Small palette-based styles for the application's controls."""

from pathlib import Path

from PySide6.QtGui import QColor, QPalette


def blend(base: QColor, accent: QColor, amount: float) -> QColor:
    """Mix two palette colors without losing the active light or dark theme."""
    channels = (
        round(a * (1 - amount) + b * amount)
        for a, b in zip(base.getRgb()[:3], accent.getRgb()[:3])
    )
    return QColor.fromRgb(*channels)


def is_light(palette: QPalette) -> bool:
    return palette.color(QPalette.ColorRole.Window).lightness() >= 128


def menu_border(palette: QPalette) -> str:
    surface = palette.color(QPalette.ColorRole.Button)
    text = palette.color(QPalette.ColorRole.WindowText)
    return blend(surface, text, 0.32 if is_light(palette) else 0.27).name()


def button_stylesheet(palette: QPalette) -> str:
    light = is_light(palette)
    window = palette.color(QPalette.ColorRole.Window)
    surface = palette.color(QPalette.ColorRole.Base if light else QPalette.ColorRole.Button)
    text = palette.color(QPalette.ColorRole.ButtonText)
    accent = palette.color(QPalette.ColorRole.Highlight)
    border = blend(surface, text, 0.22 if light else 0.24).name()
    hover = blend(surface, accent, 0.10 if light else 0.18).name()
    hover_border = blend(surface, accent, 0.40).name()
    pressed = blend(surface, accent, 0.20 if light else 0.26).name()
    disabled_bg = blend(window, surface, 0.30).name()
    disabled_text = blend(window, text, 0.42).name()
    divider = blend(window, text, .14 if light else .22).name()
    check_icon = (Path(__file__).parent / "icons" / "check.svg").as_posix()
    arrow_tone = "dark" if light else "light"
    down_icon = (Path(__file__).parent / "icons" / f"chevron-down-{arrow_tone}.svg").as_posix()
    spin_down_icon = (Path(__file__).parent / "icons" / f"spin-down-{arrow_tone}.svg").as_posix()
    up_icon = (Path(__file__).parent / "icons" / f"chevron-up-{arrow_tone}.svg").as_posix()
    return f"""
        QDialogButtonBox {{ dialogbuttonbox-buttons-have-icons: 0; }}
        QMenuBar {{ border-bottom: 1px solid {divider}; }}
        QStatusBar {{ border-top: 1px solid {divider}; }}
        QStatusBar::item {{ border: none; }}
        QSplitter::handle {{ background-color: {divider}; }}
        SettingsPanel QGroupBox {{
            border: 1px solid {border};
            border-radius: 7px;
            margin-top: 10px;
            padding-top: 8px;
        }}
        SettingsPanel QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0px 4px;
        }}
        QPushButton {{
            background-color: {surface.name()};
            color: {text.name()};
            border: 1px solid {border};
            border-radius: 5px;
            padding: 6px 10px;
            min-height: 18px;
        }}
        QPushButton:hover {{
            background-color: {hover};
            border-color: {hover_border};
        }}
        QPushButton:pressed {{ background-color: {pressed}; }}
        QPushButton:focus {{ border-color: {accent.name()}; }}
        QPushButton:disabled {{
            background-color: {disabled_bg};
            color: {disabled_text};
            border-color: {border};
        }}
        QToolButton {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 5px;
        }}
        QToolButton:hover {{
            background-color: {hover};
            border-color: {border};
        }}
        QToolButton:pressed {{ background-color: {pressed}; }}
        QToolButton:focus {{ border-color: {accent.name()}; }}
        QComboBox {{
            combobox-popup: 0;
            background-color: {surface.name()}; color: {text.name()};
            border: 1px solid {border}; border-radius: 5px;
            padding: 5px 24px 6px 8px; min-height: 18px;
        }}
        QComboBox:hover {{ background-color: {hover}; border-color: {hover_border}; }}
        QComboBox:focus {{ border-color: {accent.name()}; }}
        QComboBox:on {{ background-color: {pressed}; border-color: {accent.name()}; }}
        QComboBox:disabled {{ background-color: {disabled_bg}; color: {disabled_text}; border-color: {border}; }}
        QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right; width: 22px; border: none; }}
        QComboBox::down-arrow {{ image: url("{down_icon}"); width: 12px; height: 12px; }}
        QComboBox QAbstractItemView {{
            background-color: {surface.name()}; color: {text.name()};
            border: 1px solid {border}; border-radius: 5px; padding: 3px;
            outline: none; selection-background-color: {accent.name()};
            selection-color: palette(highlighted-text);
        }}
        QComboBox QAbstractItemView::item {{ min-height: 24px; padding: 2px 6px; margin: 1px; border-radius: 4px; }}
        QComboBox QAbstractItemView::item:hover:!selected {{ background-color: {hover}; }}
        QComboBox QAbstractItemView::item:selected {{ background-color: {accent.name()}; color: palette(highlighted-text); }}
        QAbstractSpinBox {{
            background-color: {surface.name()}; color: {text.name()};
            border: 1px solid {border}; border-radius: 5px;
            padding: 3px 24px 3px 8px; min-height: 18px;
            selection-background-color: {accent.name()};
            selection-color: palette(highlighted-text);
        }}
        QAbstractSpinBox:hover {{ border-color: {hover_border}; }}
        QAbstractSpinBox:focus {{ border-color: {accent.name()}; }}
        QAbstractSpinBox:disabled {{ background-color: {disabled_bg}; color: {disabled_text}; }}
        QAbstractSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 22px; border: none; border-top-right-radius: 5px; margin: 1px; }}
        QAbstractSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 22px; border: none; border-bottom-right-radius: 5px; margin: 1px; }}
        QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background-color: {hover}; }}
        QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {{ background-color: {pressed}; }}
        QAbstractSpinBox::up-arrow {{ image: url("{up_icon}"); width: 12px; height: 12px; position: relative; top: 2px; }}
        QAbstractSpinBox::down-arrow {{ image: url("{spin_down_icon}"); width: 12px; height: 12px; position: relative; top: -2px; }}
        QCheckBox {{ spacing: 7px; min-height: 22px; }}
        QCheckBox::indicator {{
            width: 16px; height: 16px; border: 1px solid {border};
            border-radius: 4px; background-color: {surface.name()};
        }}
        QCheckBox::indicator:hover {{ background-color: {hover}; border-color: {hover_border}; }}
        QCheckBox::indicator:checked {{ background-color: {accent.name()}; border-color: {accent.name()}; image: url("{check_icon}"); }}
        QCheckBox::indicator:checked:hover {{ background-color: {accent.lighter(110).name()}; }}
        QCheckBox::indicator:focus {{ border-color: {accent.name()}; }}
        QCheckBox::indicator:disabled {{ background-color: {disabled_bg}; border-color: {border}; }}
        QCheckBox::indicator:checked:disabled {{ background-color: {disabled_text}; }}
    """


def presets_stylesheet(palette: QPalette) -> str:
    """Use the shared controls with spacious preset navigation and editor tabs."""
    light = is_light(palette)
    surface = palette.color(QPalette.ColorRole.Base if light else QPalette.ColorRole.Button)
    text = palette.color(QPalette.ColorRole.ButtonText)
    accent = palette.color(QPalette.ColorRole.Highlight)
    border = blend(surface, text, .22 if light else .24).name()
    hover = blend(surface, accent, .10 if light else .18).name()
    return button_stylesheet(palette) + scrollbar_stylesheet(palette) + f"""
        QLineEdit[presetField="true"] {{
            background-color: {surface.name()}; color: {text.name()};
            border: 1px solid {border}; border-radius: 5px;
            padding: 3px 8px; min-height: 18px;
            selection-background-color: {accent.name()};
            selection-color: palette(highlighted-text);
        }}
        QLineEdit[presetField="true"]:hover {{ border-color: {accent.name()}; }}
        QLineEdit[presetField="true"]:focus {{ border-color: {accent.name()}; }}
        QTabWidget#presetEditor::pane {{
            border: 1px solid {border}; border-radius: 8px;
            background-color: palette(base); padding: 4px;
        }}
        QScrollArea#presetScroll {{ border: none; background-color: palette(base); }}
        QWidget#presetPage {{ background-color: palette(base); }}
        QTabWidget#presetEditor QTabBar::tab {{
            background-color: {surface.name()}; color: {text.name()};
            border: 1px solid {border}; border-radius: 7px;
            padding: 7px 10px; min-height: 18px;
            margin-right: 4px; margin-bottom: 7px;
        }}
        QTabWidget#presetEditor QTabBar::tab:hover:!selected {{ background-color: {hover}; }}
        QTabWidget#presetEditor QTabBar::tab:selected {{
            background-color: {accent.name()}; color: palette(highlighted-text);
            border-color: {accent.name()};
        }}
        QListWidget#presetList {{
            background-color: palette(base); border: 1px solid {border};
            border-radius: 8px; padding: 5px; outline: none;
        }}
        QListWidget#presetList::item {{
            min-height: 26px; padding: 5px 8px; margin: 2px 0px;
            border-radius: 6px;
        }}
        QListWidget#presetList::item:hover:!selected {{ background-color: {hover}; }}
        QListWidget#presetList::item:selected {{
            background-color: {accent.name()}; color: palette(highlighted-text);
        }}
    """


def tree_stylesheet(palette: QPalette) -> str:
    return """
        QTreeWidget {
            background-color: palette(base);
            border: none;
            outline: none;
        }
        QTreeWidget::item {
            padding: 3px 1px;
            margin: 1px 0px;
            border-radius: 0px;
        }
        QTreeWidget::item:hover:!selected { background-color: transparent; }
        QTreeWidget::item:selected {
            background-color: transparent;
            color: palette(highlighted-text);
        }
    """ + scrollbar_stylesheet(palette)


def scrollbar_stylesheet(palette: QPalette) -> str:
    """Shared scrollbar geometry for navigation trees and image viewports."""
    base = palette.color(QPalette.ColorRole.Base)
    text = palette.color(QPalette.ColorRole.Text)
    handle = blend(base, text, .30).name()
    handle_hover = blend(base, text, .46).name()
    return f"""
        QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: {handle}; min-height: 28px; border-radius: 4px; }}
        QScrollBar::handle:vertical:hover {{ background: {handle_hover}; }}
        QScrollBar::handle:vertical:pressed {{ background: {handle_hover}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; border: none; }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
        QScrollBar::handle:horizontal {{ background: {handle}; min-width: 28px; border-radius: 4px; }}
        QScrollBar::handle:horizontal:hover, QScrollBar::handle:horizontal:pressed {{ background: {handle_hover}; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; border: none; }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
        QAbstractScrollArea::corner {{ background: transparent; border: none; }}
    """
