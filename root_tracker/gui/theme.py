"""Small palette-based styles for the application's controls."""

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
    return f"""
        QPushButton {{
            background-color: {surface.name()};
            color: {text.name()};
            border: 1px solid {border};
            border-radius: 5px;
            padding: 3px 8px;
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
    """


def tree_stylesheet(palette: QPalette) -> str:
    base = palette.color(QPalette.ColorRole.Base)
    accent = palette.color(QPalette.ColorRole.Highlight)
    hover = blend(base, accent, 0.08 if is_light(palette) else 0.18).name()
    return f"""
        QTreeWidget {{
            background-color: palette(base);
            border: none;
            outline: none;
        }}
        QTreeWidget::item {{
            padding: 3px 2px;
            margin: 1px 3px;
            border-radius: 4px;
        }}
        QTreeWidget::item:hover:!selected {{ background-color: {hover}; }}
        QTreeWidget::item:selected {{
            background-color: palette(highlight);
            color: palette(highlighted-text);
        }}
        QTreeWidget::branch:selected {{ background-color: palette(base); }}
    """
