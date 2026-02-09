"""
Image tree widget for group/image navigation.

Displays groups as parent nodes with images as children.
Shows an empty state with load button when no images are loaded.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QStackedWidget
)
from PySide6.QtCore import Signal, Qt

from ..models import ImageSeries, ImageData


class ImageTree(QWidget):
    """
    Tree widget for navigating groups and images.
    
    Shows a hierarchical view of:
    - Groups (image series) as parent nodes
    - Individual images as child nodes
    
    When empty, shows a placeholder with load button.
    
    Signals:
        image_selected: Emitted when an image is selected (ImageData).
        group_selected: Emitted when a group is selected (ImageSeries).
        load_requested: Emitted when user clicks load button in empty state.
    """
    
    image_selected = Signal(object)  # ImageData
    group_selected = Signal(object)  # ImageSeries
    load_requested = Signal()  # Emitted from empty state button
    
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        self._series_dict: dict[str, ImageSeries] = {}
        self._item_to_data: dict[int, ImageData | ImageSeries] = {}
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Stacked widget for empty state vs tree
        self._stack = QStackedWidget()
        
        # Empty state widget
        empty_widget = QWidget()
        empty_layout = QVBoxLayout(empty_widget)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        empty_label = QLabel("No images loaded")
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_label.setStyleSheet("color: #888; font-size: 12px; margin-bottom: 10px;")
        empty_layout.addWidget(empty_label)
        
        load_btn = QPushButton("📁 Open Folder...")
        load_btn.clicked.connect(self.load_requested.emit)
        empty_layout.addWidget(load_btn)
        
        self._stack.addWidget(empty_widget)  # Index 0: empty state
        
        # Tree widget - single column
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(1)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.setIndentation(20)
        
        self._stack.addWidget(self._tree)  # Index 1: tree
        
        layout.addWidget(self._stack)
        
        # Show empty state by default
        self._stack.setCurrentIndex(0)
    
    def set_series(self, series_dict: dict[str, ImageSeries]) -> None:
        """
        Populate the tree with image series.
        
        Args:
            series_dict: Dictionary mapping group names to ImageSeries.
        """
        self._series_dict = series_dict
        self._item_to_data.clear()
        self._tree.clear()
        
        if not series_dict:
            self._stack.setCurrentIndex(0)  # Show empty state
            return
        
        for group_name, series in sorted(series_dict.items()):
            # Create group item with count (add warning icon if mismatches exist)
            warning_prefix = "⚠️ " if series.has_barcode_warning else ""
            group_item = QTreeWidgetItem([f"{warning_prefix}{group_name} ({len(series.images)})"])
            group_item.setFlags(group_item.flags() | Qt.ItemFlag.ItemIsSelectable)
            self._item_to_data[id(group_item)] = series
            
            # Add image children - just show date
            for image_data in series.images:
                date_str = image_data.date.strftime("%Y-%m-%d") if image_data.date else "Unknown"
                image_item = QTreeWidgetItem([date_str])
                self._item_to_data[id(image_item)] = image_data
                group_item.addChild(image_item)
            
            self._tree.addTopLevelItem(group_item)
        
        # Expand all groups
        self._tree.expandAll()
        
        # Show tree
        self._stack.setCurrentIndex(1)
    
    def _on_selection_changed(self) -> None:
        """Handle tree selection change."""
        items = self._tree.selectedItems()
        if not items:
            return
        
        item = items[0]
        data = self._item_to_data.get(id(item))
        
        if data is None:
            return
        
        if isinstance(data, ImageData):
            self.image_selected.emit(data)
        elif isinstance(data, ImageSeries):
            self.group_selected.emit(data)
    
    def select_first_image(self) -> None:
        """Select the first image in the tree."""
        if self._tree.topLevelItemCount() > 0:
            first_group = self._tree.topLevelItem(0)
            if first_group and first_group.childCount() > 0:
                first_image = first_group.child(0)
                self._tree.setCurrentItem(first_image)
    
    def get_selected_data(self) -> ImageData | ImageSeries | None:
        """Get the currently selected data object."""
        items = self._tree.selectedItems()
        if not items:
            return None
        return self._item_to_data.get(id(items[0]))
    
    def get_selected_series(self) -> ImageSeries | None:
        """Get the series containing the selected item."""
        items = self._tree.selectedItems()
        if not items:
            return None
        
        item = items[0]
        data = self._item_to_data.get(id(item))
        
        if isinstance(data, ImageSeries):
            return data
        elif isinstance(data, ImageData):
            # Find parent series
            parent = item.parent()
            if parent:
                return self._item_to_data.get(id(parent))
        
        return None
    
    def refresh(self) -> None:
        """Refresh the tree display, preserving current selection and focus."""
        if not self._series_dict:
            return
        
        # Remember current selection before rebuilding
        selected_data = self.get_selected_data()
        had_focus = self._tree.hasFocus()
        
        # Rebuild tree
        self.set_series(self._series_dict)
        
        # Restore selection
        if isinstance(selected_data, ImageData):
            self.select_image(selected_data)
        elif isinstance(selected_data, ImageSeries):
            self.select_series(selected_data)
        
        # Restore focus if tree had it before
        if had_focus:
            self._tree.setFocus()

    
    def select_image(self, image_data: ImageData) -> None:
        """
        Select a specific image in the tree.
        
        Args:
            image_data: The image to select.
        """
        # Find the tree item for this image
        for item_id, data in self._item_to_data.items():
            if data is image_data:
                # Find the item with this id
                for i in range(self._tree.topLevelItemCount()):
                    group = self._tree.topLevelItem(i)
                    for j in range(group.childCount()):
                        child = group.child(j)
                        if id(child) == item_id:
                            self._tree.blockSignals(True)
                            self._tree.setCurrentItem(child)
                            self._tree.blockSignals(False)
                            return
    
    def select_series(self, series: 'ImageSeries') -> None:
        """
        Select a group (series) header in the tree.
        
        Args:
            series: The series to select.
        """
        # Find the tree item for this series
        for item_id, data in self._item_to_data.items():
            if data is series:
                # Find the group item with this id
                for i in range(self._tree.topLevelItemCount()):
                    group = self._tree.topLevelItem(i)
                    if id(group) == item_id:
                        self._tree.blockSignals(True)
                        self._tree.setCurrentItem(group)
                        self._tree.blockSignals(False)
                        return
    
    def is_empty(self) -> bool:
        """Check if no images are loaded."""
        return len(self._series_dict) == 0
    
    def setFocus(self) -> None:
        """Set focus to the internal tree widget for keyboard navigation."""
        self._tree.setFocus()
