"""
Image tree widget for group/image navigation.

Displays groups as parent nodes with images as children.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QHeaderView
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QIcon

from ..models import ImageSeries, ImageData


class ImageTree(QWidget):
    """
    Tree widget for navigating groups and images.
    
    Shows a hierarchical view of:
    - Groups (image series) as parent nodes
    - Individual images as child nodes
    
    Signals:
        image_selected: Emitted when an image is selected (ImageData).
        group_selected: Emitted when a group is selected (ImageSeries).
    """
    
    image_selected = Signal(object)  # ImageData
    group_selected = Signal(object)  # ImageSeries
    
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        self._series_dict: dict[str, ImageSeries] = {}
        self._item_to_data: dict[int, ImageData | ImageSeries] = {}
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header label
        header = QLabel("Images")
        header.setStyleSheet("""
            QLabel {
                font-weight: bold;
                padding: 5px;
                background-color: #f0f0f0;
                border-bottom: 1px solid #ccc;
            }
        """)
        layout.addWidget(header)
        
        # Tree widget
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Name", "Date"])
        self._tree.setColumnCount(2)
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.setIndentation(20)
        
        layout.addWidget(self._tree)
    
    def set_series(self, series_dict: dict[str, ImageSeries]) -> None:
        """
        Populate the tree with image series.
        
        Args:
            series_dict: Dictionary mapping group names to ImageSeries.
        """
        self._series_dict = series_dict
        self._item_to_data.clear()
        self._tree.clear()
        
        for group_name, series in sorted(series_dict.items()):
            # Create group item
            group_item = QTreeWidgetItem([group_name, f"{len(series.images)} images"])
            group_item.setFlags(group_item.flags() | Qt.ItemFlag.ItemIsSelectable)
            self._item_to_data[id(group_item)] = series
            
            # Add image children
            for image_data in series.images:
                date_str = image_data.date.strftime("%Y-%m-%d") if image_data.date else ""
                image_item = QTreeWidgetItem([image_data.barcode or "Unknown", date_str])
                self._item_to_data[id(image_item)] = image_data
                group_item.addChild(image_item)
            
            self._tree.addTopLevelItem(group_item)
        
        # Expand all groups
        self._tree.expandAll()
    
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
        """Refresh the tree display."""
        if self._series_dict:
            self.set_series(self._series_dict)
    
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
