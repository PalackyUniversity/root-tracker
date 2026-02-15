"""
Image tree widget for group/image navigation.

Displays groups as parent nodes with images as children.
Shows an empty state with load button when no images are loaded.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QStackedWidget, QMenu, QHBoxLayout
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QAction, QFont, QColor

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
    
    # Context menu signals
    set_aside_requested = Signal(object)  # ImageData or ImageSeries
    unset_aside_requested = Signal(object)  # ImageData or ImageSeries
    delete_requested = Signal(object)  # ImageData or ImageSeries
    
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        self._series_dict: dict[str, ImageSeries] = {}
        self._item_to_data: dict[int, ImageData | ImageSeries] = {}
        self._filter_aside: bool = False  # If True, hide items that are set aside
        self._current_folder: str = ""
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header widget with folder name and open button
        self._header = QWidget()
        self._header.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border-bottom: 1px solid #3a3a3a;
            }
        """)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(4)
        
        self._folder_label = QLabel("No folder")
        self._folder_label.setStyleSheet("color: #ffffff; font-size: 11px; border: none;")
        self._folder_label.setWordWrap(False)
        header_layout.addWidget(self._folder_label, 1)  # stretch to fill
        
        self._open_folder_btn = QPushButton("Open...")
        self._open_folder_btn.setToolTip("Open different folder")
        self._open_folder_btn.setFixedHeight(24)
        self._open_folder_btn.clicked.connect(self.load_requested.emit)
        self._open_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                padding: 4px 8px;
                color: #ccc;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border: 1px solid #5a5a5a;
                color: #fff;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """)
        header_layout.addWidget(self._open_folder_btn, 0)
        
        self._header.hide()  # Hidden by default until folder is loaded
        layout.addWidget(self._header)
        
        # Stacked widget for empty state vs tree
        self._stack = QStackedWidget()
        
        # Empty state widget
        empty_widget = QWidget()
        empty_layout = QVBoxLayout(empty_widget)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add stretch before to push content down
        empty_layout.addStretch(1)
        
        # Container for centered content
        empty_label = QLabel("No images loaded")
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_label.setStyleSheet("color: #888; font-size: 12px; margin-bottom: 10px;")
        empty_layout.addWidget(empty_label, 0, Qt.AlignmentFlag.AlignHCenter)
        
        load_btn = QPushButton("Open Folder...")
        load_btn.setFixedWidth(120)
        load_btn.clicked.connect(self.load_requested.emit)
        empty_layout.addWidget(load_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        
        # Add stretch after to push content up (centering vertically)
        empty_layout.addStretch(1)
        
        self._stack.addWidget(empty_widget)  # Index 0: empty state
        
        # Tree widget - single column
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(1)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.setIndentation(20)
        
        # Enable context menu
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        
        self._stack.addWidget(self._tree)  # Index 1: tree
        
        layout.addWidget(self._stack)
        
        # Show empty state by default
        self._stack.setCurrentIndex(0)
    
    def set_filter_aside(self, filter_aside: bool) -> None:
        """Set whether to hide items that are set aside."""
        if self._filter_aside != filter_aside:
            self._filter_aside = filter_aside
            self.refresh()

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
            self._header.hide()
        else:
            self._stack.setCurrentIndex(1)  # Show tree
            self._header.show()
            self._populate_tree()
    
    def set_folder_path(self, folder_path: str) -> None:
        """Set the current folder path to display in the header."""
        import os
        self._current_folder = folder_path
        if folder_path:
            # Show just the folder name, not the full path
            folder_name = os.path.basename(folder_path.rstrip(os.sep))
            self._folder_label.setText(folder_name)
            self._folder_label.setToolTip(folder_path)  # Full path on hover
    
    def _populate_tree(self) -> None:
        """Populate the tree with the current series data."""
        for group_name, series in sorted(self._series_dict.items()):
            is_group_aside = series.is_set_aside
            
            # Skip if we're filtering aside items and this group is aside
            if self._filter_aside and is_group_aside:
                continue
                
            # Create group item with status
            
            # Determine status
            prefix = ""
            color = None
            tooltip = None
            
            if series.has_barcode_error:
                prefix = "❌ "
                error_count = series.barcode_error_count
                suffix = f" (Err: {error_count})"
                color = Qt.GlobalColor.red
                tooltip = f"Barcode mismatch in {error_count} image(s)"
            elif series.has_barcode_warning:
                prefix = "⚠️ "
                suffix = ""
                # Orange-ish color for warning
                color = QColor(255, 140, 0)
                tooltip = "Barcode not detected in some images"
            else:
                suffix = ""
            
            group_text = f"{prefix}{group_name} ({len(series.images)}){suffix}"
            group_item = QTreeWidgetItem([group_text])
            group_item.setFlags(group_item.flags() | Qt.ItemFlag.ItemIsSelectable)
            
            if color:
                group_item.setForeground(0, color)
            
            if tooltip:
                group_item.setToolTip(0, tooltip)
            
            # Strikethrough if set aside
            if is_group_aside:
                font = group_item.font(0)
                font.setStrikeOut(True)
                group_item.setFont(0, font)
                group_item.setForeground(0, Qt.GlobalColor.gray)
                
            self._item_to_data[id(group_item)] = series
            
            # Add image children - just show date
            for image_data in series.images:
                # If group is aside, images are aside.
                # If group is NOT aside, individual images might still be aside (if we supported individual move, 
                # but currently we support per-file or per-group. If per-file move makes group mixed, `is_set_aside` on group might be false/complex)
                # But `is_set_aside` on Series returns True ONLY if all are aside. 
                # Let's check individual image aside status.
                is_img_aside = image_data.is_set_aside
                
                if self._filter_aside and is_img_aside:
                    continue
                    
                date_str = image_data.date.strftime("%Y-%m-%d") if image_data.date else "Unknown"
                
                # Image status
                img_prefix = ""
                img_color = None
                img_tooltip = None
                
                if image_data.barcode_mismatch:
                    img_prefix = "❌ "
                    img_color = Qt.GlobalColor.red
                    img_tooltip = f"Barcode mismatch: Read '{image_data.barcode_read}', Expected '{image_data.barcode}'"
                elif image_data.barcode_not_found and image_data.barcode_detected:
                    img_prefix = "⚠️ "
                    img_color = QColor(255, 140, 0)
                    img_tooltip = "No barcode detected"
                
                image_item = QTreeWidgetItem([f"{img_prefix}{date_str}"])
                
                if img_color:
                    image_item.setForeground(0, img_color)
                
                if img_tooltip:
                    image_item.setToolTip(0, img_tooltip)
                
                if is_img_aside:
                    font = image_item.font(0)
                    font.setStrikeOut(True)
                    image_item.setFont(0, font)
                    image_item.setForeground(0, Qt.GlobalColor.gray)
                    
                self._item_to_data[id(image_item)] = image_data
                group_item.addChild(image_item)
            
            # Only add group if it has visible children or if it's the group itself that matched visibility
            if group_item.childCount() > 0 or (not self._filter_aside and is_group_aside):
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

    def clear_selection(self) -> None:
        """Clear current selection."""
        self._tree.clearSelection()

    def _show_context_menu(self, position) -> None:
        """Show context menu for tree items."""
        item = self._tree.itemAt(position)
        if item is None:
            return
            
        data = self._item_to_data.get(id(item))
        if data is None:
            return
            
        menu = QMenu(self)
        
        # Determine if item is already set aside
        is_aside = data.is_set_aside
        
        if is_aside:
            action_text = "Restore from Aside"
            action_handler = self.unset_aside_requested
        else:
            action_text = "Set Aside"
            action_handler = self.set_aside_requested
            
        aside_action = QAction(action_text, self)
        aside_action.triggered.connect(lambda: action_handler.emit(data))
        menu.addAction(aside_action)
        
        # Detailed tooltip for "Set Aside"
        if not is_aside:
            aside_action.setToolTip(
                "Moves this item to an 'aside' subfolder.\n"
                "It will be excluded from computation and export.\n"
                "You can see it in 'Load' tab as crossed out."
            )
            
        menu.addSeparator()
        
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self.delete_requested.emit(data))
        menu.addAction(delete_action)
        
        menu.exec(self._tree.viewport().mapToGlobal(position))
