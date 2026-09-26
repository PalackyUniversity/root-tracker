"""
Image tree widget for group/image navigation.

Displays groups as parent nodes with images as children.
Shows an empty state with load button when no images are loaded.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QStackedWidget, QHBoxLayout,
    QToolButton, QStyle, QHeaderView, QStyleOption, QApplication
)
from PySide6.QtCore import Signal, Qt, QRect
from PySide6.QtGui import QAction, QFont, QColor, QPalette

from ..models import ImageSeries, ImageData
from ..config import Config
from .menus import RoundedMenu
from .theme import tree_stylesheet
from .workflow_bar import WorkflowStep, STEP_NAMES


class ImageNavigationTree(QTreeWidget):
    """Keep branch controls neutral and arrow navigation on visible images."""

    def drawBranches(self, painter, rect, index):
        painter.fillRect(rect, self.palette().brush(QPalette.ColorRole.Base))
        if not self.model().hasChildren(index):
            return
        option = QStyleOption()
        option.initFrom(self)
        option.rect = QRect(rect.right() - self.indentation() + 1, rect.top(),
                            self.indentation(), rect.height())
        option.state = QStyle.StateFlag.State_Enabled | QStyle.StateFlag.State_Children | QStyle.StateFlag.State_Item
        if self.isExpanded(index):
            option.state |= QStyle.StateFlag.State_Open
        QApplication.style().drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorBranch,
                                   option, painter, self)

    def moveCursor(self, action, modifiers):
        if action not in (self.CursorAction.MoveUp, self.CursorAction.MoveDown):
            return super().moveCursor(action, modifiers)
        current = self.currentItem()
        direction = self.itemAbove if action == self.CursorAction.MoveUp else self.itemBelow
        candidate = direction(current) if current is not None else self.topLevelItem(0)
        while candidate is not None:
            if candidate.parent() is not None and not candidate.isHidden():
                return self.indexFromItem(candidate)
            candidate = direction(candidate)
        return self.currentIndex()


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
        self._step = WorkflowStep.LOAD
        self._status_config = Config()
        self._processing_series_ids = set()
        self._processing_step = None
        self._pending_series_ids = set()
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Source header above the image list
        self._header = QWidget()
        self._header.setObjectName("imageTreeHeader")
        self._header.setStyleSheet("""
            QWidget#imageTreeHeader {
                background-color: palette(base);
                border-bottom: 1px solid palette(midlight);
            }
        """)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(5, 6, 3, 6)
        header_layout.setSpacing(4)

        self._folder_label = QLabel("No folder")
        self._folder_label.setStyleSheet("font-size: 13px; border: none;")
        self._folder_label.setWordWrap(False)
        header_layout.addWidget(self._folder_label, 1)

        self._open_folder_btn = QToolButton()
        self._open_folder_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self._open_folder_btn.setAutoRaise(True)
        self._open_folder_btn.setFixedSize(24, 24)
        self._open_folder_btn.setToolTip("Change image folder")
        self._open_folder_btn.setAccessibleName("Change image folder")
        self._open_folder_btn.clicked.connect(self.load_requested.emit)
        header_layout.addWidget(self._open_folder_btn)
        
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
        empty_label.setStyleSheet("font-size: 12px; margin-bottom: 10px;")
        empty_layout.addWidget(empty_label, 0, Qt.AlignmentFlag.AlignHCenter)
        
        load_btn = QPushButton("Open Folder...")
        load_btn.setFixedWidth(120)
        load_btn.clicked.connect(self.load_requested.emit)
        empty_layout.addWidget(load_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        
        # Add stretch after to push content up (centering vertically)
        empty_layout.addStretch(1)
        
        self._stack.addWidget(empty_widget)  # Index 0: empty state
        
        # A stable trailing column keeps completion separate from filenames and warnings.
        self._tree = ImageNavigationTree()
        self._tree.setStyleSheet(tree_stylesheet(self.palette()))
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(['Image', 'Step status'])
        self._tree.header().setStretchLastSection(False)
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._tree.header().setMinimumSectionSize(20)
        self._tree.setColumnWidth(1, 30)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.setIndentation(14)
        
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

    def set_step(self, step: WorkflowStep, config: Config) -> None:
        """Display completion for the selected step and applied configuration."""
        self._step = step
        self._status_config = config
        self.refresh_status()

    def set_pending_changes(self, series: ImageSeries | None) -> None:
        self.set_pending_groups([series] if series is not None else [])

    def set_pending_groups(self, groups) -> None:
        pending = {id(series) for series in groups}
        if pending != self._pending_series_ids:
            self._pending_series_ids = pending
            self.refresh_status()

    def set_processing(self, series: ImageSeries | list[ImageSeries] | None, step: WorkflowStep | None) -> None:
        """Mark queued/running groups without reporting partial results as done."""
        groups = series if isinstance(series, list) else ([series] if series else [])
        self._processing_series_ids = {id(group) for group in groups}
        self._processing_step = step
        self.refresh_status()

    def _image_status(self, image, series, preprocess_hash, tracking_hash):
        if image.is_set_aside:
            return 'excluded', 'Set aside; excluded from processing'
        if self._step == WorkflowStep.LOAD:
            if not self._status_config.data.detect_barcodes:
                return 'done', 'Image loaded; barcode checking is disabled'
            if image.barcode_detected:
                return 'done', 'Barcode checked'
        if id(series) in self._processing_series_ids and self._step == self._processing_step:
            return 'running', 'Queued or processing this group; waiting for the step to finish'
        if self._step == WorkflowStep.LOAD:
            return 'pending', 'Barcode not checked yet'

        state = series.pipeline_state
        prepared = state.preprocessed and bool(image.positions_x)
        if self._step == WorkflowStep.PREPROCESS:
            has_result = prepared
            current = state.preprocess_config_hash == preprocess_hash
        else:
            has_result = state.tracked and image.total_length is not None
            current = (prepared and state.preprocess_config_hash == preprocess_hash and
                       state.tracking_config_hash == tracking_hash)
        if has_result and current:
            return 'done', 'Complete with the current settings'
        if has_result:
            return 'outdated', 'Settings changed; run this step again'
        return 'pending', 'Not processed for this step'

    def refresh_status(self) -> None:
        """Update only status cells; preserve focus, selection, expansion and scroll."""
        config = self._status_config
        preprocess_hash = config.preprocess_config_hash()
        tracking_hash = config.tracking_config_hash()
        label = STEP_NAMES[self._step]
        verb = 'done'
        if self._step == WorkflowStep.LOAD:
            label = 'Barcodes' if config.data.detect_barcodes else 'Load'
            verb = 'checked' if config.data.detect_barcodes else 'loaded'
        light = self.palette().color(QPalette.ColorRole.Base).lightness() >= 128
        colors = {
            'done': QColor('#237a45' if light else '#74c69d'),
            'warning': QColor('#9a5b00' if light else '#ffad42'),
            'modified': QColor('#d97706' if light else '#ffad42'),
            'pending': self.palette().color(QPalette.ColorRole.PlaceholderText),
            'outdated': QColor('#9a5b00' if light else '#ffad42'),
            'running': QColor('#1766a5' if light else '#80bfff'),
            'excluded': self.palette().color(QPalette.ColorRole.PlaceholderText),
        }
        symbols = {'done': '✓', 'pending': '○', 'outdated': '↻', 'running': '…', 'excluded': '—'}
        for index in range(self._tree.topLevelItemCount()):
            group = self._tree.topLevelItem(index)
            series = self._item_to_data[id(group)]
            statuses = []
            warnings = []
            for child_index in range(group.childCount()):
                child = group.child(child_index)
                image = self._item_to_data[id(child)]
                status, detail = self._image_status(image, series, preprocess_hash, tracking_hash)
                statuses.append(status)
                warning = ''
                if config.data.detect_barcodes and status != 'excluded':
                    if image.barcode_mismatch:
                        warning = f"Barcode mismatch: read '{image.barcode_read}', expected '{image.barcode}'"
                    elif image.barcode_detected and image.barcode_not_found:
                        warning = 'No barcode detected'
                if warning:
                    warnings.append(warning)
                    detail += f'; {warning}'
                pending = id(series) in self._pending_series_ids and status != 'excluded'
                if pending:
                    detail += '; Unapplied settings changes'
                child.setText(1, '⚠' if warning else '●' if pending else symbols[status])
                child.setTextAlignment(1, Qt.AlignmentFlag.AlignCenter)
                child.setForeground(1, colors['warning' if warning else 'modified' if pending else status])
                child.setData(1, Qt.ItemDataRole.UserRole, status)
                child.setToolTip(1, f'{label}: {detail}')
                child.setData(1, Qt.ItemDataRole.AccessibleTextRole, f'{image.filename}: {label}. {detail}')
            count = sum(s != 'excluded' for s in statuses)
            done = statuses.count('done')
            status = ('excluded' if not count else 'done' if done == count else
                      'running' if 'running' in statuses else
                      'outdated' if 'outdated' in statuses else 'pending')
            pending = id(series) in self._pending_series_ids and bool(count)
            group.setText(1, '⚠' if warnings else '●' if pending else symbols[status])
            group.setTextAlignment(1, Qt.AlignmentFlag.AlignCenter)
            group.setForeground(1, colors['warning' if warnings else 'modified' if pending else status])
            group.setData(1, Qt.ItemDataRole.UserRole, status)
            detail = f'{label}: {done} of {count} images {verb}' if count else 'Set aside; excluded from processing'
            if status == 'running':
                detail += '; queued or processing'
            elif status == 'outdated':
                detail += '; settings changed'
            if warnings:
                detail += f'; barcode warnings in {len(warnings)} image(s)'
            if pending:
                detail += '; Unapplied settings changes'
            group.setToolTip(1, detail)
            group.setData(1, Qt.ItemDataRole.AccessibleTextRole, f'{series.group}: {detail}')

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
                
            group_item = QTreeWidgetItem([group_name])
            group_item.setFlags(group_item.flags() | Qt.ItemFlag.ItemIsSelectable)

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
                
                image_item = QTreeWidgetItem([date_str])

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
        self.refresh_status()
    
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
            
        menu = RoundedMenu(self)
        
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
