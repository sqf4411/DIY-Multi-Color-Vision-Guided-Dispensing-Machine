"""
多颜色路径编辑器主对话框
支持4色选择、点线绘制、路径列表管理、批量编辑
"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import cv2
import numpy as np
import json
import os

from .path_segment import PathSegment, PathSegmentList
from .path_canvas import PathCanvas

class MultiColorPathEditor(QDialog):
    """多颜色路径编辑器"""
    
    def __init__(self, template_img, existing_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("多颜色路径编辑器")
        self.resize(1400, 900)
        
        # 数据
        self.template = template_img
        self.path_list = PathSegmentList()
        
        # 加载已有数据
        if existing_data:
            self.load_existing_data(existing_data)
        
        # 当前设置
        self.current_color = 0
        self.current_dwell = 100
        self.current_mode = 'dot'
        
        # 加载颜色配置
        self.load_color_config()
        
        self.init_ui()
        self.update_canvas()
        self.refresh_table()
    
    def init_ui(self):
        """初始化UI"""
        layout = QHBoxLayout()
        
        # 左侧：画布（带滚动和缩放）
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)  # 不自动调整大小
        scroll.setAlignment(Qt.AlignCenter)  # 居中显示
        self.canvas = PathCanvas(self.template, self)
        self.canvas.segment_added.connect(self.on_segment_added)
        self.canvas.segment_clicked.connect(self.on_canvas_segment_clicked)  # 画布点击->列表选择
        self.canvas.batch_edit_requested.connect(self.on_batch_edit_requested)  # 右键批量编辑
        scroll.setWidget(self.canvas)
        layout.addWidget(scroll, stretch=2)  # 左侧占2
        
        # 右侧：控制面板
        right_panel = self.create_control_panel()
        layout.addWidget(right_panel, stretch=1)  # 占满剩余
        
        self.setLayout(layout)
        
        # 启用键盘事件
        self.setFocusPolicy(Qt.StrongFocus)
    
    def create_control_panel(self):
        """创建右侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 1. 绘制模式
        layout.addWidget(self.create_mode_group())
        
        # 2. 颜色选择
        layout.addWidget(self.create_color_group())
        
        # 3. 时间设置
        layout.addWidget(self.create_time_group())
        
        # 4. 路径列表
        layout.addWidget(self.create_path_list_group())
        
        # 5. 批量编辑
        layout.addWidget(self.create_batch_group())
        
        # 6. 操作按钮
        layout.addWidget(self.create_action_buttons())
        
        panel.setLayout(layout)
        return panel
    
    def create_mode_group(self):
        """绘制模式选择"""
        group = QGroupBox("绘制模式")
        layout = QHBoxLayout()
        
        self.rb_dot = QRadioButton("● 点")
        self.rb_line = QRadioButton("━ 线")
        self.rb_dot.setChecked(True)
        
        self.rb_dot.toggled.connect(lambda checked: self.on_mode_changed('dot') if checked else None)
        self.rb_line.toggled.connect(lambda checked: self.on_mode_changed('line') if checked else None)
        
        layout.addWidget(self.rb_dot)
        layout.addWidget(self.rb_line)
        
        group.setLayout(layout)
        return group
    
    def create_color_group(self):
        """颜色选择"""
        group = QGroupBox("颜色选择")
        layout = QGridLayout()
        
        self.color_btns = []
        names = PathSegment.COLOR_NAMES
        
        for i in range(4):
            name = names[i]
            # 获取当前颜色
            r, g, b = PathSegment.COLOR_RGB[i]
            color = QColor(r, g, b)
            bg_color = color.name()
            
            # 计算文字颜色 (YIQ公式)
            text_color = 'white' if (r*0.299 + g*0.587 + b*0.114) < 128 else 'black'
            
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg_color};
                    color: {text_color};
                    padding: 12px;
                    border: 2px solid #888888;
                    border-radius: 4px;
                    font-weight: bold;
                }}
                QPushButton:checked {{
                    border: 4px solid #000000;
                }}
            """)
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(lambda pos, b=btn, c=i: self.on_color_context_menu(pos, b, c))
            
            btn.clicked.connect(lambda checked, c=i: self.on_color_changed(c))
            
            layout.addWidget(btn, i // 2, i % 2)
            self.color_btns.append(btn)
        
        self.color_btns[0].setChecked(True)
        
        group.setLayout(layout)
        return group
    
    def create_time_group(self):
        """时间设置"""
        group = QGroupBox("出胶时间")
        layout = QFormLayout()
        
        self.spn_dwell = QSpinBox()
        self.spn_dwell.setRange(10, 5000)
        self.spn_dwell.setValue(100)
        self.spn_dwell.setSuffix(" ms")
        self.spn_dwell.valueChanged.connect(self.on_dwell_changed)
        
        layout.addRow("当前时间:", self.spn_dwell)
        
        # 提示
        lbl_hint = QLabel("左键: 添加点/线点\n右键: 结束当前线")
        lbl_hint.setStyleSheet("color: gray; font-size: 11px;")
        layout.addRow(lbl_hint)
        
        group.setLayout(layout)
        return group
    
    def create_path_list_group(self):
        """路径列表"""
        group = QGroupBox("路径列表")
        layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['#', '类型', '颜色', '坐标', '时间(ms)'])
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(1, 50)
        self.table.setColumnWidth(2, 80)
        self.table.setColumnWidth(3, 100)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)  # 支持多选
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)     # 整行选择
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)  # 支持多选
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)     # 整行选择
        
        # 设置样式表：去除白色背景，白色文字
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #2C2C2C;
                color: white;
                gridline-color: #555555;
                border: 1px solid #555555;
                selection-background-color: #007ACC;
            }
            QHeaderView::section {
                background-color: #3C3C3C;
                color: white;
                border: 1px solid #555555;
                padding: 4px;
            }
            QTableCornerButton::section {
                background-color: #3C3C3C;
                border: 1px solid #555555;
            }
        """)

        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)  # 列表选择->画布高亮
        self.table.itemDoubleClicked.connect(self.on_item_double_clicked)  # 双击编辑
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        
        btn_del = QPushButton("🗑️ 删除")
        btn_del.clicked.connect(self.delete_selected)
        
        btn_up = QPushButton("↑")
        btn_up.setFixedWidth(40)
        btn_up.clicked.connect(self.move_up)
        
        btn_down = QPushButton("↓")
        btn_down.setFixedWidth(40)
        btn_down.clicked.connect(self.move_down)
        
        btn_layout.addWidget(btn_del)
        btn_layout.addWidget(btn_up)
        btn_layout.addWidget(btn_down)
        btn_layout.addStretch()
        
        layout.addWidget(self.table)
        layout.addLayout(btn_layout)
        
        group.setLayout(layout)
        return group
    
    def create_batch_group(self):
        """批量编辑"""
        group = QGroupBox("批量编辑")
        layout = QFormLayout()
        
        self.spn_batch = QSpinBox()
        self.spn_batch.setRange(10, 5000)
        self.spn_batch.setValue(100)
        self.spn_batch.setSuffix(" ms")
        
        btn_all = QPushButton("应用到全部")
        btn_all.clicked.connect(lambda: self.apply_batch(True))
        
        btn_sel = QPushButton("应用到选中")
        btn_sel.clicked.connect(lambda: self.apply_batch(False))
        
        layout.addRow("统一时间:", self.spn_batch)
        layout.addRow(btn_all)
        layout.addRow(btn_sel)
        
        group.setLayout(layout)
        return group
    
    def create_action_buttons(self):
        """操作按钮"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        btn_save = QPushButton("💾 保存路径")
        btn_save.setStyleSheet("background-color: #10B981; color: white; padding: 8px; font-size: 13px;")
        btn_save.clicked.connect(self.accept)
        
        btn_cancel = QPushButton("❌ 取消")
        btn_cancel.setStyleSheet("padding: 6px;")
        btn_cancel.clicked.connect(self.reject)
        
        btn_clear = QPushButton("🗑️ 清空全部")
        btn_clear.setStyleSheet("background-color: #EF4444; color: white; padding: 6px;")
        btn_clear.clicked.connect(self.clear_all)
        
        layout.addWidget(btn_save)
        layout.addWidget(btn_cancel)
        layout.addWidget(btn_clear)
        
        widget.setLayout(layout)
        return widget
    
    # ===== 事件处理 =====
    
    def on_mode_changed(self, mode):
        """模式切换"""
        self.current_mode = mode
        self.canvas.set_mode(mode)
    
    def on_color_changed(self, color):
        """颜色切换"""
        self.current_color = color
        # 更新按钮状态
        for i, btn in enumerate(self.color_btns):
            btn.setChecked(i == color)
        # 更新画布
        self.canvas.set_color(color)
    
    def on_dwell_changed(self, value):
        """时间改变"""
        self.current_dwell = value
        self.canvas.set_dwell(value)
    
    def on_segment_added(self, segment):
        """添加新路径段"""
        self.path_list.add(segment)
        self.refresh_table()
        self.update_canvas()
        
        # 清除表格选中（新建模式）
        self.table.clearSelection()
    
    def on_canvas_segment_clicked(self, index):
        """画布中的路径段被点击"""
        # 检测是否按住Ctrl键
        from PyQt5.QtWidgets import QApplication
        modifiers = QApplication.keyboardModifiers()
        from PyQt5.QtCore import Qt
        
        if modifiers & Qt.ControlModifier:
            # Ctrl点击：切换选中状态
            if self.table.item(index, 0).isSelected():
                # 已选中，取消选中
                for col in range(self.table.columnCount()):
                    item = self.table.item(index, col)
                    if item:
                        item.setSelected(False)
            else:
                # 未选中，添加到选中
                self.table.selectRow(index)
        else:
            # 普通点击：清除其他选择，只选中当前
            self.table.clearSelection()
            self.table.selectRow(index)
        
        self.table.scrollToItem(self.table.item(index, 0))
    
    def on_table_selection_changed(self):
        """表格选择改变"""
        # 获取所有选中的行
        selected_rows = set()
        for item in self.table.selectedItems():
            selected_rows.add(item.row())
        
        # 更新画布高亮
        self.canvas.set_selected_indices(selected_rows)
    
    def on_batch_edit_requested(self):
        """画布右键批量编辑"""
        # 获取所有选中的行
        selected_rows = set()
        for item in self.table.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            return
        
        # 创建批量编辑对话框
        dialog = QDialog(self)
        dialog.setWindowTitle(f"批量编辑 {len(selected_rows)} 个路径段")
        dialog.setFixedWidth(300)
        
        layout = QVBoxLayout()
        
        # 颜色选择
        color_group = QGroupBox("颜色")
        color_layout = QGridLayout()
        
        color_btns = []
        color_btns = []
        names = PathSegment.COLOR_NAMES
        for i in range(4):
            name = names[i]
            color_idx = i
            btn = QRadioButton(name)
            if i == 0:
                btn.setChecked(True)
            color_btns.append((btn, color_idx))
            color_layout.addWidget(btn, i // 2, i % 2)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        # 时间设置
        time_group = QGroupBox("时间")
        time_layout = QFormLayout()
        
        spn_time = QSpinBox()
        spn_time.setRange(10, 5000)
        spn_time.setValue(100)
        spn_time.setSuffix(" ms")
        time_layout.addRow("出胶时间:", spn_time)
        
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        # 确定/取消按钮
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("确定")
        btn_cancel = QPushButton("取消")
        btn_ok.clicked.connect(dialog.accept)
        btn_cancel.clicked.connect(dialog.reject)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            # 直接应用修改
            selected_color = None
            for btn, color_idx in color_btns:
                if btn.isChecked():
                    selected_color = color_idx
                    break
            
            selected_time = spn_time.value()
            
            for row in selected_rows:
                if 0 <= row < self.path_list.count():
                    seg = self.path_list.get_all()[row]
                    seg.color = selected_color
                    seg.dwell_ms = selected_time
            
            # 刷新显示
            self.refresh_table()
            self.update_canvas()
    
    def delete_selected(self):
        """删除选中"""
        row = self.table.currentRow()
        if row >= 0:
            self.path_list.remove(row)
            self.refresh_table()
            self.update_canvas()
    
    def move_up(self):
        """上移"""
        row = self.table.currentRow()
        if row > 0:
            self.path_list.move_up(row)
            self.refresh_table()
            self.table.setCurrentCell(row - 1, 0)
            self.update_canvas()
    
    def move_down(self):
        """下移"""
        row = self.table.currentRow()
        if row >= 0 and row < self.path_list.count() - 1:
            self.path_list.move_down(row)
            self.refresh_table()
            self.table.setCurrentCell(row + 1, 0)
            self.update_canvas()
    
    def apply_batch(self, to_all):
        """批量应用时间"""
        batch_time = self.spn_batch.value()
        
        if to_all:
            # 应用到全部
            for seg in self.path_list.get_all():
                seg.dwell_ms = batch_time
            QMessageBox.information(self, "完成", f"已将所有路径时间设为{batch_time}ms")
        else:
            # 应用到选中（支持多选）
            selected_rows = set()
            for item in self.table.selectedItems():
                selected_rows.add(item.row())
            
            if not selected_rows:
                QMessageBox.warning(self, "提示", "请先选择一个或多个路径")
                return
            
            # 应用到所有选中的行
            for row in selected_rows:
                if 0 <= row < self.path_list.count():
                    self.path_list.get_all()[row].dwell_ms = batch_time
            
            QMessageBox.information(self, "完成", 
                f"已将{len(selected_rows)}个选中路径的时间设为{batch_time}ms")
        
        self.refresh_table()
    
    def clear_all(self):
        """清空所有"""
        if self.path_list.count() == 0:
            return
        
        reply = QMessageBox.question(self, '确认', '确定要清空所有路径吗？',
                                      QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.path_list.clear()
            self.refresh_table()
            self.update_canvas()
    
    def refresh_table(self):
        """刷新表格"""
        self.table.setRowCount(self.path_list.count())
        
        for i, seg in enumerate(self.path_list.get_all()):
            # 序号
            self.table.setItem(i, 0, QTableWidgetItem(f"{i+1}"))
            
            # 类型（不可编辑）
            type_str = "●点" if seg.type == 'dot' else "━线"
            type_item = QTableWidgetItem(type_str)
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 1, type_item)
            
            # 颜色（双击可编辑）
            color_item = QTableWidgetItem(seg.get_color_name())
            r, g, b = seg.get_display_color_rgb()
            color_item.setBackground(QColor(r, g, b))
            color_item.setForeground(QColor(255, 255, 255) if seg.color in [0, 2] else QColor(0, 0, 0))
            self.table.setItem(i, 2, color_item)
            
            # 坐标（双击可编辑）
            coord_item = QTableWidgetItem(seg.get_coord_str())
            self.table.setItem(i, 3, coord_item)
            
            # 时间（SpinBox）
            spn = QSpinBox()
            spn.setRange(10, 5000)
            spn.setValue(seg.dwell_ms)
            spn.valueChanged.connect(lambda val, idx=i: self.on_seg_time_changed(idx, val))
            self.table.setCellWidget(i, 4, spn)
    
    def on_item_double_clicked(self, item):
        """双击表格项编辑"""
        row = item.row()
        col = item.column()
        
        if row < 0 or row >= self.path_list.count():
            return
        
        seg = self.path_list.get_all()[row]
        
        if col == 2:  # 颜色列
            # 创建颜色选择对话框
            dialog = QDialog(self)
            dialog.setWindowTitle(f"修改路径段 #{row+1} 的颜色")
            dialog.setFixedWidth(250)
            
            layout = QVBoxLayout()
            
            color_group = QGroupBox("选择颜色")
            color_layout = QGridLayout()
            color_btns = []
            color_btns = []
            names = PathSegment.COLOR_NAMES
            for i in range(4):
                name = names[i]
                color_idx = i
                btn = QRadioButton(name)
                if color_idx == seg.color:
                    btn.setChecked(True)
                color_btns.append((btn, color_idx))
                color_layout.addWidget(btn, i // 2, i % 2)
            color_group.setLayout(color_layout)
            layout.addWidget(color_group)
            
            btn_layout = QHBoxLayout()
            btn_ok = QPushButton("确定")
            btn_cancel = QPushButton("取消")
            btn_ok.clicked.connect(dialog.accept)
            btn_cancel.clicked.connect(dialog.reject)
            btn_layout.addWidget(btn_ok)
            btn_layout.addWidget(btn_cancel)
            layout.addLayout(btn_layout)
            
            dialog.setLayout(layout)
            
            if dialog.exec_() == QDialog.Accepted:
                for btn, color_idx in color_btns:
                    if btn.isChecked():
                        seg.color = color_idx
                        break
                self.refresh_table()
                self.update_canvas()
        
        elif col == 3:  # 坐标列
            # 暂不支持坐标编辑（需要画布交互）
            QMessageBox.information(self, "提示", "坐标暂不支持直接编辑\n请在画布上重新绘制")
    
    def on_seg_time_changed(self, index, value):
        """单个路径段时间改变"""
        if 0 <= index < self.path_list.count():
            self.path_list.get_all()[index].dwell_ms = value
    
    def update_canvas(self):
        """更新画布"""
        self.canvas.set_segments(self.path_list.get_all())
    
    def get_path_list(self):
        """获取路径列表"""
        return self.path_list
    
    # ===== 数据加载/保存 =====
    
    def load_existing_data(self, data):
        """加载已有数据"""
        if isinstance(data, dict) and 'segments' in data:
            self.path_list = PathSegmentList.from_dict_list(data['segments'])
    
    def keyPressEvent(self, event):
        """键盘事件处理"""
        key = event.key()
        
        if key == Qt.Key_Z:
            # Z: 按顺序撤销（撤销最后一个路径段）
            if self.path_list.count() > 0:
                self.path_list.segments.pop()
                self.refresh_table()
                self.update_canvas()
        
        elif key == Qt.Key_D:
            # D: 删除选中的路径段
            selected_rows = set()
            for item in self.table.selectedItems():
                selected_rows.add(item.row())
            
            if selected_rows:
                # 从后往前删除，避免索引变化
                for row in sorted(selected_rows, reverse=True):
                    if 0 <= row < self.path_list.count():
                        self.path_list.remove(row)
                
                self.refresh_table()
                self.update_canvas()
        
        elif key == Qt.Key_S:
            # S: 保存
            self.accept()
        
        elif key == Qt.Key_Escape:
            # ESC: 取消
            self.reject()
            
    def on_color_context_menu(self, pos, btn, color_idx):
        """右键点击颜色按钮：自定义颜色"""
        current_tuple = PathSegment.COLOR_RGB[color_idx]
        current_color = QColor(current_tuple[0], current_tuple[1], current_tuple[2])
        
        # 弹出颜色选择器
        new_color = QColorDialog.getColor(current_color, self, "选择新颜色")
        
        if new_color.isValid():
            # 1. 更新 PathSegment 静态配置 (影响画布显示)
            r, g, b = new_color.red(), new_color.green(), new_color.blue()
            PathSegment.COLOR_RGB[color_idx] = (r, g, b)
            # 更新BGR用于OpenCV (可选)
            PathSegment.COLOR_BGR[color_idx] = (b, g, r) 
            
            # 2. 更新按钮样式
            # 根据亮度决定文字颜色
            text_color = 'white' if (r*0.299 + g*0.587 + b*0.114) < 128 else 'black'
            
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {new_color.name()};
                    color: {text_color};
                    padding: 12px;
                    border: 2px solid #888888;
                    border-radius: 4px;
                    font-weight: bold;
                }}
                QPushButton:checked {{
                    border: 4px solid #000000;
                }}
            """)
            
            # 3. 刷新画布
            self.canvas.update()
            
            # 4. 保存颜色配置
            self.save_color_config()
            
    def load_color_config(self):
        """加载颜色配置"""
        config_path = os.path.join("config", "colors.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if 'rgb' in data and len(data['rgb']) == 4:
                    # 更新 RGB
                    for i, color in enumerate(data['rgb']):
                        PathSegment.COLOR_RGB[i] = tuple(color)
                        # 更新 BGR
                        PathSegment.COLOR_BGR[i] = (color[2], color[1], color[0])
            except Exception as e:
                print(f"Error loading color config: {e}")
                
    def save_color_config(self):
        """保存颜色配置"""
        config_dir = "config"
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        data = {
            'rgb': PathSegment.COLOR_RGB
        }
        
        try:
            with open(os.path.join(config_dir, "colors.json"), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving color config: {e}")

