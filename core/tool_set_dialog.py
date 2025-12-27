"""
自动对刀对话框
提供对刀仪位置设置和自动对刀功能
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QDoubleSpinBox, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt
import json
import os


class ToolSetDialog(QDialog):
    """对刀设置对话框"""
    
    def __init__(self, parent=None, current_pos=None):
        super().__init__(parent)
        self.setWindowTitle("🔧 对刀仪设置")
        self.resize(400, 300)
        
        self.parent_window = parent
        self.current_pos = current_pos or (0, 0, 0)
        self.config_file = "config/tool_setter.json"
        
        # 加载保存的对刀仪位置
        self.saved_pos = self.load_tool_setter_pos()
        
        # 构建界面
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        if self.saved_pos is None:
            # 首次设置：保存当前位置
            self.build_first_time_ui(layout)
        else:
            # 已有数据：显示操作选项
            self.build_existing_data_ui(layout)
        
        self.setLayout(layout)
    
    def build_first_time_ui(self, layout):
        """构建首次设置界面"""
        # 标题
        title = QLabel("📍 设置对刀仪位置")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # 当前位置显示
        pos_group = QGroupBox("当前机器位置")
        pos_layout = QGridLayout()
        
        x, y, z = self.current_pos
        pos_layout.addWidget(QLabel("X:"), 0, 0)
        pos_layout.addWidget(QLabel(f"{x:.2f} mm"), 0, 1)
        pos_layout.addWidget(QLabel("Y:"), 1, 0)
        pos_layout.addWidget(QLabel(f"{y:.2f} mm"), 1, 1)
        pos_layout.addWidget(QLabel("Z:"), 2, 0)
        pos_layout.addWidget(QLabel(f"{z:.2f} mm"), 2, 1)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # 提示
        tip = QLabel("💡 请将针头移动到对刀仪上方，然后保存位置")
        tip.setStyleSheet("color: #FFA500; padding: 10px;")
        layout.addWidget(tip)
        
        layout.addStretch()
        
        # 按钮
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("✓ 保存为对刀仪位置")
        btn_save.setStyleSheet("background-color: #2563EB; min-height: 35px;")
        btn_save.clicked.connect(self.save_first_time)
        
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
    
    def build_existing_data_ui(self, layout):
        """构建已有数据界面"""
        # 标题
        title = QLabel("🎯 自动对刀")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # 已保存位置显示
        pos_group = QGroupBox("已保存对刀仪位置")
        pos_layout = QGridLayout()
        
        x, y, z = self.saved_pos
        pos_layout.addWidget(QLabel("X:"), 0, 0)
        pos_layout.addWidget(QLabel(f"{x:.2f} mm"), 0, 1)
        pos_layout.addWidget(QLabel("Y:"), 1, 0)
        pos_layout.addWidget(QLabel(f"{y:.2f} mm"), 1, 1)
        pos_layout.addWidget(QLabel("Z:"), 2, 0)
        pos_layout.addWidget(QLabel(f"{z:.2f} mm"), 2, 1)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        layout.addStretch()
        
        # 主操作按钮
        btn_auto = QPushButton("🚀 开始自动对刀")
        btn_auto.setStyleSheet("background-color: #10B981; min-height: 40px; font-size: 14px; font-weight: bold;")
        btn_auto.clicked.connect(self.start_auto_tool_set)
        layout.addWidget(btn_auto)
        
        # 次要操作按钮
        btn_layout = QHBoxLayout()
        
        btn_recalibrate = QPushButton("📝 重新标定位置")
        btn_recalibrate.clicked.connect(self.recalibrate_position)
        
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(btn_recalibrate)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
    
    def save_first_time(self):
        """首次保存对刀仪位置"""
        x, y, z = self.current_pos
        data = {
            "x": x,
            "y": y,
            "z": z,
            "timestamp": str(__import__('datetime').datetime.now())
        }
        
        # 保存到文件
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        QMessageBox.information(self, "成功", f"✅ 已保存对刀仪位置：\nX={x:.2f}, Y={y:.2f}, Z={z:.2f}")
        self.accept()
    
    def start_auto_tool_set(self):
        """开始自动对刀"""
        self.result_action = "auto_tool_set"
        self.accept()
    
    def recalibrate_position(self):
        """重新标定位置"""
        # 打开重新标定对话框
        dialog = RecalibrateDialog(self, self.saved_pos, self.current_pos)
        if dialog.exec_() == QDialog.Accepted:
            # 重新加载数据
            self.saved_pos = self.load_tool_setter_pos()
            QMessageBox.information(self, "成功", "✅ 对刀仪位置已更新")
            self.reject()  # 关闭当前对话框
    
    def load_tool_setter_pos(self):
        """加载保存的对刀仪位置"""
        if not os.path.exists(self.config_file):
            return None
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return (data['x'], data['y'], data['z'])
        except Exception as e:
            print(f"[ERROR] 加载对刀仪位置失败: {e}")
            return None
    
    def get_tool_setter_pos(self):
        """获取对刀仪位置（供外部调用）"""
        return self.saved_pos if self.saved_pos else self.current_pos


class RecalibrateDialog(QDialog):
    """重新标定对刀仪位置对话框"""
    
    def __init__(self, parent=None, saved_pos=None, current_pos=None):
        super().__init__(parent)
        self.setWindowTitle("📝 重新标定对刀仪位置")
        self.resize(350, 250)
        
        self.saved_pos = saved_pos or (0, 0, 0)
        self.current_pos = current_pos or (0, 0, 0)
        self.config_file = "config/tool_setter.json"
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 标题
        title = QLabel("设置对刀仪坐标")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        layout.addWidget(title)
        
        # 坐标输入
        grid = QGridLayout()
        
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(-999, 999)
        self.spin_x.setDecimals(2)
        self.spin_x.setValue(self.saved_pos[0])
        self.spin_x.setSuffix(" mm")
        
        self.spin_y = QDoubleSpinBox()
        self.spin_y.setRange(-999, 999)
        self.spin_y.setDecimals(2)
        self.spin_y.setValue(self.saved_pos[1])
        self.spin_y.setSuffix(" mm")
        
        self.spin_z = QDoubleSpinBox()
        self.spin_z.setRange(-999, 999)
        self.spin_z.setDecimals(2)
        self.spin_z.setValue(self.saved_pos[2])
        self.spin_z.setSuffix(" mm")
        self.spin_z.setStyleSheet("background-color: #FEF3C7;")  # 高亮Z轴
        
        grid.addWidget(QLabel("X:"), 0, 0)
        grid.addWidget(self.spin_x, 0, 1)
        grid.addWidget(QLabel("Y:"), 1, 0)
        grid.addWidget(self.spin_y, 1, 1)
        grid.addWidget(QLabel("Z:"), 2, 0)
        grid.addWidget(self.spin_z, 2, 1)
        
        layout.addLayout(grid)
        
        # 提示当前位置
        info = QLabel(f"💡 当前机器位置：X={self.current_pos[0]:.2f}, Y={self.current_pos[1]:.2f}, Z={self.current_pos[2]:.2f}")
        info.setStyleSheet("color: #6B7280; font-size: 12px; padding: 5px;")
        layout.addWidget(info)
        
        layout.addStretch()
        
        # 按钮
        btn_layout = QHBoxLayout()
        
        btn_save = QPushButton("✓ 保存")
        btn_save.setStyleSheet("background-color: #2563EB; min-height: 30px;")
        btn_save.clicked.connect(self.save_position)
        
        btn_use_current = QPushButton("使用当前位置")
        btn_use_current.clicked.connect(self.use_current_position)
        
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_use_current)
        btn_layout.addWidget(btn_cancel)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def use_current_position(self):
        """使用当前机器位置"""
        self.spin_x.setValue(self.current_pos[0])
        self.spin_y.setValue(self.current_pos[1])
        self.spin_z.setValue(self.current_pos[2])
    
    def save_position(self):
        """保存位置"""
        x = self.spin_x.value()
        y = self.spin_y.value()
        z = self.spin_z.value()
        
        data = {
            "x": x,
            "y": y,
            "z": z,
            "timestamp": str(__import__('datetime').datetime.now())
        }
        
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.accept()
