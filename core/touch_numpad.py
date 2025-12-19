"""
触屏数字键盘对话框
专为iPad/触屏设备设计的数值输入界面
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class TouchNumberPad(QDialog):
    """触屏数字键盘对话框"""
    
    def __init__(self, title="输入数值", current_value=0.0, min_val=0.0, max_val=999.9, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        
        self.min_val = min_val
        self.max_val = max_val
        self.current_text = str(current_value) if current_value != 0.0 else ""
        self.result_value = current_value
        
        self.init_ui()
        self.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                font-weight: bold;
                min-height: 60px;
                min-width: 80px;
                border: 2px solid #3B82F6;
                border-radius: 8px;
                background-color: #EFF6FF;
                color: #1F2937;  /* 强制深色文字 */
            }
            QPushButton:pressed {
                background-color: #3B82F6;
                color: white;
            }
            QPushButton:hover {
                background-color: #DBEAFE;
                color: #1F2937;
            }
            QLabel#display {
                font-size: 32px;
                font-weight: bold;
                background-color: white;
                color: #1F2937;  /* 强制深色文字 */
                border: 3px solid #3B82F6;
                border-radius: 8px;
                padding: 15px;
                min-height: 50px;
            }
        """)
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # 显示区域
        self.display = QLabel(self.current_text if self.current_text else "0")
        self.display.setObjectName("display")
        self.display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.display)
        
        # 数字键盘网格
        grid = QGridLayout()
        grid.setSpacing(8)
        
        # 数字按钮 7-9
        for i in range(7, 10):
            btn = QPushButton(str(i))
            btn.clicked.connect(lambda checked, num=i: self.append_digit(str(num)))
            grid.addWidget(btn, 0, i - 7)
        
        # 数字按钮 4-6
        for i in range(4, 7):
            btn = QPushButton(str(i))
            btn.clicked.connect(lambda checked, num=i: self.append_digit(str(num)))
            grid.addWidget(btn, 1, i - 4)
        
        # 数字按钮 1-3
        for i in range(1, 4):
            btn = QPushButton(str(i))
            btn.clicked.connect(lambda checked, num=i: self.append_digit(str(num)))
            grid.addWidget(btn, 2, i - 1)
        
        # 第四行: 小数点, 0, 退格
        btn_dot = QPushButton(".")
        btn_dot.clicked.connect(lambda: self.append_digit("."))
        grid.addWidget(btn_dot, 3, 0)
        
        btn_zero = QPushButton("0")
        btn_zero.clicked.connect(lambda: self.append_digit("0"))
        grid.addWidget(btn_zero, 3, 1)
        
        btn_backspace = QPushButton("⌫")
        btn_backspace.clicked.connect(self.backspace)
        grid.addWidget(btn_backspace, 3, 2)
        
        layout.addLayout(grid)
        
        # 底部按钮
        bottom_layout = QHBoxLayout()
        
        btn_clear = QPushButton("清除")
        btn_clear.setStyleSheet("""
            QPushButton {
                background-color: #FEF2F2;
                border-color: #EF4444;
                color: #DC2626;
            }
            QPushButton:pressed {
                background-color: #EF4444;
                color: white;
            }
        """)
        btn_clear.clicked.connect(self.clear_all)
        bottom_layout.addWidget(btn_clear)
        
        btn_ok = QPushButton("✓ 确定")
        btn_ok.setStyleSheet("""
            QPushButton {
                background-color: #F0FDF4;
                border-color: #10B981;
                color: #059669;
            }
            QPushButton:pressed {
                background-color: #10B981;
                color: white;
            }
        """)
        btn_ok.clicked.connect(self.confirm)
        bottom_layout.addWidget(btn_ok)
        
        layout.addLayout(bottom_layout)
        
        self.setLayout(layout)
        self.resize(350, 450)
    
    def append_digit(self, digit):
        """添加数字或小数点"""
        # 防止多个小数点
        if digit == "." and "." in self.current_text:
            return
        
        # 添加数字
        self.current_text += digit
        self.display.setText(self.current_text)
    
    def backspace(self):
        """退格删除"""
        if self.current_text:
            self.current_text = self.current_text[:-1]
            self.display.setText(self.current_text if self.current_text else "0")
    
    def clear_all(self):
        """清除所有"""
        self.current_text = ""
        self.display.setText("0")
    
    def confirm(self):
        """确认输入"""
        try:
            value = float(self.current_text) if self.current_text else 0.0
            
            # 范围检查
            if value < self.min_val:
                value = self.min_val
            elif value > self.max_val:
                value = self.max_val
            
            self.result_value = value
            self.accept()
        except ValueError:
            # 输入无效，恢复到0
            self.result_value = 0.0
            self.accept()
    
    @staticmethod
    def get_number(title="输入数值", current_value=0.0, min_val=0.0, max_val=999.9, parent=None):
        """
        静态方法：弹出数字键盘获取数值
        
        Args:
            title: 对话框标题
            current_value: 当前值
            min_val: 最小值
            max_val: 最大值
            parent: 父窗口
        
        Returns:
            (value, ok) - 返回输入的值和是否确认
        """
        dialog = TouchNumberPad(title, current_value, min_val, max_val, parent)
        result = dialog.exec_()
        return dialog.result_value, result == QDialog.Accepted
