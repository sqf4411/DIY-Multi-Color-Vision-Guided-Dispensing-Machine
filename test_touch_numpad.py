"""
触屏数字键盘使用示例
演示如何在SpinBox中集成触屏数字键盘
"""

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QDoubleSpinBox, QLabel, QPushButton
from core.touch_numpad import TouchNumberPad
import sys


class TestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("触屏数字键盘测试")
        self.resize(400, 300)
        
        layout = QVBoxLayout()
        
        # 说明标签
        label = QLabel("点击下方按钮测试触屏数字键盘：")
        layout.addWidget(label)
        
        # 测试按钮
        btn_test = QPushButton("打开数字键盘")
        btn_test.clicked.connect(self.open_numpad)
        layout.addWidget(btn_test)
        
        # 显示结果
        self.result_label = QLabel("当前值: 0.0")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.result_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        self.current_value = 0.0
    
    def open_numpad(self):
        """打开触屏数字键盘"""
        value, ok = TouchNumberPad.get_number(
            title="输入数值",
            current_value=self.current_value,
            min_val=0.0,
            max_val=100.0,
            parent=self
        )
        
        if ok:
            self.current_value = value
            self.result_label.setText(f"当前值: {value:.2f}")
            print(f"用户输入: {value}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())
