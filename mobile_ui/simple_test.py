from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

class SimpleMobileMonitor(QDialog):
    """最简化的测试副屏 - 只显示文字，完全没有任何处理"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("测试副屏")
        self.resize(800, 600)
        
        layout = QVBoxLayout()
        label = QLabel("这是一个空白测试窗口\n\n没有任何图像处理\n没有任何定时器\n没有任何HTTP请求")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 24px; color: white; background-color: #1F2937;")
        
        layout.addWidget(label)
        self.setLayout(layout)
