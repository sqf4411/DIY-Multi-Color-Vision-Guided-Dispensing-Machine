"""
简化的ROI选择对话框 - 调试版本
"""

import cv2
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor


class SimpleROIDialog(QDialog):
    """简化的ROI选择对话框用于调试"""
    
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("ROI选择 (调试版)")
        
        # 加载图像
        cv_img = cv2.imread(image_path)
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        
        # 转换为QImage
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        
        # 缩放到合适大小
        qt_image = qt_image.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # 在图像上画一个明显的测试矩形
        pixmap = QPixmap.fromImage(qt_image)
        painter = QPainter(pixmap)
        
        # 画一个大的绿色矩形 (100,100) 到 (400, 400)
        test_rect = QRect(100, 100, 300, 300)
        painter.fillRect(test_rect, QColor(0, 255, 0, 100))  # 半透明填充
        pen = QPen(QColor(255, 0, 0), 10, Qt.SolidLine)  # 红色粗边框
        painter.setPen(pen)
        painter.drawRect(test_rect)
        painter.end()
        
        # 显示
        layout = QVBoxLayout()
        label = QLabel()
        label.setPixmap(pixmap)
        layout.addWidget(label)
        
        btn = QPushButton("关闭")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
        
        self.setLayout(layout)


# 测试代码
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    
    # 替换为实际的图像路径
    dlg = SimpleROIDialog("h:/CodeForGithub/DIY-Multi-Color-Vision-Guided-Dispensing-Machine/produces/new2/reference.png")
    dlg.exec_()
