"""
ROI Selection Dialog for Halcon Model Training
Allows user to select a rectangular region of interest for better model quality
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QApplication
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor


class ROISelectionDialog(QDialog):
    """对话框：在reference.png上选择ROI区域用于训练"""
    
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择训练区域 (ROI)")
        self.setModal(True)
        
        # 加载图像
        self.image_path = image_path
        self.cv_image = cv2.imread(image_path)
        if self.cv_image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换为QImage显示
        rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        self.qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
        # ROI状态
        self.roi_start = None  # QPoint
        self.roi_end = None    # QPoint
        self.is_drawing = False
        self.roi_rect = None   # QRect (x, y, w, h) in pixels
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 说明标签
        self.label_info = QLabel("用鼠标拖拽选择产品区域，松开确认")
        self.label_info.setAlignment(Qt.AlignCenter)
        self.label_info.setStyleSheet("font-size: 14px; padding: 10px; background-color: #2563EB; color: white; border-radius: 4px;")
        layout.addWidget(self.label_info)
        
        # 先缩放图片以适应屏幕
        screen = QApplication.desktop().screenGeometry()
        max_width = int(screen.width() * 0.6)  # 屏幕宽度的60%
        max_height = int(screen.height() * 0.6)  # 屏幕高度的60%
        
        if self.qt_image.width() > max_width or self.qt_image.height() > max_height:
            self.qt_image = self.qt_image.scaled(
                max_width, max_height, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        
        # 再创建ImageLabel（使用缩放后的图像）
        self.image_label = ImageLabelWithROI(self.qt_image)
        self.image_label.mouse_press.connect(self.on_mouse_press)
        self.image_label.mouse_move.connect(self.on_mouse_move)
        self.image_label.mouse_release.connect(self.on_mouse_release)
        layout.addWidget(self.image_label)
        
        # ROI信息显示
        self.label_roi_info = QLabel("未选择ROI")
        self.label_roi_info.setAlignment(Qt.AlignCenter)
        self.label_roi_info.setStyleSheet("padding: 5px; background-color: #1F2933; border-radius: 4px;")
        layout.addWidget(self.label_roi_info)
        
        # 按钮
        btn_layout = QHBoxLayout()
        
        self.btn_reset = QPushButton("重新选择")
        self.btn_reset.clicked.connect(self.reset_roi)
        btn_layout.addWidget(self.btn_reset)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)
        
        self.btn_confirm = QPushButton("确认")
        self.btn_confirm.clicked.connect(self.accept)
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.setStyleSheet("background-color: #10B981;")
        btn_layout.addWidget(self.btn_confirm)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        self.resize(max_width + 50, max_height + 200)
    
    def on_mouse_press(self, pos):
        """鼠标按下，开始选择"""
        self.is_drawing = True
        self.roi_start = pos
        self.roi_end = pos
        self.update_display()
    
    def on_mouse_move(self, pos):
        """鼠标移动，更新ROI"""
        if self.is_drawing:
            self.roi_end = pos
            self.update_display()
    
    def on_mouse_release(self, pos):
        """鼠标释放，完成选择"""
        if self.is_drawing:
            self.roi_end = pos
            self.is_drawing = False
            self.finalize_roi()
    
    def finalize_roi(self):
        """完成ROI选择"""
        if self.roi_start and self.roi_end:
            x1, y1 = self.roi_start.x(), self.roi_start.y()
            x2, y2 = self.roi_end.x(), self.roi_end.y()
            
            print(f"[ROI] 选择坐标: ({x1},{y1}) -> ({x2},{y2})")  # Debug
            
            # 确保x1 < x2, y1 < y2
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            print(f"[ROI] 计算尺寸: x={x}, y={y}, w={w}, h={h}")  # Debug
            
            if w > 10 and h > 10:  # 最小尺寸限制
                self.roi_rect = QRect(x, y, w, h)
                self.update_display()
                self.update_roi_info()
                self.btn_confirm.setEnabled(True)
                self.btn_confirm.setStyleSheet("background-color: #10B981; color: white;")
                print(f"[ROI] 选择成功！")  # Debug
            else:
                self.label_roi_info.setText("ROI区域太小，请重新选择")
                self.btn_confirm.setEnabled(False)
                print(f"[ROI] 区域太小！")  # Debug
    
    def update_display(self):
        """更新图像显示（绘制ROI矩形）"""
        if self.roi_start and self.roi_end:
            x1, y1 = self.roi_start.x(), self.roi_start.y()
            x2, y2 = self.roi_end.x(), self.roi_end.y()
            self.image_label.set_roi_rect(QRect(QPoint(x1, y1), QPoint(x2, y2)))
    
    def update_roi_info(self):
        """更新ROI信息文本"""
        if self.roi_rect:
            self.label_roi_info.setText(
                f"ROI: ({self.roi_rect.x()}, {self.roi_rect.y()}) "
                f"宽={self.roi_rect.width()}, 高={self.roi_rect.height()}"
            )
    
    def reset_roi(self):
        """重置ROI选择"""
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.is_drawing = False
        self.image_label.set_roi_rect(None)
        self.label_roi_info.setText("未选择ROI")
        self.btn_confirm.setEnabled(False)
    
    def get_roi_halcon_format(self):
        """
        获取ROI坐标（Halcon格式）
        Returns: (row1, col1, row2, col2) or None
        """
        if not self.roi_rect:
            return None
        
        # QRect: (x, y, w, h) -> Halcon: (row1, col1, row2, col2)
        # Halcon: row=Y轴, col=X轴
        x = self.roi_rect.x()
        y = self.roi_rect.y()
        w = self.roi_rect.width()
        h = self.roi_rect.height()
        
        row1 = y
        col1 = x
        row2 = y + h
        col2 = x + w
        
        return (row1, col1, row2, col2)


class ImageLabelWithROI(QLabel):
    """支持ROI绘制的图像标签"""
    
    from PyQt5.QtCore import pyqtSignal
    
    mouse_press = pyqtSignal(QPoint)
    mouse_move = pyqtSignal(QPoint)
    mouse_release = pyqtSignal(QPoint)
    
    def __init__(self, qimage):
        super().__init__()
        self.original_image = qimage
        self.roi_rect = None
        self.setMouseTracking(True)  # 启用鼠标跟踪
        self.update_pixmap()
    
    def set_roi_rect(self, rect):
        """设置ROI矩形并重绘"""
        self.roi_rect = rect
        self.update_pixmap()
    
    def update_pixmap(self):
        """更新显示的图像（绘制ROI）"""
        # 创建副本用于绘制
        pixmap = QPixmap.fromImage(self.original_image)
        
        if self.roi_rect:
            # 规范化矩形（确保width和height为正）
            rect = self.roi_rect.normalized()
            print(f"[ImageLabel] Drawing ROI: {rect}")  # Debug
            
            # 在pixmap上绘制ROI矩形
            painter = QPainter(pixmap)
            
            # 先画不透明填充（绿色）
            painter.fillRect(rect, QColor(0, 255, 0, 200))  # 几乎不透明
            
            # 再画超粗边框（红色）
            pen = QPen(QColor(255, 0, 0), 20, Qt.SolidLine)  # 红色，20像素宽
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)  # 不填充
            painter.drawRect(rect)
            
            painter.end()
        
        self.setPixmap(pixmap)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_press.emit(event.pos())
            print(f"[ImageLabel] Mouse Press: {event.pos()}")  # Debug
    
    def mouseMoveEvent(self, event):
        self.mouse_move.emit(event.pos())
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_release.emit(event.pos())
            print(f"[ImageLabel] Mouse Release: {event.pos()}")  # Debug
