"""
ROI Selection Dialog - 使用paintEvent的简化版本
支持模具边界显示和正确的坐标缩放
"""

import cv2
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QApplication
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor


class SimpleImageWidget(QWidget):
    """使用paintEvent直接绘制的图像Widget"""
    
    mouse_press = pyqtSignal(QPoint)
    mouse_move = pyqtSignal(QPoint)
    mouse_release = pyqtSignal(QPoint)
    
    def __init__(self, qimage, mold_boundary=None, scale_factor=1.0):
        super().__init__()
        self.image = qimage
        self.mold_boundary = mold_boundary
        self.scale_factor = scale_factor
        self.roi_rect = None
        self.setMouseTracking(True)
        self.setMinimumSize(qimage.width(), qimage.height())
    
    def set_roi(self, rect):
        self.roi_rect = rect
        self.update()  # 触发重绘
    
    def paintEvent(self, event):
        """直接在paintEvent中绘制"""
        painter = QPainter(self)
        
        # 画图像
        painter.drawImage(0, 0, self.image)
        
        # 画模具边界（如果有）- 绿色虚线框
        if self.mold_boundary:
            x = int(self.mold_boundary['x'] * self.scale_factor)
            y = int(self.mold_boundary['y'] * self.scale_factor)
            w = int(self.mold_boundary['width'] * self.scale_factor)
            h = int(self.mold_boundary['height'] * self.scale_factor)
            
            pen = QPen(QColor(0, 255, 0), 3, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x, y, w, h)
        
        # 画ROI - 半透明绿色填充 + 红色边框
        if self.roi_rect:
            rect = self.roi_rect.normalized()
            
            # 填充
            painter.fillRect(rect, QColor(0, 255, 0, 150))
            
            # 边框
            pen = QPen(QColor(255, 0, 0), 5, Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_press.emit(event.pos())
    
    def mouseMoveEvent(self, event):
        self.mouse_move.emit(event.pos())
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_release.emit(event.pos())


class SimpleROIDialog(QDialog):
    """使用paintEvent的简化ROI对话框"""
    
    def __init__(self, image_path, mold_boundary=None):
        super().__init__()
        self.setWindowTitle("选择训练区域 (ROI) - 请框选单个产品")
        
        # 加载并缩放图像
        cv_img = cv2.imread(image_path)
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        
        screen = QApplication.desktop().screenGeometry()
        max_w = int(screen.width() * 0.7)
        max_h = int(screen.height() * 0.7)
        
        self.scale_factor = 1.0  # 缩放比例
        if qt_image.width() > max_w or qt_image.height() > max_h:
            self.scale_factor = min(max_w / qt_image.width(), max_h / qt_image.height())
            new_w = int(qt_image.width() * self.scale_factor)
            new_h = int(qt_image.height() * self.scale_factor)
            qt_image = qt_image.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # 模具边界（如果提供）
        self.mold_boundary = mold_boundary
        
        # ROI状态
        self.roi_start = None
        self.roi_end = None
        self.is_drawing = False
        self.roi_rect = None
        
        # UI
        layout = QVBoxLayout()
        
        info_text = "用鼠标拖拽选择**单个产品**区域"
        if mold_boundary:
            info_text += " (绿色虚线框为模具边界)"
        self.info_label = QLabel(info_text)
        self.info_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #2563EB; color: white;")
        layout.addWidget(self.info_label)
        
        self.image_widget = SimpleImageWidget(qt_image, mold_boundary, self.scale_factor)
        self.image_widget.mouse_press.connect(self.on_press)
        self.image_widget.mouse_move.connect(self.on_move)
        self.image_widget.mouse_release.connect(self.on_release)
        layout.addWidget(self.image_widget)
        
        self.roi_label = QLabel("未选择ROI")
        layout.addWidget(self.roi_label)
        
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
        self.btn_confirm.setStyleSheet("background-color: #10B981; color: white;")
        btn_layout.addWidget(self.btn_confirm)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def on_press(self, pos):
        self.is_drawing = True
        self.roi_start = pos
        self.roi_end = pos
        self.image_widget.set_roi(QRect(pos, pos))
    
    def on_move(self, pos):
        if self.is_drawing:
            self.roi_end = pos
            self.image_widget.set_roi(QRect(self.roi_start, self.roi_end))
    
    def on_release(self, pos):
        if self.is_drawing:
            self.roi_end = pos
            self.is_drawing = False
            
            x1, y1 = self.roi_start.x(), self.roi_start.y()
            x2, y2 = self.roi_end.x(), self.roi_end.y()
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            
            if w > 10 and h > 10:
                self.roi_rect = QRect(x, y, w, h)
                self.image_widget.set_roi(self.roi_rect)
                
                # 转换回原始图像坐标
                orig_x = int(x / self.scale_factor)
                orig_y = int(y / self.scale_factor)
                orig_w = int(w / self.scale_factor)
                orig_h = int(h / self.scale_factor)
                
                self.roi_label.setText(f"ROI: ({orig_x},{orig_y}) 宽={orig_w} 高={orig_h}")
                self.btn_confirm.setEnabled(True)
            else:
                self.roi_label.setText("ROI区域太小，请重新选择")
    
    def reset_roi(self):
        """重置ROI"""
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.is_drawing = False
        self.image_widget.set_roi(None)
        self.roi_label.setText("未选择ROI")
        self.btn_confirm.setEnabled(False)
    
    def get_roi_rect(self):
        """返回ROI矩形（原始图像坐标）"""
        if not self.roi_rect:
            return None
        
        # 转换回原始图像坐标
        orig_x = int(self.roi_rect.x() / self.scale_factor)
        orig_y = int(self.roi_rect.y() / self.scale_factor)
        orig_w = int(self.roi_rect.width() / self.scale_factor)
        orig_h = int(self.roi_rect.height() / self.scale_factor)
        
        return QRect(orig_x, orig_y, orig_w, orig_h)
    
    def get_roi_halcon_format(self):
        """返回Halcon格式ROI（原始图像坐标）"""
        rect = self.get_roi_rect()
        if not rect:
            return None
        return (rect.y(), rect.x(), 
                rect.y() + rect.height(),
                rect.x() + rect.width())
