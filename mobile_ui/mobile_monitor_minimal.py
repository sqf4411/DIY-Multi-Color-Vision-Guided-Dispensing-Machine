from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap

class MobileMonitorDialog(QDialog):
    """
    最精简的副屏 - 只显示USB画面，不做任何其他处理
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 🚀 USB画面帧率限制
        from time import perf_counter
        self.perf_counter = perf_counter
        self.last_usb_frame_time = 0
        self.usb_frame_interval = 0.2  # 200ms一帧，5fps
        
        self.setWindowTitle("iPad 无线监控")
        self.resize(1024, 768)
        self.setStyleSheet("background-color: #000000; color: white;")
        
        self.init_ui()
    
    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # 顶部栏
        top_bar = QHBoxLayout()
        lbl_title = QLabel("📡 USB针头监控")
        lbl_title.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        lbl_title.setStyleSheet("color: white;")
        
        btn_close = QPushButton("✖ 退出")
        btn_close.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        btn_close.setStyleSheet("""
            background-color: #EF4444; 
            color: white; 
            padding: 8px 16px; 
            border-radius: 6px;
        """)
        btn_close.clicked.connect(self.close)
        
        top_bar.addWidget(lbl_title)
        top_bar.addStretch()
        top_bar.addWidget(btn_close)
        main_layout.addLayout(top_bar)
        
        # 中间：USB相机画面
        self.lbl_camera = QLabel("📹 等待USB画面...")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("""
            background-color: #1F2937; 
            border: 2px solid #374151; 
            border-radius: 16px;
            font-size: 24px;
            color: #9CA3AF;
        """)
        self.lbl_camera.setMinimumSize(800, 600)
        self.lbl_camera.setScaledContents(False)
        
        main_layout.addWidget(self.lbl_camera, stretch=1)
        
        self.setLayout(main_layout)
    
    def update_usb_image(self, qimg):
        """接收USB相机画面并显示 - 严格帧率限制"""
        if not self.isVisible():
            return
        
        # 🚀 帧率限制 - 只有每200ms才更新一次
        current_time = self.perf_counter()
        if current_time - self.last_usb_frame_time < self.usb_frame_interval:
            return  # 跳过此帧
        
        self.last_usb_frame_time = current_time
        
        try:
            # 缩放以适应标签大小
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                self.lbl_camera.size(), 
                Qt.KeepAspectRatio, 
                Qt.FastTransformation  # 快速变换
            )
            self.lbl_camera.setPixmap(scaled)
        except:
            pass
    
    def closeEvent(self, event):
        """关闭时恢复USB流到主UI"""
        try:
            if hasattr(self, 'parent') and self.parent():
                parent = self.parent()
                
                # 恢复USB相机连接
                if hasattr(parent, 'usb_cam_worker') and parent.usb_cam_worker:
                    try:
                        parent.usb_cam_worker.frame_signal.disconnect(self.update_usb_image)
                        parent.usb_cam_worker.frame_signal.connect(parent.update_usb_image)
                        if hasattr(parent, 'label_usb_cam'):
                            parent.label_usb_cam.show()
                    except:
                        pass
                
                # 更新按钮状态
                if hasattr(parent, 'btn_ipad'):
                    parent.btn_ipad.setText("📱 启动 iPad 监控副屏")
                    parent.btn_ipad.setStyleSheet(
                        "background-color: #059669; color: white; "
                        "font-weight: bold; font-size: 16px; border-radius: 6px;"
                    )
        except:
            pass
        
        event.accept()
