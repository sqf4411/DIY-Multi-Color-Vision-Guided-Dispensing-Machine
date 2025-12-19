from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QWidget, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QFont, QImage, QPixmap
import numpy as np
import cv2

class MobileFrameProcessor(QThread):
    """
    独立线程处理副屏图像，完全不阻塞主UI线程
    """
    frame_ready = pyqtSignal(QPixmap)  # 发送处理好的QPixmap
    
    def __init__(self, max_width=800, target_size=(800, 600)):
        super().__init__()
        self.max_width = max_width
        self.target_size = target_size
        self.running = True
        self.current_frame = None  # 当前待处理帧
        from threading import Lock
        self.frame_lock = Lock()  # 线程锁保护帧数据
        
    def add_frame(self, frame):
        """添加帧到处理队列（只保留最新一帧）"""
        with self.frame_lock:
            self.current_frame = frame  # 直接覆盖，总是保留最新帧
    
    def run(self):
        """后台线程：处理图像"""
        import time
        while self.running:
            # 安全地读取当前帧
            frame = None
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = self.current_frame
                    self.current_frame = None  # 清空，避免重复处理
            
            if frame is not None:
                try:
                    # 所有耗时操作都在这个后台线程完成
                    pixmap = self._process_frame(frame)
                    if pixmap:
                        self.frame_ready.emit(pixmap)
                except Exception as e:
                    pass
            else:
                time.sleep(0.01)  # 空闲时休眠10ms
    
    def _process_frame(self, frame):
        """在后台线程处理帧"""
        if isinstance(frame, np.ndarray):
            h, w = frame.shape[:2]
            
            # 降低分辨率
            if w > self.max_width:
                scale = self.max_width / w
                new_w = self.max_width
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                h, w = new_h, new_w
            
            # 转换为QImage
            if len(frame.shape) == 2:
                # 灰度图
                frame_copy = np.ascontiguousarray(frame)
                qimg = QImage(frame_copy.data, w, h, w, QImage.Format_Grayscale8)
                qimg = qimg.copy()
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                # BGR -> RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = np.ascontiguousarray(rgb_frame)
                bytes_per_line = 3 * w
                qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                qimg = qimg.copy()
            else:
                return None
            
            # 转换为QPixmap并缩放
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(self.target_size[0], self.target_size[1], 
                              Qt.KeepAspectRatio, Qt.FastTransformation)
            return scaled
        
        return None
    
    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()

class MobileMonitorDialog(QDialog):
    """
    专门为 iPad/手机 副屏设计的极简监控看板。
    特点：
    - 大字体显示 XYZ 坐标
    - 明显的电磁阀状态指示（颜色）
    - 居中的相机画面
    - 无复杂菜单，仅供观察
    """
    def __init__(self, parent=None, machine=None, valve_pin_name="VALVE_A_CTRL"):
        super().__init__(parent)
        self.machine = machine
        self.valve_pin_name = valve_pin_name
        self.is_fullscreen = False  # 全屏状态标志
        
        # 🚀 副屏不再显示相机画面，只显示数据，避免卡顿主UI
        # 移除所有图像处理相关代码
        
        # 🚀 USB画面帧率限制
        from time import perf_counter
        self.perf_counter = perf_counter
        self.last_usb_frame_time = 0
        self.usb_frame_interval = 0.2  # 200ms一帧，5fps，降低主线程占用
        
        self.setWindowTitle("iPad 无线示教监控板")
        self.resize(1024, 768)
        self.setStyleSheet("background-color: #000000; color: white;")
        
        self.init_ui()
        
        # ❌ 禁用定时器数据查询 - 会导致主UI卡顿
        # 原因：每500ms的HTTP请求会阻塞主线程并与主UI的查询冲突
        # 改为被动接收主UI推送的数据
        
        # 定时器：刷新坐标和阀门状态（500ms）
        # self.status_timer = QTimer(self)
        # self.status_timer.timeout.connect(self.poll_status)
        # self.status_timer.start(500)

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 1. 顶部栏：标题 + 全屏按钮 + 关闭
        top_bar = QHBoxLayout()
        lbl_title = QLabel("📡 远程数据监控 (iPad Mode)")
        lbl_title.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))  # 增大字号
        lbl_title.setStyleSheet("color: #9CA3AF;")
        
        # 全屏按钮
        self.btn_fullscreen = QPushButton("🔲 全屏")
        self.btn_fullscreen.setFixedSize(140, 60)  # 增大按钮
        self.btn_fullscreen.setStyleSheet("""
            QPushButton { background-color: #1E40AF; border-radius: 8px; font-size: 20px; font-weight: bold; color: white; }
            QPushButton:hover { background-color: #2563EB; }
            QPushButton:pressed { background-color: #1E3A8A; }
        """)
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        
        btn_close = QPushButton("✖ 退出")
        btn_close.setFixedSize(140, 60)  # 增大按钮
        btn_close.setStyleSheet("""
            QPushButton { background-color: #374151; border-radius: 8px; font-size: 20px; font-weight: bold; }
            QPushButton:hover { background-color: #4B5563; }
            QPushButton:pressed { background-color: #1F2937; }
        """)
        btn_close.clicked.connect(self.close)
        
        top_bar.addWidget(lbl_title)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_fullscreen)
        top_bar.addWidget(btn_close)
        main_layout.addLayout(top_bar)
        
        # 2. 中间：USB相机画面 - 从主UI转移过来
        self.lbl_camera = QLabel("📹 USB针头监控\n等待画面...")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("""
            background-color: #1F2937; 
            border: 2px solid #374151; 
            border-radius: 16px;
            font-size: 24px;
            color: #9CA3AF;
            padding: 40px;
        """)
        self.lbl_camera.setMinimumSize(600, 400)
        self.lbl_camera.setScaledContents(False)
        
        main_layout.addWidget(self.lbl_camera, stretch=1)
        
        # 3. 底部：数据面板 (Grid)
        # 包含：X, Y, Z, Valve Status
        data_panel = QFrame()
        data_panel.setStyleSheet("background-color: #1F2933; border-radius: 16px; padding: 15px;")
        data_layout = QHBoxLayout(data_panel)
        
        # 字体定义 - 增大字号
        font_label = QFont("Consolas", 20, QFont.Bold)  # 从16增大到20
        font_val = QFont("Consolas", 64, QFont.Bold)    # 从48增大到64
        
        # --- X ---
        x_layout, self.lbl_x_val = self.create_data_block("X Axis", "0.00", "#3B82F6", font_label, font_val)
        data_layout.addLayout(x_layout)
        
        # --- Y ---
        y_layout, self.lbl_y_val = self.create_data_block("Y Axis", "0.00", "#3B82F6", font_label, font_val)
        data_layout.addLayout(y_layout)
        
        # --- Z ---
        z_layout, self.lbl_z_val = self.create_data_block("Z Axis", "0.00", "#F59E0B", font_label, font_val)
        data_layout.addLayout(z_layout)
        
        # --- Valve ---
        self.lbl_valve_status = QLabel("CLOSED")
        self.lbl_valve_status.setAlignment(Qt.AlignCenter)
        self.lbl_valve_status.setFont(QFont("Arial", 40, QFont.Bold))  # 从32增大到40
        self.lbl_valve_status.setStyleSheet("color: #9CA3AF; background-color: #374151; border-radius: 12px; padding: 15px;")
        # 容器
        v_layout = QVBoxLayout()
        v_title = QLabel("VALVE")
        v_title.setFont(font_label)
        v_title.setAlignment(Qt.AlignCenter)
        v_layout.addWidget(v_title)
        v_layout.addWidget(self.lbl_valve_status)
        data_layout.addLayout(v_layout)
        
        main_layout.addWidget(data_panel)
        
        self.setLayout(main_layout)
    
    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.is_fullscreen:
            # 退出全屏
            self.showNormal()
            self.btn_fullscreen.setText("🔲 全屏")
            self.is_fullscreen = False
        else:
            # 进入全屏
            self.showFullScreen()
            self.btn_fullscreen.setText("🗗 退出全屏")
            self.is_fullscreen = True
    
    def closeEvent(self, event):
        """关闭窗口时将USB流恢复到主UI"""
        try:
            # 🚀 恢复USB流到主UI
            if hasattr(self, 'parent') and self.parent():
                parent = self.parent()
                
                # 恢复USB相机连接
                if hasattr(parent, 'usb_cam_worker') and parent.usb_cam_worker:
                    try:
                        # 断开副屏
                        parent.usb_cam_worker.frame_signal.disconnect(self.update_usb_image)
                        # 重新连接主UI
                        parent.usb_cam_worker.frame_signal.connect(parent.update_usb_image)
                        # 显示主UI的USB画面
                        if hasattr(parent, 'label_usb_cam'):
                            parent.label_usb_cam.show()
                        print("[副屏] USB相机流已恢复到主UI")
                    except Exception as e:
                        print(f"[副屏] USB流恢复失败: {e}")
                
                # 更新主窗口按钮状态
                if hasattr(parent, 'btn_ipad'):
                    parent.btn_ipad.setText("📱 启动 iPad 监控副屏")
                    parent.btn_ipad.setStyleSheet("background-color: #059669; color: white; font-weight: bold; font-size: 16px; border-radius: 6px;")
        except:
            pass
        
        event.accept()

    def create_data_block(self, title, init_val, color, f_title, f_val):
        layout = QVBoxLayout()
        lbl_t = QLabel(title)
        lbl_t.setFont(f_title)
        lbl_t.setStyleSheet("color: #9CA3AF;")
        lbl_t.setAlignment(Qt.AlignCenter)
        
        lbl_v = QLabel(init_val)
        lbl_v.setFont(f_val)
        lbl_v.setStyleSheet(f"color: {color};")
        lbl_v.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(lbl_t)
        layout.addWidget(lbl_v)
        return layout, lbl_v # 返回 layout 和 value label

    def update_data(self, x, y, z, valve_open):
        """由主UI推送数据更新（避免副屏主动查询导致卡顿）"""
        try:
            self.lbl_x_val.setText(f"{x:.2f}")
            self.lbl_y_val.setText(f"{y:.2f}")
            self.lbl_z_val.setText(f"{z:.2f}")
            
            if valve_open:
                self.lbl_valve_status.setText("OPEN")
                self.lbl_valve_status.setStyleSheet("color: white; background-color: #10B981; border-radius: 12px; padding: 10px;")
            else:
                self.lbl_valve_status.setText("CLOSED")
                self.lbl_valve_status.setStyleSheet("color: #9CA3AF; background-color: #374151; border-radius: 12px; padding: 10px;")
        except:
            pass
    
    def poll_status(self):
        """❌ 已禁用 - 主动查询会导致HTTP请求阻塞主线程"""
        # 不再使用此方法
        pass

    def update_usb_image(self, qimg):
        """接收USB相机画面并显示 - 严格帧率限制避免卡顿主UI"""
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
            scaled = pix.scaled(self.lbl_camera.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
            self.lbl_camera.setPixmap(scaled)
        except Exception as e:
            pass

