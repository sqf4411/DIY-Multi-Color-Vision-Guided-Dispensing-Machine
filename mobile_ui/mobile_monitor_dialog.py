from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QWidget, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap

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
        
        self.setWindowTitle("iPad 无线示教监控板")
        self.resize(1024, 768) # 典型的 iPad 横屏分辨率
        self.setStyleSheet("background-color: #000000; color: white;")
        
        self.init_ui()
        
        # 定时器：刷新坐标和阀门状态 (因为这是独立的 Dialog，我们自己轮询更简单，不依赖主线程的 PollingWorker)
        # 也可以通过 Signal/Slot 从主界面传过来，但为了解耦，这里自己查。
        # 考虑到 iPad 主要是看，频率不用太高，0.5s 一次即可。
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_status)
        self.timer.start(500)

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 1. 顶部栏：标题 + 关闭
        top_bar = QHBoxLayout()
        lbl_title = QLabel("📡 远程数据监控 (iPad Mode)")
        lbl_title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        lbl_title.setStyleSheet("color: #9CA3AF;")
        
        btn_close = QPushButton("退出监控")
        btn_close.setFixedSize(120, 50)
        btn_close.setStyleSheet("""
            QPushButton { background-color: #374151; border-radius: 8px; font-size: 16px; font-weight: bold; }
            QPushButton:pressed { background-color: #4B5563; }
        """)
        btn_close.clicked.connect(self.close)
        
        top_bar.addWidget(lbl_title)
        top_bar.addStretch()
        top_bar.addWidget(btn_close)
        main_layout.addLayout(top_bar)
        
        # 2. 中间：相机画面
        # 使用 QLabel 显示
        self.lbl_camera = QLabel("Wait For Stream...")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("background-color: #111827; border: 2px solid #374151; border-radius: 12px;")
        self.lbl_camera.setMinimumSize(640, 480)
        self.lbl_camera.setScaledContents(False) # 保持比例自己算，或者 True 拉伸
        # 为了保持比例，通常设为 False 并在 setPixmap 时处理，或者用 scaledContents=True 配合 KeepAspectRatio
        
        main_layout.addWidget(self.lbl_camera, stretch=1)
        
        # 3. 底部：数据面板 (Grid)
        # 包含：X, Y, Z, Valve Status
        data_panel = QFrame()
        data_panel.setStyleSheet("background-color: #1F2933; border-radius: 16px; padding: 10px;")
        data_layout = QHBoxLayout(data_panel)
        
        # 字体定义
        font_label = QFont("Consolas", 16)
        font_val = QFont("Consolas", 48, QFont.Bold)
        
        # --- X ---
        self.lbl_x = self.create_data_block("X Axis", "0.00", "#3B82F6", font_label, font_val)
        data_layout.addLayout(self.lbl_x)
        
        # --- Y ---
        self.lbl_y = self.create_data_block("Y Axis", "0.00", "#3B82F6", font_label, font_val)
        data_layout.addLayout(self.lbl_y)
        
        # --- Z ---
        self.lbl_z = self.create_data_block("Z Axis", "0.00", "#F59E0B", font_label, font_val)
        data_layout.addLayout(self.lbl_z)
        
        # --- Valve ---
        self.lbl_valve_status = QLabel("CLOSED")
        self.lbl_valve_status.setAlignment(Qt.AlignCenter)
        self.lbl_valve_status.setFont(QFont("Arial", 32, QFont.Bold))
        self.lbl_valve_status.setStyleSheet("color: #9CA3AF; background-color: #374151; border-radius: 12px; padding: 10px;")
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

    def poll_status(self):
        if not self.machine or not self.machine.connected:
            return
            
        # 1. 坐标
        x, y, z = self.machine.get_position()
        self.lbl_x[1].setText(f"{x:.2f}")
        self.lbl_y[1].setText(f"{y:.2f}")
        self.lbl_z[1].setText(f"{z:.2f}")
        
        # 2. 阀门
        # 注意: get_output_pin_status 可能需要一点时间，如果卡顿则应移到线程
        # 但本地 moonraker 应该很快 (<10ms)
        is_open = self.machine.get_output_pin_status(self.valve_pin_name)
        
        if is_open:
            self.lbl_valve_status.setText("OPEN")
            self.lbl_valve_status.setStyleSheet("color: white; background-color: #10B981; border-radius: 12px; padding: 10px;") # Green
        else:
            self.lbl_valve_status.setText("CLOSED")
            self.lbl_valve_status.setStyleSheet("color: #9CA3AF; background-color: #374151; border-radius: 12px; padding: 10px;") # Gray

    def update_image(self, qimg):
        """接收主界面传来的图像信号"""
        if self.isVisible():
            # 缩放以适应标签大小，保持比例
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(self.lbl_camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_camera.setPixmap(scaled)

