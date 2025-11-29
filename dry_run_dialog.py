from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QGroupBox, QDoubleSpinBox, QSpinBox, QTextEdit,
    QDialogButtonBox
)
import numpy as np

class DryRunDialog(QDialog):
    def __init__(self, parent, paths_px, cam2mach_matrix, machine_origin, nozzle_offset, initial_params, img_center):
        super().__init__(parent)
        self.setWindowTitle("空跑参数设置 & G-code 预览")
        self.resize(800, 600)

        self.paths_px = paths_px
        self.m = cam2mach_matrix
        self.origin_x, self.origin_y, self.origin_z = machine_origin
        self.nozzle_offset_x, self.nozzle_offset_y = nozzle_offset
        self.center_x, self.center_y = img_center
        
        # 初始参数
        self.safe_z = initial_params.get('safe_z', 20.0)
        self.work_z = initial_params.get('work_z', 5.0)
        self.speed_xy = initial_params.get('speed_xy', 2000)
        self.speed_z = initial_params.get('speed_z', 500)

        self.init_ui()
        self.generate_gcode()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # 1. 参数设置区
        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout()

        # Safe Z
        param_layout.addWidget(QLabel("安全高度 (Safe Z):"), 0, 0)
        self.spn_safe_z = QDoubleSpinBox()
        self.spn_safe_z.setRange(0, 300)
        self.spn_safe_z.setValue(self.safe_z)
        self.spn_safe_z.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_safe_z, 0, 1)

        # Work Z
        param_layout.addWidget(QLabel("点胶高度 (Work Z):"), 0, 2)
        self.spn_work_z = QDoubleSpinBox()
        self.spn_work_z.setRange(0, 300)
        self.spn_work_z.setValue(self.work_z)
        self.spn_work_z.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_work_z, 0, 3)

        # Speed XY
        param_layout.addWidget(QLabel("XY 速度 (mm/min):"), 1, 0)
        self.spn_speed_xy = QSpinBox()
        self.spn_speed_xy.setRange(100, 20000)
        self.spn_speed_xy.setValue(int(self.speed_xy))
        self.spn_speed_xy.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_speed_xy, 1, 1)

        # Speed Z
        param_layout.addWidget(QLabel("Z 速度 (mm/min):"), 1, 2)
        self.spn_speed_z = QSpinBox()
        self.spn_speed_z.setRange(100, 5000)
        self.spn_speed_z.setValue(int(self.speed_z))
        self.spn_speed_z.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_speed_z, 1, 3)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 2. 预览区
        layout.addWidget(QLabel(f"已识别 {len(self.paths_px)} 条路径。G-code 预览:"))
        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc; font-family: Consolas;")
        layout.addWidget(self.text_preview)

        # 3. 按钮区
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.button(QDialogButtonBox.Ok).setText("执行 (Run)")
        btn_box.button(QDialogButtonBox.Cancel).setText("取消")
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.setLayout(layout)

    def on_param_changed(self):
        self.safe_z = self.spn_safe_z.value()
        self.work_z = self.spn_work_z.value()
        self.speed_xy = self.spn_speed_xy.value()
        self.speed_z = self.spn_speed_z.value()
        self.generate_gcode()

    def generate_gcode(self):
        gcode_lines = []
        gcode_lines.append("G90")
        gcode_lines.append(f"G0 Z{self.safe_z:.2f} F{self.speed_z}")

        for i, p_px in enumerate(self.paths_px):
            # 像素转毫米
            delta_px = p_px - np.array([self.center_x, self.center_y])
            delta_mm_x = delta_px[:, 0] * self.m[0, 0] + delta_px[:, 1] * self.m[0, 1]
            delta_mm_y = delta_px[:, 0] * self.m[1, 0] + delta_px[:, 1] * self.m[1, 1]

            abs_x = self.origin_x + delta_mm_x
            abs_y = self.origin_y + delta_mm_y
            path_machine = np.column_stack((abs_x, abs_y))

            if len(path_machine) == 0: continue

            gcode_lines.append(f"; --- Path {i + 1} ---")
            
            # 起点
            start_x, start_y = path_machine[0]
            start_x += self.nozzle_offset_x
            start_y += self.nozzle_offset_y

            gcode_lines.append(f"G0 X{start_x:.3f} Y{start_y:.3f} F{self.speed_xy}")
            gcode_lines.append(f"G0 Z{self.work_z:.2f} F{self.speed_z}")

            # 路径点
            for pt in path_machine[1:]:
                pt_x = pt[0] + self.nozzle_offset_x
                pt_y = pt[1] + self.nozzle_offset_y
                gcode_lines.append(f"G0 X{pt_x:.3f} Y{pt_y:.3f} F{self.speed_xy}")

            gcode_lines.append(f"G0 Z{self.safe_z:.2f} F{self.speed_z}")

        gcode_lines.append(f"; --- Return to Start ---")
        gcode_lines.append(f"G0 X{self.origin_x:.3f} Y{self.origin_y:.3f} F{self.speed_xy}")
        gcode_lines.append(f"G0 Z{self.origin_z:.3f} F{self.speed_z}")

        self.text_preview.setText("\n".join(gcode_lines))

    def get_gcode(self):
        return self.text_preview.toPlainText()

from PyQt5.QtWidgets import QPushButton

class DryRunControlDialog(QDialog):
    def __init__(self, parent, machine_client):
        super().__init__(parent)
        self.machine = machine_client
        self.setWindowTitle("空跑控制")
        self.resize(300, 150)
        # 设置为非模态，或者在主窗口中以非阻塞方式显示
        # 但如果是 exec_() 就会阻塞。
        # 我们希望它阻塞主窗口的操作，但不阻塞后台线程。
        # 所以用 exec_() 是可以的，因为 GCodeWorker 是后台线程。
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        self.lbl_status = QLabel("正在执行空跑...")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #10B981;")
        layout.addWidget(self.lbl_status)

        btn_layout = QHBoxLayout()
        
        self.btn_pause = QPushButton("暂停 (Pause)")
        self.btn_pause.clicked.connect(self.on_pause)
        btn_layout.addWidget(self.btn_pause)

        self.btn_resume = QPushButton("继续 (Resume)")
        self.btn_resume.clicked.connect(self.on_resume)
        btn_layout.addWidget(self.btn_resume)

        self.btn_stop = QPushButton("停止 (Stop)")
        self.btn_stop.setStyleSheet("background-color: #EF4444; color: white; font-weight: bold;")
        self.btn_stop.clicked.connect(self.on_stop)
        btn_layout.addWidget(self.btn_stop)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def on_pause(self):
        self.lbl_status.setText("正在发送暂停指令...")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #F59E0B;")
        
        def _target():
            # 使用短超时 (5秒)，防止界面卡死
            self.machine._post_gcode("PAUSE", timeout=5)
            # 注意：子线程不能直接更新 UI，这里偷懒了，因为 PyQt5 有时允许简单的 setText
            # 严谨做法是用信号，但为了简单起见，这里假设没问题，或者用 QTimer 轮询
            # 更好的做法是定义一个 Worker，但这里代码量少，直接用 lambda
        
        import threading
        t = threading.Thread(target=_target)
        t.start()
        
        # 乐观更新 UI
        self.lbl_status.setText("已发送暂停指令 (PAUSE)")

    def on_resume(self):
        self.lbl_status.setText("正在发送继续指令...")
        
        def _target():
            self.machine._post_gcode("RESUME", timeout=5)
            
        import threading
        t = threading.Thread(target=_target)
        t.start()
        
        self.lbl_status.setText("已发送继续指令 (RESUME)")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #10B981;")

    def on_stop(self):
        self.lbl_status.setText("正在停止...")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #EF4444;")
        
        def _target():
            self.machine._post_gcode("CANCEL_PRINT", timeout=5)
            
        import threading
        t = threading.Thread(target=_target)
        t.start()

