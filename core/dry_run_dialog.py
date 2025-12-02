from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QGroupBox, QDoubleSpinBox, QSpinBox, QTextEdit,
    QDialogButtonBox, QCheckBox
)
import numpy as np

class DryRunDialog(QDialog):
    def __init__(self, parent, paths_px, cam2mach_matrix, machine_origin, nozzle_offset, initial_params, img_center, is_real_run=False, save_callback=None):
        super().__init__(parent)
        self.save_callback = save_callback
        self.is_real_run = is_real_run
        self.setWindowTitle("点胶参数设置 & G-code 预览" if is_real_run else "空跑参数设置 & G-code 预览")
        self.resize(800, 600)

        self.paths_px = paths_px
        self.m = cam2mach_matrix
        self.origin_x, self.origin_y, self.origin_z = machine_origin
        self.nozzle_offset_x, self.nozzle_offset_y = nozzle_offset
        self.center_x, self.center_y = img_center
        
        # Parse products structure
        self.products = []
        if len(self.paths_px) > 0:
            if isinstance(self.paths_px[0], list):
                # New format: List[List[path]]
                self.products = self.paths_px
            else:
                # Old format: List[path] -> Treat each path as a separate product for safety
                self.products = [[p] for p in self.paths_px]

        # 初始参数
        self.safe_z_cavity = initial_params.get('safe_z_cavity', 22.0)
        self.safe_z_product = initial_params.get('safe_z_product', 50.0)
        self.work_z = initial_params.get('work_z', 5.0)
        self.speed_xy = initial_params.get('speed_xy', 2000)
        self.speed_z = initial_params.get('speed_z', 500)
        self.dwell_ms = initial_params.get('dwell_ms', 100)      # Dot Dwell
        self.line_dwell_ms = initial_params.get('line_dwell_ms', 0) # Line Start Dwell

        self.init_ui()
        self.generate_gcode()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # 1. 参数设置区
        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout()

        # Safe Z Cavity
        param_layout.addWidget(QLabel("腔内安全高度 (Cavity Z):"), 0, 0)
        self.spn_safe_z_cavity = QDoubleSpinBox()
        self.spn_safe_z_cavity.setRange(0, 300)
        self.spn_safe_z_cavity.setValue(self.safe_z_cavity)
        self.spn_safe_z_cavity.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_safe_z_cavity, 0, 1)

        # Safe Z Product
        param_layout.addWidget(QLabel("产品安全高度 (Product Z):"), 1, 0)
        self.spn_safe_z_product = QDoubleSpinBox()
        self.spn_safe_z_product.setRange(0, 300)
        self.spn_safe_z_product.setValue(self.safe_z_product)
        self.spn_safe_z_product.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_safe_z_product, 1, 1)

        # Work Z
        param_layout.addWidget(QLabel("点胶高度 (Work Z):"), 0, 2)
        self.spn_work_z = QDoubleSpinBox()
        self.spn_work_z.setRange(0, 300)
        self.spn_work_z.setValue(self.work_z)
        self.spn_work_z.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_work_z, 0, 3)

        # Speed XY
        param_layout.addWidget(QLabel("XY 速度 (mm/min):"), 2, 0)
        self.spn_speed_xy = QSpinBox()
        self.spn_speed_xy.setRange(100, 20000)
        self.spn_speed_xy.setValue(int(self.speed_xy))
        self.spn_speed_xy.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_speed_xy, 2, 1)

        # Speed Z
        param_layout.addWidget(QLabel("Z 速度 (mm/min):"), 2, 2)
        self.spn_speed_z = QSpinBox()
        self.spn_speed_z.setRange(100, 5000)
        self.spn_speed_z.setValue(int(self.speed_z))
        self.spn_speed_z.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_speed_z, 2, 3)

        # Dot Dwell
        param_layout.addWidget(QLabel("点胶延时 (ms):"), 3, 0)
        self.spn_dwell = QSpinBox()
        self.spn_dwell.setRange(0, 5000)
        self.spn_dwell.setValue(self.dwell_ms)
        self.spn_dwell.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_dwell, 3, 1)

        # Line Dwell
        param_layout.addWidget(QLabel("画线延时 (ms):"), 3, 2)
        self.spn_line_dwell = QSpinBox()
        self.spn_line_dwell.setRange(0, 5000)
        self.spn_line_dwell.setValue(self.line_dwell_ms)
        self.spn_line_dwell.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_line_dwell, 3, 3)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 2. 产品选择区 (New)
        prod_group = QGroupBox(f"产品选择 (共 {len(self.products)} 个)")
        prod_layout = QHBoxLayout()
        self.chk_products = []
        for i in range(len(self.products)):
            chk = QCheckBox(f"产品 {i + 1}")
            chk.setChecked(True) # 默认全选
            chk.stateChanged.connect(self.generate_gcode)
            prod_layout.addWidget(chk)
            self.chk_products.append(chk)
        prod_layout.addStretch()
        prod_group.setLayout(prod_layout)
        layout.addWidget(prod_group)

        # 3. 预览区
        layout.addWidget(QLabel("G-code 预览:"))
        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc; font-family: Consolas;")
        layout.addWidget(self.text_preview)

        # 4. 按钮区
        btn_layout = QHBoxLayout()
        
        self.btn_save = QPushButton("仅保存参数 (Save)")
        self.btn_save.clicked.connect(self.on_save_clicked)
        if self.save_callback is None:
            self.btn_save.setEnabled(False)
            
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.button(QDialogButtonBox.Ok).setText("执行 (Run)")
        btn_box.button(QDialogButtonBox.Cancel).setText("取消")
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(btn_box)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def on_param_changed(self):
        self.safe_z_cavity = self.spn_safe_z_cavity.value()
        self.safe_z_product = self.spn_safe_z_product.value()
        self.work_z = self.spn_work_z.value()
        self.speed_xy = self.spn_speed_xy.value()
        self.speed_z = self.spn_speed_z.value()
        self.dwell_ms = self.spn_dwell.value()
        self.line_dwell_ms = self.spn_line_dwell.value()
        self.generate_gcode()

    def generate_gcode(self):
        gcode_lines = []
        gcode_lines.append("G90")
        # Start with Product Safe Z
        gcode_lines.append(f"G0 Z{self.safe_z_product:.2f} F{self.speed_z}")

        for prod_idx, prod_segments in enumerate(self.products):
            # Check if product is selected
            if not self.chk_products[prod_idx].isChecked():
                continue

            gcode_lines.append(f"; --- Product {prod_idx + 1} ---")
            
            for seg_idx, p_px in enumerate(prod_segments):
                # 1. Convert pixels to mm
                delta_px = p_px - np.array([self.center_x, self.center_y])
                delta_mm_x = delta_px[:, 0] * self.m[0, 0] + delta_px[:, 1] * self.m[0, 1]
                delta_mm_y = delta_px[:, 0] * self.m[1, 0] + delta_px[:, 1] * self.m[1, 1]

                abs_x = self.origin_x + delta_mm_x
                abs_y = self.origin_y + delta_mm_y
                path_machine = np.column_stack((abs_x, abs_y))

                if len(path_machine) == 0: continue

                gcode_lines.append(f";   Segment {seg_idx + 1}")

                # 2. Move to Start Point
                start_x, start_y = path_machine[0]
                start_x += self.nozzle_offset_x
                start_y += self.nozzle_offset_y

                # Move XY first (at current Z, which should be safe)
                gcode_lines.append(f"G0 X{start_x:.3f} Y{start_y:.3f} F{self.speed_xy}")
                
                # 3. Down to Work Z
                gcode_lines.append(f"G0 Z{self.work_z:.2f} F{self.speed_z}")

                # Check if it is a Dot (1 point) or Line (>1 points)
                if len(path_machine) == 1:
                    # --- Dot Dispensing ---
                    # Dwell
                    if self.dwell_ms > 0:
                        if self.is_real_run: gcode_lines.append("VALVE_ON")
                        gcode_lines.append(f"G4 P{self.dwell_ms}")
                        if self.is_real_run: gcode_lines.append("VALVE_OFF")
                else:
                    # --- Line Dispensing ---
                    # Line Start Dwell
                    if self.line_dwell_ms > 0:
                        gcode_lines.append(f"G4 P{self.line_dwell_ms}")

                    # 4. Dispense Path
                    if self.is_real_run: gcode_lines.append("VALVE_ON")
                    for pt in path_machine[1:]:
                        pt_x = pt[0] + self.nozzle_offset_x
                        pt_y = pt[1] + self.nozzle_offset_y
                        gcode_lines.append(f"G0 X{pt_x:.3f} Y{pt_y:.3f} F{self.speed_xy}")
                    if self.is_real_run: gcode_lines.append("VALVE_OFF")

                # 5. Lift after segment
                # If this is the LAST segment of the product, lift to Product Safe Z
                # Otherwise, lift to Cavity Safe Z
                if seg_idx == len(prod_segments) - 1:
                    gcode_lines.append(f"G0 Z{self.safe_z_product:.2f} F{self.speed_z}")
                else:
                    gcode_lines.append(f"G0 Z{self.safe_z_cavity:.2f} F{self.speed_z}")

        gcode_lines.append(f"; --- Return to Start ---")
        gcode_lines.append(f"G0 X{self.origin_x:.3f} Y{self.origin_y:.3f} F{self.speed_xy}")
        gcode_lines.append(f"G0 Z{self.origin_z:.3f} F{self.speed_z}")

        self.text_preview.setText("\n".join(gcode_lines))

    def get_gcode(self):
        return self.text_preview.toPlainText()

    def on_save_clicked(self):
        """点击保存按钮：调用回调保存参数，但不关闭窗口"""
        if self.save_callback:
            try:
                self.save_callback(self)
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self, "保存成功", "参数已保存！")
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "保存失败", f"保存参数时出错: {e}")

from PyQt5.QtWidgets import QPushButton, QProgressBar

class DryRunControlDialog(QDialog):
    def __init__(self, parent, worker):
        super().__init__(parent)
        self.worker = worker
        self.setWindowTitle("空跑控制 (逐行模式)")
        self.resize(400, 180)
        self.init_ui()
        
        # 连接 worker 信号
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.update_status)

    def init_ui(self):
        layout = QVBoxLayout()
        
        self.lbl_status = QLabel("正在执行空跑...")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #10B981;")
        layout.addWidget(self.lbl_status)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        
        self.btn_pause = QPushButton("暂停 (Pause)")
        self.btn_pause.clicked.connect(self.on_pause)
        btn_layout.addWidget(self.btn_pause)

        self.btn_resume = QPushButton("继续 (Resume)")
        self.btn_resume.clicked.connect(self.on_resume)
        self.btn_resume.setEnabled(False) # 初始禁用
        btn_layout.addWidget(self.btn_resume)

        self.btn_stop = QPushButton("停止 (Stop)")
        self.btn_stop.setStyleSheet("background-color: #EF4444; color: white; font-weight: bold;")
        self.btn_stop.clicked.connect(self.on_stop)
        btn_layout.addWidget(self.btn_stop)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def on_pause(self):
        self.worker.pause()
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)
        self.lbl_status.setText("已暂停 (等待当前移动完成)")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #F59E0B;")

    def on_resume(self):
        self.worker.resume()
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.lbl_status.setText("正在执行空跑...")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #10B981;")

    def on_stop(self):
        self.worker.stop()
        self.lbl_status.setText("正在停止...")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #EF4444;")
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
    def update_status(self, msg):
        # 可以在这里显示来自 worker 的日志
        # self.lbl_status.setText(msg) 
        pass

