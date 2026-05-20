from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QGroupBox, QDoubleSpinBox, QSpinBox, QTextEdit,
    QDialogButtonBox, QCheckBox, QPushButton, QProgressBar,
    QScrollArea, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox, QWidget
)
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage, QColor, QBrush, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRectF
import numpy as np
import cv2
import os
import json

class DryRunDialog(QDialog):
    def __init__(self, parent, paths_px, cam2mach_matrix, machine_origin, nozzle_offset, initial_params, img_center, is_real_run=False, save_callback=None, laser_data=None, z_base=0.0, img_dry=None):
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
        self.img_dry = img_dry
        
        # [NEW] Laser Data
        self.laser_data = laser_data if laser_data else []
        self.z_base = z_base
        
        # Parse products structure and convert to PathSegments
        from core.path_segment import PathSegment
        self.products = []
        if len(self.paths_px) > 0:
            if isinstance(self.paths_px[0], list):
                # New format: List[List[path]]
                for prod_seg_list in self.paths_px:
                    new_seg_list = []
                    for seg in prod_seg_list:
                        if isinstance(seg, PathSegment):
                            new_seg_list.append(seg)
                        else:
                            pts = seg.tolist() if isinstance(seg, np.ndarray) else seg
                            t = 'line' if len(pts) > 1 else 'dot'
                            new_seg_list.append(PathSegment(t, 0, pts))
                    self.products.append(new_seg_list)
            else:
                # Old format: List[path] -> Treat each path as a separate product for safety
                for seg in self.paths_px:
                    if isinstance(seg, PathSegment):
                        self.products.append([seg])
                    else:
                        pts = seg.tolist() if isinstance(seg, np.ndarray) else seg
                        t = 'line' if len(pts) > 1 else 'dot'
                        self.products.append([PathSegment(t, 0, pts)])

        # 初始参数
        self.safe_z_cavity = initial_params.get('safe_z_cavity', 22.0)
        self.safe_z_product = initial_params.get('safe_z_product', 50.0)
        self.work_z = initial_params.get('work_z', 5.0)
        self.speed_xy = initial_params.get('speed_xy', 2000)
        self.speed_z = initial_params.get('speed_z', 500)
        self.dwell_ms = initial_params.get('dwell_ms', 100)      # Dot Dwell
        self.line_dwell_ms = initial_params.get('line_dwell_ms', 0) # Line Start Dwell
        self.color_delay = initial_params.get('color_delay', 500)   # Color Supply Delay
        self.line_speed_xy = initial_params.get('line_speed_xy', 800)  # Line Dispensing Speed

        # Load customized heights from product config
        self.product_name = None
        self.product_dir = None
        self.segment_heights_config = {}
        
        if parent:
            if hasattr(parent, 'halcon_product_manager') and parent.halcon_product_manager:
                self.product_name = parent.halcon_product_manager.get_current_product_name()
                if self.product_name:
                    self.product_dir = parent.halcon_product_manager.products.get_product_path(self.product_name)
            elif hasattr(parent, 'mw') and parent.mw and hasattr(parent.mw, 'halcon_product_manager') and parent.mw.halcon_product_manager:
                self.product_name = parent.mw.halcon_product_manager.get_current_product_name()
                if self.product_name:
                    self.product_dir = parent.mw.halcon_product_manager.products.get_product_path(self.product_name)

        if self.product_dir:
            config_path = os.path.join(self.product_dir, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        cfg = json.load(f)
                    self.segment_heights_config = cfg.get("segment_heights", {})
                except Exception as e:
                    print(f"[DryRunDialog] Error loading heights config: {e}")

        # Apply heights from config, or default to global parameters
        for prod_idx, prod_segs in enumerate(self.products):
            for seg_idx, seg in enumerate(prod_segs):
                key = f"prod_{prod_idx}_seg_{seg_idx}"
                custom = self.segment_heights_config.get(key, {})
                if custom:
                    seg.custom_safe_z_product = custom.get("safe_z_product", self.safe_z_product)
                    seg.custom_safe_z_cavity = custom.get("safe_z_cavity", self.safe_z_cavity)
                    seg.custom_work_z = custom.get("work_z", self.work_z)
                    seg.has_custom_height = True
                else:
                    seg.custom_safe_z_product = self.safe_z_product
                    seg.custom_safe_z_cavity = self.safe_z_cavity
                    seg.custom_work_z = self.work_z
                    seg.has_custom_height = False

        self.init_ui()
        self.generate_gcode()

    def init_ui(self):
        self.machine_paths = [] # [NEW] Store calculated machine coordinates
        layout = QVBoxLayout()
        
        # 1. 参数设置区
        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout()

        # Safe Z Cavity — Row 0
        param_layout.addWidget(QLabel("腔内安全高度 (Cavity Z):"), 0, 0)
        self.spn_safe_z_cavity = QDoubleSpinBox()
        self.spn_safe_z_cavity.setRange(0, 300)
        self.spn_safe_z_cavity.setValue(self.safe_z_cavity)
        self.spn_safe_z_cavity.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_safe_z_cavity, 0, 1)

        # Work Z — Row 0 right column
        param_layout.addWidget(QLabel("点胶高度 (Work Z):"), 0, 2)
        self.spn_work_z = QDoubleSpinBox()
        self.spn_work_z.setRange(0, 300)
        self.spn_work_z.setValue(self.work_z)
        self.spn_work_z.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_work_z, 0, 3)

        # Safe Z Product — Row 1 (was being hidden by laser checkbox before)
        param_layout.addWidget(QLabel("产品安全高度 (Product Z):"), 1, 0)
        self.spn_safe_z_product = QDoubleSpinBox()
        self.spn_safe_z_product.setRange(0, 300)
        self.spn_safe_z_product.setValue(self.safe_z_product)
        self.spn_safe_z_product.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_safe_z_product, 1, 1)

        # 详细调整按钮 - Row 1, Column 2, 3
        self.btn_detail_adjust = QPushButton("详细调整...")
        self.btn_detail_adjust.setStyleSheet("background-color: #3B82F6; color: white; font-weight: bold;")
        self.btn_detail_adjust.clicked.connect(self.on_detail_adjust_clicked)
        param_layout.addWidget(self.btn_detail_adjust, 1, 2, 1, 2)

        # Laser Z checkbox — Row 2, full width (no longer conflicts with Product Z)
        self.chk_use_laser = QCheckBox(f"使用激光测量高度 (基准Z={self.z_base:.1f} + 测量深度 + 工作Z)")
        if not self.laser_data:
            self.chk_use_laser.setText("使用激光测量高度 (无数据)")
            self.chk_use_laser.setEnabled(False)
        else:
            self.chk_use_laser.setChecked(False)  # 默认不启用激光测高，需要时手动勾选
        self.chk_use_laser.stateChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.chk_use_laser, 2, 0, 1, 4)

        # Speed XY — Row 3
        param_layout.addWidget(QLabel("XY 速度 (mm/min):"), 3, 0)
        self.spn_speed_xy = QSpinBox()
        self.spn_speed_xy.setRange(100, 20000)
        self.spn_speed_xy.setValue(int(self.speed_xy))
        self.spn_speed_xy.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_speed_xy, 3, 1)

        # Speed Z — Row 3 right column
        param_layout.addWidget(QLabel("Z 速度 (mm/min):"), 3, 2)
        self.spn_speed_z = QSpinBox()
        self.spn_speed_z.setRange(100, 5000)
        self.spn_speed_z.setValue(int(self.speed_z))
        self.spn_speed_z.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_speed_z, 3, 3)

        # Dot Dwell — Row 4
        param_layout.addWidget(QLabel("点胶延时 (ms):"), 4, 0)
        self.spn_dwell = QSpinBox()
        self.spn_dwell.setRange(0, 5000)
        self.spn_dwell.setValue(self.dwell_ms)
        self.spn_dwell.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_dwell, 4, 1)

        # Line Dwell — Row 4 right column
        param_layout.addWidget(QLabel("画线延时 (ms):"), 4, 2)
        self.spn_line_dwell = QSpinBox()
        self.spn_line_dwell.setRange(0, 5000)
        self.spn_line_dwell.setValue(self.line_dwell_ms)
        self.spn_line_dwell.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_line_dwell, 4, 3)

        # Color Delay — Row 5
        param_layout.addWidget(QLabel("供料延时 (ms):"), 5, 0)
        self.spn_color_delay = QSpinBox()
        self.spn_color_delay.setRange(0, 5000)
        self.spn_color_delay.setValue(int(self.color_delay))
        self.spn_color_delay.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_color_delay, 5, 1)

        # Line Speed — Row 5 right column
        param_layout.addWidget(QLabel("画线速度 (mm/min):"), 5, 2)
        self.spn_line_speed = QSpinBox()
        self.spn_line_speed.setRange(100, 5000)
        self.spn_line_speed.setValue(self.line_speed_xy)
        self.spn_line_speed.valueChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.spn_line_speed, 5, 3)


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
        
        # [NEW] Simulation Checkbox
        self.chk_simulate = QCheckBox("模拟出胶 (关闭阀门)")
        self.chk_simulate.setChecked(True) # Default Checked (Safety)
        self.chk_simulate.setStyleSheet("color: #F59E0B; font-weight: bold;")
        self.chk_simulate.stateChanged.connect(self.generate_gcode)
        prod_layout.addWidget(self.chk_simulate)
        
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
        self.color_delay = self.spn_color_delay.value()
        self.line_speed_xy = self.spn_line_speed.value()
        
        # Keep non-customized segments in sync with global parameter spinboxes
        for prod_idx, prod_segs in enumerate(self.products):
            for seg_idx, seg in enumerate(prod_segs):
                if not getattr(seg, 'has_custom_height', False):
                    seg.custom_safe_z_product = self.safe_z_product
                    seg.custom_safe_z_cavity = self.safe_z_cavity
                    seg.custom_work_z = self.work_z
        self.generate_gcode()

    def on_detail_adjust_clicked(self):
        global_params = {
            'safe_z_cavity': self.spn_safe_z_cavity.value(),
            'safe_z_product': self.spn_safe_z_product.value(),
            'work_z': self.spn_work_z.value()
        }
        dlg = DetailedHeightAdjustDialog(self, self.products, self.img_dry, global_params, self.product_dir)
        if dlg.exec_() == QDialog.Accepted:
            self.generate_gcode()

    def find_nearest_depth(self, x, y):
        """
        查找最近的激光测高点深度
        :param x: 机器坐标 X
        :param y: 机器坐标 Y
        :return: depth (mm)
        """
        if not self.laser_data:
            return 0.0
            
        points = []
        # Handle 1: laser_data might be a dict with key "points" (loaded from config JSON)
        if isinstance(self.laser_data, dict):
            points = self.laser_data.get('points', [])
        # Handle 2: laser_data might be a list directly
        elif isinstance(self.laser_data, list):
            points = self.laser_data
            
        if not points:
            return 0.0
            
        min_dist = float('inf')
        nearest_depth = 0.0
        found = False

        for pt in points:
            # Support both dict item and possible tuple (legacy)
            if isinstance(pt, dict):
                px = pt.get('x')
                py = pt.get('y')
                d = pt.get('depth', 0.0)
            else:
                continue

            if px is None or py is None:
                continue
                
            dist = ((px - x)**2 + (py - y)**2)**0.5
            
            # Simple nearest neighbor
            if dist < min_dist:
                min_dist = dist
                nearest_depth = d
                found = True
        
        # Optional: Add a max distance threshold (e.g. 5mm) if needed
        # if min_dist > 5.0: return 0.0
        
        return nearest_depth if found else 0.0

    def generate_gcode(self):
        self.machine_paths = [] # [NEW] Reset paths
        gcode_lines = []
        gcode_lines.append("G90")
        # 初始：确保 Z 在安全高度（取当前位置与产品安全高度的最大值，保证 XY 移动时不撞车）
        max_safe_z_product = self.safe_z_product
        for prod_idx, prod_segments in enumerate(self.products):
            if not self.chk_products[prod_idx].isChecked():
                continue
            for seg in prod_segments:
                seg_z = seg.custom_safe_z_product if getattr(seg, 'has_custom_height', False) else self.safe_z_product
                if seg_z > max_safe_z_product:
                    max_safe_z_product = seg_z
        initial_safe_z = max(getattr(self, 'origin_z', max_safe_z_product), max_safe_z_product)
        gcode_lines.append(f"G0 Z{initial_safe_z:.2f} F{self.speed_z}  ; 确保初始安全高度 ≥ 产品安全高度")

        # [NEW] Color-First Logic
        all_colors = set()
        for prod_idx, prod_segments in enumerate(self.products):
            if not self.chk_products[prod_idx].isChecked():
                continue
            for seg in prod_segments:
                seg_color = getattr(seg, 'color', 0) if hasattr(seg, 'color') else 0
                all_colors.add(seg_color)
        
        sorted_colors = sorted(list(all_colors)) if all_colors else [0]
        active_color_valve = None
        
        for color_idx in sorted_colors:
            gcode_lines.append(f"; ===== Color {color_idx+1} =====")
            
            for prod_idx, prod_segments in enumerate(self.products):
                if not self.chk_products[prod_idx].isChecked():
                    continue

                gcode_lines.append(f"; --- Product {prod_idx + 1} (Color {color_idx+1}) ---")
                
                color_segments = [seg for seg in prod_segments 
                                  if getattr(seg, 'color', 0) == color_idx]
                
                if not color_segments:
                    continue

                for seg_idx, p_px in enumerate(color_segments):

                    # 1. Convert pixels to mm
                    # [FIX] Handle PathSegment objects
                    if hasattr(p_px, 'points'):
                        path_array = np.array(p_px.points)
                    else:
                        path_array = p_px
                    
                    delta_px = path_array - np.array([self.center_x, self.center_y])
                    delta_mm_x = delta_px[:, 0] * self.m[0, 0] + delta_px[:, 1] * self.m[0, 1]
                    delta_mm_y = delta_px[:, 0] * self.m[1, 0] + delta_px[:, 1] * self.m[1, 1]

                    abs_x = self.origin_x + delta_mm_x
                    abs_y = self.origin_y + delta_mm_y
                    path_machine = np.column_stack((abs_x, abs_y))

                    if len(path_machine) == 0: continue

                    # [NEW] Collect machine coordinates for saving
                    # Determine type based on length or original segment type if available
                    seg_type = getattr(p_px, 'type', 'line' if len(path_machine) > 1 else 'dot')
                    seg_color = getattr(p_px, 'color', color_idx)
                    
                    self.machine_paths.append({
                        'type': seg_type,
                        'color': seg_color,
                        'points': path_machine.tolist() # [[x,y], [x,y]...]
                    })

                    gcode_lines.append(f";   Segment {seg_idx + 1}")

                    # [TAG REMOVED] Logic moved to dispense block

                    # 2. Move to Start Point
                    start_x, start_y = path_machine[0]
                    # Apply offset
                    start_x += self.nozzle_offset_x
                    start_y += self.nozzle_offset_y

                    # ① XY 移动到点位（Z 不变：产品间在 product_safe，点间在 cavity_safe）
                    gcode_lines.append(f"G0 X{start_x:.3f} Y{start_y:.3f} F{self.speed_xy}")

                    # Determine target heights for this segment
                    has_custom = getattr(p_px, 'has_custom_height', False)
                    seg_safe_z_product = p_px.custom_safe_z_product if has_custom else self.safe_z_product
                    seg_safe_z_cavity = p_px.custom_safe_z_cavity if has_custom else self.safe_z_cavity
                    seg_work_z = p_px.custom_work_z if has_custom else self.work_z

                    # ② 计算目标 Z（支持激光补偿）
                    calc_z = seg_work_z
                    use_laser = getattr(self, 'chk_use_laser', None) and self.chk_use_laser.isChecked()
                    if use_laser and seg_type == 'dot':
                        depth = self.find_nearest_depth(abs_x[0], abs_y[0])
                        if depth != 0.0:
                            calc_z = self.z_base + depth + seg_work_z
                            gcode_lines.append(f"; [3D] Depth={depth:.3f} -> Z={calc_z:.2f}")

                    # ③ 三级下降（产品第一段：完整三级；后续段：仅 work_z）
                    if seg_idx == 0:
                        gcode_lines.append(f"G0 Z{seg_safe_z_product:.2f} F{self.speed_z}  ; ③-1 降至产品安全高度")
                        gcode_lines.append(f"G0 Z{seg_safe_z_cavity:.2f} F{self.speed_z}   ; ③-2 降至腔内安全高度")
                    gcode_lines.append(f"G0 Z{calc_z:.2f} F{self.speed_z}                    ; ③-3 降至工作高度")


                    # Check if it is a Dot (1 point) or Line (>1 points)
                    # Prepare Valve Commands
                    # Default to A1 if no color info
                    dwell_ms = getattr(p_px, 'dwell_ms', 0)
                    # 手动换色模式：固定使用 A1_VALVE，不按颜色 index 切换阀门
                    # 颜色顺序由用户手动换料保证；软件始终控制 A1_VALVE + VALVE_A_CTRL
                    target_valve = "A1_VALVE"

                    # --- Valve State Logic ---
                    # Check if we need to switch color valve
                    if active_color_valve != target_valve:
                        # Close previous if open
                        if active_color_valve is not None:
                            is_sim = self.chk_simulate.isChecked()
                            if not is_sim:
                                gcode_lines.append(f"SET_PIN PIN={active_color_valve} VALUE=0")
                            else:
                                gcode_lines.append(f"; [Tag] {active_color_valve} (OFF) - Switch Color")
                        
                        # Open new
                        is_sim = self.chk_simulate.isChecked()
                        if not is_sim:
                            gcode_lines.append(f"SET_PIN PIN={target_valve} VALUE=1")
                            if self.color_delay > 0:
                                gcode_lines.append(f"G4 P{self.color_delay}")
                        else:
                            gcode_lines.append(f"; [Tag] {target_valve} (ON)")
                            if self.color_delay > 0:
                                gcode_lines.append(f"; [Tag] G4 P{self.color_delay}")
                        
                        active_color_valve = target_valve
                    
                    # Master Valve A Control (Always toggle per segment)
                    master_pin = "VALVE_A_CTRL"
                    
                    # Helper to append commands
                    def append_master_cmd(is_on):
                        val = 1 if is_on else 0
                        action = "ON" if is_on else "OFF"
                        
                        # [MOD] Simulation Check
                        is_sim = self.chk_simulate.isChecked()
                        
                        # Allow real output if simulation is unchecked
                        if not is_sim:
                            gcode_lines.append(f"SET_PIN PIN={master_pin} VALUE={val}")
                        else:
                            gcode_lines.append(f"; [Tag] {master_pin} ({action})")

                    if len(path_machine) == 1:
                        # --- Dot Dispensing ---
                        # [MOD] Prioritize individual segment dwell time
                        actual_dwell = dwell_ms if dwell_ms > 0 else self.dwell_ms
                        
                        if actual_dwell > 0:
                            append_master_cmd(True) # Open A
                            gcode_lines.append(f"G4 P{actual_dwell}")
                            append_master_cmd(False) # Close A
                    else:
                        # --- Line Dispensing ---
                        # 1. 先开阀
                        append_master_cmd(True) # Open A
                        
                        # 2. 延时等待供料建压
                        line_dwell = dwell_ms if dwell_ms > 0 else self.line_dwell_ms
                        if line_dwell > 0:
                            gcode_lines.append(f"G4 P{line_dwell}")

                        # 3. 沿路径移动出胶
                        for pt in path_machine[1:]:
                            pt_x = pt[0] + self.nozzle_offset_x
                            pt_y = pt[1] + self.nozzle_offset_y
                            # [FIX] Use G1 for controlled speed, and use line_speed_xy
                            gcode_lines.append(f"G1 X{pt_x:.3f} Y{pt_y:.3f} F{self.line_speed_xy}")
                        
                        # 4. 终点延时（与起点相同）
                        if line_dwell > 0:
                            gcode_lines.append(f"G4 P{line_dwell}")
                        
                        # 5. 关阀
                        append_master_cmd(False) # Close A

                    # 5. Lift after segment to Cavity Safe Z
                    gcode_lines.append(f"G0 Z{seg_safe_z_cavity:.2f} F{self.speed_z}")
                
                # ✅ Product完成：升到产品安全Z
                if color_segments:
                    last_seg = color_segments[-1]
                    last_has_custom = getattr(last_seg, 'has_custom_height', False)
                    last_safe_z_product = last_seg.custom_safe_z_product if last_has_custom else self.safe_z_product
                    gcode_lines.append(f"G0 Z{last_safe_z_product:.2f} F{self.speed_z}")
            
            # End of Color: Close Color Valve after all products of this color
            if active_color_valve is not None:
                gcode_lines.append(f"; [Tag] {active_color_valve} (OFF) - End Color")
                is_sim = self.chk_simulate.isChecked()
                if not is_sim:
                    gcode_lines.append(f"SET_PIN PIN={active_color_valve} VALUE=0")
                active_color_valve = None

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
    def __init__(self, parent, worker, full_gcode_text=""):
        super().__init__(parent)
        self.worker = worker
        self.full_gcode_text = full_gcode_text
        self.setWindowTitle("空跑控制 (逐行模式)")
        self.resize(600, 500) # [MOD] Increase size for text area
        self.init_ui()
        
        # 连接 worker 信号
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.update_status)
        self.worker.current_index_signal.connect(self.highlight_current_line)

        # Init display
        self.update_gcode_display(self.full_gcode_text)

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

        # [NEW] G-code Realtime View
        layout.addWidget(QLabel("实时代码执行:"))
        self.gcode_box = QTextEdit()
        self.gcode_box.setReadOnly(True)
        self.gcode_box.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc; font-family: Consolas; font-size: 10pt;")
        layout.addWidget(self.gcode_box)

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
        self.lbl_status.setText(msg) 

    def update_gcode_display(self, full_gcode):
        """显示完整的 G-code"""
        self.gcode_box.setText(full_gcode)

    def highlight_current_line(self, line_index):
        """高亮当前执行的行 (根据行号)"""
        # 移动光标到指定行
        cursor = self.gcode_box.textCursor()
        cursor.movePosition(cursor.Start)
        
        # 向下移动 line_index 行
        cursor.movePosition(cursor.Down, cursor.MoveAnchor, line_index)
        
        # 选中该行
        cursor.select(cursor.LineUnderCursor)
        self.gcode_box.setTextCursor(cursor)
        self.gcode_box.ensureCursorVisible()


# =========================================================================
# HeightAdjustCanvas & DetailedHeightAdjustDialog for Fine-tuning heights
# =========================================================================

class HeightAdjustCanvas(QWidget):
    segment_clicked = pyqtSignal(int, int) # (prod_idx, seg_idx)

    def __init__(self, template_img, products, global_params, parent=None):
        super().__init__(parent)
        self.template = template_img # BGR numpy array
        self.products = products # list of lists of PathSegment
        self.global_params = global_params
        self.active_height_type = "work_z"
        self.selected_prod_idx = None
        self.selected_seg_idx = None
        
        # Zoom/Pan parameters
        self.scale = 1.0
        self.calculate_scale()
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #1e1e1e;")
        
        self.is_panning = False
        self.last_pan_pos = QPoint()

    def calculate_scale(self):
        if self.template is None:
            self.setFixedSize(800, 600)
            return
        h, w = self.template.shape[:2]
        self.scale = 1.0
        self.setFixedSize(int(w * self.scale), int(h * self.scale))

    def set_active_height_type(self, h_type):
        self.active_height_type = h_type
        self.update()

    def set_selected_segment(self, prod_idx, seg_idx):
        self.selected_prod_idx = prod_idx
        self.selected_seg_idx = seg_idx
        self.update()

    def mousePressEvent(self, event):
        x = event.x() / self.scale
        y = event.y() / self.scale
        
        if event.button() == Qt.MiddleButton:
            self.setCursor(Qt.ClosedHandCursor)
            self.last_pan_pos = event.globalPos()
            self.is_panning = True
            return
            
        if event.button() == Qt.LeftButton:
            prod_idx, seg_idx = self.find_segment_at(x, y)
            if prod_idx is not None and seg_idx is not None:
                self.segment_clicked.emit(prod_idx, seg_idx)
                self.set_selected_segment(prod_idx, seg_idx)
            else:
                self.segment_clicked.emit(-1, -1)
                self.set_selected_segment(None, None)

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.globalPos() - self.last_pan_pos
            self.last_pan_pos = event.globalPos()
            
            parent = self.parent()
            while parent:
                if isinstance(parent, QScrollArea):
                    scrollbar_h = parent.horizontalScrollBar()
                    scrollbar_v = parent.verticalScrollBar()
                    scrollbar_h.setValue(scrollbar_h.value() - delta.x())
                    scrollbar_v.setValue(scrollbar_v.value() - delta.y())
                    break
                parent = parent.parent()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setCursor(Qt.ArrowCursor)
            self.is_panning = False

    def wheelEvent(self, event):
        if self.template is None:
            return
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        new_scale = self.scale * zoom_factor
        
        new_scale = max(0.1, min(10.0, new_scale))
        if new_scale != self.scale:
            self.scale = new_scale
            h, w = self.template.shape[:2]
            self.setFixedSize(int(w * self.scale), int(h * self.scale))
            self.update()

    def find_segment_at(self, x, y):
        threshold = 15 / self.scale
        for prod_idx, prod_segs in enumerate(self.products):
            for seg_idx, seg in enumerate(prod_segs):
                if seg.type == 'dot':
                    px, py = seg.points[0]
                    dist = ((px - x)**2 + (py - y)**2) ** 0.5
                    if dist < threshold:
                        return prod_idx, seg_idx
                else:
                    for j in range(len(seg.points) - 1):
                        x1, y1 = seg.points[j]
                        x2, y2 = seg.points[j+1]
                        dist = self.point_to_line_distance(x, y, x1, y1, x2, y2)
                        if dist < threshold:
                            return prod_idx, seg_idx
        return None, None

    def point_to_line_distance(self, px, py, x1, y1, x2, y2):
        line_len = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
        if line_len == 0:
            return ((px-x1)**2 + (py-y1)**2) ** 0.5
        t = max(0, min(1, ((px-x1)*(x2-x1) + (py-y1)*(y2-y1)) / (line_len**2)))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        return ((px-proj_x)**2 + (py-proj_y)**2) ** 0.5

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. Background image
        if self.template is not None:
            h, w = self.template.shape[:2]
            scaled_w = int(w * self.scale)
            scaled_h = int(h * self.scale)
            
            if len(self.template.shape) == 2:
                img_color = cv2.cvtColor(self.template, cv2.COLOR_GRAY2RGB)
            else:
                img_color = cv2.cvtColor(self.template, cv2.COLOR_BGR2RGB)
                
            qimg = QImage(img_color.data, w, h, w*3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, scaled_pixmap)
        else:
            painter.fillRect(self.rect(), QColor(30, 30, 30))

        # 2. Draw all products & segments
        for prod_idx, prod_segs in enumerate(self.products):
            for seg_idx, seg in enumerate(prod_segs):
                is_selected = (prod_idx == self.selected_prod_idx and seg_idx == self.selected_seg_idx)
                has_custom = getattr(seg, 'has_custom_height', False)
                
                r, g, b = seg.get_display_color_rgb()
                base_color = QColor(r, g, b)
                if has_custom:
                    base_color = QColor(34, 211, 238) # Cyan
                
                # Draw selection highlight
                if is_selected:
                    glow_pen = QPen(QColor(234, 179, 8), 6) # Yellow glow
                    painter.setPen(glow_pen)
                    if seg.type == 'dot':
                        x, y = seg.points[0]
                        painter.drawEllipse(int(x*self.scale)-8, int(y*self.scale)-8, 16, 16)
                    else:
                        for i in range(len(seg.points)-1):
                            p1 = QPoint(int(seg.points[i][0]*self.scale), int(seg.points[i][1]*self.scale))
                            p2 = QPoint(int(seg.points[i+1][0]*self.scale), int(seg.points[i+1][1]*self.scale))
                            painter.drawLine(p1, p2)

                # Draw segment itself
                pen = QPen(base_color, 2 if not is_selected else 3)
                painter.setPen(pen)
                painter.setBrush(QBrush(base_color))
                
                if seg.type == 'dot':
                    x, y = seg.points[0]
                    painter.drawEllipse(int(x*self.scale)-4, int(y*self.scale)-4, 8, 8)
                else:
                    for i in range(len(seg.points)-1):
                        p1 = QPoint(int(seg.points[i][0]*self.scale), int(seg.points[i][1]*self.scale))
                        p2 = QPoint(int(seg.points[i+1][0]*self.scale), int(seg.points[i+1][1]*self.scale))
                        painter.drawLine(p1, p2)
                    for x, y in seg.points:
                        painter.drawEllipse(int(x*self.scale)-3, int(y*self.scale)-3, 6, 6)

                # 3. Draw Height Label
                if seg.type == 'dot':
                    lx, ly = seg.points[0]
                else:
                    mid = len(seg.points) // 2
                    lx, ly = seg.points[mid]
                
                if self.active_height_type == "safe_z_product":
                    val = seg.custom_safe_z_product if has_custom else self.global_params['safe_z_product']
                elif self.active_height_type == "safe_z_cavity":
                    val = seg.custom_safe_z_cavity if has_custom else self.global_params['safe_z_cavity']
                else:
                    val = seg.custom_work_z if has_custom else self.global_params['work_z']
                
                label_text = f"{val:.2f}"
                if has_custom:
                    label_text += "*"

                lbl_px = int(lx * self.scale)
                lbl_py = int(ly * self.scale) - 10
                
                painter.setFont(QFont('Arial', 9, QFont.Bold))
                fm = painter.fontMetrics()
                rect = fm.boundingRect(label_text)
                
                box_x = lbl_px - rect.width() // 2 - 4
                box_y = lbl_py - rect.height() - 2
                box_w = rect.width() + 8
                box_h = rect.height() + 4
                
                painter.setPen(Qt.NoPen)
                if is_selected:
                    painter.setBrush(QBrush(QColor(234, 179, 8, 220)))
                elif has_custom:
                    painter.setBrush(QBrush(QColor(8, 145, 178, 220)))
                else:
                    painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
                
                painter.drawRoundedRect(QRectF(box_x, box_y, box_w, box_h), 3, 3)
                
                if is_selected:
                    painter.setPen(QColor(0, 0, 0))
                else:
                    painter.setPen(QColor(255, 255, 255))
                painter.drawText(box_x + 4, box_y + fm.ascent() + 2, label_text)

                # Segment index tag
                tag_text = f"P{prod_idx+1}-{seg_idx+1}"
                tag_rect = fm.boundingRect(tag_text)
                tag_x = lbl_px - tag_rect.width() // 2 - 4
                tag_y = box_y - tag_rect.height() - 2
                tag_w = tag_rect.width() + 8
                tag_h = tag_rect.height() + 4
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(75, 85, 99, 180)))
                painter.drawRoundedRect(QRectF(tag_x, tag_y, tag_w, tag_h), 2, 2)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(tag_x + 4, tag_y + fm.ascent() + 2, tag_text)


class DetailedHeightAdjustDialog(QDialog):
    def __init__(self, parent, products, img_dry, global_params, product_dir):
        super().__init__(parent)
        self.setWindowTitle("产品高度细微调整 (模具误差补偿)")
        self.resize(1200, 800)
        self.products = products
        self.img_dry = img_dry
        self.global_params = global_params
        self.product_dir = product_dir
        self.active_height_type = "work_z"
        
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: #121212; border: 1px solid #374151;")
        
        self.canvas = HeightAdjustCanvas(self.img_dry, self.products, self.global_params, self)
        self.canvas.segment_clicked.connect(self.on_canvas_segment_clicked)
        scroll_area.setWidget(self.canvas)
        
        sidebar = QWidget()
        sidebar.setMaximumWidth(400)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        
        type_group = QGroupBox("高度类型切换")
        type_layout = QHBoxLayout()
        
        self.btn_type_product = QPushButton("产品安全 (Product Z)")
        self.btn_type_product.setCheckable(True)
        self.btn_type_product.clicked.connect(lambda: self.change_active_height_type("safe_z_product"))
        
        self.btn_type_cavity = QPushButton("腔内安全 (Cavity Z)")
        self.btn_type_cavity.setCheckable(True)
        self.btn_type_cavity.clicked.connect(lambda: self.change_active_height_type("safe_z_cavity"))
        
        self.btn_type_work = QPushButton("点胶高度 (Work Z)")
        self.btn_type_work.setCheckable(True)
        self.btn_type_work.setChecked(True)
        self.btn_type_work.clicked.connect(lambda: self.change_active_height_type("work_z"))
        
        self.height_buttons = [self.btn_type_product, self.btn_type_cavity, self.btn_type_work]
        for btn in self.height_buttons:
            type_layout.addWidget(btn)
        
        type_group.setLayout(type_layout)
        sidebar_layout.addWidget(type_group)
        
        self.lbl_global_default = QLabel()
        self.lbl_global_default.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        self.update_global_default_label()
        sidebar_layout.addWidget(self.lbl_global_default)
        
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["产品", "段", "类型", "颜色", "高度值 (mm)", "自定义"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        self.table.itemChanged.connect(self.on_table_item_changed)
        sidebar_layout.addWidget(self.table)
        
        self.populate_table()

        adjust_group = QGroupBox("快捷调整")
        adjust_layout = QVBoxLayout()
        
        self.btn_apply_default = QPushButton("重置为全局默认高度")
        self.btn_apply_default.clicked.connect(self.on_reset_to_default)
        adjust_layout.addWidget(self.btn_apply_default)
        
        self.btn_apply_to_product = QPushButton("应用当前高度到本穴所有段")
        self.btn_apply_to_product.clicked.connect(self.on_apply_to_product)
        adjust_layout.addWidget(self.btn_apply_to_product)
        
        adjust_group.setLayout(adjust_layout)
        sidebar_layout.addWidget(adjust_group)
        
        sidebar_layout.addStretch()
        
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("保存到此配方 (Save)")
        self.btn_save.setStyleSheet("background-color: #10B981; color: white; font-weight: bold; min-height: 35px;")
        self.btn_save.clicked.connect(self.save_to_recipe)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setStyleSheet("background-color: #4B5563; color: white; min-height: 35px;")
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_cancel)
        sidebar_layout.addLayout(btn_layout)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll_area)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)

    def change_active_height_type(self, h_type):
        self.active_height_type = h_type
        for btn in self.height_buttons:
            btn.setChecked(btn.text().lower().startswith(h_type[:4]) or 
                            (h_type == "safe_z_product" and "产品" in btn.text()) or
                            (h_type == "safe_z_cavity" and "腔内" in btn.text()) or
                            (h_type == "work_z" and "点胶" in btn.text()))
        
        self.update_global_default_label()
        self.canvas.set_active_height_type(h_type)
        self.populate_table()

    def update_global_default_label(self):
        val = self.global_params.get(self.active_height_type, 0.0)
        self.lbl_global_default.setText(f"当前高度类型的全局默认值: {val:.2f} mm\n(勾选【自定义】以修改，取消勾选将恢复默认值)")

    def populate_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        
        row_idx = 0
        for prod_idx, prod_segs in enumerate(self.products):
            for seg_idx, seg in enumerate(prod_segs):
                self.table.insertRow(row_idx)
                
                item_prod = QTableWidgetItem(f"产品 {prod_idx + 1}")
                item_prod.setFlags(item_prod.flags() & ~Qt.ItemIsEditable)
                item_prod.setData(Qt.UserRole, prod_idx)
                item_prod.setData(Qt.UserRole + 1, seg_idx)
                self.table.setItem(row_idx, 0, item_prod)
                
                item_seg = QTableWidgetItem(f"{seg_idx + 1}")
                item_seg.setFlags(item_seg.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row_idx, 1, item_seg)
                
                type_str = "点" if seg.type == 'dot' else "线"
                item_type = QTableWidgetItem(type_str)
                item_type.setFlags(item_type.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row_idx, 2, item_type)
                
                color_str = f"颜色 {seg.color + 1}"
                item_color = QTableWidgetItem(color_str)
                item_color.setFlags(item_color.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row_idx, 3, item_color)
                
                has_custom = getattr(seg, 'has_custom_height', False)
                if self.active_height_type == "safe_z_product":
                    val = seg.custom_safe_z_product if has_custom else self.global_params['safe_z_product']
                elif self.active_height_type == "safe_z_cavity":
                    val = seg.custom_safe_z_cavity if has_custom else self.global_params['safe_z_cavity']
                else:
                    val = seg.custom_work_z if has_custom else self.global_params['work_z']
                
                item_val = QTableWidgetItem(f"{val:.2f}")
                if not has_custom:
                    item_val.setFlags(item_val.flags() & ~Qt.ItemIsEditable)
                    item_val.setForeground(QColor(156, 163, 175))
                self.table.setItem(row_idx, 4, item_val)
                
                item_custom = QTableWidgetItem()
                item_custom.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                item_custom.setCheckState(Qt.Checked if has_custom else Qt.Unchecked)
                self.table.setItem(row_idx, 5, item_custom)
                
                if prod_idx == self.canvas.selected_prod_idx and seg_idx == self.canvas.selected_seg_idx:
                    self.table.selectRow(row_idx)
                
                row_idx += 1
                
        self.table.blockSignals(False)

    def on_canvas_segment_clicked(self, prod_idx, seg_idx):
        if prod_idx == -1 or seg_idx == -1:
            self.table.clearSelection()
            return
            
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item.data(Qt.UserRole) == prod_idx and item.data(Qt.UserRole + 1) == seg_idx:
                self.table.blockSignals(True)
                self.table.selectRow(r)
                self.table.blockSignals(False)
                break

    def on_table_selection_changed(self):
        selected_ranges = self.table.selectedRanges()
        if not selected_ranges:
            self.canvas.set_selected_segment(None, None)
            return
        row = selected_ranges[0].topRow()
        item = self.table.item(row, 0)
        prod_idx = item.data(Qt.UserRole)
        seg_idx = item.data(Qt.UserRole + 1)
        self.canvas.set_selected_segment(prod_idx, seg_idx)

    def on_table_item_changed(self, item):
        row = item.row()
        col = item.column()
        
        item_prod = self.table.item(row, 0)
        if not item_prod:
            return
        prod_idx = item_prod.data(Qt.UserRole)
        seg_idx = item_prod.data(Qt.UserRole + 1)
        seg = self.products[prod_idx][seg_idx]
        
        if col == 5:
            is_checked = (item.checkState() == Qt.Checked)
            seg.has_custom_height = is_checked
            
            if is_checked:
                if not hasattr(seg, 'custom_safe_z_product'):
                    seg.custom_safe_z_product = self.global_params['safe_z_product']
                if not hasattr(seg, 'custom_safe_z_cavity'):
                    seg.custom_safe_z_cavity = self.global_params['safe_z_cavity']
                if not hasattr(seg, 'custom_work_z'):
                    seg.custom_work_z = self.global_params['work_z']
            
            self.populate_table()
            self.canvas.update()
            
        elif col == 4:
            try:
                val = float(item.text())
                seg.has_custom_height = True
                
                self.table.blockSignals(True)
                self.table.item(row, 5).setCheckState(Qt.Checked)
                self.table.blockSignals(False)
                
                if self.active_height_type == "safe_z_product":
                    seg.custom_safe_z_product = val
                elif self.active_height_type == "safe_z_cavity":
                    seg.custom_safe_z_cavity = val
                elif self.active_height_type == "work_z":
                    seg.custom_work_z = val
                
                self.canvas.update()
            except ValueError:
                self.populate_table()

    def on_reset_to_default(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "未选中", "请先在列表中或画布上选择一个段！")
            return
            
        item_prod = self.table.item(row, 0)
        prod_idx = item_prod.data(Qt.UserRole)
        seg_idx = item_prod.data(Qt.UserRole + 1)
        seg = self.products[prod_idx][seg_idx]
        
        seg.has_custom_height = False
        
        self.populate_table()
        self.canvas.update()

    def on_apply_to_product(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "未选中", "请先在列表中或画布上选择一个段！")
            return
            
        item_prod = self.table.item(row, 0)
        prod_idx = item_prod.data(Qt.UserRole)
        seg_idx = item_prod.data(Qt.UserRole + 1)
        seg = self.products[prod_idx][seg_idx]
        
        has_custom = getattr(seg, 'has_custom_height', False)
        if self.active_height_type == "safe_z_product":
            val = seg.custom_safe_z_product if has_custom else self.global_params['safe_z_product']
        elif self.active_height_type == "safe_z_cavity":
            val = seg.custom_safe_z_cavity if has_custom else self.global_params['safe_z_cavity']
        else:
            val = seg.custom_work_z if has_custom else self.global_params['work_z']
            
        for target_seg in self.products[prod_idx]:
            target_seg.has_custom_height = True
            
            if not hasattr(target_seg, 'custom_safe_z_product'):
                target_seg.custom_safe_z_product = self.global_params['safe_z_product']
            if not hasattr(target_seg, 'custom_safe_z_cavity'):
                target_seg.custom_safe_z_cavity = self.global_params['safe_z_cavity']
            if not hasattr(target_seg, 'custom_work_z'):
                target_seg.custom_work_z = self.global_params['work_z']
                
            if self.active_height_type == "safe_z_product":
                target_seg.custom_safe_z_product = val
            elif self.active_height_type == "safe_z_cavity":
                target_seg.custom_safe_z_cavity = val
            else:
                target_seg.custom_work_z = val
                
        self.populate_table()
        self.canvas.update()

    def save_to_recipe(self):
        segment_heights = {}
        for prod_idx, prod_segs in enumerate(self.products):
            for seg_idx, seg in enumerate(prod_segs):
                if getattr(seg, 'has_custom_height', False):
                    key = f"prod_{prod_idx}_seg_{seg_idx}"
                    segment_heights[key] = {
                        "safe_z_product": getattr(seg, 'custom_safe_z_product', self.global_params['safe_z_product']),
                        "safe_z_cavity": getattr(seg, 'custom_safe_z_cavity', self.global_params['safe_z_cavity']),
                        "work_z": getattr(seg, 'custom_work_z', self.global_params['work_z'])
                    }
                    
        if self.product_dir:
            config_path = os.path.join(self.product_dir, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        cfg = json.load(f)
                except Exception:
                    cfg = {}
            else:
                cfg = {}
                
            cfg["segment_heights"] = segment_heights
            
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                print(f"[DetailedHeightAdjustDialog] Saved customized segment heights: {segment_heights}")
                QMessageBox.information(self, "保存成功", f"细微高度参数已成功保存到配方配置中！")
                self.accept()
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存参数到配方配置失败:\n{e}")
        else:
            QMessageBox.warning(self, "无法保存", "未找到加载的产品配方目录，无法持久化保存高度！")
            self.accept()

