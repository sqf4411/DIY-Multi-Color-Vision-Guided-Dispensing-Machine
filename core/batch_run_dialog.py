import json
import os
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
    QSpinBox, QPushButton, QGroupBox, QFormLayout, QMessageBox, QTextEdit,
    QProgressBar
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from core.mold_detector import get_calculated_paths, get_paths_and_image

class BatchWorker(QThread):
    """
    批量运行工作线程：
    循环：移动 -> 拍照 -> 识别 -> 点胶
    """
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    finished_signal = pyqtSignal(bool, str)
    progress_signal = pyqtSignal(int, int)  # current, total
    gcode_signal = pyqtSignal(str)          # 新增：发送生成的完整 G-code
    current_line_signal = pyqtSignal(str)   # 新增：发送当前执行的 G-code 行
    current_index_signal = pyqtSignal(int)  # 新增：发送当前执行的 G-code 行号 (用于准确高亮)

    def __init__(self, main_window, grid_config, mode='real'):
        super().__init__()
        self.mw = main_window
        self.cfg = grid_config
        self.mode = mode  # 'real', 'camera_dry', 'nozzle_dry'
        self.running = True
        self.paused = False

    def pause(self):
        self.paused = True
        self.log_signal.emit("⏸ 已暂停...")

    def resume(self):
        self.paused = False
        self.log_signal.emit("▶ 继续运行...")

    def run(self):
        rows = self.cfg['rows']
        cols = self.cfg['cols']
        start_x = self.cfg['start_x']
        start_y = self.cfg['start_y']
        start_z = self.cfg.get('start_z', 0.0) # 新增 Start Z
        space_x = self.cfg['space_x']
        space_y = self.cfg['space_y']
        
        total_molds = rows * cols
        molds_finished = 0
        RESOLUTION = 1000
        global_total_steps = total_molds * RESOLUTION

        mode_names = {
            'real': "真实批量点胶",
            'camera_dry': "相机批量空跑",
            'nozzle_dry': "针头批量空跑"
        }
        title = mode_names.get(self.mode, "未知模式")

        self.log_signal.emit(f"🚀 开始【{title}】：{rows}x{cols}，共 {total_molds} 个")
        
        # 0. 开启十字光标 (方便观察)
        # 注意：主界面使用的是 show_center_crosshair
        self.mw.show_center_crosshair = True
        self.mw.render_current_frame() # 立即刷新

        # 1. 确保 Z 轴抬起 (根据模式选择安全高度)
        if self.mode == 'camera_dry':
            safe_z = float(self.cfg['cam_safe_z_cavity'])
        else:
            safe_z = float(self.cfg['nozzle_safe_z_cavity'])

        self.mw.machine.move_to(z=safe_z, speed=self.mw.feedrate_z)
        self.log_signal.emit(f"🔼 抬升至安全高度 Z={safe_z}")

        for r in range(rows):
            for c in range(cols):
                if not self.running:
                    self.finished_signal.emit(False, "用户停止")
                    return

                # 检查暂停
                while self.paused:
                    if not self.running:
                        self.finished_signal.emit(False, "用户停止")
                        return
                    self.msleep(100)

                # count += 1 # 移除旧计数
                # self.progress_signal.emit(count, total) # 移除旧的进度更新
                
                # 更新当前模具的起始进度
                current_base_progress = molds_finished * RESOLUTION
                self.progress_signal.emit(current_base_progress, global_total_steps)
                
                # --- 1. 计算目标坐标 (相机位置) ---
                target_x = start_x + c * space_x
                target_y = start_y + r * space_y
                
                self.log_signal.emit(f"📍 [{molds_finished + 1}/{total_molds}] 移动到 (R{r}, C{c}): X{target_x:.2f}, Y{target_y:.2f}")
                
                # --- 2. 移动相机到位 ---
                self.mw.machine.move_to(x=target_x, y=target_y, speed=self.mw.feedrate_xy)
                self.mw.machine.wait_until_idle()
                
                # 等待相机稳定
                time.sleep(0.5) 

                # --- 3. 拍照 & 识别 ---
                if self.mw.last_frame is None:
                    self.log_signal.emit("❌ 无法获取图像，跳过")
                    continue

                img = self.mw.last_frame.copy()
                img = cv2.flip(img, -1) # 保持和主界面一致的翻转

                self.log_signal.emit("👀 正在识别模具...")
                # 使用新函数同时获取路径和图像，但只使用路径
                # 图像显示交给主界面的 live_detect
                paths_groups, _ = get_paths_and_image(img)
                
                if not paths_groups:
                    self.log_signal.emit(f"⚠ 未识别到模具 (R{r}, C{c})，跳过")
                    molds_finished += 1
                    continue
                
                # Do NOT flatten groups. Keep structure: List[List[Path]]
                # paths_groups = [Product1_Paths, Product2_Paths, ...]
                
                try:
                    gcode_lines = self.generate_dispense_gcode(paths_groups, target_x, target_y)
                    
                    if not gcode_lines:
                        self.log_signal.emit("⚠ 生成 G-code 失败，跳过")
                        molds_finished += 1
                        continue

                    # 发送完整 G-code 到界面显示
                    full_gcode_str = "\n".join(gcode_lines)
                    self.gcode_signal.emit(full_gcode_str)

                    # --- 5. 执行点胶 ---
                    self.log_signal.emit(f"💉 开始执行 (R{r}, C{c})...")
                    
                    # 逐行发送
                    total_lines = len(gcode_lines)
                    self.progress_signal.emit(0, total_lines) # 初始化进度条
                    
                    for i, line in enumerate(gcode_lines):
                        # 发送当前行信号 (用于高亮)
                        self.current_line_signal.emit(line)
                        self.current_index_signal.emit(i)
                        
                        # 更新进度条 (全局行粒度)
                        # 当前进度 = 已完成模具进度 + 当前模具内进度
                        current_mold_progress = int((i + 1) / total_lines * RESOLUTION)
                        global_current = (molds_finished * RESOLUTION) + current_mold_progress
                        self.progress_signal.emit(global_current, global_total_steps)
                        # 1. 检查停止
                        if not self.running:
                            self.finished_signal.emit(False, "用户停止")
                            return

                        # 2. 检查暂停 (立即生效)
                        while self.paused:
                            if not self.running:
                                self.finished_signal.emit(False, "用户停止")
                                return
                            self.msleep(100)

                        self.mw.machine._post_gcode(line)
                    
                    self.mw.machine.wait_until_idle()
                    self.log_signal.emit("✅ 单个完成")
                    
                    # 关键修复：单个完成后，确保 Z 轴抬升到【移动安全高度】
                    # 这样移动到下一个阵列点时（XY移动）就不会撞到模具壁
                    # generate_dispense_gcode 最后一行是 safe_z_cavity，可能不够高
                    if self.mode == 'camera_dry':
                        safe_z_move = float(self.cfg['cam_safe_z_product'])
                    else:
                        safe_z_move = float(self.cfg['nozzle_safe_z_product'])
                        
                    self.mw.machine.move_to(z=safe_z_move, speed=self.mw.feedrate_z)
                    self.mw.machine.wait_until_idle()

                except Exception as e:
                    self.log_signal.emit(f"❌ 执行出错: {e}")
                
                molds_finished += 1

        # 结束
        # 结束 - 返回起始点
        self.log_signal.emit("🔙 正在返回起始点...")
        # 1. 确保 Z 在最高安全位
        if self.mode == 'camera_dry':
            final_safe_z = float(self.cfg['cam_safe_z_product'])
        else:
            final_safe_z = float(self.cfg['nozzle_safe_z_product'])
        self.mw.machine.move_to(z=final_safe_z, speed=self.mw.feedrate_z)
        
        # 2. 移动回起始 XY
        self.mw.machine.move_to(x=start_x, y=start_y, speed=self.mw.feedrate_xy)
        self.mw.machine.wait_until_idle()

        # 3. 恢复起始 Z (用户要求)
        self.log_signal.emit(f"🔙 恢复起始高度 Z={start_z:.2f}")
        self.mw.machine.move_to(z=start_z, speed=self.mw.feedrate_z)
        self.mw.machine.wait_until_idle()

        self.finished_signal.emit(True, f"{title}完成")

    def generate_dispense_gcode(self, paths_groups, current_x, current_y):
        """
        根据识别到的像素路径组，生成绝对坐标 G-code
        paths_groups: List[List[Path]]，每个内部List代表一个产品的多条路径
        """
        M = self.mw.cam2mach_matrix
        if M is None:
            raise Exception("未标定坐标系")

        h_img, w_img = self.mw.last_frame.shape[:2]
        center_x, center_y = w_img / 2, h_img / 2
        
        # 根据模式选择参数
        if self.mode == 'camera_dry':
            # 相机空跑：无偏移，用相机高度
            off_x, off_y = 0.0, 0.0
            safe_z_cavity = float(self.cfg['cam_safe_z_cavity'])
            safe_z_product = float(self.cfg['cam_safe_z_product'])
            work_z = float(self.cfg['cam_work_z'])
            valve_on = False
        else:
            # 针头空跑 / 真实点胶：有偏移，用主界面高度
            off_x = self.mw.nozzle_offset_x
            off_y = self.mw.nozzle_offset_y
            safe_z_cavity = float(self.cfg['nozzle_safe_z_cavity'])
            safe_z_product = float(self.cfg['nozzle_safe_z_product'])
            work_z = float(self.cfg['nozzle_work_z'])
            valve_on = (self.mode == 'real')

        speed_xy = int(self.mw.feedrate_xy)
        speed_z = int(self.mw.feedrate_z)

        lines = []
        lines.append("G90") # 绝对模式
        
        # 初始抬升到产品安全高度 (最高)
        lines.append(f"G1 Z{safe_z_product:.2f} F{speed_z}")

        for prod_idx, product_paths in enumerate(paths_groups):
            lines.append(f"; --- Product {prod_idx + 1} ---")
            
            for seg_idx, path in enumerate(product_paths):
                first_pt = True
                for (px, py) in path:
                    # 像素 -> 相对 mm
                    dx_pix = px - center_x
                    dy_pix = py - center_y
                    
                    dx_mm = M[0, 0] * dx_pix + M[0, 1] * dy_pix
                    dy_mm = M[1, 0] * dx_pix + M[1, 1] * dy_pix
                    
                    # 目标机器坐标 = 相机中心 + 视觉偏移 - 针头偏移
                    target_mach_x = current_x + dx_mm - off_x
                    target_mach_y = current_y + dy_mm - off_y
                    
                    if first_pt:
                        # 1. 移动到第一个点上方 (保持当前高度，可能是 Product Z 或 Cavity Z)
                        lines.append(f"G1 X{target_mach_x:.3f} Y{target_mach_y:.3f} F{speed_xy}")
                        # 2. 下降到工作高度
                        lines.append(f"G1 Z{work_z:.2f} F{speed_z}")
                        # 3. 开阀 (仅真实模式)
                        if valve_on:
                            lines.append("M106 S255")
                        first_pt = False
                    else:
                        # 移动到后续点
                        lines.append(f"G1 X{target_mach_x:.3f} Y{target_mach_y:.3f} F{speed_xy}")
                
                # 一条路径(Segment)结束：关阀
                if valve_on:
                    lines.append("M107") # 关阀
                
                # 抬升逻辑：
                # 如果是该产品的最后一段 -> 抬升到【产品安全高度】(准备去下一个产品)
                # 否则 -> 抬升到【腔内安全高度】(准备去同产品的下一段)
                if seg_idx == len(product_paths) - 1:
                    lines.append(f"G1 Z{safe_z_product:.2f} F{speed_z}")
                else:
                    lines.append(f"G1 Z{safe_z_cavity:.2f} F{speed_z}")

        # 最后确保在最高安全高度
        lines.append(f"G1 Z{safe_z_product:.2f} F{speed_z}")
        
        return lines

    def stop(self):
        self.running = False


class BatchRunDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.setWindowTitle("批量运行 (Batch Run)")
        self.resize(600, 750) # 增加高度
        
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        layout = QVBoxLayout()

        # --- 1. 阵列配置 ---
        grp_grid = QGroupBox("阵列配置 (Grid Config)")
        form = QFormLayout()

        # 起始点
        self.spn_start_x = QDoubleSpinBox()
        self.spn_start_x.setRange(-500, 500)
        self.spn_start_y = QDoubleSpinBox()
        self.spn_start_y.setRange(-500, 500)
        
        self.spn_start_z = QDoubleSpinBox() # 新增 Z
        self.spn_start_z.setRange(-500, 500)

        btn_teach_start = QPushButton("示教当前位置")
        btn_teach_start.clicked.connect(self.on_teach_start)
        
        h_start = QHBoxLayout()
        h_start.addWidget(QLabel("X:"))
        h_start.addWidget(self.spn_start_x)
        h_start.addWidget(QLabel("Y:"))
        h_start.addWidget(self.spn_start_y)
        h_start.addWidget(QLabel("Z:"))     # 新增 Z
        h_start.addWidget(self.spn_start_z)
        h_start.addWidget(btn_teach_start)
        
        form.addRow("起始点 (Start Pos):", h_start)

        # 行列数
        self.spn_rows = QSpinBox()
        self.spn_rows.setRange(1, 100)
        self.spn_rows.setValue(4)
        
        self.spn_cols = QSpinBox()
        self.spn_cols.setRange(1, 100)
        self.spn_cols.setValue(4)
        
        h_grid = QHBoxLayout()
        h_grid.addWidget(QLabel("行 (Rows):"))
        h_grid.addWidget(self.spn_rows)
        h_grid.addWidget(QLabel("列 (Cols):"))
        h_grid.addWidget(self.spn_cols)
        
        form.addRow("阵列规模:", h_grid)

        # 间距
        self.spn_space_x = QDoubleSpinBox()
        self.spn_space_x.setRange(-200, 200)
        self.spn_space_x.setValue(50.0)
        
        self.spn_space_y = QDoubleSpinBox()
        self.spn_space_y.setRange(-200, 200)
        self.spn_space_y.setValue(50.0)
        
        h_space = QHBoxLayout()
        h_space.addWidget(QLabel("X 间距:"))
        h_space.addWidget(self.spn_space_x)
        h_space.addWidget(QLabel("Y 间距:"))
        h_space.addWidget(self.spn_space_y)
        
        form.addRow("间距 (Spacing):", h_space)
        
        grp_grid.setLayout(form)
        layout.addWidget(grp_grid)

        # --- 2. 高度配置 (新增) ---
        grp_z = QGroupBox("高度配置 (Z-Heights)")
        form_z = QFormLayout()
        
        # 相机空跑高度
        # 相机空跑高度
        # 智能默认值：如果主界面参数看起来是默认值(<=20.0)，但 Z 基准很大(>100)，则使用 Z 基准
        # 这样可以防止用户忘记设置参数时机器撞向底部
        def_cam_safe = self.mw.cam_safe_z_cavity
        if def_cam_safe <= 20.0 and self.mw.z_base > 100:
            def_cam_safe = self.mw.z_base
            
        self.spn_cam_safe_cavity = QDoubleSpinBox()
        self.spn_cam_safe_cavity.setRange(-500, 500)
        self.spn_cam_safe_cavity.setValue(def_cam_safe)
        
        self.spn_cam_safe_product = QDoubleSpinBox()
        self.spn_cam_safe_product.setRange(-500, 500)
        self.spn_cam_safe_product.setValue(self.mw.cam_safe_z_product)

        self.spn_cam_work = QDoubleSpinBox()
        self.spn_cam_work.setRange(-500, 500)
        self.spn_cam_work.setValue(self.mw.cam_work_z)
        
        h_cam_z = QHBoxLayout()
        h_cam_z.addWidget(QLabel("腔内安全:"))
        h_cam_z.addWidget(self.spn_cam_safe_cavity)
        h_cam_z.addWidget(QLabel("移动安全:"))
        h_cam_z.addWidget(self.spn_cam_safe_product)
        h_cam_z.addWidget(QLabel("工作:"))
        h_cam_z.addWidget(self.spn_cam_work)
        form_z.addRow("相机高度:", h_cam_z)

        # 针头/真实高度
        self.spn_nozzle_safe_cavity = QDoubleSpinBox()
        self.spn_nozzle_safe_cavity.setRange(-500, 500)
        self.spn_nozzle_safe_cavity.setValue(self.mw.safe_z_cavity)
        
        self.spn_nozzle_safe_product = QDoubleSpinBox()
        self.spn_nozzle_safe_product.setRange(-500, 500)
        self.spn_nozzle_safe_product.setValue(self.mw.safe_z_product)

        self.spn_nozzle_work = QDoubleSpinBox()
        self.spn_nozzle_work.setRange(-500, 500)
        self.spn_nozzle_work.setValue(self.mw.work_z)
        
        h_nozzle_z = QHBoxLayout()
        h_nozzle_z.addWidget(QLabel("腔内安全:"))
        h_nozzle_z.addWidget(self.spn_nozzle_safe_cavity)
        h_nozzle_z.addWidget(QLabel("移动安全:"))
        h_nozzle_z.addWidget(self.spn_nozzle_safe_product)
        h_nozzle_z.addWidget(QLabel("工作:"))
        h_nozzle_z.addWidget(self.spn_nozzle_work)
        form_z.addRow("针头高度:", h_nozzle_z)
        
        grp_z.setLayout(form_z)
        layout.addWidget(grp_z)

        # --- 3. 运行日志 ---
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        # --- 3.5 进度条 ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%") # 显示百分比
        layout.addWidget(self.progress_bar)

        # G-code 预览框 (新增)
        layout.addWidget(QLabel("G-code 实时预览:"))
        self.gcode_box = QTextEdit()
        self.gcode_box.setReadOnly(True)
        self.gcode_box.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc; font-family: Consolas; font-size: 10pt;")
        layout.addWidget(self.gcode_box)

        # --- 4. 按钮 ---
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("保存配置")
        self.btn_save.clicked.connect(self.save_settings)
        
        # 新增按钮
        self.btn_cam_dry = QPushButton("相机批量")
        self.btn_cam_dry.clicked.connect(lambda: self.on_start('camera_dry'))
        
        self.btn_nozzle_dry = QPushButton("针头批量")
        self.btn_nozzle_dry.clicked.connect(lambda: self.on_start('nozzle_dry'))

        self.btn_start = QPushButton("真实批量执行")
        self.btn_start.setStyleSheet("background-color: #10B981; color: white; font-weight: bold; font-size: 14px;")
        self.btn_start.clicked.connect(lambda: self.on_start('real'))
        
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setStyleSheet("background-color: #F59E0B; color: white; font-weight: bold;")
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_pause.setEnabled(False)

        self.btn_resume = QPushButton("继续")
        self.btn_resume.setStyleSheet("background-color: #3B82F6; color: white; font-weight: bold;")
        self.btn_resume.clicked.connect(self.on_resume)
        self.btn_resume.setEnabled(False)
        self.btn_resume.setVisible(False) # 初始隐藏

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setStyleSheet("background-color: #EF4444; color: white;")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)

        btn_layout.addWidget(self.btn_save)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cam_dry)
        btn_layout.addWidget(self.btn_nozzle_dry)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_resume)
        btn_layout.addWidget(self.btn_stop)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def on_teach_start(self):
        x, y, z = self.mw.machine.get_position()
        self.spn_start_x.setValue(x)
        self.spn_start_y.setValue(y)
        self.spn_start_z.setValue(z) # 新增 Z
        self.log(f"已示教起始点: X{x:.2f}, Y{y:.2f}, Z{z:.2f}")

    def get_config(self):
        return {
            'start_x': self.spn_start_x.value(),
            'start_y': self.spn_start_y.value(),
            'start_z': self.spn_start_z.value(), # 新增 Z
            'rows': self.spn_rows.value(),
            'cols': self.spn_cols.value(),
            'space_x': self.spn_space_x.value(),
            'space_y': self.spn_space_y.value(),
            # 新增高度参数
            'cam_safe_z_cavity': self.spn_cam_safe_cavity.value(),
            'cam_safe_z_product': self.spn_cam_safe_product.value(),
            'cam_work_z': self.spn_cam_work.value(),
            'nozzle_safe_z_cavity': self.spn_nozzle_safe_cavity.value(),
            'nozzle_safe_z_product': self.spn_nozzle_safe_product.value(),
            'nozzle_work_z': self.spn_nozzle_work.value(),
        }

    def on_start(self, mode):
        # 自动保存配置 (符合用户直觉：运行即保存)
        self.save_settings()
        
        cfg = self.get_config()
        
        # ====== 安全检查 ======
        # 1. Z 基准检查 (防止原点在底部时撞车)
        if self.mw.z_base > 100:
            safe_z = cfg['cam_safe_z_cavity'] if 'camera' in mode else cfg['nozzle_safe_z_cavity']
            if safe_z < 50:
                ret = QMessageBox.warning(self, "高度警告", 
                                          f"检测到安全高度 ({safe_z:.1f}) 远小于 Z 基准 ({self.mw.z_base:.1f})！\n\n"
                                          "这可能导致机器撞向底部。\n"
                                          "是否确实要继续？",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if ret == QMessageBox.No:
                    return

        # 2. 相机与针头高度差异检查 (防止忘记设置相机高度)
        # 如果针头安全高度很高 (>80)，但相机安全高度很低 (<30)，提示用户
        if 'camera' in mode:
            nz_safe = cfg['nozzle_safe_z_cavity']
            cam_safe = cfg['cam_safe_z_cavity']
            if nz_safe > 80 and cam_safe < 30:
                ret = QMessageBox.warning(self, "参数可疑", 
                                          f"检测到相机安全高度 ({cam_safe:.1f}) 远小于 针头安全高度 ({nz_safe:.1f})！\n\n"
                                          "您可能忘记设置【相机空跑】的高度参数了。\n"
                                          "继续运行可能会导致撞车。\n\n"
                                          "是否确实要继续？",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if ret == QMessageBox.No:
                    return
        # =====================
        # 不再反向更新主界面参数，保持独立性
        # self.mw.cam_safe_z_cavity = cfg['cam_safe_z'] ...
        
        self.worker = BatchWorker(self.mw, cfg, mode)
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.gcode_signal.connect(self.update_gcode_display)
        self.worker.current_index_signal.connect(self.highlight_current_line)
        
        # 关闭实时识别 (防止卡顿和Z轴变化导致的误识别)
        self.mw.live_detect = False
        self.mw.btn_detect.setText("识别模具")
        
        # 开启中心十字 (辅助观察)
        self.mw.show_center_crosshair = True
        self.mw.btn_show_crosshair.setText("关闭中心")
        self.mw.log("✅ 批量运行中：关闭实时识别，开启中心十字")
        
        self.btn_cam_dry.setEnabled(False)
        self.btn_nozzle_dry.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setVisible(True)
        self.btn_resume.setVisible(False)
        
        # 设置进度条最大值
        # total = cfg['rows'] * cfg['cols']
        # self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        
        self.worker.start()

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_pause(self):
        if hasattr(self, 'worker'):
            self.worker.pause()
            self.btn_pause.setVisible(False)
            self.btn_resume.setVisible(True)
            self.btn_resume.setEnabled(True)

    def on_resume(self):
        if hasattr(self, 'worker'):
            self.worker.resume()
            self.btn_resume.setVisible(False)
            self.btn_pause.setVisible(True)

    def on_stop(self):
        if hasattr(self, 'worker'):
            self.worker.stop()
            self.log("正在停止...")

    def on_finished(self, success, msg):
        self.btn_cam_dry.setEnabled(True)
        self.btn_nozzle_dry.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setVisible(False)
        self.btn_pause.setVisible(True)
        
        # 关闭中心十字
        self.mw.show_center_crosshair = False
        self.mw.btn_show_crosshair.setText("显示中心")
        self.mw.log("⏹ 批量运行结束：关闭中心十字")
        if success:
            QMessageBox.information(self, "完成", msg)
        else:
            # 如果是用户停止，更新状态显示
            if msg == "用户停止":
                self.log("🛑 已停止运行")
                QMessageBox.warning(self, "已停止", "批量运行已停止")
            else:
                QMessageBox.warning(self, "中止", msg)

    def log(self, msg):
        self.log_box.append(msg)
        # 也可以同步到主界面日志
        # self.mw.log(f"[Batch] {msg}")

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

    def load_settings(self):
        """
        从 app_settings.json 加载批量配置
        """
        settings_file = "config/app_settings.json"
        if not os.path.exists(settings_file):
            return

        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                
                # 加载阵列参数
                if 'batch_start_x' in settings: self.spn_start_x.setValue(float(settings['batch_start_x']))
                if 'batch_start_y' in settings: self.spn_start_y.setValue(float(settings['batch_start_y']))
                if 'batch_start_z' in settings: self.spn_start_z.setValue(float(settings['batch_start_z'])) # 新增 Z
                if 'batch_rows' in settings: self.spn_rows.setValue(int(settings['batch_rows']))
                if 'batch_cols' in settings: self.spn_cols.setValue(int(settings['batch_cols']))
                if 'batch_space_x' in settings: self.spn_space_x.setValue(float(settings['batch_space_x']))
                if 'batch_space_y' in settings: self.spn_space_y.setValue(float(settings['batch_space_y']))
                
                # Z轴参数通常跟随主界面，但如果用户在批量界面改了并保存，也可以加载
                if 'cam_safe_z_cavity' in settings: self.spn_cam_safe_cavity.setValue(float(settings['cam_safe_z_cavity']))
                if 'cam_safe_z_product' in settings: self.spn_cam_safe_product.setValue(float(settings['cam_safe_z_product']))
                if 'cam_work_z' in settings: self.spn_cam_work.setValue(float(settings['cam_work_z']))
                
                if 'safe_z_cavity' in settings: self.spn_nozzle_safe_cavity.setValue(float(settings['safe_z_cavity']))
                if 'safe_z_product' in settings: self.spn_nozzle_safe_product.setValue(float(settings['safe_z_product']))
                if 'work_z' in settings: self.spn_nozzle_work.setValue(float(settings['work_z']))
                
        except Exception as e:
            self.log(f"❌ 加载配置失败: {e}")

    def save_settings(self):
        """
        保存配置到 app_settings.json
        仅保存阵列参数 (Grid Config)
        """
        settings_file = "config/app_settings.json"
        settings = {}
        
        # 先读取旧配置，避免覆盖其他参数
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
            except:
                pass

        # 1. 更新阵列参数
        settings['batch_start_x'] = self.spn_start_x.value()
        settings['batch_start_y'] = self.spn_start_y.value()
        settings['batch_start_z'] = self.spn_start_z.value() # 新增 Z
        settings['batch_rows'] = self.spn_rows.value()
        settings['batch_cols'] = self.spn_cols.value()
        settings['batch_space_x'] = self.spn_space_x.value()
        settings['batch_space_y'] = self.spn_space_y.value()

        # 2. 保存高度参数 (用户希望在批量界面修改后能保存)
        settings['cam_safe_z_cavity'] = self.spn_cam_safe_cavity.value()
        settings['cam_safe_z_product'] = self.spn_cam_safe_product.value()
        settings['cam_work_z'] = self.spn_cam_work.value()
        
        settings['safe_z_cavity'] = self.spn_nozzle_safe_cavity.value()
        settings['safe_z_product'] = self.spn_nozzle_safe_product.value()
        settings['work_z'] = self.spn_nozzle_work.value()

        # 同时更新主界面变量，以便下次打开时同步
        self.mw.cam_safe_z_cavity = settings['cam_safe_z_cavity']
        self.mw.cam_safe_z_product = settings['cam_safe_z_product']
        self.mw.cam_work_z = settings['cam_work_z']
        self.mw.safe_z_cavity = settings['safe_z_cavity']
        self.mw.safe_z_product = settings['safe_z_product']
        self.mw.work_z = settings['work_z']

        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            self.log("✅ 批量阵列配置已保存")
            QMessageBox.information(self, "保存成功", "批量阵列配置已保存！")
        except Exception as e:
            self.log(f"❌ 保存失败: {e}")
