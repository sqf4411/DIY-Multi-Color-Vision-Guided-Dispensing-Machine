import json
import os
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
    QSpinBox, QPushButton, QGroupBox, QFormLayout, QMessageBox, QTextEdit
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from core.mold_detector import get_calculated_paths

class BatchWorker(QThread):
    """
    批量运行工作线程：
    循环：移动 -> 拍照 -> 识别 -> 点胶
    """
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    progress_signal = pyqtSignal(int, int)  # current, total

    def __init__(self, main_window, grid_config, mode='real'):
        super().__init__()
        self.mw = main_window
        self.cfg = grid_config
        self.mode = mode  # 'real', 'camera_dry', 'nozzle_dry'
        self.running = True
        self.paused = False

    def run(self):
        rows = self.cfg['rows']
        cols = self.cfg['cols']
        start_x = self.cfg['start_x']
        start_y = self.cfg['start_y']
        space_x = self.cfg['space_x']
        space_y = self.cfg['space_y']
        
        total = rows * cols
        count = 0

        mode_names = {
            'real': "真实批量点胶",
            'camera_dry': "相机批量空跑",
            'nozzle_dry': "针头批量空跑"
        }
        title = mode_names.get(self.mode, "未知模式")

        self.log_signal.emit(f"🚀 开始【{title}】：{rows}x{cols}，共 {total} 个")

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

                count += 1
                self.progress_signal.emit(count, total)
                
                # --- 1. 计算目标坐标 (相机位置) ---
                target_x = start_x + c * space_x
                target_y = start_y + r * space_y
                
                self.log_signal.emit(f"📍 [{count}/{total}] 移动到 (R{r}, C{c}): X{target_x:.2f}, Y{target_y:.2f}")
                
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
                paths_groups = get_calculated_paths(img)
                
                if not paths_groups:
                    self.log_signal.emit(f"⚠ 未识别到模具 (R{r}, C{c})，跳过")
                    continue
                
                # Flatten groups to single list of paths
                paths_px = [p for group in paths_groups for p in group]

                try:
                    gcode_lines = self.generate_dispense_gcode(paths_px, target_x, target_y)
                    
                    if not gcode_lines:
                        self.log_signal.emit("⚠ 生成 G-code 失败，跳过")
                        continue

                    # --- 5. 执行点胶 ---
                    self.log_signal.emit(f"💉 开始执行 (R{r}, C{c})...")
                    
                    # 逐行发送
                    for line in gcode_lines:
                        if not self.running: return
                        self.mw.machine._post_gcode(line)
                    
                    self.mw.machine.wait_until_idle()
                    self.log_signal.emit("✅ 单个完成")

                except Exception as e:
                    self.log_signal.emit(f"❌ 执行出错: {e}")

        # 结束
        self.finished_signal.emit(True, f"{title}完成")

    def generate_dispense_gcode(self, paths_px, current_x, current_y):
        """
        根据识别到的像素路径，生成绝对坐标 G-code
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
        
        for path in paths_px:
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
                    # 1. 抬起
                    lines.append(f"G1 Z{safe_z_cavity:.2f} F{speed_z}")
                    # 2. 移动到第一个点上方
                    lines.append(f"G1 X{target_mach_x:.3f} Y{target_mach_y:.3f} F{speed_xy}")
                    # 3. 下降到工作高度
                    lines.append(f"G1 Z{work_z:.2f} F{speed_z}")
                    # 4. 开阀 (仅真实模式)
                    if valve_on:
                        lines.append("M106 S255")
                    first_pt = False
                else:
                    # 移动到后续点
                    lines.append(f"G1 X{target_mach_x:.3f} Y{target_mach_y:.3f} F{speed_xy}")
            
            # 一条路径结束：关阀，抬起
            if valve_on:
                lines.append("M107") # 关阀
            lines.append(f"G1 Z{safe_z_product:.2f} F{speed_z}")

        # 最后抬起到最高安全高度
        lines.append(f"G1 Z{safe_z_cavity:.2f} F{speed_z}")
        
        return lines

    def stop(self):
        self.running = False


class BatchRunDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.setWindowTitle("批量运行 (Batch Run)")
        self.resize(600, 500)
        
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
        
        btn_teach_start = QPushButton("示教当前位置")
        btn_teach_start.clicked.connect(self.on_teach_start)
        
        h_start = QHBoxLayout()
        h_start.addWidget(QLabel("X:"))
        h_start.addWidget(self.spn_start_x)
        h_start.addWidget(QLabel("Y:"))
        h_start.addWidget(self.spn_start_y)
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
        self.spn_cam_safe_cavity = QDoubleSpinBox()
        self.spn_cam_safe_cavity.setRange(-500, 500)
        self.spn_cam_safe_cavity.setValue(self.mw.cam_safe_z_cavity)
        
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
        
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setStyleSheet("background-color: #EF4444; color: white;")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)

        btn_layout.addWidget(self.btn_save)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cam_dry)
        btn_layout.addWidget(self.btn_nozzle_dry)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def on_teach_start(self):
        x, y, z = self.mw.machine.get_position()
        self.spn_start_x.setValue(x)
        self.spn_start_y.setValue(y)
        self.log(f"已示教起始点: {x:.2f}, {y:.2f}")

    def get_config(self):
        return {
            'start_x': self.spn_start_x.value(),
            'start_y': self.spn_start_y.value(),
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
        cfg = self.get_config()
        # 不再反向更新主界面参数，保持独立性
        # self.mw.cam_safe_z_cavity = cfg['cam_safe_z'] ...
        
        self.worker = BatchWorker(self.mw, cfg, mode)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        
        self.btn_cam_dry.setEnabled(False)
        self.btn_nozzle_dry.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.worker.start()

    def on_stop(self):
        if hasattr(self, 'worker'):
            self.worker.stop()
            self.log("正在停止...")

    def on_finished(self, success, msg):
        self.btn_cam_dry.setEnabled(True)
        self.btn_nozzle_dry.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if success:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.warning(self, "中止", msg)

    def log(self, msg):
        self.log_box.append(msg)
        # 也可以同步到主界面日志
        # self.mw.log(f"[Batch] {msg}")

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
                if 'batch_rows' in settings: self.spn_rows.setValue(int(settings['batch_rows']))
                if 'batch_cols' in settings: self.spn_cols.setValue(int(settings['batch_cols']))
                if 'batch_space_x' in settings: self.spn_space_x.setValue(float(settings['batch_space_x']))
                if 'batch_space_y' in settings: self.spn_space_y.setValue(float(settings['batch_space_y']))
                
                # Z轴参数通常跟随主界面，但如果用户在批量界面改了并保存，也可以加载
                # 这里我们选择：初始化时用主界面参数(已在init_ui完成)，
                # 但如果 JSON 里有专门的 batch_cam_safe_z 等，也可以覆盖。
                # 现在的逻辑是：init_ui 用 self.mw 的值初始化，这里就不覆盖了，
                # 除非我们想让批量界面的修改独立保存。
                # 用户说：“点击相机批量那一定直接使用相机的3个高度参数啊”
                # 所以最好保持同步。这里就不加载 Z 了，让它跟随 self.mw。
                
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
        settings['batch_rows'] = self.spn_rows.value()
        settings['batch_cols'] = self.spn_cols.value()
        settings['batch_space_x'] = self.spn_space_x.value()
        settings['batch_space_y'] = self.spn_space_y.value()

        # 注意：Z轴高度参数由主界面管理，这里不保存到文件，
        # 避免混淆。用户在批量界面修改的高度仅对本次运行有效，
        # 若要永久修改，请在主界面或通过“保存全局配置”进行。

        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            self.log("✅ 批量阵列配置已保存")
            QMessageBox.information(self, "保存成功", "批量阵列配置已保存！")
        except Exception as e:
            self.log(f"❌ 保存失败: {e}")
