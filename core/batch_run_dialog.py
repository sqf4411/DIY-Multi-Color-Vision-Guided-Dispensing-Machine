import json
import os
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
    QSpinBox, QPushButton, QGroupBox, QFormLayout, QMessageBox, QTextEdit,
    QProgressBar, QCheckBox
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
    progress_signal = pyqtSignal(int, int)  # current, total
    gcode_signal = pyqtSignal(str)          # 新增：发送生成的完整 G-code
    current_line_signal = pyqtSignal(str)   # 新增：发送当前执行的 G-code 行
    current_index_signal = pyqtSignal(int)  # 新增：发送当前执行的 G-code 行号 (用于准确高亮)
    path_update_signal = pyqtSignal(list)   # 新增：发送当前产品的路径数据用于实时显示
    color_change_request_signal = pyqtSignal(str, str, float, float)  # finished_color, next_color, park_x, park_y

    def __init__(self, main_window, grid_config, mode='real'):
        super().__init__()
        self.mw = main_window
        self.cfg = grid_config
        self.mode = mode  # 'real', 'camera_dry', 'nozzle_dry'
        self.running = True
        self.paused = False
        self._color_change_decision = None

    def pause(self):
        self.paused = True
        self.log_signal.emit("⏸ 已暂停...")

    def resume(self):
        self.paused = False
        self.log_signal.emit("▶ 继续运行...")

    def confirm_color_change_continue(self):
        self._color_change_decision = "continue"

    def confirm_color_change_stop(self):
        self._color_change_decision = "stop"

    def _color_name(self, color_idx: int) -> str:
        return ["红色(A1)", "绿色(A2)", "蓝色(A3)", "黄色(A4)"][color_idx] if 0 <= color_idx < 4 else f"Color {color_idx}"

    def _await_color_change_confirmation(self, finished_color: str, next_color: str) -> bool:
        """
        Color-change checkpoint:
        1) Move to configured XY park position.
        2) Ask UI thread to show dialog: test-dispense / continue / stop.
        """
        park_x = float(self.cfg.get('color_change_park_x', self.cfg.get('start_x', 0.0)))
        park_y = float(self.cfg.get('color_change_park_y', self.cfg.get('start_y', 0.0)))

        # Ensure safe lift before moving to park point
        self._safe_lift_z()
        self.mw.machine.move_to(x=park_x, y=park_y, speed=self.mw.feedrate_xy)
        self.mw.machine.wait_until_idle()

        self.log_signal.emit(
            f"🔁 {finished_color} 已完成，停靠到换胶位 X={park_x:.2f}, Y={park_y:.2f}，等待确认..."
        )

        self._color_change_decision = None
        self.color_change_request_signal.emit(finished_color, next_color, park_x, park_y)

        while self.running and self._color_change_decision is None:
            self.msleep(100)

        if not self.running:
            return False
        if self._color_change_decision != "continue":
            self.running = False
            return False

        self.log_signal.emit(f"▶ 继续执行下一颜色：{next_color}")
        return True

    def run(self):
        rows = self.cfg['rows']
        cols = self.cfg['cols']
        start_x = self.cfg['start_x']
        start_y = self.cfg['start_y']
        start_z = self.cfg.get('start_z', 0.0) 
        space_x = self.cfg['space_x']
        space_y = self.cfg['space_y']
        
        total_molds = rows * cols
        # Calculate total colors to determine total steps (molds * colors)
        # We need to know segments first. 
        # But we only get segments AFTER first vision or from main window.
        # Assuming all molds have same colors. We can check self.mw.mold_segments for colors.
        
        # 0. Analyze Colors to Process
        active_colors = set()
        is_mixed = self.cfg.get('mixed_mode', False)
        
        if is_mixed:
            # Mixed Mode: Initially assume all 4 colors to set up progress bar
            # We will accurately recalculate this after Phase 1 finishes!
            active_colors = {0, 1, 2, 3}
            self.mw.log(f"[Batch] Mixed Mode Enabled: Initially reserving 4 colors.")
        else:
            # Standard Mode: Check current loaded template
            if not getattr(self.mw, 'halcon_path_template', None):
                 self.mw.log("[Batch] Computing Halcon template to determine colors...")
                 try:
                     self.mw._compute_halcon_template()
                 except Exception as e:
                     self.mw.log(f"[Batch] Warning: Template computation failed ({e}). Defaulting to Color 0.")

            template_segments = getattr(self.mw, 'halcon_path_template', [])
            if template_segments:
                 for seg in template_segments:
                     if hasattr(seg, 'color'): 
                         active_colors.add(seg.color)
        
        sorted_colors = sorted(list(active_colors)) if active_colors else [0]
        if not sorted_colors: sorted_colors = [0]
        
        total_scan_ops = total_molds
        total_dispense_ops = total_molds * len(sorted_colors)
        total_ops = total_scan_ops + total_dispense_ops
        ops_finished = 0
        RESOLUTION = 1000
        global_total_steps = total_ops * RESOLUTION

        mode_names = {
            'real': "真实批量(颜色优先)",
            'camera_dry': "相机批量(颜色优先)",
            'nozzle_dry': "针头批量(颜色优先)"
        }
        title = mode_names.get(self.mode, "未知模式")

        self.log_signal.emit(f"🚀 开始【{title}】：{rows}x{cols}，共 {len(sorted_colors)} 种颜色")
        self.log_signal.emit(f"📸 阶段1：全局扫描所有模具 ({total_molds} 个)...")
        
        self.mw.show_center_crosshair = True
        self.mw.render_current_frame() 

        # Vision Cache: Stores (r, c) -> (paths_groups, current_x, current_y, product_name)
        vision_cache = {}

        # 1. 扫描阶段始终用相机移动安全高度（无论 camera_dry 还是 nozzle_dry）
        #    Phase1 扫描由相机完成，XY 移动必须在相机移动安全高度
        cam_scan_safe_z = float(self.mw.cam_safe_z_product or 0.0)
        self.mw.machine.move_to(z=cam_scan_safe_z, speed=self.mw.feedrate_z)
        self.log_signal.emit(f"🔼 [扫描阶段] 抬升至相机移动安全高度 Z={cam_scan_safe_z}")


        # ========================================================
        # Phase 1: Global Scan Loop
        # ========================================================
        for r in range(rows):
            for c in range(cols):
                if not self.running:
                    self.finished_signal.emit(False, "用户停止")
                    return

                while self.paused:
                    if not self.running: return
                    self.msleep(100)

                current_base_progress = ops_finished * RESOLUTION
                self.progress_signal.emit(current_base_progress, global_total_steps)

                cache_key = (r, c)
                target_x = start_x + c * space_x
                target_y = start_y + r * space_y
                
                self.log_signal.emit(f"📍 [{ops_finished + 1}/{total_ops}] (R{r}, C{c}) 首次定位 & 识别...")
                
                # 移动到扫描位置（XY在相机移动安全高度，不会触碰模具）
                self.mw.machine.move_to(x=target_x, y=target_y, speed=self.mw.feedrate_xy)

                # 降到相机工作高度（对焦高度）进行扫描
                cam_work_z = float(self.mw.cam_work_z or 0.0)
                self.mw.machine.move_to(z=cam_work_z, speed=self.mw.feedrate_z)
                self.mw.machine.wait_until_idle()
                
                # 增加稳定时间，保证相机对焦和曝光稳定 (由 0.5 改为 1.2)
                time.sleep(1.2) 

                if self.mw.last_frame is None:
                    self.log_signal.emit("❌ 无法获取图像，跳过")
                    self._safe_lift_z()
                    vision_cache[cache_key] = None
                    ops_finished += 1
                    continue

                img = self.mw.last_frame.copy()
                img = cv2.flip(img, -1)
                if len(img.shape) == 3: gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else: gray_img = img

                # [NEW] Mixed Mode Logic: Detect Product First
                is_mixed = self.cfg.get('mixed_mode', False)
                if is_mixed:
                    self.log_signal.emit(f"🔍 [混合模式] 正在识别最佳产品...")
                    all_prods = self.mw.product_manager.list_products()
                    best_prod, score = self.mw.halcon_product_manager.find_best_match_product(gray_img, all_prods)
                    
                    if best_prod and score > 0.4: # Lower threshold to 0.4
                        self.log_signal.emit(f"✅ (R{r}C{c}) 识别到产品 [{best_prod}] (Score: {score:.2f})")
                        # 加载该产品
                        ok, msg = self.mw.halcon_product_manager.load_product(best_prod)
                        if ok:
                            # 重要：更新主界面模板缓存，否则后续识别会用旧模板
                            self.mw._compute_halcon_template()
                        else:
                            self.log_signal.emit(f"❌ 加载产品失败: {msg}")
                    else:
                        self.log_signal.emit(f"⚠️ (R{r}C{c}) 未能识别已知产品 (Best: {best_prod}, Score: {score:.2f})")
                        # 可能会继续用当前产品尝试，或者跳过
                
                paths_groups, _ = self.mw.get_vision_result(gray_img)
                
                if not paths_groups:
                    self.log_signal.emit(f"⚠ 未识别到模具 (R{r}, C{c})")
                    vision_cache[cache_key] = None # Mark as processed but empty
                else:
                    self.path_update_signal.emit(paths_groups)
                    # 应用当前配方的段级自定义高度到路径对象
                    self._apply_segment_heights_to_paths(paths_groups)
                    # Store in cache (including product name for mixed mode)
                    current_product = getattr(self.mw.halcon_product_manager, 'current_product', None)
                    vision_cache[cache_key] = (paths_groups, target_x, target_y, current_product)
                    
                # 扫描后抬回相机移动安全高度（为下一个模具的 XY 移动做准备）
                self.mw.machine.move_to(z=cam_scan_safe_z, speed=self.mw.feedrate_z)
                ops_finished += 1
                
        self.log_signal.emit("✅ 阶段1扫描完成，准备开始点胶...")

        # --- 重新统计所有模具中实际存在的颜色 ---
        actual_colors = set()
        for key, cached_data in vision_cache.items():
            if cached_data:
                paths_groups, _, _, _ = cached_data
                for group in paths_groups:
                    if hasattr(group, 'color'):
                        actual_colors.add(group.color)
        
        if actual_colors:
            sorted_colors = sorted(list(actual_colors))
            self.log_signal.emit(f"📊 扫描汇总：实际需要点胶的颜色共 {len(sorted_colors)} 种。")
            # 动态调整进度条总量
            total_dispense_ops = total_molds * len(sorted_colors)
            total_ops = total_scan_ops + total_dispense_ops
            global_total_steps = total_ops * RESOLUTION

        # ========================================================
        # Phase 2: Dispense by Color Loop
        # ========================================================
        for color_pos, color_idx in enumerate(sorted_colors):
            color_name = self._color_name(color_idx)
            self.log_signal.emit(f"🎨 阶段2：正在处理颜色: {color_name} ...")
            
            # --- Inner Loop: Grid ---
            for r in range(rows):
                for c in range(cols):
                    if not self.running:
                        self.finished_signal.emit(False, "用户停止")
                        return

                    while self.paused:
                        if not self.running: return
                        self.msleep(100)

                    current_base_progress = ops_finished * RESOLUTION
                    self.progress_signal.emit(current_base_progress, global_total_steps)

                    cache_key = (r, c)
                    cached_data = vision_cache.get(cache_key)
                    
                    if not cached_data:
                        ops_finished += 1
                        continue
                        
                    # Unpack cache (now includes product name)
                    if len(cached_data) == 4:
                        paths_groups, cached_x, cached_y, cached_product = cached_data
                    else:
                        # Backward compatibility (old cache format)
                        paths_groups, cached_x, cached_y = cached_data
                        cached_product = None
                    
                    current_x = cached_x
                    current_y = cached_y
                    
                    # [NEW] In Mixed Mode, reload product and update path visualization
                    is_mixed = self.cfg.get('mixed_mode', False)
                    if is_mixed and cached_product:
                        # Check if we need to switch products
                        current_loaded = getattr(self.mw.halcon_product_manager, 'current_product', None)
                        if current_loaded != cached_product:
                            # Load cached product
                            ok, msg = self.mw.halcon_product_manager.load_product(cached_product)
                            if ok:
                                self.mw._compute_halcon_template()
                                
                        # Always update path visualization when returning to a position
                        # Use cached paths_groups for visualization
                        if paths_groups:
                            self.path_update_signal.emit(paths_groups)

                    if not paths_groups:
                        ops_finished += 1
                        continue

                    # Filter paths for CURRENT COLOR
                    filtered_groups = []
                    has_work = False
                    for prod_paths in paths_groups:
                        prod_filtered = []
                        for p in prod_paths:
                            p_color = getattr(p, 'color', 0)
                            if p_color == color_idx:
                                prod_filtered.append(p)
                        if prod_filtered:
                            filtered_groups.append(prod_filtered)
                            has_work = True
                    
                    if not has_work:
                        # No paths for this color in this mold
                        ops_finished += 1
                        continue

                    self.log_signal.emit(f"💉 (R{r}, C{c}) {color_name} 出胶...")
                    
                    # [DEBUG]
                    debug_colors = []
                    for fg in filtered_groups:
                        for p in fg:
                            debug_colors.append(getattr(p, 'color', 'N/A'))
                    self.log_signal.emit(f"[DEBUG] CIdx={color_idx} R{r}C{c} | Filtered Colors: {debug_colors}")

                    try:
                        # Generate G-code ONLY for this color
                        gcode_lines = self.generate_dispense_gcode(filtered_groups, current_x, current_y, target_color=color_idx)
                        
                        if gcode_lines:
                             # Send G-code, Execute...
                            full_gcode_str = "\n".join(gcode_lines)
                            self.gcode_signal.emit(full_gcode_str)

                            total_lines = len(gcode_lines)
                            self.progress_signal.emit(0, total_lines) 
                            
                            for i, line in enumerate(gcode_lines):
                                self.current_line_signal.emit(line)
                                self.current_index_signal.emit(i)
                                
                                current_mold_progress = int((i + 1) / total_lines * RESOLUTION)
                                global_current = (ops_finished * RESOLUTION) + current_mold_progress
                                self.progress_signal.emit(global_current, global_total_steps)
                                
                                if not self.running: return
                                while self.paused: 
                                    if not self.running: return
                                    self.msleep(100)

                                self.mw.machine._post_gcode(line)
                            
                            self.mw.machine.wait_until_idle()
                    except Exception as e:
                        self.log_signal.emit(f"❌ 执行出错: {e}")
                        self._safe_lift_z()

                    ops_finished += 1
            
            # End of Grid Loop (One Color Finished)
            self._safe_lift_z()
            self.log_signal.emit(f"✅ {color_name} 批次完成")

            # Optional color-change checkpoint for both nozzle and camera modes.
            need_checkpoint = (
                bool(self.cfg.get('enable_color_change_pause', True))
                and color_pos < (len(sorted_colors) - 1)
            )
            if need_checkpoint:
                next_color_name = self._color_name(sorted_colors[color_pos + 1])
                if not self._await_color_change_confirmation(color_name, next_color_name):
                    self.finished_signal.emit(False, "用户停止")
                    return

        # 结束 - 返回起始点
        self.log_signal.emit("🔙 全部完成，返回起始点...")
        self.progress_signal.emit(global_total_steps, global_total_steps)
        
        if self.mode == 'camera_dry':
            final_safe_z = float(self.cfg['cam_safe_z_product'])
        else:
            final_safe_z = float(self.cfg['nozzle_safe_z_product'])
        self.mw.machine.move_to(z=final_safe_z, speed=self.mw.feedrate_z)
        self.mw.machine.move_to(x=start_x, y=start_y, speed=self.mw.feedrate_xy)
        self.mw.machine.wait_until_idle()
        
        # ====== ✅ 修改：返回到视觉最佳高度，而不是起始Z ======
        vision_z = getattr(self.mw, 'vision_optimal_z', start_z)
        self.log_signal.emit(f"📷 返回视觉最佳高度 Z={vision_z:.2f}")
        self.mw.machine.move_to(z=vision_z, speed=self.mw.feedrate_z)
        # ====================================================

        self.finished_signal.emit(True, f"{title}完成")


    def _safe_lift_z(self):
        """辅助函数：抬升到移动安全高度"""
        if self.mode == 'camera_dry':
            safe_z_move = float(self.cfg['cam_safe_z_product'])
        else:
            safe_z_move = float(self.cfg['nozzle_safe_z_product'])
        self.mw.machine.move_to(z=safe_z_move, speed=self.mw.feedrate_z)


    def generate_dispense_gcode(self, paths_groups, current_x, current_y, target_color=None):
        """
        根据识别到的像素路径组，生成绝对坐标 G-code
        paths_groups: List[List[Path]]，每个内部List代表一个产品的多条路径
        target_color: strictly filter for this color index if provided (Int)
        """
        M = self.mw.cam2mach_matrix
        if M is None:
            raise Exception("未标定坐标系")

        h_img, w_img = self.mw.last_frame.shape[:2]
        center_x, center_y = w_img / 2, h_img / 2
        
        # 根据模式选择参数（始终从主界面/当前配方读取，支持混合模式切换产品）
        if self.mode == 'camera_dry':
            # 相机空跑：无偏移，用相机高度
            off_x, off_y = 0.0, 0.0
            safe_z_cavity  = float(self.mw.cam_safe_z_cavity  or 0.0)
            safe_z_product = float(self.mw.cam_safe_z_product or 0.0)
            work_z         = float(self.mw.cam_work_z         or 0.0)
            valve_on = False
        else:
            # 针头空跑 / 真实点胶：有偏移，用针头高度
            off_x = self.mw.nozzle_offset_x
            off_y = self.mw.nozzle_offset_y
            safe_z_cavity  = float(self.mw.safe_z_cavity  or 0.0)
            safe_z_product = float(self.mw.safe_z_product or 0.0)
            work_z         = float(self.mw.work_z         or 0.0)
            # [FIX] 如果取消了“模拟出胶”，针头模式下也应该真实出胶
            valve_on = not self.cfg.get('disable_valves', False)

        speed_xy = int(self.mw.feedrate_xy)
        speed_z = int(self.mw.feedrate_z)

        lines = []
        lines.append("G90") # 绝对模式
        
        # 先 XY 移动到本模具中心上方（此时仍在相机扫描安全高度，最高最安全）
        # 再下探到产品安全高度 —— 保证横向移动时 Z 不会碰到模具
        mold_center_x = current_x + off_x
        mold_center_y = current_y + off_y
        lines.append(f"G0 X{mold_center_x:.3f} Y{mold_center_y:.3f} F{speed_xy}  ; XY 先到模具中心上方")
        lines.append(f"G1 Z{safe_z_product:.2f} F{speed_z}  ; 到位后再下探至产品安全高度")


        # [Fix] Initialize state
        active_color_valve = None

        for prod_idx, product_paths in enumerate(paths_groups):
            lines.append(f"; --- Product {prod_idx + 1} ---")
            
            for seg_idx, path_obj in enumerate(product_paths):
                # 统一获取数据 (兼容 PathSegment 对象和旧列表)
                if hasattr(path_obj, 'points'):
                    # PathSegment 对象
                    path_points = path_obj.points
                    seg_type = path_obj.type
                    seg_color = path_obj.color
                    # color_name = path_obj.get_color_name()
                    
                    # [MOD] Prioritize individual dwell, fallback to 100ms default
                    val = getattr(path_obj, 'dwell_ms', 0)
                    dwell_ms = val if val > 0 else 100
                else:
                    # 旧格式 (List or Array)
                    path_points = path_obj
                    seg_type = 'line' if len(path_points) > 1 else 'dot'
                    seg_color = 0
                    seg_color = 0
                    dwell_ms = 100 # Default

                # [CRITICAL] Double-check Color (Redundancy)
                if target_color is not None:
                     if int(seg_color) != int(target_color):
                         continue


                # 读取段级自定义高度（已由 _apply_segment_heights_to_paths 写入到段对象）
                if self.mode == 'camera_dry':
                    # 相机空跑模式下，强制使用相机高度，忽略段级自定义高度
                    seg_safe_z_product = safe_z_product
                    seg_safe_z_cavity  = safe_z_cavity
                    seg_work_z         = work_z
                else:
                    seg_safe_z_product = getattr(path_obj, 'custom_safe_z_product', safe_z_product)
                    seg_safe_z_cavity  = getattr(path_obj, 'custom_safe_z_cavity',  safe_z_cavity)
                    seg_work_z         = getattr(path_obj, 'custom_work_z',         work_z)

                # Color/Valve Change Comments
                lines.append(f"; Segment {seg_idx+1}: Type={seg_type}, Color={seg_color}, Dwell={dwell_ms}ms, Z=[product={seg_safe_z_product:.2f}, cavity={seg_safe_z_cavity:.2f}, work={seg_work_z:.2f}]")
                
                # Prepare Valve Commands
                # 手动换色模式：固定使用 A1_VALVE，不按颜色 index 切换阀门
                # 颜色顺序由用户手动换料保证；软件始终控制 A1_VALVE + VALVE_A_CTRL
                target_valve = "A1_VALVE"

                # --- Valve State Logic ---
                # Check if we need to switch color valve
                if active_color_valve != target_valve:
                    # Close previous if open
                    if active_color_valve is not None:
                         # [MOD] Simulation Check
                        is_sim = self.cfg.get('disable_valves', False)
                        if not is_sim:
                             lines.append(f"SET_PIN PIN={active_color_valve} VALUE=0")
                        else:
                             lines.append(f"; [Tag] {active_color_valve} (OFF) - Switch Color")
                        
                        if self.mode == 'real':
                             pass
                    
                    # Open new
                    # [MOD] Simulation Check
                    is_sim = self.cfg.get('disable_valves', False)
                    color_delay = self.cfg.get('color_delay', 500)
                    
                    if not is_sim:
                        lines.append(f"SET_PIN PIN={target_valve} VALUE=1")
                        if color_delay > 0:
                            lines.append(f"G4 P{color_delay}")
                    else:
                         lines.append(f"; [Tag] {target_valve} (ON)")
                         if color_delay > 0:
                            lines.append(f"; [Tag] G4 P{color_delay} (Supply Delay)")
                    
                    active_color_valve = target_valve
                
                # Master Valve A Control (Always toggle per segment)
                master_pin = "VALVE_A_CTRL"

                # Helper to append commands
                def append_master_cmd(is_on):
                    val = 1 if is_on else 0
                    action = "ON" if is_on else "OFF"
                    # [MOD] Simulation Check
                    is_sim = self.cfg.get('disable_valves', False)
                    
                    if not is_sim:
                        lines.append(f"SET_PIN PIN={master_pin} VALUE={val}")
                    else:
                        lines.append(f"; [Tag] {master_pin} ({action})")

                first_pt = True
                for (px, py) in path_points:
                    # 像素 -> 相对 mm
                    dx_pix = px - center_x
                    dy_pix = py - center_y
                    
                    dx_mm = M[0, 0] * dx_pix + M[0, 1] * dy_pix
                    dy_mm = M[1, 0] * dx_pix + M[1, 1] * dy_pix
                    
                    # 目标机器坐标
                    target_mach_x = current_x + dx_mm + off_x
                    target_mach_y = current_y + dy_mm + off_y
                    
                    if first_pt:
                        # ① XY 移动到起点上方（快速，Z 不变）
                        lines.append(f"G0 X{target_mach_x:.3f} Y{target_mach_y:.3f} F{speed_xy}")
                        
                        # ② 三级下降（每个产品第一段完整三级，后续段直接工作高度）
                        #    产品第一段: safe_z_product(已在) → safe_z_cavity → work_z
                        #    后续段:     safe_z_cavity(已在) → work_z
                        if seg_idx == 0:
                            lines.append(f"G1 Z{seg_safe_z_cavity:.2f} F{speed_z}  ; ②-1 降至腔内安全高度")
                        lines.append(f"G1 Z{seg_work_z:.2f} F{speed_z}  ; ②-2 降至工作高度")
                        
                        # ③ 开阀逻辑
                        if valve_on:
                            lines.append(f"; Open Master Valve A")
                            append_master_cmd(True)
                            if dwell_ms > 0:
                                lines.append(f"G4 P{dwell_ms}")
                            
                        first_pt = False

                    else:
                        # 移动到后续点 - 使用画线速度
                        line_speed = self.cfg.get('line_speed_xy', speed_xy)
                        lines.append(f"G1 X{target_mach_x:.3f} Y{target_mach_y:.3f} F{line_speed}")
                
                # [FIX] Add end dwell before closing valve
                if valve_on and dwell_ms > 0:
                    lines.append(f"; End point dwell")
                    lines.append(f"G4 P{dwell_ms}")  # 终点延时与起点相同
                
                # 一条路径段结束：关阀
                if valve_on:
                    lines.append(f"; Close Master Valve A")
                    append_master_cmd(False)
                
                # ✅ 段结束：抬到腔内安全Z
                lines.append(f"G1 Z{seg_safe_z_cavity:.2f} F{speed_z}")
            
            # ✅ Product完成：升到产品安全Z（取当前产品最后一段的自定义产品安全高度）
            last_seg_safe_z_product = safe_z_product
            if product_paths:
                last_seg = product_paths[-1]
                if self.mode == 'camera_dry':
                    last_seg_safe_z_product = safe_z_product
                else:
                    last_seg_safe_z_product = getattr(last_seg, 'custom_safe_z_product', safe_z_product)
            lines.append(f"G1 Z{last_seg_safe_z_product:.2f} F{speed_z}")
        
        # End of Processing: Close Color Valve if open
        if active_color_valve is not None:
            lines.append(f"; [Tag] {active_color_valve} (OFF) - End Batch Item")
            is_sim = self.cfg.get('disable_valves', False)
            if not is_sim:
                lines.append(f"SET_PIN PIN={active_color_valve} VALUE=0")
            active_color_valve = None

        # 最后确保在最高安全高度
        lines.append(f"G1 Z{safe_z_product:.2f} F{speed_z}")
        
        return lines

    def stop(self):
        self.running = False

    def _apply_segment_heights_to_paths(self, paths_groups):
        """从当前配方的 config.json 加载 segment_heights，
        并将自定义高度属性写入各路径段对象。
        这样在后续按颜色过滤后，段对象仍然携带正确的高度信息。"""
        segment_heights = {}
        try:
            pm = getattr(self.mw, 'halcon_product_manager', None)
            current_product = getattr(pm, 'current_product', None)
            if current_product and pm and hasattr(pm, 'products'):
                prod_dir = pm.products.get_product_path(current_product)
                if prod_dir:
                    config_path = os.path.join(prod_dir, 'config.json')
                    if os.path.exists(config_path):
                        with open(config_path, 'r', encoding='utf-8') as f:
                            cfg = json.load(f)
                        segment_heights = cfg.get('segment_heights', {})
                        if segment_heights:
                            self.log_signal.emit(
                                f"[Batch] 已加载配方段高度微调: {len(segment_heights)} 个段已自定义")
        except Exception as e:
            self.log_signal.emit(f"[WARN] 无法加载段高度配置: {e}")

        global_safe_z_cavity  = float(self.mw.safe_z_cavity  or 0.0)
        global_safe_z_product = float(self.mw.safe_z_product or 0.0)
        global_work_z         = float(self.mw.work_z         or 0.0)

        for p_idx, prod_paths in enumerate(paths_groups):
            for s_idx, seg in enumerate(prod_paths):
                key = f"prod_{p_idx}_seg_{s_idx}"
                custom = segment_heights.get(key, {})
                if custom:
                    seg.has_custom_height = True
                    seg.custom_safe_z_product = custom.get('safe_z_product', global_safe_z_product)
                    seg.custom_safe_z_cavity  = custom.get('safe_z_cavity',  global_safe_z_cavity)
                    seg.custom_work_z         = custom.get('work_z',         global_work_z)
                else:
                    seg.has_custom_height     = False
                    seg.custom_safe_z_product = global_safe_z_product
                    seg.custom_safe_z_cavity  = global_safe_z_cavity
                    seg.custom_work_z         = global_work_z

class BatchRunDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.setWindowTitle("批量运行 (Batch Run)")
        self.resize(1200, 750) # 增加宽度和高度以适应左右布局
        
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        main_layout = QVBoxLayout()
        h_split = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        mid_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # --- Left: 运行日志 ---
        left_layout.addWidget(QLabel("运行日志:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumWidth(350) # 强制最小宽度
        left_layout.addWidget(self.log_box)

        # --- Mid: 阵列配置 & 其它设置 ---
        grp_grid = QGroupBox("阵列配置 (Grid Config)")
        form = QFormLayout()

        # 起始点
        self.spn_start_x = QDoubleSpinBox()
        self.spn_start_x.setRange(-2000, 2000)
        self.spn_start_y = QDoubleSpinBox()
        self.spn_start_y.setRange(-2000, 2000)
        
        self.spn_start_z = QDoubleSpinBox() # 新增 Z
        self.spn_start_z.setRange(-2000, 2000)

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
        mid_layout.addWidget(grp_grid)

        # 高度配置
        grp_z = QGroupBox("高度配置 (Z-Heights)  ← 自动同步配方参数，不可手动修改")
        grp_z.setStyleSheet("QGroupBox { color: #F59E0B; }")
        form_z = QFormLayout()

        def _make_readonly_spn():
            spn = QDoubleSpinBox()
            spn.setRange(-500, 500)
            spn.setDecimals(2)
            spn.setReadOnly(True)
            spn.setButtonSymbols(QDoubleSpinBox.NoButtons)
            spn.setStyleSheet("background-color: #2d2d2d; color: #aaaaaa;")
            return spn

        # 相机高度 (只读)
        self.spn_cam_safe_cavity  = _make_readonly_spn()
        self.spn_cam_safe_product = _make_readonly_spn()
        self.spn_cam_work         = _make_readonly_spn()

        h_cam_z = QHBoxLayout()
        h_cam_z.addWidget(QLabel("腔内安全:"))
        h_cam_z.addWidget(self.spn_cam_safe_cavity)
        h_cam_z.addWidget(QLabel("移动安全:"))
        h_cam_z.addWidget(self.spn_cam_safe_product)
        h_cam_z.addWidget(QLabel("工作:"))
        h_cam_z.addWidget(self.spn_cam_work)
        form_z.addRow("相机高度:", h_cam_z)

        # 针头高度 (只读)
        self.spn_nozzle_safe_cavity  = _make_readonly_spn()
        self.spn_nozzle_safe_product = _make_readonly_spn()
        self.spn_nozzle_work         = _make_readonly_spn()

        h_nozzle_z = QHBoxLayout()
        h_nozzle_z.addWidget(QLabel("腔内安全:"))
        h_nozzle_z.addWidget(self.spn_nozzle_safe_cavity)
        h_nozzle_z.addWidget(QLabel("移动安全:"))
        h_nozzle_z.addWidget(self.spn_nozzle_safe_product)
        h_nozzle_z.addWidget(QLabel("工作:"))
        h_nozzle_z.addWidget(self.spn_nozzle_work)
        form_z.addRow("针头高度:", h_nozzle_z)

        grp_z.setLayout(form_z)
        mid_layout.addWidget(grp_z)

        self.update_z_display()

        # 模拟设置
        self.chk_simulate = QCheckBox("模拟出胶 (关闭阀门)")
        self.chk_simulate.setChecked(True) # Default Checked
        self.chk_simulate.setStyleSheet("color: #F59E0B; font-weight: bold;")
        mid_layout.addWidget(self.chk_simulate)

        # 混合模式
        self.chk_mixed_mode = QCheckBox("启用混合模式 (Mixed Mode - Detect All Products)")
        self.chk_mixed_mode.setChecked(False)
        self.chk_mixed_mode.setStyleSheet("color: #3B82F6; font-weight: bold;")
        self.chk_mixed_mode.setToolTip("开启后，每个位置都会尝试匹配所有已训练的产品，\n并自动加载最佳匹配的产品路径。")
        mid_layout.addWidget(self.chk_mixed_mode)

        # Color-change checkpoint
        self.chk_color_change_pause = QCheckBox("颜色切换停靠确认（测试出胶后继续）")
        self.chk_color_change_pause.setChecked(True)
        self.chk_color_change_pause.setStyleSheet("color: #F97316; font-weight: bold;")
        self.chk_color_change_pause.setToolTip("每种颜色完成后，自动停靠到换胶位并弹窗确认。")
        mid_layout.addWidget(self.chk_color_change_pause)

        h_color_park = QHBoxLayout()
        h_color_park.addWidget(QLabel("换胶停靠 X:"))
        self.spn_color_change_x = QDoubleSpinBox()
        self.spn_color_change_x.setRange(-2000, 2000)
        self.spn_color_change_x.setDecimals(3)
        h_color_park.addWidget(self.spn_color_change_x)
        h_color_park.addWidget(QLabel("Y:"))
        self.spn_color_change_y = QDoubleSpinBox()
        self.spn_color_change_y.setRange(-2000, 2000)
        self.spn_color_change_y.setDecimals(3)
        h_color_park.addWidget(self.spn_color_change_y)
        h_color_park.addStretch()
        mid_layout.addLayout(h_color_park)

        # Color Supply Delay
        h_color_delay = QHBoxLayout()
        h_color_delay.addWidget(QLabel("供料延时 (ms):"))
        self.spn_color_delay = QSpinBox()
        self.spn_color_delay.setRange(0, 5000)
        self.spn_color_delay.setValue(500)
        h_color_delay.addWidget(self.spn_color_delay)
        h_color_delay.addStretch()
        mid_layout.addLayout(h_color_delay)

        # Line Speed
        h_line_speed = QHBoxLayout()
        h_line_speed.addWidget(QLabel("画线速度 (mm/min):"))
        self.spn_line_speed = QSpinBox()
        self.spn_line_speed.setRange(100, 5000)
        self.spn_line_speed.setValue(800)
        h_line_speed.addWidget(self.spn_line_speed)
        h_line_speed.addStretch()
        mid_layout.addLayout(h_line_speed)

        mid_layout.addStretch()

        # --- Right: G-code 预览 ---
        right_layout.addWidget(QLabel("G-code 实时预览:"))
        self.gcode_box = QTextEdit()
        self.gcode_box.setReadOnly(True)
        self.gcode_box.setMinimumWidth(350) # 强制最小宽度
        self.gcode_box.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc; font-family: Consolas; font-size: 10pt;")
        right_layout.addWidget(self.gcode_box)

        # Add to split layout
        h_split.addLayout(left_layout, stretch=1)
        h_split.addLayout(mid_layout, stretch=1)
        h_split.addLayout(right_layout, stretch=1)
        
        main_layout.addLayout(h_split)

        # --- 3.5 进度条 ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%") # 显示百分比
        main_layout.addWidget(self.progress_bar)

        # --- 4. 按钮 ---
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("保存配置")
        self.btn_save.clicked.connect(self.save_settings)
        
        # 新增按钮
        self.btn_cam_dry = QPushButton("相机批量")
        self.btn_cam_dry.clicked.connect(lambda: self.on_start('camera_dry'))
        
        self.btn_nozzle_dry = QPushButton("针头批量")
        self.btn_nozzle_dry.clicked.connect(lambda: self.on_start('nozzle_dry'))

        self.btn_test_valve = QPushButton("测试出胶")
        self.btn_test_valve.setStyleSheet("background-color: #8B5CF6; color: white;") # Purple
        self.btn_test_valve.clicked.connect(self.on_test_dispense)

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
        btn_layout.addWidget(self.btn_test_valve) # [NEW]
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_resume)
        btn_layout.addWidget(self.btn_stop)
        
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    def on_teach_start(self):
        x, y, z = self.mw.machine.get_position()
        self.spn_start_x.setValue(x)
        self.spn_start_y.setValue(y)
        self.spn_start_z.setValue(z) 
        # V3.1 default: keep color-change park point aligned with taught start.
        self.spn_color_change_x.setValue(x)
        self.spn_color_change_y.setValue(y)
        self.log(f"已示教起始点: X{x:.2f}, Y{y:.2f}, Z{z:.2f}")
        # 立即保存
        self.save_settings(silent=True)

    def update_z_display(self):
        """从主界面（当前配方）同步 Z 高度到只读展示框"""
        mw = self.mw
        self.spn_cam_safe_cavity.setValue(mw.cam_safe_z_cavity  or 0.0)
        self.spn_cam_safe_product.setValue(mw.cam_safe_z_product or 0.0)
        self.spn_cam_work.setValue(mw.cam_work_z             or 0.0)
        self.spn_nozzle_safe_cavity.setValue(mw.safe_z_cavity    or 0.0)
        self.spn_nozzle_safe_product.setValue(mw.safe_z_product   or 0.0)
        self.spn_nozzle_work.setValue(mw.work_z               or 0.0)

    def get_config(self):
        return {
            'start_x': self.spn_start_x.value(),
            'start_y': self.spn_start_y.value(),
            'start_z': self.spn_start_z.value(), 
            'rows': self.spn_rows.value(),
            'cols': self.spn_cols.value(),
            'space_x': self.spn_space_x.value(),
            'space_y': self.spn_space_y.value(),
            # 高度参数：始终从主界面（当前配方）读取，而非 spinbox
            'cam_safe_z_cavity':  self.mw.cam_safe_z_cavity  or 0.0,
            'cam_safe_z_product': self.mw.cam_safe_z_product or 0.0,
            'cam_work_z':         self.mw.cam_work_z         or 0.0,
            'nozzle_safe_z_cavity':  self.mw.safe_z_cavity   or 0.0,
            'nozzle_safe_z_product': self.mw.safe_z_product  or 0.0,
            'nozzle_work_z':         self.mw.work_z          or 0.0,
            # [NEW] Simulation
            'disable_valves': self.chk_simulate.isChecked(),
            'mixed_mode': self.chk_mixed_mode.isChecked(),
            'enable_color_change_pause': self.chk_color_change_pause.isChecked(),
            'color_change_park_x': self.spn_color_change_x.value(),
            'color_change_park_y': self.spn_color_change_y.value(),
            'color_delay': self.spn_color_delay.value(),
            'line_speed_xy': self.spn_line_speed.value()
        }

    def on_start(self, mode):
        # 自动保存配置 (符合用户直觉：运行即保存)
        self.save_settings(silent=True)
        
        cfg = self.get_config()
        
        # ====== 安全检查 ======
        # 0. 检查参数是否未设置 (为0)
        safe_z = cfg['cam_safe_z_cavity'] if 'camera' in mode else cfg['nozzle_safe_z_cavity']
        if safe_z == 0.0:
             QMessageBox.warning(self, "参数未设置", "检测到安全高度为 0！\n\n请先设置【安全高度】，否则可能会撞车。")
             return

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
        self.worker.path_update_signal.connect(self.mw.on_batch_path_update)  # [NEW] Dynamic path visualization
        self.worker.color_change_request_signal.connect(self.on_color_change_request)  # [NEW V3.1]
        
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

    # [NEW] Check Dispense
    def on_test_dispense(self):
        """调用主界面的测试出胶功能"""
        if hasattr(self.mw, 'on_test_valve'):
            # self.hide() # 可选：是否隐藏对话框？如果不隐藏，可以在对话框里看日志（如果日志同步）
            # 测试出胶通常很快，不一定要隐藏
            self.log("正在执行测试出胶...")
            try:
                self.mw.on_test_valve()
                self.log("✅ 测试出胶指令已发送")
            except Exception as e:
                self.log(f"❌ 测试出胶失败: {e}")
        else:
            QMessageBox.warning(self, "错误", "主窗口未找到测试出胶功能")

    def on_color_change_request(self, finished_color, next_color, park_x, park_y):
        """
        V3.1: Called by worker at color checkpoints.
        Shows a modal dialog with:
        - 测试出胶
        - 继续执行
        - 停止任务
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("颜色切换确认")
        box.setText(
            f"{finished_color} 已完成。\n"
            f"当前停靠位: X={park_x:.2f}, Y={park_y:.2f}\n\n"
            f"请完成换胶/排胶后继续执行 {next_color}。"
        )
        btn_test = box.addButton("测试出胶", QMessageBox.ActionRole)
        btn_continue = box.addButton("继续执行", QMessageBox.AcceptRole)
        btn_stop = box.addButton("停止任务", QMessageBox.DestructiveRole)
        box.setDefaultButton(btn_continue)

        while True:
            box.exec_()
            clicked = box.clickedButton()

            if clicked == btn_test:
                self.on_test_dispense()
                continue

            if hasattr(self, 'worker'):
                if clicked == btn_continue:
                    self.worker.confirm_color_change_continue()
                else:
                    # Includes explicit stop button and close-window fallback.
                    self.worker.confirm_color_change_stop()
            break

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
        
        # [NEW] 清除路径显示（批量运行结束，已回到原点）
        self.mw.dry_run_paths_to_draw = []
        self.mw.render_current_frame()
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
        从 ConfigManager 加载批量配置，防止文件冲突
        """
        try:
            cfg = self.mw.config
            
            # Helper to get float/int
            def get_float(key, default=0.0):
                val = cfg.get(key, default)
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default

            def get_int(key, default=0):
                val = cfg.get(key, default)
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return default

            # 加载阵列参数
            self.spn_start_x.setValue(get_float('batch_start_x', 0.0))
            self.spn_start_y.setValue(get_float('batch_start_y', 0.0))
            self.spn_start_z.setValue(get_float('batch_start_z', 0.0))
            self.spn_rows.setValue(get_int('batch_rows', 1))
            self.spn_cols.setValue(get_int('batch_cols', 1))
            self.spn_space_x.setValue(get_float('batch_space_x', 0.0))
            self.spn_space_y.setValue(get_float('batch_space_y', 0.0))

            self.spn_color_change_x.setValue(get_float('batch_color_change_x', self.spn_start_x.value()))
            self.spn_color_change_y.setValue(get_float('batch_color_change_y', self.spn_start_y.value()))
            self.chk_color_change_pause.setChecked(bool(get_int('batch_color_change_pause', 1)))
            
            # Z轴参数：始终从主界面（当前配方）读取，不从全局 cfg 读旧展示値
            self.update_z_display()
            
            # [NEW] Line speed
            self.spn_line_speed.setValue(get_int('line_speed_xy', 800))
            
        except Exception as e:
            self.log(f"❌ 加载配置失败: {e}")

    def showEvent(self, event):
        """每次打开对话框时刷新 Z 高度展示（同步当前配方的最新值）"""
        super().showEvent(event)
        self.update_z_display()

    def save_settings(self, silent=False):
        """
        保存配置到 ConfigManager，并持久化到文件
        """
        try:
            cfg = self.mw.config
            
            # 1. 更新阵列参数
            cfg.set('batch_start_x', self.spn_start_x.value())
            cfg.set('batch_start_y', self.spn_start_y.value())
            cfg.set('batch_start_z', self.spn_start_z.value())
            cfg.set('batch_rows', self.spn_rows.value())
            cfg.set('batch_cols', self.spn_cols.value())
            cfg.set('batch_space_x', self.spn_space_x.value())
            cfg.set('batch_space_y', self.spn_space_y.value())
            cfg.set('batch_color_change_x', self.spn_color_change_x.value())
            cfg.set('batch_color_change_y', self.spn_color_change_y.value())
            cfg.set('batch_color_change_pause', 1 if self.chk_color_change_pause.isChecked() else 0)

            # 2. 保存高度参数
            val_csc = self.spn_cam_safe_cavity.value()
            val_csp = self.spn_cam_safe_product.value()
            val_cw = self.spn_cam_work.value()
            
            val_sc = self.spn_nozzle_safe_cavity.value()
            val_sp = self.spn_nozzle_safe_product.value()
            val_w = self.spn_nozzle_work.value()

            cfg.set('cam_safe_z_cavity', val_csc)
            cfg.set('cam_safe_z_product', val_csp)
            cfg.set('cam_work_z', val_cw)
            
            cfg.set('safe_z_cavity', val_sc)
            cfg.set('safe_z_product', val_sp)
            cfg.set('work_z', val_w)

            # 同时更新主界面变量
            self.mw.cam_safe_z_cavity = val_csc
            self.mw.cam_safe_z_product = val_csp
            self.mw.cam_work_z = val_cw
            self.mw.safe_z_cavity = val_sc
            self.mw.safe_z_product = val_sp
            self.mw.work_z = val_w # 这里虽然 self.mw.work_z 也在 ui 上，但最好保持一致

            # [NEW] Line speed
            cfg.set('line_speed_xy', self.spn_line_speed.value())

            # 3. 持久化
            cfg.save_app_settings()

            self.log("✅ 批量阵列配置已保存")
            if not silent:
                QMessageBox.information(self, "保存成功", "批量阵列配置已保存！")
                
        except Exception as e:
            self.log(f"❌ 保存失败: {e}")
            if not silent:
                QMessageBox.warning(self, "保存失败", f"无法保存配置: {e}")

    def closeEvent(self, event):
        """关闭窗口时自动保存"""
        self.save_settings(silent=True)
        event.accept()
