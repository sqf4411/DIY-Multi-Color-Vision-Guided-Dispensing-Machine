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
    finished_signal = pyqtSignal(bool, str)
    progress_signal = pyqtSignal(int, int)  # current, total
    gcode_signal = pyqtSignal(str)          # 新增：发送生成的完整 G-code
    current_line_signal = pyqtSignal(str)   # 新增：发送当前执行的 G-code 行
    current_index_signal = pyqtSignal(int)  # 新增：发送当前执行的 G-code 行号 (用于准确高亮)
    path_update_signal = pyqtSignal(list)   # 新增：发送当前产品的路径数据用于实时显示

    def __init__(self, main_window, grid_config, mode='real'):
        super().__init__()
        self.mw = main_window
        self.cfg = grid_config
        self.mode = mode  # 'real', 'camera_dry', 'nozzle_dry'
        self.running = True
        self.paused = False
        
    def _load_product_params(self, product_name):
        """
        从产品的 config.json 加载 Z-height 参数
        根据 self.mode 决定加载相机参数还是针头参数
        
        Returns:
            dict: {'safe_z_cavity', 'safe_z_product', 'work_z'}
        """
        try:
            # Get product directory
            prod_dir = os.path.join(self.mw.product_base_dir, product_name)
            config_path = os.path.join(prod_dir, "config.json")
            
            if not os.path.exists(config_path):
                self.log_signal.emit(f"⚠️ 产品 {product_name} 的配置文件不存在: {config_path}")
                return None
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if self.mode == 'camera_dry':
                # 加载相机空跑参数
                params = {
                    'safe_z_cavity': config.get('cam_safe_z_cavity', 0.0),
                    'safe_z_product': config.get('cam_safe_z_product', 0.0),
                    'work_z': config.get('cam_work_z', 0.0)
                }
                self.log_signal.emit(f"📂 已加载【{product_name}】的相机参数: {params}")
            else:  # nozzle_dry or real
                # 加载针头空跑参数
                params = {
                    'safe_z_cavity': config.get('safe_z_cavity', 0.0),
                    'safe_z_product': config.get('safe_z_product', 0.0),
                    'work_z': config.get('work_z', 0.0)
                }
                self.log_signal.emit(f"📂 已加载【{product_name}】的针头参数: {params}")
            
            # Validation
            if params['safe_z_cavity'] == 0.0 or params['work_z'] == 0.0:
                self.log_signal.emit(f"⚠️ 警告: 产品 {product_name} 的参数未设置或为0，可能导致撞车！")
                return None
            
            return params
            
        except Exception as e:
            self.log_signal.emit(f"❌ 加载产品参数失败: {e}")
            return None

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
            # Mixed Mode: Assume all colors (0-3) available
            active_colors = {0, 1, 2, 3}
            self.mw.log(f"[Batch] Mixed Mode Enabled: Using all color channels [0, 1, 2, 3]")
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
        
        total_ops = total_molds * len(sorted_colors)
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
        
        self.mw.show_center_crosshair = True
        self.mw.render_current_frame() 

        # Vision Cache: Stores (r, c) -> (paths_groups, current_x, current_y)
        vision_cache = {}

        # 1. 加载当前产品的参数并抬升 Z 轴
        current_product = getattr(self.mw.halcon_product_manager, 'current_product', None)
        if not current_product:
            self.log_signal.emit("❌ 未加载产品，无法执行批量运行")
            self.finished_signal.emit(False, "未加载产品")
            return
        
        # Load Z-height parameters from product config
        params = self._load_product_params(current_product)
        if params is None:
            self.log_signal.emit(f"❌ 产品 [{current_product}] 参数加载失败，请先执行空跑设置参数")
            self.finished_signal.emit(False, "产品参数缺失")
            return
        
        safe_z = params['safe_z_cavity']
        safe_z_product = params['safe_z_product']
        work_z = params['work_z']
        
        self.mw.machine.move_to(z=safe_z, speed=self.mw.feedrate_z)
        self.log_signal.emit(f"🔼 抬升至安全高度 Z={safe_z} [产品: {current_product}]")

        # --- Outer Loop: Colors ---
        for color_idx in sorted_colors:
            color_name = ["红色(A1)", "绿色(A2)", "蓝色(A3)", "黄色(A4)"][color_idx] if 0 <= color_idx < 4 else f"Color {color_idx}"
            self.log_signal.emit(f"🎨 正在处理颜色: {color_name} ...")
            
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

                    # Determine if we have cached data
                    cache_key = (r, c)
                    has_cache = cache_key in vision_cache
                    
                    target_x = start_x + c * space_x
                    target_y = start_y + r * space_y
                    
                    paths_groups = []
                    
                    if not has_cache:
                        # --- First Pass (Vision) ---
                        self.log_signal.emit(f"📍 [{ops_finished + 1}/{total_ops}] (R{r}, C{c}) 首次定位 & 识别...")
                        
                        # [FIX] Set current_x/y for this iteration
                        current_x = target_x
                        current_y = target_y
                        
                        # Move Camera
                        self.mw.machine.move_to(x=target_x, y=target_y, speed=self.mw.feedrate_xy)
                        
                        # Use work_z from product params
                        self.mw.machine.move_to(z=work_z, speed=self.mw.feedrate_z)
                        self.mw.machine.wait_until_idle()
                        time.sleep(0.5) 

                        if self.mw.last_frame is None:
                            self.log_signal.emit("❌ 无法获取图像，跳过")
                            self._safe_lift_z()
                            ops_finished += 1
                            continue

                        img = self.mw.last_frame.copy()
                        img = cv2.flip(img, -1)
                        if len(img.shape) == 3: gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        else: gray_img = img

                        paths_groups, _ = self.mw.get_vision_result(gray_img)
                        
                        if not paths_groups:
                            self.log_signal.emit(f"⚠ 未识别到模具 (R{r}, C{c})")
                            vision_cache[cache_key] = None # Mark as processed but empty
                        else:
                            # Store in cache (including product name for mixed mode)
                            current_product = getattr(self.mw.halcon_product_manager, 'current_product', None)
                            vision_cache[cache_key] = (paths_groups, target_x, target_y, current_product)
                            

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
                                    
                                    # [NEW] 发送路径更新信号，动态显示当前产品的路径
                                    # 重新识别获取当前产品的路径
                                    paths_px_preview, _ = self.mw.get_vision_result(gray_img)
                                    if paths_px_preview:
                                        self.path_update_signal.emit(paths_px_preview)
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
                            # Store in cache (including product name for mixed mode)
                            current_product = getattr(self.mw.halcon_product_manager, 'current_product', None)
                            vision_cache[cache_key] = (paths_groups, target_x, target_y, current_product)
                            
                        # Lift after vision
                        self._safe_lift_z()
                        
                    else:
                        # --- Subsequent Passes (Use Cache) ---
                        cached_data = vision_cache[cache_key]
                        if cached_data is None:
                            # Previous vision failed or empty
                            ops_finished += 1
                            continue
                        
                        # Unpack cache (now includes product name)
                        if len(cached_data) == 4:
                            paths_groups, cached_x, cached_y, cached_product = cached_data
                        else:
                            # Backward compatibility (old cache format)
                            paths_groups, cached_x, cached_y = cached_data
                            cached_product = None
                        
                        # [FIX] Update current_x/y to cached coordinates
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
                                    # Reload parameters for this product
                                    params = self._load_product_params(cached_product)
                                    if params:
                                        safe_z = params['safe_z_cavity']
                                        safe_z_product = params['safe_z_product']
                                        work_z = params['work_z']
                                        self.log_signal.emit(f"🔄 切换到产品 [{cached_product}]，参数已更新")
                                    
                            # Always update path visualization when returning to a position
                            # Use cached paths_groups for visualization
                            if paths_groups:
                                self.path_update_signal.emit(paths_groups)

                    if not paths_groups and has_cache:
                         cached_data = vision_cache[cache_key]
                         if cached_data:
                             # Handle both old (3-tuple) and new (4-tuple) cache formats
                             if len(cached_data) >= 3:
                                 paths_groups = cached_data[0]  # paths_groups is always first element
                    
                    if not paths_groups:
                        ops_finished += 1
                        continue

                    # Filter paths for CURRENT COLOR
                    # paths_groups is List[List[Path]]. Usually 1 product per vision result unless multi-mold in one view.
                    # Flatten or iterate?
                    # The `generate_dispense_gcode` expects `paths_groups`.
                    # We can filter IN PLACE or pass color to generator.
                    # Better: Filter here.
                    
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
                        # [FIX] Use current_x / current_y (which are updated from cache or current target)
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
                            # Lift after dispense (already handled in generate? check safe lift)
                            # self._safe_lift_z() # generate_dispense_gcode ends with safe lift
                    except Exception as e:
                        self.log_signal.emit(f"❌ 执行出错: {e}")
                        self._safe_lift_z()

                    ops_finished += 1
            
            # End of Grid Loop (One Color Finished)
            # Maybe return to Safe Z or Origin?
            self._safe_lift_z()
            self.log_signal.emit(f"✅ {color_name} 批次完成")

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


                # Color/Valve Change Comments
                lines.append(f"; Segment {seg_idx+1}: Type={seg_type}, Color={seg_color}, Dwell={dwell_ms}ms")
                
                # Prepare Valve Commands
                # Default to A1 if no color info
                # Prepare Valve Commands
                # Default to A1 if no color info
                valve_macros = ["A1_VALVE", "A2_VALVE", "A3_VALVE", "A4_VALVE"]
                target_valve = valve_macros[seg_color] if 0 <= seg_color < 4 else "A1_VALVE"

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
                        # 1. 移动到起点上方 (快速 G0)
                        lines.append(f"G0 X{target_mach_x:.3f} Y{target_mach_y:.3f} F{speed_xy}")
                        # 2. 下降到工作高度 (慢速 G1)
                        lines.append(f"G1 Z{work_z:.2f} F{speed_z}")
                        
                        # 3. 开阀逻辑 (如果需要出胶)
                        if valve_on:
                            lines.append(f"; Open Master Valve A")
                            append_master_cmd(True)
                            # [FIX] 无论点还是线，都先延时等待压力建立
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
                lines.append(f"G1 Z{safe_z_cavity:.2f} F{speed_z}")
            
            # ✅ Product完成：升到产品安全Z
            lines.append(f"G1 Z{safe_z_product:.2f} F{speed_z}")
        
        # End of Processing: Close Color Valve if open
        if active_color_valve is not None:
            lines.append(f"; [Tag] {active_color_valve} (OFF) - End Batch Item")
            if self.mode == 'real':
                lines.append(active_color_valve)
            active_color_valve = None

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
        layout.addWidget(grp_grid)



        # --- 2.5 激光测量参考 (Laser Measurement Reference) ---
        grp_laser = QGroupBox("激光测量参考 (Laser Reference)")
        laser_layout = QVBoxLayout()
        
        # Check if laser data exists
        laser_data = getattr(self.mw, 'laser_depth_map', None)
        z_base = getattr(self.mw, 'z_base', 0.0)
        
        if laser_data and len(laser_data) > 0:
            # laser_depth_map is a dictionary: {(x,y): depth}
            try:
                depths = list(laser_data.values())
                if depths:
                    avg_depth = sum(depths) / len(depths)
                    min_depth = min(depths)
                    max_depth = max(depths)
                    avg_z = z_base + avg_depth
                    
                    info_text = f"""<strong>测量数据:</strong><br>
测量点数: {len(depths)} 个<br>
均Z: {avg_z:.2f} mm<br>
均深: {avg_depth:.2f} mm<br>
深度范围: {min_depth:.3f} ~ {max_depth:.3f} mm<br>
<br>
<span style='color: #10B981;'>✓ 激光数据可用，可作为高度参考</span>"""
                    
                    lbl_laser_info = QLabel(info_text)
                    lbl_laser_info.setStyleSheet("padding: 10px; background-color: #1a1a1a; border: 2px solid #10B981; border-radius: 4px; color: #00FF00;")
                    laser_layout.addWidget(lbl_laser_info)
                else:
                    lbl_no_data = QLabel("⚠️ 未找到有效的激光测量数据")
                    lbl_no_data.setStyleSheet("color: #FFA500; padding: 10px; background-color: #2a2a2a; border: 2px solid #FFA500; border-radius: 4px;")
                    laser_layout.addWidget(lbl_no_data)
            except Exception as e:
                lbl_error = QLabel(f"❌ 读取激光数据失败: {e}")
                lbl_error.setStyleSheet("color: #FF4444; padding: 10px; background-color: #2a2a2a; border: 2px solid #FF4444; border-radius: 4px;")
                laser_layout.addWidget(lbl_error)
        else:
            lbl_no_data = QLabel("❌ 未进行激光测量\n\n请在主界面中执行激光测量后再使用针头批量功能")
            lbl_no_data.setStyleSheet("color: #FF6B6B; padding: 10px; background-color: #2a2a2a; border: 2px solid #FF6B6B; border-radius: 4px;")
            laser_layout.addWidget(lbl_no_data)
        
        grp_laser.setLayout(laser_layout)
        layout.addWidget(grp_laser)

        # --- 3. 运行日志 ---
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        # --- 3.1 模拟设置 (New) ---
        self.chk_simulate = QCheckBox("模拟出胶 (关闭阀门)")
        self.chk_simulate.setChecked(True) # Default Checked
        self.chk_simulate.setStyleSheet("color: #F59E0B; font-weight: bold;")
        layout.addWidget(self.chk_simulate)

        # --- 3.2 混合模式 (New) ---
        self.chk_mixed_mode = QCheckBox("启用混合模式 (Mixed Mode - Detect All Products)")
        self.chk_mixed_mode.setChecked(False)
        self.chk_mixed_mode.setStyleSheet("color: #3B82F6; font-weight: bold;")
        self.chk_mixed_mode.setToolTip("开启后，每个位置都会尝试匹配所有已训练的产品，\n并自动加载最佳匹配的产品路径。")
        layout.addWidget(self.chk_mixed_mode)

        # [NEW] Color Supply Delay
        h_color_delay = QHBoxLayout()
        h_color_delay.addWidget(QLabel("供料延时 (ms):"))
        self.spn_color_delay = QSpinBox()
        self.spn_color_delay.setRange(0, 5000)
        self.spn_color_delay.setValue(500)
        h_color_delay.addWidget(self.spn_color_delay)
        h_color_delay.addStretch()
        layout.addLayout(h_color_delay)

        # [NEW] Line Speed
        h_line_speed = QHBoxLayout()
        h_line_speed.addWidget(QLabel("画线速度 (mm/min):"))
        self.spn_line_speed = QSpinBox()
        self.spn_line_speed.setRange(100, 5000)
        self.spn_line_speed.setValue(800)  # Default 800mm/min
        h_line_speed.addWidget(self.spn_line_speed)
        h_line_speed.addStretch()
        layout.addLayout(h_line_speed)

        # --- 3.5 进度条 ---
        self.progress_bar = QProgressBar()
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
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def on_teach_start(self):
        x, y, z = self.mw.machine.get_position()
        self.spn_start_x.setValue(x)
        self.spn_start_y.setValue(y)
        self.spn_start_z.setValue(z) 
        self.log(f"已示教起始点: X{x:.2f}, Y{y:.2f}, Z{z:.2f}")
        # 立即保存
        self.save_settings(silent=True)

    def get_config(self):
        return {
            'start_x': self.spn_start_x.value(),
            'start_y': self.spn_start_y.value(),
            'start_z': self.spn_start_z.value(),
            'rows': self.spn_rows.value(),
            'cols': self.spn_cols.value(),
            'space_x': self.spn_space_x.value(),
            'space_y': self.spn_space_y.value(),
            # Z-height parameters removed - will be loaded from product config.json
            'disable_valves': self.chk_simulate.isChecked(),
            'mixed_mode': self.chk_mixed_mode.isChecked(),
            'color_delay': self.spn_color_delay.value(),
            'line_speed_xy': self.spn_line_speed.value()
        }

    def on_start(self, mode):
        # 自动保存配置 (符合用户直觉：运行即保存)
        self.save_settings(silent=True)
        
        cfg = self.get_config()
        
        # Note: Z-height parameters will be loaded from product config.json
        # No safety checks needed here as they will be validated when loading
        
        self.worker = BatchWorker(self.mw, cfg, mode)
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.gcode_signal.connect(self.update_gcode_display)
        self.worker.current_index_signal.connect(self.highlight_current_line)
        self.worker.path_update_signal.connect(self.mw.on_batch_path_update)  # [NEW] Dynamic path visualization
        
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
                try: return float(val)
                except: return default

            def get_int(key, default=0):
                val = cfg.get(key, default)
                try: return int(val)
                except: return default

            # 加载阵列参数
            self.spn_start_x.setValue(get_float('batch_start_x', 0.0))
            self.spn_start_y.setValue(get_float('batch_start_y', 0.0))
            self.spn_start_z.setValue(get_float('batch_start_z', 0.0))
            self.spn_rows.setValue(get_int('batch_rows', 1))
            self.spn_cols.setValue(get_int('batch_cols', 1))
            self.spn_space_x.setValue(get_float('batch_space_x', 0.0))
            self.spn_space_y.setValue(get_float('batch_space_y', 0.0))
            
            # Z-height parameters removed - loaded from product config.json
            
            # [NEW] Line speed
            self.spn_line_speed.setValue(get_int('line_speed_xy', 800))
            
        except Exception as e:
            self.log(f"❌ 加载配置失败: {e}")

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

            # Z-height parameters removed - loaded from product config.json

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
