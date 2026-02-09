"""
Halcon产品管理器
负责Halcon模式下的产品训练、加载、路径管理和识别逻辑
将业务逻辑从UI层分离，提高代码复用性和可维护性
"""

import os
import cv2
import numpy as np
import shutil
from typing import Optional, Dict, List, Tuple
from core.path_segment import PathSegmentList, PathSegment
import json


class HalconProductManager:
    """Halcon模式专用的产品管理逻辑"""
    
    def __init__(self, halcon_manager, product_manager):
        """
        初始化Halcon产品管理器
        
        Args:
            halcon_manager: HalconManager实例，负责Halcon引擎交互
            product_manager: ProductManager实例，负责产品文件管理
        """
        self.halcon = halcon_manager
        self.products = product_manager
        self.current_product = None
        self.current_paths = {'dots': [], 'polys': []}
        self.model_ref_row = 0
        self.model_ref_col = 0
        self.vision_offset_x = 0.0
        self.vision_offset_y = 0.0
        self.current_segments: Optional[PathSegmentList] = None # [NEW] Store rich segments
        
    # ========== 产品操作 ==========
    
    def train_new_product(self, image_path: str, product_name: str, 
                         config_data: dict) -> Tuple[bool, str]:
        """
        训练新产品模型
        
        Args:
            image_path: 训练图像路径（captures中的原图）
            product_name: 产品名称
            config_data: 初始配置数据
            
        Returns:
            (成功, 消息)
        """
        try:
            # 1. 创建产品目录
            product_dir = self.products.get_product_path(product_name)
            os.makedirs(product_dir, exist_ok=True)
            
            # 2. 保存训练图片（翻转为正向）
            dest_img_path = os.path.join(product_dir, 'reference.png')
            raw_img = cv2.imread(image_path)
            if raw_img is None:
                return False, f"无法读取图像: {image_path}"
            
            # 翻转180度，与路径编辑器的warp图方向一致
            upright_img = cv2.flip(raw_img, -1)
            cv2.imwrite(dest_img_path, upright_img)
            
            # 3. 使用OpenCV检测模具外框
            from core.mold_detector import detect_mold_and_cells
            
            print(f"[HalconProductManager] 检测模具边界...")
            gray = cv2.cvtColor(upright_img, cv2.COLOR_BGR2GRAY)
            _, big_box, _, _ = detect_mold_and_cells(gray)
            
            if big_box is None:
                print(f"[HalconProductManager] ⚠️ 未检测到模具边界，使用整张图")
                mold_boundary = None
            else:
                # big_box是4个角点 (x,y)
                x_coords = big_box[:, 0]
                y_coords = big_box[:, 1]
                mold_boundary = {
                    'x': int(x_coords.min()),
                    'y': int(y_coords.min()),
                    'width': int(x_coords.max() - x_coords.min()),
                    'height': int(y_coords.max() - y_coords.min())
                }
                print(f"[HalconProductManager] ✅ 检测到模具边界: {mold_boundary}")
            
            # 4. 弹出ROI选择对话框（在模具边界内选择）
            from core.simple_roi_dialog import SimpleROIDialog
            from PyQt5.QtWidgets import QDialog
            
            print(f"[HalconProductManager] 弹出ROI选择对话框（请框选单个产品）...")
            dlg = SimpleROIDialog(dest_img_path, mold_boundary)
            if dlg.exec_() == QDialog.Accepted:
                roi_rect = dlg.get_roi_rect()  # QRect格式
                if roi_rect is None:
                    return False, "未选择有效的ROI区域"
                print(f"[HalconProductManager] ROI选择成功: ({roi_rect.x()}, {roi_rect.y()}, {roi_rect.width()}, {roi_rect.height()})")
            else:
                return False, "用户取消ROI选择"
            
            # 5. 裁剪ROI区域并保存
            x, y, w, h = roi_rect.x(), roi_rect.y(), roi_rect.width(), roi_rect.height()
            roi_image = upright_img[y:y+h, x:x+w]
            roi_img_path = os.path.join(product_dir, 'roi_template.png')
            cv2.imwrite(roi_img_path, roi_image)
            print(f"[HalconProductManager] ✅ ROI图像已保存: {roi_img_path}, 尺寸: {w}x{h}")
            
            # 6. 用裁剪后的ROI图像训练模型（不需要roi参数了！）
            model_path = os.path.join(product_dir, f'{product_name}.shm')
            result = self.halcon.train_model(roi_img_path, model_path, roi=None)
            
            # 解析训练结果
            if isinstance(result, dict):
                success = True
                # origin是相对于裁剪图的坐标，需要转换到reference.png坐标
                origin_row_in_roi = result.get('origin_row', h / 2)
                origin_col_in_roi = result.get('origin_col', w / 2)
                
                # 转换到reference.png的坐标：ROI左上角 + origin在裁剪图中的位置
                origin_row = y + origin_row_in_roi
                origin_col = x + origin_col_in_roi
                
                print(f"[HalconProductManager] 模型Origin在ROI中: ({origin_row_in_roi:.2f}, {origin_col_in_roi:.2f})")
                print(f"[HalconProductManager] 模型Origin在reference.png中: ({origin_row:.2f}, {origin_col:.2f})")
                
            elif isinstance(result, tuple):
                success, msg = result
                # 默认使用ROI中心转换到reference.png坐标
                origin_row, origin_col = y + h/2, x + w/2
                if not success:
                    return False, msg
            else:
                success = result
                origin_row, origin_col = y + h/2, x + w/2
            
            if not success:
                return False, "模型训练失败"
            
            # 7. 更新配置（保存ROI信息用于记录）
            roi_info = {
                'x': x, 'y': y, 'width': w, 'height': h
            }
            config_data['model_file'] = f'{product_name}.shm'
            config_data['model_row'] = origin_row
            config_data['model_col'] = origin_col
            config_data['product_name'] = product_name
            config_data['roi_in_reference'] = roi_info  # 记录ROI在reference.png中的位置
            
            # 8. 保存配置（路径为空）
            success, msg = self.products.save_product(
                product_name,
                config_data=config_data,
                path_points=None,
                path_polys=None
            )
            
            if not success:
                return False, f"保存配置失败: {msg}"
            
            # 9. 加载新产品
            self.current_product = product_name
            self.current_paths = {'dots': [], 'polys': []}
            self.model_ref_row = origin_row
            self.model_ref_col = origin_col
            
            # [FIX] Initialize roi_in_reference in memory immediately
            self.roi_in_reference = roi_info
            
            # 10. 加载模型
            self.halcon.set_model_path(model_path)
            
            return True, f"产品 '{product_name}' 创建成功（使用ROI训练）"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"训练产品失败: {e}"
    
    def load_product(self, product_name: str, force: bool = False) -> Tuple[bool, str]:
        """
        加载产品（模型+路径+参数）
        
        Args:
            product_name: 产品名称
            force:是否强制从磁盘重新加载
            
        Returns:
            (成功, 消息)
        """
        try:
            print(f"[HalconProductManager] Loading product '{product_name}' (Force={force})...")
            
            # 1. 加载配置
            config = self.products.load_product_config(product_name)
            if not config:
                return False, f"配置加载失败: {product_name}"
            
            # 2. 加载模型
            product_dir = self.products.get_product_path(product_name)
            model_file = config.get('model_file', f'{product_name}.shm')
            model_path = os.path.join(product_dir, model_file)
            
            print(f"[DEBUG] product_dir={product_dir}")
            print(f"[DEBUG] model_file={model_file}")
            print(f"[DEBUG] model_path={model_path}")
            print(f"[DEBUG] file exists={os.path.exists(model_path)}")
            
            if os.path.exists(model_path):
                ok = self.halcon.set_model_path(model_path)
                if not ok:
                    return False, f"模型加载失败: {model_file}"
            else:
                return False, f"模型文件不存在: {model_file}"
            
            # 3. 加载路径
            self.current_segments = None  # [FIX] 防止状态泄漏：先清空上一产品的富文本段
            path_data, poly_data = self.products.load_product_paths(product_name)
            
            # [NEW] 尝试加载 multicolor_path.json
            json_path = os.path.join(product_dir, "multicolor_path.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    segments = PathSegmentList.from_dict_list(data.get('segments', []))
                    self.current_segments = segments # [NEW] Store it
                    # 转换回 Legacy 格式供 HalconManager 使用
                    poly_data, path_data = segments.get_legacy_format()
                    # print(f"[HalconProductManager] 已加载多色路径: {len(segments.segments)} 段")
                except Exception as e:
                    print(f"[HalconProductManager] 多色路径加载失败: {e}")
            
            
            self.current_paths['dots'] = path_data if path_data is not None else []
            self.current_paths['polys'] = poly_data if poly_data is not None else []
            
            # 4. 加载参数
            self.model_ref_row = config.get('model_row', 0)
            self.model_ref_col = config.get('model_col', 0)
            self.vision_offset_x = config.get('vision_offset_x', 0.0)
            self.vision_offset_y = config.get('vision_offset_y', 0.0)
            
            # ⭐ 加载 ROI 偏移信息（用于路径坐标转换）
            self.roi_in_reference = config.get('roi_in_reference', None)
            
            # 4.5 ⭐ 应用阀门参数到全局配置
            from pathlib import Path
            from datetime import datetime
            
            valve_params = {
                "version": "1.0",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "parameters": {
                    "prefeed_time": config.get("valve_prefeed_time", 0.10),
                    "valve_a_dispense_time": config.get("valve_a_dispense_time", 0.50),
                    "valve_b_dispense_time": config.get("valve_b_dispense_time", 0.50),
                    "line_speed_xy": config.get("line_speed_xy", 800)  # [NEW] 画线速度
                },
                "notes": f"Loaded from product '{product_name}'"
            }
            
            print(f"[HalconProductManager] 加载产品 '{product_name}' 的点胶参数:")
            print(f"[HalconProductManager]   供料提前: {valve_params['parameters']['prefeed_time']}")
            print(f"[HalconProductManager]   阀A时间: {valve_params['parameters']['valve_a_dispense_time']}")
            print(f"[HalconProductManager]   阀B时间: {valve_params['parameters']['valve_b_dispense_time']}")
            print(f"[HalconProductManager]   画线速度: {valve_params['parameters']['line_speed_xy']} mm/min")
            
            # [CRITICAL Fix for Mixed Mode]
            # Ensure MainWindow state is consistent with the loaded product.
            # BatchWorker calls self.mw._compute_halcon_template() which relies on:
            #   self.mw.current_segments (Priority 1)
            #   self.mw.polys / self.mw.dots (Priority 2)
            # If we don't update them here, MainWindow keeps the OLD product's paths (e.g. 7777),
            # causing 'path leakage'.
            
            if hasattr(self, 'mw') and self.mw:
                 # Update Priority 1: Multicolor Segments
                 if self.current_segments:
                     self.mw.current_segments = self.current_segments
                     
                 # Update Priority 2: Legacy Lists (converted in Step 3 above)
                 self.mw.polys = self.current_paths.get('polys', [])
                 self.mw.dots = self.current_paths.get('dots', [])
                 
                 # Optional: Clear cached template to force re-computation
                 if hasattr(self.mw, 'halcon_path_template'):
                     self.mw.halcon_path_template = None

            # 5. 设置当前产品
            self.current_product = product_name
            
            return True, f"✅ 产品 '{product_name}' 加载成功"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"加载产品失败: {e}"
    
    def save_product(self, config_data: dict) -> Tuple[bool, str]:
        """
        保存当前产品配置
        
        Args:
            config_data: 配置数据
            
        Returns:
            (成功, 消息)
        """
        if not self.current_product:
            return False, "没有加载的产品"
        
        try:
            # 更新配置中的模型参数
            config_data['model_row'] = self.model_ref_row
            config_data['model_col'] = self.model_ref_col
            config_data['product_name'] = self.current_product
            config_data['model_file'] = f'{self.current_product}.shm'
            
            # [FIX] Ensure roi_in_reference is preserved if not present in config_data
            if 'roi_in_reference' not in config_data:
                if hasattr(self, 'roi_in_reference') and self.roi_in_reference:
                    config_data['roi_in_reference'] = self.roi_in_reference
                    print(f"[HalconProductManager] ✅ Preserved roi_in_reference: {self.roi_in_reference}")
                else:
                    # Try to reload from disk if memory is empty (prevent data loss)
                    old_config = self.products.load_product_config(self.current_product)
                    if old_config and 'roi_in_reference' in old_config:
                        config_data['roi_in_reference'] = old_config['roi_in_reference']
                        # Sync back to memory
                        self.roi_in_reference = old_config['roi_in_reference']
                        print(f"[HalconProductManager] ✅ Recovered roi_in_reference from disk: {self.roi_in_reference}")
                    else:
                        print(f"[HalconProductManager] ⚠️ Warning: roi_in_reference missing during save!")

            # 保存配置和路径
            success, msg = self.products.save_product(
                self.current_product,
                config_data=config_data,
                path_points=self.current_paths['dots'],
                path_polys=self.current_paths['polys']
            )
            
            return success, msg
            
        except Exception as e:
            return False, f"保存产品失败: {e}"

    def find_best_match_product(self, image_np: np.ndarray, candidate_products: List[str]) -> Tuple[Optional[str], float]:
        """
        在给定的候选产品列表中寻找最佳匹配。
        返回 (best_product_name, best_score)。如果没有匹配或分数过低，返回 (None, 0.0)。
        """
        best_score = -1.0
        best_product = None
        
        # 备份当前模型路径，以便恢复
        original_model = self.halcon.current_model_path
        
        try:
            for prod in candidate_products:
                # 获取路径
                prod_dir = self.products.get_product_path(prod)
                model_file = f'{prod}.shm'
                # 尝试读取 config 确认文件名 (可选)
                config = self.products.load_product_config(prod)
                if config: 
                    model_file = config.get('model_file', model_file)
                
                model_path = os.path.join(prod_dir, model_file)
                if not os.path.exists(model_path):
                    continue
                    
                # 加载模型
                if not self.halcon.set_model_path(model_path):
                    continue
                    
                # 运行检测
                result = self.halcon.process_image(image_np)
                if result and result.get('success') and result.get('matches'):
                    # [CHANGE] 使用第一个匹配（左上角产品，训练参考）而不是最高分
                    # 第一个匹配最可靠，因为它对应训练时使用的产品
                    first_score = result['matches'][0]['score']
                    print(f"[BestMatch] Candidate '{prod}' First Match Score: {first_score:.2f} (Total: {len(result['matches'])} matches)")
                    if first_score > best_score:
                        best_score = first_score
                        best_product = prod
                else:
                    print(f"[BestMatch] Candidate '{prod}' Score: 0.00 (No Match)")
                        
            return best_product, best_score
            
        finally:
            # 恢复之前的模型
            if original_model and os.path.exists(original_model):
                self.halcon.set_model_path(original_model)
    
    # ========== 路径管理 ==========
    
    def load_paths_from_product(self, product_name: Optional[str] = None):
        """
        从产品目录加载路径
        
        Args:
            product_name: 产品名称，None表示使用当前产品
        """
        name = product_name or self.current_product
        if not name:
            return
        
        try:
            path_data, poly_data = self.products.load_product_paths(name)
            self.current_paths['dots'] = path_data if path_data is not None else []
            self.current_paths['polys'] = poly_data if poly_data is not None else []
        except Exception as e:
            print(f"[HalconProductManager] 加载路径失败: {e}")
    
    def save_paths_to_product(self, dots: list, polys: list):
        """
        保存路径到当前产品
        
        Args:
            dots: 点路径
            polys: 线路径
        """
        if not self.current_product:
            return
        
        self.current_paths['dots'] = dots
        self.current_paths['polys'] = polys
    
    def get_current_paths(self) -> dict:
        """获取当前加载的路径"""
        return self.current_paths.copy()
    
    def get_product_directory(self) -> Optional[str]:
        """获取当前产品目录"""
        if not self.current_product:
            return None
        return self.products.get_product_path(self.current_product)
    
    # ========== 识别处理 ==========
    
    def process_image(self, gray_img: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        处理图像并绘制路径（完整识别逻辑）
        
        Args:
            gray_img: 灰度图像
            
        Returns:
            (found_paths, debug_img)
        """
        # 1. 翻转图像为正向（与训练图和路径编辑器一致）
        # [Fix] UI already passes upright image (from last_frame which is flipped).
        # So check if we really need to flip again. 
        # Usually, input gray_img is already upright.
        gray_img_upright = gray_img
        
        # 2. 运行Halcon识别
        res = self.halcon.process_image(gray_img_upright)
        debug_img = cv2.cvtColor(gray_img_upright, cv2.COLOR_GRAY2BGR)
        found_paths = []
        
        if not res or 'matches' not in res:
            return {'found_paths': [], 'debug_img': debug_img, 'success': False}
        
        # 2. 获取参考点
        ref_row, ref_col = self.model_ref_row, self.model_ref_col
        if ref_row == 0 and ref_col == 0:
            # Fallback to image center (assuming 5MP camera 1944x2592)
            # Center: Row 972, Col 1296
            ref_row, ref_col = 972.0, 1296.0
        
        # 3. 处理每个匹配
        for match in res['matches']:
            cx, cy = match['col'], match['row']  # Halcon: Col=X, Row=Y
            score = match['score']
            angle = match['angle']  # 弧度
            
            # 绘制匹配中心
            cx_i, cy_i = int(cx), int(cy)
            cv2.drawMarker(debug_img, (cx_i, cy_i), (0, 255, 0), 
                          cv2.MARKER_CROSS, 40, 3)
            
            # 转换并绘制路径
            mold_segments = self._transform_and_draw_paths(
                debug_img, cx, cy, angle, ref_row, ref_col
            )
            
            if mold_segments:
                found_paths.append(mold_segments)
                
                # ✅ 新增：保存识别结果到产品配方
                self._save_recognized_coordinates(
                    mold_segments, cx, cy, angle, score
                )
            
            # 绘制分数
            text = f"S:{score:.2f}"
            cv2.putText(debug_img, text, (cx_i + 25, cy_i), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return {
            'found_paths': found_paths,
            'debug_img': debug_img,
            'matches': res['matches'],
            'success': True
        }
    
    def get_current_segments(self):
        """
        Get the current PathSegmentList object (contains colors/metadata).
        Returns: PathSegmentList or None
        """
        return getattr(self, 'current_segments', None)

    def _transform_and_draw_paths(self, debug_img: np.ndarray, 
                                  cx: float, cy: float, angle: float,
                                  ref_row: float, ref_col: float) -> List[PathSegment]:
        """
        转换并绘制路径 (返回 PathSegment 对象列表)
        """
        mold_segments = []
        
        try:
            # 1. 确定源路径 (优先使用 rich segments)
            source_segments = []
            if self.current_segments and self.current_segments.count() > 0:
                source_segments = self.current_segments.get_all()
            else:
                # Legacy fallback (create dummy segments)
                if self.current_paths.get('polys'):
                    for p in self.current_paths['polys']:
                        # p is numpy array (N,2) => Convert to List[Tuple]
                        pts = [tuple(x) for x in p]
                        source_segments.append(PathSegment('line', 0, pts))
                if self.current_paths.get('dots') and len(self.current_paths['dots']) > 0:
                    for p in self.current_paths['dots']:
                        # p is dot (x,y)
                        if isinstance(p, np.ndarray): p = tuple(p)
                        else: p = tuple(p)
                        source_segments.append(PathSegment('dot', 0, [p]))
            
            if not source_segments:
                # Return center point as dummy dot?
                return [PathSegment('dot', 2, [(cx, cy)])] # Color 2=Blue
            
            # 2. 转换每个段
            for seg in source_segments:
                pts = np.array(seg.points, dtype=np.float32)
                if len(pts) == 0: continue
                
                # ⭐ ROI 坐标系转换
                # 路径现在保存在 ROI 坐标系（编辑器使用 roi_template.png）
                # 需要转换：ROI -> Reference (training) -> Relative (相对origin) -> Rotated -> Current Match
                
                # 2.1 从 ROI 坐标转换到 Reference 坐标（训练位置）
                # 获取 ROI 偏移（从 config.json）
                roi_offset_x = 0
                roi_offset_y = 0
                
                # [Fix] 确保 roi_in_reference 已加载
                if not hasattr(self, 'roi_in_reference') or not self.roi_in_reference:
                    print(f"[HalconProductManager] ⚠️ Warning: roi_in_reference missing. Try reloading for '{self.current_product}'...")
                    if self.current_product:
                        cfg = self.products.load_product_config(self.current_product)
                        if cfg:
                            self.roi_in_reference = cfg.get('roi_in_reference')
                            print(f"[HalconProductManager] ✅ Reloaded ROI: {self.roi_in_reference}")
                        else:
                            print(f"[HalconProductManager] ❌ Failed to load config for '{self.current_product}'")
                
                # 强制打印调试信息 (无条件)
                if len(pts) > 0:
                    roi_status = self.roi_in_reference if hasattr(self, 'roi_in_reference') else "MISSING"
                    print(f"[DEBUG] Product={self.current_product}")
                    print(f"[DEBUG] ROI_Info={roi_status}")
                    print(f"[DEBUG] RefPoint=({ref_col}, {ref_row}) Match=({cx}, {cy})")
                    print(f"[DEBUG] Pts[0]={pts[0]}")

                if hasattr(self, 'roi_in_reference') and self.roi_in_reference:
                    roi_offset_x = self.roi_in_reference.get('x', 0)
                    roi_offset_y = self.roi_in_reference.get('y', 0)
                
                # 应用偏移：ROI -> Reference
                pts_in_reference = pts + np.array([roi_offset_x, roi_offset_y])
                
                if len(pts) > 0 and hasattr(self, 'roi_in_reference') and self.roi_in_reference:
                    print(f"[坐标转换] 第1个点(Ref): {pts_in_reference[0]}")
                
                # 2.2 转换为相对于训练原点的坐标
                pts_centered = pts_in_reference - np.array([ref_col, ref_row])
                
                if len(pts) > 0 and hasattr(self, 'roi_in_reference') and self.roi_in_reference:
                    print(f"[坐标转换] 第1个点(相对): {pts_centered[0]}")
                
                # 2.3 旋转
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                x_new = pts_centered[:, 0] * cos_a + pts_centered[:, 1] * sin_a
                y_new = -pts_centered[:, 0] * sin_a + pts_centered[:, 1] * cos_a
                
                pts_rotated = np.stack([x_new, y_new], axis=1)
                
                # 2.4 平移到匹配中心
                pts_transformed = pts_rotated + np.array([cx, cy])
                
                # 2.5 应用视觉偏移
                pts_transformed[:, 0] += self.vision_offset_x
                pts_transformed[:, 1] += self.vision_offset_y
                
                # 2.5 创建新段 (保留原有属性)
                new_pts = [tuple(p) for p in pts_transformed]
                new_seg = PathSegment(seg.type, seg.color, new_pts, seg.dwell_ms)
                
                # [DEBUG] Log segment transformation
                if len(source_segments) <= 5:  # Only log first few segments to avoid spam
                    print(f"[_transform_and_draw_paths DEBUG] Transformed segment: "
                          f"type={seg.type}, color={seg.color}, dwell_ms={seg.dwell_ms}ms → "
                          f"new_seg.dwell_ms={new_seg.dwell_ms}ms")
                
                mold_segments.append(new_seg)
                
                # 2.6 绘制 (使用段的颜色)
                pts_draw = pts_transformed.astype(np.int32)
                color = seg.get_display_color_bgr() # (B,G,R)
                
                if seg.type == 'dot':
                    cv2.circle(debug_img, tuple(pts_draw[0]), 4, color, -1)
                else:
                    cv2.polylines(debug_img, [pts_draw], False, color, 2, cv2.LINE_AA)
                
        except Exception as e:
            print(f"[HalconProductManager] 路径转换失败: {e}")
            import traceback
            traceback.print_exc()
        
        return mold_segments if mold_segments else []
        """
        转换并绘制路径
        
        Args:
            debug_img: 调试图像
            cx, cy: 匹配中心坐标
            angle: 旋转角度（弧度）
            ref_row, ref_col: 参考点
            
        Returns:
            转换后的路径段列表
        """
        mold_segments = []
        
        try:
            # 合并点路径和线路径
            all_paths = []
            
            # 添加线路径
            if self.current_paths['polys']:
                all_paths.extend(self.current_paths['polys'])
            
            # 添加点路径
            if self.current_paths['dots'] and len(self.current_paths['dots']) > 0:
                all_paths.append(self.current_paths['dots'])
            
            if not all_paths:
                # 没有路径，返回中心点
                return [np.array([[cx, cy]])]
            
            # 转换每个路径段
            for pts in all_paths:
                if len(pts) == 0:
                    continue
                
                pts = np.array(pts, dtype=np.float32)
                
                # 1. 居中（相对于ROI参考点）
                pts_centered = pts - np.array([ref_col, ref_row])
                
                # 2. 旋转
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                x_new = pts_centered[:, 0] * cos_a + pts_centered[:, 1] * sin_a
                y_new = -pts_centered[:, 0] * sin_a + pts_centered[:, 1] * cos_a
                pts_rotated = np.stack([x_new, y_new], axis=1)
                
                # 3. 平移到匹配中心
                pts_transformed = pts_rotated + np.array([cx, cy])
                
                # 4. 应用视觉偏移
                pts_transformed[:, 0] += self.vision_offset_x
                pts_transformed[:, 1] += self.vision_offset_y
                
                # 5. 绘制
                pts_draw = pts_transformed.astype(np.int32)
                if len(pts_draw) == 1:
                    # 单点
                    cv2.circle(debug_img, tuple(pts_draw[0]), 4, (255, 0, 255), -1)
                else:
                    # 线段
                    cv2.polylines(debug_img, [pts_draw], False, (255, 0, 255), 4, cv2.LINE_AA)
                
                mold_segments.append(pts_transformed)
                
        except Exception as e:
            print(f"[HalconProductManager] 路径转换失败: {e}")
            import traceback
            traceback.print_exc()
        
        return mold_segments if mold_segments else [np.array([[cx, cy]])]
    
    # ========== 工具方法 ==========
    
    def is_product_loaded(self) -> bool:
        """检查是否有加载的产品"""
        return self.current_product is not None
    
    def get_current_product_name(self) -> Optional[str]:
        """获取当前产品名称"""
        return self.current_product
    
    
    def unload_product(self):
        """卸载当前产品"""
        self.current_product = None
        self.current_paths = {'dots': [], 'polys': []}
        self.model_ref_row = 0
        self.model_ref_col = 0
    
    def _save_recognized_coordinates(self, segments, cx, cy, angle, score):
        """保存识别后的坐标到产品配方"""
        if not self.current_product:
            return  # 没有加载产品
        
        import time
        
        # 准备数据
        recognized_data = {
            "segments": [seg.to_dict() for seg in segments],
            "timestamp": time.time(),
            "transform_data": {
                "cx": float(cx),
                "cy": float(cy),
                "angle": float(angle),
                "score": float(score)
            }
        }
        
        # 读取现有配置
        product_dir = self.products.get_product_path(self.current_product)
        config_path = os.path.join(product_dir, "config.json")
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # 更新配置
            config["recognized_coordinates"] = recognized_data
            
            # 保存
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"[HalconProductManager] ✅ 已保存识别坐标到 {self.current_product} ({len(segments)}个段)")
        except Exception as e:
            print(f"[HalconProductManager] ❌ 保存识别坐标失败: {e}")
