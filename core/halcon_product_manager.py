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
        
    # ========== 产品操作 ==========
    
    def train_new_product(self, image_path: str, product_name: str, 
                         config_data: dict) -> Tuple[bool, str]:
        """
        训练新产品模型
        
        Args:
            image_path: 训练图像路径
            product_name: 产品名称
            config_data: 初始配置数据
            
        Returns:
            (成功, 消息)
        """
        try:
            # 1. 创建产品目录
            product_dir = self.products.get_product_path(product_name)
            os.makedirs(product_dir, exist_ok=True)
            
            # 2. 保存并翻转训练图片
            dest_img_path = os.path.join(product_dir, 'reference.png')
            raw_img = cv2.imread(image_path)
            if raw_img is not None:
                # 翻转180度以匹配相机方向
                upright_img = cv2.flip(raw_img, -1)
                cv2.imwrite(dest_img_path, upright_img)
            else:
                return False, f"无法读取图像: {image_path}"
            
            # 3. 训练Halcon模型
            model_path = os.path.join(product_dir, f'{product_name}.shm')
            result = self.halcon.train_model(dest_img_path, model_path)
            
            # 解析训练结果
            if isinstance(result, dict):
                success = True
                origin_row = result.get('origin_row', 0)
                origin_col = result.get('origin_col', 0)
            elif isinstance(result, tuple):
                success, msg = result
                origin_row, origin_col = 0, 0
                if not success:
                    return False, msg
            else:
                success = result
                origin_row, origin_col = 0, 0
            
            if not success:
                return False, "模型训练失败"
            
            # 4. 更新配置
            config_data['model_file'] = f'{product_name}.shm'
            config_data['model_row'] = origin_row
            config_data['model_col'] = origin_col
            config_data['product_name'] = product_name
            
            # 5. 保存配置（路径为空）
            success, msg = self.products.save_product(
                product_name,
                config_data=config_data,
                path_points=None,
                path_polys=None
            )
            
            if not success:
                return False, f"保存配置失败: {msg}"
            
            # 6. 加载新产品
            self.current_product = product_name
            self.current_paths = {'dots': [], 'polys': []}
            self.model_ref_row = origin_row
            self.model_ref_col = origin_col
            
            # 7. 加载模型
            self.halcon.set_model_path(model_path)
            
            return True, f"产品 '{product_name}' 创建成功"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"训练产品失败: {e}"
    
    def load_product(self, product_name: str) -> Tuple[bool, str]:
        """
        加载产品（模型+路径+参数）
        
        Args:
            product_name: 产品名称
            
        Returns:
            (成功, 消息)
        """
        try:
            # 1. 加载配置
            config = self.products.load_product_config(product_name)
            if not config:
                return False, f"配置加载失败: {product_name}"
            
            # 2. 加载模型
            product_dir = self.products.get_product_path(product_name)
            model_file = config.get('model_file', f'{product_name}.shm')
            model_path = os.path.join(product_dir, model_file)
            
            if os.path.exists(model_path):
                ok = self.halcon.set_model_path(model_path)
                if not ok:
                    return False, f"模型加载失败: {model_file}"
            else:
                return False, f"模型文件不存在: {model_file}"
            
            # 3. 加载路径
            path_data, poly_data = self.products.load_product_paths(product_name)
            
            self.current_paths['dots'] = path_data if path_data is not None else []
            self.current_paths['polys'] = poly_data if poly_data is not None else []
            
            # 4. 加载参数
            self.model_ref_row = config.get('model_row', 0)
            self.model_ref_col = config.get('model_col', 0)
            self.vision_offset_x = config.get('vision_offset_x', 0.0)
            self.vision_offset_y = config.get('vision_offset_y', 0.0)
            
            # 5. 设置当前产品
            self.current_product = product_name
            
            return True, f"产品 '{product_name}' 加载成功"
            
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
        # 1. 运行Halcon识别
        res = self.halcon.process_image(gray_img)
        debug_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        found_paths = []
        
        if not res or 'matches' not in res:
            return found_paths, debug_img
        
        # 2. 获取参考点
        ref_row, ref_col = self.model_ref_row, self.model_ref_col
        if ref_row == 0 and ref_col == 0:
            ref_row, ref_col = 651.6765, 1113.3245  # Fallback
        
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
            
            # 绘制分数
            text = f"S:{score:.2f}"
            cv2.putText(debug_img, text, (cx_i + 25, cy_i), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return found_paths, debug_img
    
    def _transform_and_draw_paths(self, debug_img: np.ndarray, 
                                  cx: float, cy: float, angle: float,
                                  ref_row: float, ref_col: float) -> List:
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
