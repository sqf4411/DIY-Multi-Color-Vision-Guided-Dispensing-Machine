"""
路径编辑器包装函数
兼容OpenCV和Halcon两种模式，集成多颜色编辑器
"""
import cv2
import numpy as np
import os
import json
from PyQt5.QtWidgets import QDialog, QMessageBox

from .multicolor_path_editor import MultiColorPathEditor
from .path_segment import PathSegmentList


def open_multicolor_path_editor(parent, template_img, existing_data=None, save_callback=None):
    """
    打开多颜色路径编辑器（通用接口）
    
    参数:
        parent: 父窗口
        template_img: 模板图像（灰度或彩色）
        existing_data: 已有数据字典 {'segments': [...]}
        save_callback: 保存回调函数 callback(path_list) -> bool
    
    返回:
        success: 是否保存成功
    """
    if template_img is None:
        QMessageBox.critical(parent, "错误", "模板图像为空")
        return False
    
    # 打开编辑器
    print("[DEBUG WRAPPER] Opening Editor Dialog...") # Debug
    editor = MultiColorPathEditor(template_img, existing_data, parent)
    
    res = editor.exec_()
    print(f"[DEBUG WRAPPER] Editor Closed. Result: {res}") # Debug

    if res == QDialog.Accepted:
        # 获取路径列表
        path_list = editor.get_path_list()
        
        if path_list.count() == 0:
            reply = QMessageBox.question(parent, '确认', 
                '路径为空，确定要保存吗？',
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return False
        
        # 调用保存回调
        if save_callback:
            try:
                success = save_callback(path_list)
                if success:
                    QMessageBox.information(parent, "成功", 
                        f"路径已保存！\n共{path_list.count()}个路径段")
                return success
            except Exception as e:
                QMessageBox.critical(parent, "错误", f"保存失败: {e}")
                return False
        
        return True
    
    return False


def edit_path_opencv_mode(parent, cell_index=0, img_path="config/mold2.png", save_dir=None):
    """
    OpenCV模式：编辑黄金示教的小格路径
    
    参数:
        parent: 父窗口
        cell_index: 小格索引
        img_path: 训练图路径
        save_dir: 保存目录
    
    返回:
        success: 是否保存成功
    """
    # 基础路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, "config")
    
    if save_dir is None:
        save_dir = config_dir
    
    # 解析图像路径
    if not os.path.isabs(img_path):
        img_path = os.path.join(base_dir, img_path)
    
    # 加载黄金模板
    try:
        path_slot = os.path.join(config_dir, "slot_templates.npy")
        path_warp = os.path.join(config_dir, "warp_size.npy")
        
        templates = np.load(path_slot)
        warp_size = np.load(path_warp).astype(int)
    except Exception as e:
        QMessageBox.critical(parent, "错误", 
            f"需要先完成黄金示教！\n未找到 slot_templates.npy / warp_size.npy\n{e}")
        return False
    
    goldW, goldH = int(warp_size[0]), int(warp_size[1])
    
    if not (0 <= cell_index < templates.shape[0]):
        QMessageBox.critical(parent, "错误", f"cell_index 越界: {cell_index}")
        return False
    
    # 读取训练图并生成黄金视角
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        QMessageBox.critical(parent, "错误", f"无法读取图像: {img_path}")
        return False
    
    # 找大模具并透视变换（复用path_teach.py逻辑）
    h_img, w_img = gray.shape
    img_area = h_img * w_img
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th_bg = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed_bg = cv2.morphologyEx(th_bg, cv2.MORPH_CLOSE, kernel_big)
    cnts_bg, _ = cv2.findContours(closed_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    big_mold_cnt = None
    max_area = 0
    for c in cnts_bg:
        area = cv2.contourArea(c)
        if area > img_area * 0.15 and area > max_area:
            max_area = area
            big_mold_cnt = c
    
    if big_mold_cnt is None:
        QMessageBox.critical(parent, "错误", "未找到大模具，无法生成黄金视角")
        return False
    
    big_rect = cv2.minAreaRect(big_mold_cnt)
    big_box = cv2.boxPoints(big_rect).astype(np.float32)
    
    # 排序四点
    s = big_box.sum(axis=1)
    tl = big_box[np.argmin(s)]
    br = big_box[np.argmax(s)]
    diff = np.diff(big_box, axis=1)
    tr = big_box[np.argmin(diff)]
    bl = big_box[np.argmax(diff)]
    rect_big = np.array([tl, tr, br, bl], dtype=np.float32)
    
    dst_big = np.array([
        [0, 0],
        [goldW - 1, 0],
        [goldW - 1, goldH - 1],
        [0, goldH - 1]
    ], dtype=np.float32)
    
    # 原图 -> 黄金整板
    M_big = cv2.getPerspectiveTransform(rect_big, dst_big)
    warped_full = cv2.warpPerspective(gray, M_big, (goldW, goldH))
    
    # 取出cell的黄金坐标
    cell_box_gold = templates[cell_index].astype(np.float32)
    
    # 排序函数
    def order_points(pts):
        pts = np.array(pts, dtype=np.float32)
        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)
    
    rect_cell = order_points(cell_box_gold)
    
    # 计算cell大小
    widthA = np.linalg.norm(rect_cell[2] - rect_cell[3])
    widthB = np.linalg.norm(rect_cell[1] - rect_cell[0])
    maxW = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(rect_cell[1] - rect_cell[2])
    heightB = np.linalg.norm(rect_cell[0] - rect_cell[3])
    maxH = int(max(heightA, heightB))
    
    dst_cell = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype=np.float32)
    
    # 黄金整板 -> 小格局部图
    M_cell = cv2.getPerspectiveTransform(rect_cell, dst_cell)
    template = cv2.warpPerspective(warped_full, M_cell, (maxW, maxH))
    
    # ✅ 修复图像颠倒问题：旋转180度
    template = cv2.flip(template, -1)
    
    # 加载已有数据
    data_file = os.path.join(save_dir, f"multicolor_path_cell{cell_index}.json")
    existing_data = None
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    
    # 保存回调
    def save_opencv(path_list):
        # 保存新格式
        data = {'segments': path_list.to_dict_list()}
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # 兼容旧格式
        polylines, dots = path_list.get_legacy_format()
        np.save(os.path.join(save_dir, f"cell_{cell_index}_polylines.npy"),
               np.array(polylines, dtype=object) if polylines else np.array([]))
        np.save(os.path.join(save_dir, f"cell_{cell_index}_dots.npy"),
               np.array(dots, dtype=np.float32) if dots else np.array([]))
        
        return True
    
    return open_multicolor_path_editor(parent, template, existing_data, save_opencv)


def edit_path_halcon_mode(parent, product_name, product_dir):
    """
    Halcon模式：编辑Halcon产品的路径
    在reference.png上直接绘制
    
    参数:
        parent: 父窗口
        product_name: 产品名称
        product_dir: 产品目录
    
    返回:
        success: 是否保存成功
    """
    
    # Load Warp Size
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, "config")
    
    # 1. Load Reference    # 加载训练图
    ref_img_path = os.path.join(product_dir, "reference.png")
    if not os.path.exists(ref_img_path):
        QMessageBox.critical(parent, "错误", 
            f"未找到训练图: reference.png\n目录: {product_dir}")
        return False
    
    gray_raw = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    if gray_raw is None:
        QMessageBox.critical(parent, "错误", f"无法读取图像: {ref_img_path}")
        return False

    # --- 核心修改：视角统一 ---
    # 目标：让用户在"黄金单格(Cell 0)"上编辑，与 OpenCV 模式一致，方便操作。
    # 运行时：identify 会输出 Warped Space 坐标 (或者 Raw 坐标，但模板匹配是基于 Warped 的特征)
    # 我们保存时，必须把 Cell 0 坐标转换回 Warped Global 坐标 (因为 warp_size 是对应整个模具的)
    
    # 1. 加载透视变换参数
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, "config")
    
    path_warp = os.path.join(config_dir, "warp_size.npy")
    path_slot = os.path.join(config_dir, "slot_templates.npy")
    
    use_cell_view = False
    M_cell = None
    M_big = None
    warped_full = None
    
    try:
        if os.path.exists(path_warp) and os.path.exists(path_slot):
             warp_size = np.load(path_warp).astype(int)
             goldW, goldH = int(warp_size[0]), int(warp_size[1])
             templates = np.load(path_slot)
             
             # A. 寻找大模具并透视变换 (Raw -> Warped Full)
             # 复用 big mold logic
             h_img, w_img = gray_raw.shape
             img_area = h_img * w_img
             blur = cv2.GaussianBlur(gray_raw, (7, 7), 0)
             _, th_bg = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
             kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
             closed_bg = cv2.morphologyEx(th_bg, cv2.MORPH_CLOSE, kernel_big)
             cnts_bg, _ = cv2.findContours(closed_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             
             big_mold_cnt = None
             max_area = 0
             for c in cnts_bg:
                 area = cv2.contourArea(c)
                 if area > img_area * 0.15 and area > max_area:
                     max_area = area
                     big_mold_cnt = c
             
             if big_mold_cnt is None:
                  print("Warning: Could not find Big Mold for warping. Using Raw.")
                  template = gray_raw
             else:
                  # Compute M_big
                  big_rect = cv2.minAreaRect(big_mold_cnt)
                  big_box = cv2.boxPoints(big_rect).astype(np.float32)
                  
                  # Order points
                  s = big_box.sum(axis=1)
                  tl = big_box[np.argmin(s)]
                  br = big_box[np.argmax(s)]
                  diff = np.diff(big_box, axis=1)
                  tr = big_box[np.argmin(diff)]
                  bl = big_box[np.argmax(diff)]
                  rect_big = np.array([tl, tr, br, bl], dtype=np.float32)
                  
                  dst_big = np.array([[0, 0], [goldW-1, 0], [goldW-1, goldH-1], [0, goldH-1]], dtype=np.float32)
                  M_big = cv2.getPerspectiveTransform(rect_big, dst_big)
                  
                  warped_full = cv2.warpPerspective(gray_raw, M_big, (goldW, goldH))
                  
                  # B. 截取 Cell 0 (Warped Full -> Cell 0)
                  # 这一步是为了给用户展示"小格子"
                  cell_box_gold = templates[0].astype(np.float32) # 使用第一个格子
                  
                  # Order Points function (inline simpl)
                  def order_pts(pts):
                      s = pts.sum(axis=1)
                      tl = pts[np.argmin(s)]
                      br = pts[np.argmax(s)]
                      
                      diff_arr = np.diff(pts, axis=1)
                      tr = pts[np.argmin(diff_arr)]
                      bl = pts[np.argmax(diff_arr)]
                      
                      return np.array([tl, tr, br, bl], dtype=np.float32)

                  rect_cell = order_pts(cell_box_gold)
                  
                  # Cell Size
                  wA = np.linalg.norm(rect_cell[2] - rect_cell[3])
                  wB = np.linalg.norm(rect_cell[1] - rect_cell[0])
                  mW = int(max(wA, wB))
                  hA = np.linalg.norm(rect_cell[1] - rect_cell[2])
                  hB = np.linalg.norm(rect_cell[0] - rect_cell[3])
                  mH = int(max(hA, hB))
                  
                  dst_cell = np.array([[0,0], [mW-1, 0], [mW-1, mH-1], [0, mH-1]], dtype=np.float32)
                  
                  M_cell = cv2.getPerspectiveTransform(rect_cell, dst_cell)
                  
                  # Generate Small Template
                  template = cv2.warpPerspective(warped_full, M_cell, (mW, mH))
                  
                  # [Corrected] Do NOT Flip. 
                  # reference.png is already upright (flipped during training).
                  # warped_full preserves this.
                  # template = cv2.flip(template, -1) <-- REMOVED
                  
                  use_cell_view = True
                  print("[Halcon Editor] Using Cell 0 View (Small Box).")

        else:
             print("Warning: Missing warp/slot templates. Using Raw Image.")
             template = gray_raw
             
    except Exception as e:
        print(f"[Halcon Editor] Warp Error: {e}")
        template = gray_raw
    
    # 加载已有数据
    data_file = os.path.join(product_dir, "multicolor_path.json")
    existing_data = None
    
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            
            # [Fix] Transform Loaded Data: Warped Full -> Cell 0
            # Because we display Cell 0, we must map the saved Warped Coords back to Cell Space.
            if use_cell_view and existing_data and 'segments' in existing_data:
                try:
                    print("[Halcon Editor] Transforming loaded data: Warped -> Cell 0")
                    for seg in existing_data['segments']:
                        if 'points' in seg:
                            pts_warped = np.array(seg['points'], dtype=np.float32)
                            if len(pts_warped) > 0:
                                # Transform: P_cell = M_cell * P_warped
                                # cv2.perspectiveTransform requires (N, 1, 2) shape
                                pts_w_reshaped = pts_warped.reshape(-1, 1, 2)
                                pts_cell = cv2.perspectiveTransform(pts_w_reshaped, M_cell)
                                
                                # Convert back to list of lists [ [x,y], ... ]
                                seg['points'] = pts_cell.reshape(-1, 2).tolist()
                except Exception as e:
                     print(f"[Halcon Editor] Load Transform Failed: {e}")
                     # If fail, keep original (might look shifted but preserves data)

    # 保存回调
    def save_halcon(path_list):
        # path_list contains coords in "Template Space" (Cell 0)
        
        if use_cell_view:
             # Convert: Cell 0 -> Warped Full Space
             # We REMOVED the flip, so no inverse flip needed.
             
             for segment in path_list.segments:
                 new_pts = []
                 for x, y in segment.points:
                     # 1. Direct Inverse Perspective (Cell -> Warped Full)
                     # p_dst = M * p_src  => p_src = M_inv * p_dst
                     # p_dst is (x, y) in Cell Destination
                     
                     M_inv = np.linalg.inv(M_cell)
                     px = (M_inv[0,0]*x + M_inv[0,1]*y + M_inv[0,2])
                     py = (M_inv[1,0]*x + M_inv[1,1]*y + M_inv[1,2])
                     w  = (M_inv[2,0]*x + M_inv[2,1]*y + M_inv[2,2])
                     
                     x_w = px / w
                     y_w = py / w
                     
                     new_pts.append([x_w, y_w])
                 
                 segment.points = new_pts
                 
             print(f"[Halcon Editor] Converted {path_list.count()} segments from Cell 0 -> Warped Full Space")

        else:
             print(f"[Halcon Editor] Saving in Raw Space (No Warp found)")
        
        # 保存新格式
        data = {'segments': path_list.to_dict_list()}
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # 兼容旧格式
        polylines, dots = path_list.get_legacy_format()
        np.save(os.path.join(product_dir, "path_poly.npy"),
               np.array(polylines, dtype=object) if polylines else np.array([]))
        np.save(os.path.join(product_dir, "path.npy"),
               np.array(dots, dtype=np.float32) if dots else np.array([]))
        
        return True
    
    return open_multicolor_path_editor(parent, template, existing_data, save_halcon)

