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
        
        # 兼容旧格式（安全JSON，不再写 object npy）
        polylines, dots = path_list.get_legacy_format()
        poly_json_file = os.path.join(save_dir, f"cell_{cell_index}_polylines.json")
        poly_payload = {
            "version": 2,
            "segments": [np.asarray(seg, dtype=np.float32).tolist() for seg in polylines if len(seg) > 0]
        }
        with open(poly_json_file, "w", encoding="utf-8") as f:
            json.dump(poly_payload, f, ensure_ascii=False, indent=2)

        legacy_poly_npy = os.path.join(save_dir, f"cell_{cell_index}_polylines.npy")
        if os.path.exists(legacy_poly_npy):
            os.remove(legacy_poly_npy)

        np.save(os.path.join(save_dir, f"cell_{cell_index}_dots.npy"),
               np.array(dots, dtype=np.float32) if dots else np.array([]))
        
        return True
    
    return open_multicolor_path_editor(parent, template, existing_data, save_opencv)


def edit_path_halcon_mode(parent, product_name, product_dir):
    """
    Halcon模式：编辑Halcon产品的路径
    直接在 roi_template.png 上编辑，路径保存在 ROI 坐标系
    
    参数:
        parent: 父窗口
        product_name: 产品名称
        product_dir: 产品目录
    
    返回:
        success: 是否保存成功
    """
    
    # 1. 加载 ROI 图像（用户手动选择的区域）
    roi_img_path = os.path.join(product_dir, 'roi_template.png')
    if not os.path.exists(roi_img_path):
        QMessageBox.critical(parent, "错误", 
            f"未找到ROI图像\\n路径: {roi_img_path}\\n\\n请先训练模型并选择ROI区域")
        return False
    
    template = cv2.imread(roi_img_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        QMessageBox.critical(parent, "错误", f"无法读取ROI图像: {roi_img_path}")
        return False
    
    print(f"[Halcon Editor] 使用 ROI 图像作为背景: {roi_img_path}")
    print(f"[Halcon Editor] ROI 尺寸: {template.shape[1]} x {template.shape[0]}")
    
    # 2. 加载已有路径数据（已在 ROI 坐标系，无需转换）
    data_file = os.path.join(product_dir, "multicolor_path.json")
    existing_data = None
    
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"[Halcon Editor] 已加载路径数据（ROI 坐标系）")
        except Exception as e:
            print(f"[Halcon Editor] 加载路径数据失败: {e}")
    
    # 3. 保存回调（路径直接保存在 ROI 坐标系）
    def save_halcon(path_list):
        """
        保存路径数据（ROI 坐标系）
        无需任何坐标转换
        """
        # 保存新格式（JSON）
        data = {'segments': path_list.to_dict_list()}
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Halcon Editor] ✅ 已保存 {path_list.count()} 个路径段（ROI 坐标系）")
        
        # 兼容旧格式（安全JSON + 数值npy）
        polylines, dots = path_list.get_legacy_format()
        poly_payload = {
            "version": 2,
            "segments": [np.asarray(seg, dtype=np.float32).tolist() for seg in polylines if len(seg) > 0]
        }
        with open(os.path.join(product_dir, "path_poly.json"), "w", encoding="utf-8") as f:
            json.dump(poly_payload, f, ensure_ascii=False, indent=2)

        legacy_poly_npy = os.path.join(product_dir, "path_poly.npy")
        if os.path.exists(legacy_poly_npy):
            os.remove(legacy_poly_npy)

        np.save(os.path.join(product_dir, "path.npy"),
                np.array(dots, dtype=np.float32) if dots else np.array([]))
        
        return True
    
    # 4. 打开编辑器
    return open_multicolor_path_editor(parent, template, existing_data, save_halcon)
