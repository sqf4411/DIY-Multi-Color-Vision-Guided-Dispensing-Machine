# mold_detector.py
import cv2
import numpy as np
import os

# ====== 全局缓存：黄金示教数据 / 点胶路径 ======
_TEMPLATES = None      # (4,4,2)  小框在黄金坐标系里的四点
_WARP_SIZE = None      # (2,)    [W,H] 黄金图尺寸

_PATH_POLY = None      # (N,2)   只存 cell0 的折线（黄金坐标系）
_PATH_DOTS = None      # (K,2)   只存 cell0 的点集（黄金坐标系）


def _order_points(pts: np.ndarray) -> np.ndarray:
    """把四边形点统一排序为 tl,tr,br,bl"""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _load_golden_and_paths():
    """第一次调用时加载黄金模板和 cell0 的点胶路径"""
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS
    if _TEMPLATES is None:
        try:
            _TEMPLATES = np.load("slot_templates.npy")
            _WARP_SIZE = np.load("warp_size.npy").astype(int)
            print(f"[mold_detector] 黄金模板 slot_templates.npy 已加载 {_TEMPLATES.shape}, warp={_WARP_SIZE}")
        except Exception as e:
            print("[mold_detector] 未找到黄金模板，仅画大框。", e)
            _TEMPLATES = None
            _WARP_SIZE = None

    # cell0 的折线
    if _PATH_POLY is None and os.path.exists("disp_poly_cell0.npy"):
        try:
            _PATH_POLY = np.load("disp_poly_cell0.npy").astype(np.float32)
            print(f"[mold_detector] 已加载点胶折线 disp_poly_cell0.npy, 点数={_PATH_POLY.shape[0]}")
        except Exception as e:
            print("[mold_detector] 加载 disp_poly_cell0.npy 失败：", e)

    # cell0 的点
    if _PATH_DOTS is None and os.path.exists("disp_dots_cell0.npy"):
        try:
            _PATH_DOTS = np.load("disp_dots_cell0.npy").astype(np.float32)
            print(f"[mold_detector] 已加载点胶点集 disp_dots_cell0.npy, 点数={_PATH_DOTS.shape[0]}")
        except Exception as e:
            print("[mold_detector] 加载 disp_dots_cell0.npy 失败：", e)


def reload_golden_templates():
    """UI 调用：黄金示教 / 路径示教之后，强制重新加载所有缓存"""
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS
    _TEMPLATES = None
    _WARP_SIZE = None
    _PATH_POLY = None
    _PATH_DOTS = None
    _load_golden_and_paths()


def _find_big_mold(gray: np.ndarray):
    """只负责找大模具轮廓与外接四边形"""
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
        return None, None

    big_rect = cv2.minAreaRect(big_mold_cnt)
    big_box = cv2.boxPoints(big_rect).astype(np.int32)
    return big_rect, big_box


def detect_mold_and_cells(gray: np.ndarray):
    """主入口：找到大模具 + 四个小框 + 叠加点胶路径"""
    _load_golden_and_paths()

    color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 1. 大模具
    big_rect, big_box = _find_big_mold(gray)
    if big_box is None:
        print("[mold_detector] 未找到大模具")
        return color_img, None, [], []

    # 画大绿框
    cv2.drawContours(color_img, [big_box], 0, (0, 255, 0), 3)

    # 画十字
    pts = big_box.astype(np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    tl = tl.astype(np.int32)
    tr = tr.astype(np.int32)
    br = br.astype(np.int32)
    bl = bl.astype(np.int32)

    mid_top = ((tl[0] + tr[0]) // 2, (tl[1] + tr[1]) // 2)
    mid_bottom = ((bl[0] + br[0]) // 2, (bl[1] + br[1]) // 2)
    mid_left = ((tl[0] + bl[0]) // 2, (tl[1] + bl[1]) // 2)
    mid_right = ((tr[0] + br[0]) // 2, (tr[1] + br[1]) // 2)

    cv2.line(color_img, mid_top, mid_bottom, (0, 255, 255), 1)
    cv2.line(color_img, mid_left, mid_right, (0, 255, 255), 1)

    cx, cy = map(int, big_rect[0])
    cv2.drawMarker(color_img, (cx, cy), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS,
                   markerSize=24,
                   thickness=2)

    # 2. 如果没有黄金模板，就到这里结束
    if _TEMPLATES is None or _WARP_SIZE is None:
        return color_img, big_box, [], []

    goldW, goldH = int(_WARP_SIZE[0]), int(_WARP_SIZE[1])

    rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [goldW - 1, 0],
        [goldW - 1, goldH - 1],
        [0, goldH - 1]
    ], dtype=np.float32)

    # 当前帧 <-> 黄金 的透视变换
    M_cur = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    Minv_cur = cv2.getPerspectiveTransform(dst_pts, rect_pts)

    # 3. 把 4 个黄金小框映射到当前图像
    cell_boxes = []
    cell_centers = []
    for idx in range(_TEMPLATES.shape[0]):
        box_w = _TEMPLATES[idx].astype(np.float32)        # 黄金坐标系
        pts_w = box_w.reshape(-1, 1, 2)
        pts_o = cv2.perspectiveTransform(pts_w, Minv_cur).reshape(-1, 2)
        box_o = pts_o.astype(np.int32)
        cell_boxes.append(box_o)

        cx_o = int(box_o[:, 0].mean())
        cy_o = int(box_o[:, 1].mean())
        cell_centers.append((cx_o, cy_o))

        cv2.drawContours(color_img, [box_o], 0, (0, 255, 255), 2)
        cv2.drawMarker(color_img, (cx_o, cy_o), (0, 0, 255),
                       cv2.MARKER_CROSS, 18, 2)
        cv2.putText(color_img, str(idx), (cx_o + 5, cy_o - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 4. 叠加点胶路径：从 cell0(黄金) -> 各 cell(当前图像) 做一次完整投影
    if _PATH_POLY is not None or _PATH_DOTS is not None:
        if _TEMPLATES.shape[0] > 0:
            base_box_gold = _order_points(_TEMPLATES[0])  # cell0 在黄金中的四点

            for idx, box_img in enumerate(cell_boxes):
                dst_box_img = _order_points(box_img)
                # 关键：从「黄金 cell0」直接投影到「当前帧 cell_i」
                H = cv2.getPerspectiveTransform(base_box_gold, dst_box_img)

                # 折线
                if _PATH_POLY is not None and _PATH_POLY.size > 0:
                    pts_w = _PATH_POLY.reshape(-1, 1, 2)
                    pts_o = cv2.perspectiveTransform(pts_w, H).reshape(-1, 2)
                    pts = pts_o.astype(np.int32)
                    for i in range(len(pts) - 1):
                        cv2.line(color_img,
                                 tuple(pts[i]),
                                 tuple(pts[i + 1]),
                                 (255, 0, 0), 1, cv2.LINE_AA)

                # 点
                if _PATH_DOTS is not None and _PATH_DOTS.size > 0:
                    pts_w = _PATH_DOTS.reshape(-1, 1, 2)
                    pts_o = cv2.perspectiveTransform(pts_w, H).reshape(-1, 2)
                    pts = pts_o.astype(np.int32)
                    for (x, y) in pts:
                        cv2.circle(color_img, (int(x), int(y)),
                                   3, (255, 0, 255), -1)

    return color_img, big_box, cell_boxes, cell_centers
