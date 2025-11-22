# mold_detector.py
import cv2
import numpy as np
import os

_TEMPLATES = None      # (4,4,2)
_WARP_SIZE = None      # (2,)
_PATH_POLY = {}
_PATH_DOTS = {}


def _order_points(pts: np.ndarray) -> np.ndarray:
    """统一将四边形点排序为 tl,tr,br,bl."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _ensure_paths_for_all_cells():
    """
    如果只给 cell0 做了路径示教，就自动复制到其它 cell。
    关键改动：对每个小框先 _order_points，避免 90° 旋转。
    """
    global _PATH_POLY, _PATH_DOTS, _TEMPLATES
    if _TEMPLATES is None:
        return

    # 如果 0 号本身没有路径，就不用复制
    if (0 not in _PATH_POLY) and (0 not in _PATH_DOTS):
        return

    base_box = _order_points(_TEMPLATES[0])   # 0 号小框（排序后的四点）

    for idx in range(len(_TEMPLATES)):
        if idx == 0:
            continue

        tgt_box = _order_points(_TEMPLATES[idx])
        H, status = cv2.findHomography(base_box, tgt_box)
        if H is None:
            continue

        # 复制折线
        if idx not in _PATH_POLY and 0 in _PATH_POLY:
            pts0 = _PATH_POLY[0].reshape(-1, 1, 2)
            ptsi = cv2.perspectiveTransform(pts0, H).reshape(-1, 2)
            _PATH_POLY[idx] = ptsi

        # 复制点
        if idx not in _PATH_DOTS and 0 in _PATH_DOTS:
            pts0 = _PATH_DOTS[0].reshape(-1, 1, 2)
            ptsi = cv2.perspectiveTransform(pts0, H).reshape(-1, 2)
            _PATH_DOTS[idx] = ptsi


def _load_all():
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS

    if _TEMPLATES is None or _WARP_SIZE is None:
        try:
            _TEMPLATES = np.load("slot_templates.npy")
            _WARP_SIZE = np.load("warp_size.npy").astype(int)
            print(f"[mold_detector] 加载黄金模板 {_TEMPLATES.shape}, warp={_WARP_SIZE}")
        except Exception as e:
            print("[mold_detector] 未找到黄金模板，仅画大框。", e)
            _TEMPLATES = None
            _WARP_SIZE = None

    if not _PATH_POLY and not _PATH_DOTS:
        for idx in range(4):
            poly_file = f"disp_poly_cell{idx}.npy"
            dots_file = f"disp_dots_cell{idx}.npy"

            if os.path.exists(poly_file):
                try:
                    arr = np.load(poly_file).astype(np.float32)
                    if arr.size > 0:
                        _PATH_POLY[idx] = arr.reshape(-1, 2)
                        print(f"[mold_detector] 加载点胶折线 {poly_file}, 点数={_PATH_POLY[idx].shape[0]}")
                except Exception as e:
                    print(f"[mold_detector] 加载 {poly_file} 失败：", e)

            if os.path.exists(dots_file):
                try:
                    arr = np.load(dots_file).astype(np.float32)
                    if arr.size > 0:
                        _PATH_DOTS[idx] = arr.reshape(-1, 2)
                        print(f"[mold_detector] 加载点胶点集 {dots_file}, 点数={_PATH_DOTS[idx].shape[0]}")
                except Exception as e:
                    print(f"[mold_detector] 加载 {dots_file} 失败：", e)

    # ⭐ 这里会按新的排序逻辑，把 0 号路径复制到 1/2/3
    _ensure_paths_for_all_cells()


def reload_golden_templates():
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS
    _TEMPLATES = None
    _WARP_SIZE = None
    _PATH_POLY = {}
    _PATH_DOTS = {}
    _load_all()


def _find_big_mold(gray: np.ndarray):
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
    _load_all()

    color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    big_rect, big_box = _find_big_mold(gray)
    if big_box is None:
        print("[mold_detector] 未找到大模具")
        return color_img, None, [], []

    # 画大框
    cv2.drawContours(color_img, [big_box], 0, (0, 255, 0), 3)

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

    M_cur = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    Minv_cur = cv2.getPerspectiveTransform(dst_pts, rect_pts)

    # 画 4 个小框
    cell_boxes = []
    cell_centers = []
    for idx in range(_TEMPLATES.shape[0]):
        box_w = _TEMPLATES[idx].astype(np.float32)
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

    # 折线
    for idx, poly in _PATH_POLY.items():
        if poly is None or poly.size == 0:
            continue
        pts_w = poly.reshape(-1, 1, 2)
        pts_o = cv2.perspectiveTransform(pts_w, Minv_cur).reshape(-1, 2)
        pts = pts_o.astype(np.int32)
        for i in range(len(pts) - 1):
            cv2.line(color_img, tuple(pts[i]), tuple(pts[i + 1]),
                     (255, 0, 0), 1, cv2.LINE_AA)

    # 点
    for idx, dots in _PATH_DOTS.items():
        if dots is None or dots.size == 0:
            continue
        pts_w = dots.reshape(-1, 1, 2)
        pts_o = cv2.perspectiveTransform(pts_w, Minv_cur).reshape(-1, 2)
        pts = pts_o.astype(np.int32)
        for (x, y) in pts:
            cv2.circle(color_img, (int(x), int(y)),
                       3, (255, 0, 255), -1)

    return color_img, big_box, cell_boxes, cell_centers
