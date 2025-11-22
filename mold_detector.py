# mold_detector.py
import cv2
import numpy as np
import os

# ====== 全局缓存 ======
_TEMPLATES = None  # (4,4,2) 黄金矫正空间下的坐标
_WARP_SIZE = None  # (2,)    黄金矫正图尺寸 [W, H]

_PATH_POLY = None  # (N,2)   Cell 0 路径 (黄金矫正空间)
_PATH_DOTS = None  # (K,2)   Cell 0 点集 (黄金矫正空间)

# ⭐ 只存 0 号模板 (Master Template) ⭐
_MASTER_TEMPLATE = None
_MASTER_TL = None  # 0号模板在黄金图中的左上角坐标 (x, y)


def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _find_big_mold(gray: np.ndarray):
    h_img, w_img = gray.shape
    img_area = h_img * w_img
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th_bg = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(th_bg, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    big_cnt = None
    max_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > img_area * 0.15 and area > max_area:
            max_area = area
            big_cnt = c

    if big_cnt is None: return None, None
    rect = cv2.minAreaRect(big_cnt)
    box = cv2.boxPoints(rect).astype(np.int32)
    return rect, box


def _get_perspective_matrix(box, w, h):
    pts = box.astype(np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(rect_pts, dst_pts)


def _load_golden_and_paths():
    """加载数据并提取 0 号主模板"""
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS, _MASTER_TEMPLATE, _MASTER_TL

    if _TEMPLATES is None:
        try:
            _TEMPLATES = np.load("slot_templates.npy")
            _WARP_SIZE = np.load("warp_size.npy").astype(int)
            print(f"[mold_detector] 黄金框数据加载: {_TEMPLATES.shape}")
        except:
            print("[mold_detector] 缺少 slot_templates.npy")
            return

    # 安全加载路径
    if _PATH_POLY is None and os.path.exists("disp_poly_cell0.npy"):
        try:
            arr = np.load("disp_poly_cell0.npy").astype(np.float32)
            _PATH_POLY = arr if arr.size > 0 else None
        except:
            pass
    if _PATH_DOTS is None and os.path.exists("disp_dots_cell0.npy"):
        try:
            arr = np.load("disp_dots_cell0.npy").astype(np.float32)
            _PATH_DOTS = arr if arr.size > 0 else None
        except:
            pass

    # ⭐ 生成 0 号主模板 ⭐
    if _MASTER_TEMPLATE is None and os.path.exists("mold2.png"):
        try:
            gold_raw = cv2.imread("mold2.png", cv2.IMREAD_GRAYSCALE)
            if gold_raw is None: return

            _, gold_big_box = _find_big_mold(gold_raw)
            if gold_big_box is None: return

            goldW, goldH = int(_WARP_SIZE[0]), int(_WARP_SIZE[1])
            M_gold = _get_perspective_matrix(gold_big_box, goldW, goldH)
            gold_warped = cv2.warpPerspective(gold_raw, M_gold, (goldW, goldH))

            # 只提取 0 号框
            box0 = _order_points(_TEMPLATES[0])
            x0, y0 = int(min(box0[:, 0])), int(min(box0[:, 1]))
            x1, y1 = int(max(box0[:, 0])), int(max(box0[:, 1]))

            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(goldW, x1), min(goldH, y1)

            _MASTER_TEMPLATE = gold_warped[y0:y1, x0:x1].copy()
            _MASTER_TL = np.array([x0, y0], dtype=np.float32)

            print(f"[mold_detector] 0号主模板提取成功，尺寸: {_MASTER_TEMPLATE.shape}")

        except Exception as e:
            print(f"[mold_detector] 模板生成出错: {e}")


def reload_golden_templates():
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS, _MASTER_TEMPLATE
    _TEMPLATES = None
    _WARP_SIZE = None
    _PATH_POLY = None
    _PATH_DOTS = None
    _MASTER_TEMPLATE = None
    _load_golden_and_paths()


def detect_mold_and_cells(gray: np.ndarray):
    _load_golden_and_paths()
    color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 1. 找大框
    big_rect, big_box = _find_big_mold(gray)
    if big_box is None:
        return color_img, None, [], []

    cv2.drawContours(color_img, [big_box], 0, (0, 255, 0), 3)

    if _TEMPLATES is None or _WARP_SIZE is None:
        return color_img, big_box, [], []

    # 2. 矫正当前帧
    goldW, goldH = int(_WARP_SIZE[0]), int(_WARP_SIZE[1])
    pts = big_box.astype(np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([[0, 0], [goldW - 1, 0], [goldW - 1, goldH - 1], [0, goldH - 1]], dtype=np.float32)

    M_cur = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    Minv_cur = cv2.getPerspectiveTransform(dst_pts, rect_pts)

    curr_warped = cv2.warpPerspective(gray, M_cur, (goldW, goldH))

    cell_boxes = []
    cell_centers = []

    # UI显示黄框
    for idx in range(_TEMPLATES.shape[0]):
        box_w = _TEMPLATES[idx].astype(np.float32)
        pts_w = box_w.reshape(-1, 1, 2)
        if pts_w.size > 0:
            pts_o = cv2.perspectiveTransform(pts_w, Minv_cur)
            if pts_o is not None:
                box_o = pts_o.reshape(-1, 2).astype(np.int32)
                cell_boxes.append(box_o)
                cx, cy = int(box_o[:, 0].mean()), int(box_o[:, 1].mean())
                cell_centers.append((cx, cy))
                cv2.drawContours(color_img, [box_o], 0, (0, 255, 255), 2)

    # 3. ⭐ Master Template 全局匹配 ⭐
    if (_PATH_POLY is not None or _PATH_DOTS is not None) and _MASTER_TEMPLATE is not None:

        th, tw = _MASTER_TEMPLATE.shape[:2]
        search_margin = 50  # 搜索范围宽容度

        for idx in range(_TEMPLATES.shape[0]):
            # 获取该位置的“理论”搜索区域 (基于手动画的框，仅用于圈定搜索范围)
            # 这里的框画得准不准不影响最终精度，只要能把模具包进去就行
            guess_box = _order_points(_TEMPLATES[idx])
            bx, by = int(min(guess_box[:, 0])), int(min(guess_box[:, 1]))

            # 定义 ROI
            rx = max(0, bx - search_margin)
            ry = max(0, by - search_margin)
            rw = tw + 2 * search_margin
            rh = th + 2 * search_margin

            if rx + rw > goldW: rw = goldW - rx
            if ry + rh > goldH: rh = goldH - ry

            # 截取 ROI 并用 0号模板 搜索
            roi = curr_warped[ry:ry + rh, rx:rx + rw]

            if roi.shape[0] >= th and roi.shape[1] >= tw:
                res = cv2.matchTemplate(roi, _MASTER_TEMPLATE, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > 0.5:
                    # 找到了! 这是模具的真实位置
                    match_x_in_warped = rx + max_loc[0]
                    match_y_in_warped = ry + max_loc[1]

                    # 计算绝对偏移：当前真实位置 - 0号原始位置
                    # 这一步完全无视了 idx 号的手动黄框
                    shift_x = match_x_in_warped - _MASTER_TL[0]
                    shift_y = match_y_in_warped - _MASTER_TL[1]

                    total_shift = np.array([shift_x, shift_y], dtype=np.float32)

                    # 绘制折线
                    if _PATH_POLY is not None:
                        pts_gold = _PATH_POLY + total_shift
                        pts_w = pts_gold.reshape(-1, 1, 2)
                        res_o = cv2.perspectiveTransform(pts_w, Minv_cur)
                        if res_o is not None:
                            pts = res_o.reshape(-1, 2).astype(np.int32)
                            for i in range(len(pts) - 1):
                                cv2.line(color_img, tuple(pts[i]), tuple(pts[i + 1]), (255, 0, 0), 1, cv2.LINE_AA)

                    # 绘制点
                    if _PATH_DOTS is not None:
                        pts_gold = _PATH_DOTS + total_shift
                        pts_w = pts_gold.reshape(-1, 1, 2)
                        res_o = cv2.perspectiveTransform(pts_w, Minv_cur)
                        if res_o is not None:
                            pts = res_o.reshape(-1, 2).astype(np.int32)
                            for px, py in pts:
                                cv2.circle(color_img, (int(px), int(py)), 3, (255, 0, 255), -1)

    return color_img, big_box, cell_boxes, cell_centers