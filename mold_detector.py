# mold_detector.py
import cv2
import numpy as np
import os

# mold_detector.py
import cv2
import numpy as np
import os

# ====== 全局缓存：黄金示教数据 / 点胶路径 ======
_TEMPLATES = None      # (4,4,2)
_WARP_SIZE = None      # (2,)  [W,H]

_PATH_POLY = None      # (N,2)  cell0 的折线路径
_PATH_DOTS = None      # (K,2)  cell0 的点集


# ====================================================
#  加载 黄金四小框 + warp_size + 点胶路径
# ====================================================
def _load_golden_and_paths():
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS

    # ---------- 1. 黄金模板 ----------
    if _TEMPLATES is None:
        try:
            _TEMPLATES = np.load("slot_templates.npy")
            _WARP_SIZE = np.load("warp_size.npy").astype(int)
            print(f"✅ 黄金模板 slot_templates.npy 已加载 {_TEMPLATES.shape}, warp={_WARP_SIZE}")
        except Exception as e:
            print("⚠ 未找到黄金模板，仅画大框。", e)
            _TEMPLATES = None
            _WARP_SIZE = None

    # ---------- 2. 点胶折线 ----------
    if _PATH_POLY is None and os.path.exists("disp_poly_cell0.npy"):
        try:
            _PATH_POLY = np.load("disp_poly_cell0.npy").astype(np.float32)
            print(f"✅ 已加载点胶折线 disp_poly_cell0.npy，点数 = {_PATH_POLY.shape[0]}")
        except Exception as e:
            print("⚠ 加载 disp_poly_cell0.npy 失败：", e)

    # ---------- 3. 点胶点 ----------
    if _PATH_DOTS is None and os.path.exists("disp_dots_cell0.npy"):
        try:
            _PATH_DOTS = np.load("disp_dots_cell0.npy").astype(np.float32)
            print(f"✅ 已加载点胶点集 disp_dots_cell0.npy，点数 = {_PATH_DOTS.shape[0]}")
        except Exception as e:
            print("⚠ 加载 disp_dots_cell0.npy 失败：", e)


def reload_golden_templates():
    """UI 调用：强制重新加载（黄金示教或路径编辑后）"""
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS
    _TEMPLATES = None
    _WARP_SIZE = None
    _PATH_POLY = None
    _PATH_DOTS = None
    _load_golden_and_paths()


def _load_golden_templates():
    """只在第一次调用时加载一次黄金示教数据"""
    global _TEMPLATES, _WARP_SIZE
    if _TEMPLATES is not None:
        return

    try:
        _TEMPLATES = np.load("slot_templates.npy")          # (4,4,2)
        _WARP_SIZE = np.load("warp_size.npy").astype(int)   # (2,)
        print(f"✅ 黄金模板 slot_templates.npy 加载成功, 形状: {_TEMPLATES.shape},{_WARP_SIZE}")
        print(f"✅ 黄金坐标系尺寸 warp_size = {_WARP_SIZE}")
    except Exception as e:
        print("⚠ 未找到黄金示教文件，仅识别大框。错误：", e)
        _TEMPLATES = None
        _WARP_SIZE = None

def reload_golden_templates():
    """
    外部调用：强制重新从硬盘加载黄金示教数据。
    黄金示教完成后，在 UI 里调用一次即可。
    """
    global _TEMPLATES, _WARP_SIZE
    _TEMPLATES = None
    _WARP_SIZE = None
    _load_golden_templates()

def detect_mold_and_cells(gray: np.ndarray):
    """
    输入:
        gray: 单通道灰度图 (H, W)

    输出:
        color_img   : 画好大绿框 + 十字 + (可选)小黄框 的 BGR 图
        big_box     : 大模具外框四点，shape (4,2)，np.int32
        cell_boxes  : 小方框列表，每个 (4,2) np.int32   —— 若没黄金模板则为空列表
        cell_centers: 小方框中心点列表，每个是 (x,y) 元组
    """
    _load_golden_templates()  # 先尝试加载黄金模板

    h_img, w_img = gray.shape
    img_area = h_img * w_img

    # ================== 1. 找大模具轮廓 ==================
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

    color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if big_mold_cnt is None:
        print("❌ 未找到大模具，请调整光照/视野")
        return color_img, None, [], []

    # 外接旋转矩形
    big_rect = cv2.minAreaRect(big_mold_cnt)      # ((cx,cy),(w,h),angle)
    big_box = cv2.boxPoints(big_rect).astype(np.int32)
    cv2.drawContours(color_img, [big_box], 0, (0, 255, 0), 3)  # 绿色大框

    # ================== 2. 画“跟着大框转”的十字 ==================
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

    # ================== 3. 若没有黄金模板，只返回大框 ==================
    if _TEMPLATES is None or _WARP_SIZE is None:
        return color_img, big_box, [], []

    goldW, goldH = int(_WARP_SIZE[0]), int(_WARP_SIZE[1])

    # 当前图像中的大模具四个点 -> 黄金坐标系 (0~goldW-1, 0~goldH-1)
    rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [goldW - 1, 0],
        [goldW - 1, goldH - 1],
        [0, goldH - 1]
    ], dtype=np.float32)

    # 当前帧的 透视矩阵 / 逆矩阵
    M_cur = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    Minv_cur = cv2.getPerspectiveTransform(dst_pts, rect_pts)

    # ================== 4. 把 4 个黄金小框映射回当前画面 ==================
    cell_boxes = []
    cell_centers = []

    for idx in range(4):
        box_w = _TEMPLATES[idx].astype(np.float32)   # (4,2) in 黄金坐标系

        # 4 个点 -> 当前图像
        pts_w = box_w.reshape(-1, 1, 2)
        pts_o = cv2.perspectiveTransform(pts_w, Minv_cur).reshape(-1, 2)
        box_o = pts_o.astype(np.int32)
        cell_boxes.append(box_o)

        # 中心点
        cx_o = int(box_o[:, 0].mean())
        cy_o = int(box_o[:, 1].mean())
        cell_centers.append((cx_o, cy_o))

        # 画黄框 + 红十字 + 序号
        cv2.drawContours(color_img, [box_o], 0, (0, 255, 255), 2)
        cv2.drawMarker(color_img, (cx_o, cy_o), (0, 0, 255),
                       cv2.MARKER_CROSS, 18, 2)
        cv2.putText(color_img, str(idx), (cx_o + 5, cy_o - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return color_img, big_box, cell_boxes, cell_centers
