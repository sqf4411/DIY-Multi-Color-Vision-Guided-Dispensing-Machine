# mold_detector.py
import cv2
import numpy as np
import os

# ====== 全局缓存 ======
# 这些变量用于在多次调用中保存数据，避免重复计算，提高效率。
_TEMPLATES = None  # (4,4,2) 存储4个模具的黄框坐标 (用户手画的示教框)
_WARP_SIZE = None  # (2,)    [W, H] 存储黄金示教图透视矫正后的尺寸

_PATH_POLY = None  # (N,2)   Cell 0 的点胶路径折线数据
_PATH_DOTS = None  # (K,2)   Cell 0 的点胶点集数据

# 0 号【去边缘】主模板 (Core Template)
# 这是整个算法的核心：我们只用这一个经过裁剪的、去除了阴影干扰的模板去匹配所有位置。
_CORE_TEMPLATE = None
# 记录核心模板在黄金图中的绝对坐标 (x, y)，作为所有相对偏移计算的基准点。
_CORE_REF_TL = None


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    辅助函数：将四边形的四个点按固定顺序排序：左上, 右上, 右下, 左下。
    这是透视变换的前提，否则图片会扭曲。
    """
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]  # x+y 最小的是左上
    br = pts[np.argmax(s)]  # x+y 最大的是右下
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]  # y-x 最小的是右上
    bl = pts[np.argmax(diff)]  # y-x 最大的是左下
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _find_big_mold(gray: np.ndarray):
    """
    在输入图像中寻找最大的矩形轮廓 (即整个模具的大外框)。
    步骤：高斯模糊 -> 二值化 -> 形态学闭运算 -> 轮廓查找 -> 筛选最大轮廓。
    """
    h_img, w_img = gray.shape
    img_area = h_img * w_img

    # 1. 预处理：去噪和平滑
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # 2. 二值化：将灰度图转为黑白图
    _, th_bg = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
    # 3. 形态学闭运算：填补轮廓中的小黑洞，使大轮廓闭合
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(th_bg, cv2.MORPH_CLOSE, kernel)
    # 4. 查找轮廓
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    big_cnt = None
    max_area = 0
    # 5. 筛选：找面积最大的那个
    for c in cnts:
        area = cv2.contourArea(c)
        if area > img_area * 0.15 and area > max_area:
            max_area = area
            big_cnt = c

    if big_cnt is None: return None, None
    # 6. 计算最小外接矩形 (带旋转角度)
    rect = cv2.minAreaRect(big_cnt)
    # 获取矩形的4个顶点坐标
    box = cv2.boxPoints(rect).astype(np.int32)
    return rect, box


def _get_perspective_matrix(box, w, h):
    """
    计算透视变换矩阵 (Perspective Matrix)。
    将输入的任意四边形 (box) 映射到一个标准的矩形 (w x h)。
    这是矫正相机视角偏差的关键步骤。
    """
    pts = box.astype(np.float32)
    # 排序四个点
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # 源点 (Source Points): 当前图像中的四个角
    rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    # 目标点 (Destination Points): 理想矩形的四个角
    dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    # 计算矩阵
    return cv2.getPerspectiveTransform(rect_pts, dst_pts)


def _load_golden_and_paths():
    """
    【初始化函数】
    1. 加载黄金示教数据 (slot_templates.npy)。
    2. 加载点胶路径数据 (disp_poly_cell0.npy)。
    3. ⭐ 关键步骤：从 mold2.png 中提取去除侧壁干扰的 '核心模板' (_CORE_TEMPLATE)。
    """
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS, _CORE_TEMPLATE, _CORE_REF_TL

    # --- 1. 加载基础数据 ---
    if _TEMPLATES is None:
        try:
            _TEMPLATES = np.load("slot_templates.npy")
            _WARP_SIZE = np.load("warp_size.npy").astype(int)
            print(f"[mold_detector] 黄金框数据加载: {_TEMPLATES.shape}")
        except:
            print("[mold_detector] 缺少 slot_templates.npy")
            return

    # --- 2. 加载路径文件 ---
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

    # --- 3. ⭐ 核心逻辑：提取去边模板 ---
    if _CORE_TEMPLATE is None and os.path.exists("mold2.png"):
        try:
            gold_raw = cv2.imread("mold2.png", cv2.IMREAD_GRAYSCALE)
            if gold_raw is None: return

            # 3.1 在黄金图上找大框，并做透视矫正
            _, gold_big_box = _find_big_mold(gold_raw)
            if gold_big_box is None: return

            goldW, goldH = int(_WARP_SIZE[0]), int(_WARP_SIZE[1])
            M_gold = _get_perspective_matrix(gold_big_box, goldW, goldH)
            gold_warped = cv2.warpPerspective(gold_raw, M_gold, (goldW, goldH))

            # 3.2 获取用户手画的 0 号框
            box0 = _order_points(_TEMPLATES[0])
            x0, y0 = int(min(box0[:, 0])), int(min(box0[:, 1]))
            x1, y1 = int(max(box0[:, 0])), int(max(box0[:, 1]))

            # 3.3 ⭐ "去壁存芯" (Inner Crop) ⭐
            # 计算内缩的边距，切掉四周 15% 的区域。
            # 目的是去除受视差影响的侧壁阴影，只保留平面的底部图案。
            w_box = x1 - x0
            h_box = y1 - y0
            crop_ratio = 0.15

            margin_x = int(w_box * crop_ratio)
            margin_y = int(h_box * crop_ratio)

            core_x0 = max(0, x0 + margin_x)
            core_y0 = max(0, y0 + margin_y)
            core_x1 = min(goldW, x1 - margin_x)
            core_y1 = min(goldH, y1 - margin_y)

            # 保存这个纯净的模板
            _CORE_TEMPLATE = gold_warped[core_y0:core_y1, core_x0:core_x1].copy()
            # 记录它在黄金图里的绝对坐标 (非常重要，这是路径偏移的基准点)
            _CORE_REF_TL = np.array([core_x0, core_y0], dtype=np.float32)

            print(f"[mold_detector] 0号核心模板提取成功 (已去边): {_CORE_TEMPLATE.shape}")

        except Exception as e:
            print(f"[mold_detector] 模板生成出错: {e}")


def reload_golden_templates():
    """
    重置所有缓存，强制重新加载。
    通常在重新示教后调用。
    """
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS, _CORE_TEMPLATE
    _TEMPLATES = None
    _WARP_SIZE = None
    _PATH_POLY = None
    _PATH_DOTS = None
    _CORE_TEMPLATE = None
    _load_golden_and_paths()


def detect_mold_and_cells(gray: np.ndarray):
    """
    【主处理函数】每帧调用一次。
    1. 找大模具位置。
    2. 矫正透视。
    3. 用核心模板匹配每个单元的位置。
    4. 根据位置计算偏移，绘制点胶路径。
    """
    _load_golden_and_paths()
    color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # --- 1. 找大框 ---
    big_rect, big_box = _find_big_mold(gray)
    if big_box is None:
        return color_img, None, [], []

    # 画大框 (绿色)
    cv2.drawContours(color_img, [big_box], 0, (0, 255, 0), 3)

    if _TEMPLATES is None or _WARP_SIZE is None:
        return color_img, big_box, [], []

    # --- 2. 矫正当前帧 ---
    # 计算当前帧到黄金尺寸的透视变换矩阵
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

    # M_cur: 当前帧 -> 黄金矫正图
    M_cur = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    # Minv_cur: 黄金矫正图 -> 当前帧 (用于把计算结果投射回去显示)
    Minv_cur = cv2.getPerspectiveTransform(dst_pts, rect_pts)

    # 生成矫正后的当前画面 (在此画面上做模板匹配最准)
    curr_warped = cv2.warpPerspective(gray, M_cur, (goldW, goldH))

    cell_boxes = []
    cell_centers = []

    # --- UI 显示: 把当初手画的黄框画出来 (绿色粗线) ---
    for idx in range(_TEMPLATES.shape[0]):
        box_w = _TEMPLATES[idx].astype(np.float32)
        pts_w = box_w.reshape(-1, 1, 2)
        if pts_w.size > 0:
            # 使用 Minv 将黄金坐标投影回当前帧坐标
            pts_o = cv2.perspectiveTransform(pts_w, Minv_cur)
            if pts_o is not None:
                box_o = pts_o.reshape(-1, 2).astype(np.int32)
                cell_boxes.append(box_o)
                cx, cy = int(box_o[:, 0].mean()), int(box_o[:, 1].mean())
                cell_centers.append((cx, cy))
                # 画框，粗细设为 5
                cv2.drawContours(color_img, [box_o], 0, (0, 255, 0), 5)

    # --- 3. 核心模板匹配 & 路径生成 ---
    if (_PATH_POLY is not None or _PATH_DOTS is not None) and _CORE_TEMPLATE is not None:

        th, tw = _CORE_TEMPLATE.shape[:2]
        search_pad = 60  # 搜索范围向外扩 60 像素

        for idx in range(_TEMPLATES.shape[0]):
            # 获取搜索中心 (用户手画框的中心)
            # 我们只用它来确定一个大概范围，不依赖它的精确位置
            guess_box = _order_points(_TEMPLATES[idx])
            bx = int((min(guess_box[:, 0]) + max(guess_box[:, 0])) / 2)
            by = int((min(guess_box[:, 1]) + max(guess_box[:, 1])) / 2)

            # 确定搜索区域 ROI (Region of Interest)
            rx = max(0, bx - search_pad - tw // 2)
            ry = max(0, by - search_pad - th // 2)
            rw = tw + 2 * search_pad
            rh = th + 2 * search_pad

            # 边界检查，防止越界报错
            if rx + rw > goldW: rw = goldW - rx
            if ry + rh > goldH: rh = goldH - ry

            # 截取搜索区域
            roi = curr_warped[ry:ry + rh, rx:rx + rw]

            match_success = False
            offset = np.array([0, 0], dtype=np.float32)

            # --- 执行模板匹配 ---
            if roi.shape[0] >= th and roi.shape[1] >= tw:
                res = cv2.matchTemplate(roi, _CORE_TEMPLATE, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                # 匹配阈值设为 0.3
                # 因为去掉了边缘干扰，只要是正确的图案，匹配度通常都很高
                if max_val > 0.3:
                    match_success = True
                    # 还原匹配点到全图坐标 (模板左上角)
                    match_x = rx + max_loc[0]
                    match_y = ry + max_loc[1]
                    current_live_tl = np.array([match_x, match_y], dtype=np.float32)

                    # ⭐ 核心公式：计算物理偏移量 ⭐
                    # 偏移量 = (当前找到的坐标) - (0号基准坐标)
                    # 这实现了真正的“位置吸附”，完全脱离了手画框的误差。
                    offset = current_live_tl - _CORE_REF_TL
                else:
                    # 匹配失败
                    pass

            # --- 兜底逻辑 ---
            # 如果实在没匹配到 (极少情况)，退回到使用手画框的相对位移来估算
            # 保证画面不会空着，虽然位置可能不准
            if not match_success:
                box0 = _order_points(_TEMPLATES[0])
                c0_x = (min(box0[:, 0]) + max(box0[:, 0])) / 2
                c0_y = (min(box0[:, 1]) + max(box0[:, 1])) / 2

                manual_offset_x = bx - c0_x
                manual_offset_y = by - c0_y
                offset = np.array([manual_offset_x, manual_offset_y], dtype=np.float32)

            # --- 4. 绘制路径 ---
            # 将 0 号的路径加上偏移量，再投射回屏幕

            # 绘制折线 (粗细 5，绿色)
            if _PATH_POLY is not None:
                pts_gold = _PATH_POLY + offset
                pts_w = pts_gold.reshape(-1, 1, 2)
                res_o = cv2.perspectiveTransform(pts_w, Minv_cur)
                if res_o is not None:
                    pts = res_o.reshape(-1, 2).astype(np.int32)
                    for i in range(len(pts) - 1):
                        cv2.line(color_img, tuple(pts[i]), tuple(pts[i + 1]), (0, 255, 0), 5, cv2.LINE_AA)

            # 绘制点 (半径 10，绿色)
            if _PATH_DOTS is not None:
                pts_gold = _PATH_DOTS + offset
                pts_w = pts_gold.reshape(-1, 1, 2)
                res_o = cv2.perspectiveTransform(pts_w, Minv_cur)
                if res_o is not None:
                    pts = res_o.reshape(-1, 2).astype(np.int32)
                    for px, py in pts:
                        cv2.circle(color_img, (int(px), int(py)), 10, (0, 255, 0), -1)

    return color_img, big_box, cell_boxes, cell_centers


# ====== 请将此函数添加到 mold_detector.py 的最末尾 ======

def get_calculated_paths(gray: np.ndarray):
    """
    专门为 UI 提供的接口：
    返回当前画面中，所有识别到的单元格的点胶路径（像素坐标）。
    返回格式: [ path1, path2, ... ]
              其中 path1 是 (N, 2) 的 numpy 数组
    """
    # 确保环境已加载
    _load_golden_and_paths()

    # 1. 找大框 & 矫正矩阵
    rect, big_box = _find_big_mold(gray)
    if big_box is None or _TEMPLATES is None or _WARP_SIZE is None:
        return []

    goldW, goldH = int(_WARP_SIZE[0]), int(_WARP_SIZE[1])

    # 计算当前帧 -> 黄金图的透视矩阵
    pts = big_box.astype(np.float32)
    s = pts.sum(axis=1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
    rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([[0, 0], [goldW - 1, 0], [goldW - 1, goldH - 1], [0, goldH - 1]], dtype=np.float32)

    # 我们需要 Minv: 黄金坐标 -> 当前像素坐标
    Minv_cur = cv2.getPerspectiveTransform(dst_pts, rect_pts)

    # 2. 准备矫正后的图用于匹配
    M_cur = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    curr_warped = cv2.warpPerspective(gray, M_cur, (goldW, goldH))

    all_paths_pixels = []

    # 3. 遍历每个格子，计算路径
    if (_PATH_POLY is not None) and (_CORE_TEMPLATE is not None):
        th, tw = _CORE_TEMPLATE.shape[:2]
        search_pad = 60

        for idx in range(_TEMPLATES.shape[0]):
            # --- A. 模板匹配找偏移 ---
            guess_box = _order_points(_TEMPLATES[idx])
            bx = int((min(guess_box[:, 0]) + max(guess_box[:, 0])) / 2)
            by = int((min(guess_box[:, 1]) + max(guess_box[:, 1])) / 2)

            rx = max(0, bx - search_pad - tw // 2)
            ry = max(0, by - search_pad - th // 2)
            rw, rh = tw + 2 * search_pad, th + 2 * search_pad
            if rx + rw > goldW: rw = goldW - rx
            if ry + rh > goldH: rh = goldH - ry

            roi = curr_warped[ry:ry + rh, rx:rx + rw]

            offset = np.array([0, 0], dtype=np.float32)
            match_success = False

            if roi.shape[0] >= th and roi.shape[1] >= tw:
                res = cv2.matchTemplate(roi, _CORE_TEMPLATE, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > 0.3:
                    match_success = True
                    match_x = rx + max_loc[0]
                    match_y = ry + max_loc[1]
                    current_live_tl = np.array([match_x, match_y], dtype=np.float32)
                    offset = current_live_tl - _CORE_REF_TL

            # 匹配失败兜底
            if not match_success:
                box0 = _order_points(_TEMPLATES[0])
                c0_x = (min(box0[:, 0]) + max(box0[:, 0])) / 2
                c0_y = (min(box0[:, 1]) + max(box0[:, 1])) / 2
                offset = np.array([bx - c0_x, by - c0_y], dtype=np.float32)

            # --- B. 生成像素坐标路径 ---
            # 黄金坐标 = 原始路径 + 偏移
            pts_gold = _PATH_POLY + offset
            # 投射回当前画面的像素坐标
            pts_w = pts_gold.reshape(-1, 1, 2)
            res_o = cv2.perspectiveTransform(pts_w, Minv_cur)  # (N, 1, 2)

            if res_o is not None:
                all_paths_pixels.append(res_o.reshape(-1, 2))  # (N, 2)

    return all_paths_pixels