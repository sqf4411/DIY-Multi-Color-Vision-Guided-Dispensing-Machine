# mold_detector.py
import cv2
import numpy as np
import os

# ====== 全局缓存 ======
_TEMPLATES = None  # (4,4,2) 黄框坐标 (用户手画的)
_WARP_SIZE = None  # (2,)    [W, H]

_PATH_POLY = None  # (N,2)   Cell 0 路径
_PATH_DOTS = None  # (K,2)   Cell 0 点集

# ⭐ 智能生成的独立模板库 ⭐
# 结构: { idx: {'img': template_image, 'offset': (dx, dy)} }
# offset 表示：该模板左上角相对于“标准路径原点”的偏移量
_SMART_TEMPLATES = {}


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
    """加载数据并执行【自动校准】生成智能模板"""
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS, _SMART_TEMPLATES

    if _TEMPLATES is None:
        try:
            _TEMPLATES = np.load("slot_templates.npy")
            _WARP_SIZE = np.load("warp_size.npy").astype(int)
            print(f"[mold_detector] 黄金框数据加载: {_TEMPLATES.shape}")
        except:
            print("[mold_detector] 缺少 slot_templates.npy")
            return

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

    # ⭐ 核心：启动时自动校准 ⭐
    # 这一步解决“手画框不准”和“光照不同”两个问题
    if not _SMART_TEMPLATES and os.path.exists("mold2.png"):
        try:
            gold_raw = cv2.imread("mold2.png", cv2.IMREAD_GRAYSCALE)
            if gold_raw is None: return

            _, gold_big_box = _find_big_mold(gold_raw)
            if gold_big_box is None: return

            goldW, goldH = int(_WARP_SIZE[0]), int(_WARP_SIZE[1])
            M_gold = _get_perspective_matrix(gold_big_box, goldW, goldH)
            gold_warped = cv2.warpPerspective(gold_raw, M_gold, (goldW, goldH))

            # 1. 提取 0 号作为“基准特征”
            box0 = _order_points(_TEMPLATES[0])
            x0, y0 = int(min(box0[:, 0])), int(min(box0[:, 1]))
            x1, y1 = int(max(box0[:, 0])), int(max(box0[:, 1]))

            # 收缩一点边缘，避免包含边框线，提高匹配准确度
            margin = 5
            x0, y0 = max(0, x0 + margin), max(0, y0 + margin)
            x1, y1 = min(goldW, x1 - margin), min(goldH, y1 - margin)

            master_tpl = gold_warped[y0:y1, x0:x1].copy()
            master_ref_tl = np.array([x0, y0], dtype=np.float32)  # 0号的绝对坐标

            # 2. 遍历所有位置，进行【内部自对齐】
            for idx in range(_TEMPLATES.shape[0]):
                # 获取该位置的粗略区域
                box_i = _order_points(_TEMPLATES[idx])
                ix, iy = int(min(box_i[:, 0])), int(min(box_i[:, 1]))

                # 在该区域附近搜索 0 号模板
                search_pad = 60
                sx = max(0, ix - search_pad)
                sy = max(0, iy - search_pad)
                sw = (x1 - x0) + 2 * search_pad
                sh = (y1 - y0) + 2 * search_pad

                if sx + sw > goldW: sw = goldW - sx
                if sy + sh > goldH: sh = goldH - sy

                roi = gold_warped[sy:sy + sh, sx:sx + sw]

                # 在黄金图内部找对齐点
                res = cv2.matchTemplate(roi, master_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > 0.4:  # 在黄金图内部，相似度应该很高
                    match_x = sx + max_loc[0]
                    match_y = sy + max_loc[1]

                    # 重点：在对齐的位置，提取【该位置独有】的模板
                    # 这样模板就包含了该位置独特的光照信息
                    new_tpl = gold_warped[match_y: match_y + (y1 - y0), match_x: match_x + (x1 - x0)].copy()

                    # 计算这个新模板相对于 0 号基准的偏移
                    # 如果 idx=0，这个偏移应该是 (0,0)
                    # 这里的 offset 是为了将来把 0 号路径搬过来用的
                    # 逻辑：Live_Pos + Offset = Path_Pos ? 不对
                    # 逻辑：路径是基于 master_ref_tl 的。
                    # 我们找到了 master_tpl 在 idx 处的坐标是 (match_x, match_y)
                    # 这意味着：(match_x, match_y) 等价于 0号的 (x0, y0)
                    # 所以，只要我们在 Live 图里找到了这个 new_tpl，那个位置就等价于 (x0, y0)
                    # 路径偏移量 = (Path - (x0, y0))

                    _SMART_TEMPLATES[idx] = {
                        'img': new_tpl,
                        'ref_origin': np.array([x0, y0], dtype=np.float32)  # 始终记录 0 号的原点
                    }
                else:
                    print(f"[mold_detector] 位置 {idx} 自动对齐失败，将使用原始框")
                    # 降级处理：直接切原始框
                    pass

            print(f"[mold_detector] 自动校准完成，生成 {len(_SMART_TEMPLATES)} 个智能模板")

        except Exception as e:
            print(f"[mold_detector] 自动校准出错: {e}")


def reload_golden_templates():
    global _TEMPLATES, _WARP_SIZE, _PATH_POLY, _PATH_DOTS, _SMART_TEMPLATES
    _TEMPLATES = None
    _WARP_SIZE = None
    _PATH_POLY = None
    _PATH_DOTS = None
    _SMART_TEMPLATES = {}
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

    # 3. 绘制搜索区域 (黄框) - 修复之前消失的问题
    for idx in range(_TEMPLATES.shape[0]):
        # 这里依然用手画框做可视化，告诉用户大概在哪里搜
        box_w = _TEMPLATES[idx].astype(np.float32)
        pts_w = box_w.reshape(-1, 1, 2)
        if pts_w.size > 0:
            pts_o = cv2.perspectiveTransform(pts_w, Minv_cur)
            if pts_o is not None:
                box_o = pts_o.reshape(-1, 2).astype(np.int32)
                cell_boxes.append(box_o)
                cx, cy = int(box_o[:, 0].mean()), int(box_o[:, 1].mean())
                cell_centers.append((cx, cy))
                cv2.drawContours(color_img, [box_o], 0, (0, 255, 255), 1)  # 细黄线

    # 4. 智能匹配与路径绘制
    if (_PATH_POLY is not None or _PATH_DOTS is not None) and len(_SMART_TEMPLATES) > 0:

        search_pad = 50

        for idx, data in _SMART_TEMPLATES.items():
            tpl = data['img']
            ref_origin = data['ref_origin']  # 这是 0 号在黄金图里的绝对坐标 (x0, y0)

            th, tw = tpl.shape[:2]

            # 确定搜索区域 (基于手画框中心)
            guess_box = _order_points(_TEMPLATES[idx])
            bx, by = int(min(guess_box[:, 0])), int(min(guess_box[:, 1]))

            rx = max(0, bx - search_pad)
            ry = max(0, by - search_pad)
            rw = tw + 2 * search_pad
            rh = th + 2 * search_pad

            if rx + rw > goldW: rw = goldW - rx
            if ry + rh > goldH: rh = goldH - ry

            roi = curr_warped[ry:ry + rh, rx:rx + rw]

            match_success = False
            if roi.shape[0] >= th and roi.shape[1] >= tw:
                res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > 0.35:  # 降低阈值，适应性更强
                    match_success = True
                    live_x = rx + max_loc[0]
                    live_y = ry + max_loc[1]

                    # 核心逻辑：
                    # live_x, live_y 是当前模具的特征点位置
                    # ref_origin 是 0 号模具的特征点位置
                    # 路径是基于 ref_origin 的
                    # 所以：Current_Path = Path_Data - ref_origin + (live_x, live_y)

                    shift = np.array([live_x, live_y], dtype=np.float32) - ref_origin

                    # 绘制折线
                    if _PATH_POLY is not None:
                        pts_gold = _PATH_POLY + shift
                        pts_w = pts_gold.reshape(-1, 1, 2)
                        res_o = cv2.perspectiveTransform(pts_w, Minv_cur)
                        if res_o is not None:
                            pts = res_o.reshape(-1, 2).astype(np.int32)
                            for i in range(len(pts) - 1):
                                cv2.line(color_img, tuple(pts[i]), tuple(pts[i + 1]), (255, 0, 0), 1, cv2.LINE_AA)

                    # 绘制点
                    if _PATH_DOTS is not None:
                        pts_gold = _PATH_DOTS + shift
                        pts_w = pts_gold.reshape(-1, 1, 2)
                        res_o = cv2.perspectiveTransform(pts_w, Minv_cur)
                        if res_o is not None:
                            pts = res_o.reshape(-1, 2).astype(np.int32)
                            for px, py in pts:
                                cv2.circle(color_img, (int(px), int(py)), 3, (255, 0, 255), -1)

            if match_success:
                # 匹配成功画绿框 (调试用)
                # p1 = (int(live_x), int(live_y))
                # p2 = (int(live_x+tw), int(live_y+th))
                # cv2.rectangle(curr_warped, p1, p2, 255, 2) # 注意这是在 warped 图上画的，外面看不到
                pass

    return color_img, big_box, cell_boxes, cell_centers