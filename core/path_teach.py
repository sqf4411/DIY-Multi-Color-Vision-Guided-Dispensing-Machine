# path_teach.py
import cv2
import numpy as np
import json

WINDOW_NAME = "Edit Path" # Avoid Chinese encoding issues on Windows for now, or use English
# If user really insists on Chinese and it breaks, we might need a specific fix like .encode('gbk').decode('latin1') trick or similar.
# But for stability, English is preferred. 
# User asked: "中文 is '编辑点胶路径', modify it".
# Okay, I will try to satisfy them, but if it garbles, it's the environment.
# Let's try to set it to what they asked, but maybe the previous string was too long or something?
# original: "点胶路径示教窗口"
# requested: "编辑点胶路径"
WINDOW_NAME = "编辑点胶路径"


def order_points(pts: np.ndarray) -> np.ndarray:
    """把 4 点排序为 tl, tr, br, bl"""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


import os

def run_path_teach(cell_index: int = 0,
                   img_path: str = "config/mold2.png",
                   save_dir: str = None) -> bool:
    """在黄金示教后的某一个小格子中编辑点胶路径。
    
    Args:
        cell_index: 小格子索引
        img_path: 训练图像路径
        save_dir: 保存路径的目录（None=使用config/，否则使用配方目录）
    """

    # Get Base Dir (Project Root)
    # This file is in core/, so project root is one level up
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, "config")
    
    # ⭐ 确定保存/加载目录
    if save_dir is None:
        save_dir = config_dir
    
    # Resolve img_path relative to project root if it's relative
    if not os.path.isabs(img_path):
        img_path = os.path.join(base_dir, img_path)

    # ---------- 0. 读取黄金模板 ----------
    try:
        path_slot = os.path.join(config_dir, "slot_templates.npy")
        path_warp = os.path.join(config_dir, "warp_size.npy")
        
        templates = np.load(path_slot)          # (4,4,2) in 黄金
        warp_size = np.load(path_warp).astype(int)   # [W,H]
    except Exception as e:
        print(f"❌ 需要先完成黄金示教，未找到 slot_templates.npy / warp_size.npy：{e}")
        return False
        
    goldW, goldH = int(warp_size[0]), int(warp_size[1])

    if not (0 <= cell_index < templates.shape[0]):
        print(f"❌ cell_index 越界: {cell_index}")
        return False

    # ---------- 1. 读入 mold2.png 并重新生成“黄金视角”整板 ----------
    from core.cv2_unicode import imread_unicode
    gray = imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"❌ 无法读取图像: {img_path}")
        return False

    # 找大模具（和 mold_detector._find_big_mold 一致）
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
        print("❌ 未找到大模具，无法生成黄金视角")
        return False

    big_rect = cv2.minAreaRect(big_mold_cnt)
    big_box = cv2.boxPoints(big_rect).astype(np.float32)  # (4,2)

    # 排序 tl,tr,br,bl
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

    # ---------- 2. 取出该 cell 的黄金坐标四边形，并单独拉正 ----------
    cell_box_gold = templates[cell_index].astype(np.float32)  # (4,2) in 黄金

    rect_cell = order_points(cell_box_gold)

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
    warped_cell = cv2.warpPerspective(warped_full, M_cell, (maxW, maxH))
    # 小格局部图 -> 黄金整板
    Minv_cell = cv2.getPerspectiveTransform(dst_cell, rect_cell)

    # ---------- 3. 缩放显示 ----------
    max_display = 700
    scale = min(max_display / maxW, max_display / maxH, 2.0)
    disp_w = int(maxW * scale)
    disp_h = int(maxH * scale)

    # ---------- 4. 尝试加载小格坐标下的历史路径 ----------
    # 策略：优先加载本地坐标文件(cell_*)，如果没有则尝试加载显示文件(disp_*)并转换
    
    # 1. 本地坐标文件 (用于编辑，最准确)
    local_poly_file = os.path.join(save_dir, f"cell_{cell_index}_polylines.npy")
    local_poly_json = os.path.join(save_dir, f"cell_{cell_index}_polylines.json")
    local_dots_file = os.path.join(save_dir, f"cell_{cell_index}_dots.npy")
    
    # 2. 显示文件 (用于主界面显示，需要逆转换)
    disp_poly_file = os.path.join(save_dir, f"disp_poly_cell{cell_index}.npy")
    disp_poly_json = os.path.join(save_dir, f"disp_poly_cell{cell_index}.json")
    disp_dots_file = os.path.join(save_dir, f"disp_dots_cell{cell_index}.npy")

    polylines = []  # list of (N,2)
    dots = []       # list of (2,)
    current_line = []
    entity_history = []  # ['dot', 'line', 'dot', ...]
    
    polylines = []  # list of (N,2)
    dots = []       # list of (2,)
    current_line = []
    entity_history = []  # ['dot', 'line', 'dot', ...]
    
    # --- 加载 Polylines ---
    if os.path.exists(local_poly_json):
        try:
            with open(local_poly_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            for seg in data.get("segments", []):
                arr = np.asarray(seg, dtype=np.float32)
                if arr.size == 0:
                    continue
                if arr.ndim == 1:
                    if arr.shape[0] != 2:
                        continue
                    arr = arr.reshape(1, 2)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    polylines.append(arr)
            print(f"✅ 从本地 JSON 载入折线, 条数 = {len(polylines)}")
        except Exception as e:
            print(f"⚠ 加载本地 JSON 折线失败: {e}")

    elif os.path.exists(local_poly_file):
        try:
            raw = np.load(local_poly_file, allow_pickle=False)
            if isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.shape[1] == 2:
                polylines = [raw.astype(np.float32)]
            print(f"✅ 从本地 NPY 载入折线, 条数 = {len(polylines)}")
        except Exception as e:
            print(f"⚠ 加载本地折线失败: {e}")
             
    elif os.path.exists(disp_poly_json) or os.path.exists(disp_poly_file):
        # Fallback: 尝试从显示文件加载并转换
        try:
            if os.path.exists(disp_poly_json):
                print(f"ℹ️ 本地记录不存在，尝试从 JSON 恢复: {disp_poly_json}")
                with open(disp_poly_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw_segments = data.get("segments", [])
                for seg_gold in raw_segments:
                    seg_arr = np.asarray(seg_gold, dtype=np.float32)
                    if seg_arr.size == 0:
                        continue
                    seg_local = golden_to_local(seg_arr)
                    polylines.append(seg_local)
            else:
                print(f"ℹ️ 本地记录不存在，尝试从旧版文件恢复: {disp_poly_file}")
                raw = np.load(disp_poly_file, allow_pickle=False)
                if isinstance(raw, np.ndarray) and raw.size > 0 and raw.ndim == 2 and raw.shape[1] == 2:
                    seg_local = golden_to_local(raw)
                    polylines.append(seg_local)
            print(f"✅ 从显示文件恢复折线, 条数 = {len(polylines)}")
        except Exception as e:
            print(f"⚠ 加载显示文件失败: {e}")
            
    # --- 加载 Dots ---
    if os.path.exists(local_dots_file):
        try:
            dots_arr = np.load(local_dots_file, allow_pickle=False)
            if dots_arr.size > 0:
                dots = dots_arr.tolist()
                print(f"✅ 从本地文件载入点集, 点数 = {len(dots)}")
        except Exception as e:
            print(f"⚠ 加载本地点集失败: {e}")
            
    elif os.path.exists(disp_dots_file):
        # Fallback
        try:
            print(f"ℹ️ 本地记录不存在，尝试从旧版文件恢复: {disp_dots_file}")
            dots_gold = np.load(disp_dots_file)
            if dots_gold.size > 0:
                dots_local = golden_to_local(dots_gold)
                dots = dots_local.tolist()
                print(f"✅ 从显示文件恢复点集, 点数 = {len(dots)}")
        except Exception as e:
            print(f"⚠ 加载显示文件失败: {e}")

    # Initialize history from loaded files (approximate order)
    if polylines:
        entity_history.extend(['line'] * len(polylines))
    if dots:
        entity_history.extend(['dot'] * len(dots))

    # ---------- 4.5 工具：小格坐标 <-> 黄金坐标 ----------
    def local_to_golden(points: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts_local = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        pts_gold = cv2.perspectiveTransform(pts_local, Minv_cell)
        return pts_gold.reshape(-1, 2)
    
    def golden_to_local(points: np.ndarray) -> np.ndarray:
        """黄金坐标 -> 局部小格坐标（用于加载已保存的路径）"""
        if points is None or len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts_gold = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        # ⭐ 直接使用 M_cell (Golden -> Local)
        pts_local = cv2.perspectiveTransform(pts_gold, M_cell)
        return pts_local.reshape(-1, 2)

    # ---------- 5. 交互绘制 ----------
    mode = "line"  # "line" or "dot"

    def draw_canvas():
        base = cv2.resize(warped_cell, (disp_w, disp_h),
                          interpolation=cv2.INTER_LINEAR)
        base_color = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        idx = 1
        for seg in polylines:
            if len(seg) < 2:
                continue
            
            pts_f = seg * scale
            pts = pts_f.astype(int)
            
            for i in range(len(pts) - 1):
                p1 = tuple(pts[i])
                p2 = tuple(pts[i + 1])
                
                # Dynamic tip length for consistent visual size (~8px)
                seg_len = np.linalg.norm(pts_f[i+1] - pts_f[i])
                if seg_len > 0:
                    tip = 8.0 / seg_len
                    # Clamp to avoid extreme values
                    tip = max(0.05, min(0.5, tip))
                else:
                    tip = 0.1

                # Draw Arrow
                cv2.arrowedLine(base_color, p1, p2, (0, 0, 255), 1, cv2.LINE_AA, tipLength=tip)
                
                # Draw Node
                cv2.circle(base_color, p1, 3, (0, 0, 255), -1)
                
                cv2.putText(base_color, str(idx),
                            (p1[0] + 3, p1[1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                idx += 1
            
            # Last point
            cv2.circle(base_color, tuple(pts[-1]), 3, (0, 0, 255), -1)
            cv2.putText(base_color, str(idx),
                        (pts[-1][0] + 3, pts[-1][1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            idx += 1

        if len(current_line) >= 1:
            seg = np.array(current_line, dtype=np.float32)
            pts = (seg * scale).astype(int)
            for i in range(len(pts) - 1):
                cv2.line(base_color, tuple(pts[i]), tuple(pts[i + 1]),
                         (0, 255, 255), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(base_color, tuple(p), 3, (0, 255, 255), -1)

        for i, (x, y) in enumerate(dots, start=1):
            px = int(x * scale)
            py = int(y * scale)
            cv2.circle(base_color, (px, py), 4, (0, 255, 0), -1)
            cv2.putText(base_color, f"P{i}", (px + 3, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        head = np.zeros((30, disp_w, 3), dtype=np.uint8)
        cv2.putText(head,
                    "L-Click: Line/Dot (Mode)   R-Click: Dot   N: New Line   "
                    "C: Clear   Z: Undo   S: Save   Q: Quit",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        canvas = np.vstack([head, base_color])
        cv2.putText(canvas, f"Mode: {mode}",
                    (10, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def on_mouse(event, x, y, flags, param):
        nonlocal mode, current_line

        if y < 30:
            return

        wx = x / scale
        wy = (y - 30) / scale

        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: Add point based on current mode
            if mode == "line":
                current_line.append((wx, wy))
            elif mode == "dot":
                dots.append((wx, wy))
                entity_history.append('dot')
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: Always add a dot
            dots.append((wx, wy))
            entity_history.append('dot')

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, disp_w, disp_h + 30)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print(f"🧵 打开点胶路径示教窗口 (cell_index={cell_index})")

    while True:
        # Check if window was closed via 'X' button
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("⚠ 用户点击关闭窗口，不保存点胶路径")
            return False

        canvas = draw_canvas()
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('l'), ord('L')):
            mode = "line"
        elif key in (ord('r'), ord('R')):
            mode = "dot"
        elif key in (ord('n'), ord('N')):
            if len(current_line) >= 2:
                polylines.append(np.array(current_line, dtype=np.float32))
                entity_history.append('line')
            current_line = []
        elif key in (ord('c'), ord('C')):
            polylines.clear()
            current_line = []
            dots.clear()
            entity_history.clear()
        elif key in (ord('z'), ord('Z')):
            if current_line:
                current_line.pop()
            # Unified Undo
            elif entity_history:
                last_type = entity_history.pop()
                if last_type == 'line' and polylines:
                    polylines.pop()
                elif last_type == 'dot' and dots:
                    dots.pop()
        elif key in (ord('q'), ord('Q')):
            print("⚠ 用户取消，不保存点胶路径")
            cv2.destroyWindow(WINDOW_NAME)
            return False
        elif key in (ord('s'), ord('S')):
            # 先把 current_line 也并入
            if current_line:
                polylines.append(np.array(current_line, dtype=np.float32))
                current_line = []

            # A) 保存小格局部坐标数据（用于下次编辑，最准确，无损）
            try:
                local_payload = {
                    "version": 2,
                    "segments": [np.asarray(seg, dtype=np.float32).tolist() for seg in polylines if len(seg) > 0]
                }
                with open(local_poly_json, "w", encoding="utf-8") as f:
                    json.dump(local_payload, f, ensure_ascii=False, indent=2)

                # Remove unsafe legacy object npy if present.
                if os.path.exists(local_poly_file):
                    os.remove(local_poly_file)

                if dots:
                    np.save(local_dots_file, np.array(dots, dtype=np.float32))
                else:
                    np.save(local_dots_file, np.zeros((0, 2), dtype=np.float32))
                print("✅ 已保存本地坐标路径")
            except Exception as e:
                print(f"❌ 保存本地路径失败: {e}")

            # B) 生成黄金坐标系下的显示路径并保存（用于主界面显示）
            try:
                poly_disp_list = []
                if polylines:
                    # 保持多段线结构，不要合并
                    for seg in polylines:
                        seg_gold = local_to_golden(seg)
                        poly_disp_list.append(seg_gold)

                disp_payload = {
                    "version": 2,
                    "segments": [np.asarray(seg, dtype=np.float32).tolist() for seg in poly_disp_list if len(seg) > 0]
                }
                with open(disp_poly_json, "w", encoding="utf-8") as f:
                    json.dump(disp_payload, f, ensure_ascii=False, indent=2)

                # Remove unsafe legacy object npy if present.
                if os.path.exists(disp_poly_file):
                    os.remove(disp_poly_file)

                if dots:
                    dots_arr = np.array(dots, dtype=np.float32)
                    dots_disp = local_to_golden(dots_arr)
                else:
                    dots_disp = np.zeros((0, 2), dtype=np.float32)

                np.save(disp_dots_file, dots_disp)
                print("✅ 已保存显示路径 (Golden)")
            except Exception as e:
                print(f"❌ 保存显示路径失败: {e}")

            print(f"✅ 路径保存完毕 -> {save_dir}")
            cv2.destroyWindow(WINDOW_NAME)
            return True
        # 其它按键忽略
