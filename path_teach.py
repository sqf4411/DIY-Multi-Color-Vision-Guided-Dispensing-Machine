# path_teach.py
import cv2
import numpy as np

WINDOW_NAME = "点胶路径示教窗口"


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


def run_path_teach(cell_index: int = 0,
                   img_path: str = "mold2.png") -> bool:
    """在黄金示教后的某一个小格子中编辑点胶路径。"""

    # ---------- 0. 读取黄金模板 ----------
    try:
        templates = np.load("slot_templates.npy")          # (4,4,2) in 黄金
        warp_size = np.load("warp_size.npy").astype(int)   # [W,H]
    except Exception as e:
        print("❌ 需要先完成黄金示教，未找到 slot_templates.npy / warp_size.npy：", e)
        return False

    goldW, goldH = int(warp_size[0]), int(warp_size[1])

    if not (0 <= cell_index < templates.shape[0]):
        print(f"❌ cell_index 越界: {cell_index}")
        return False

    # ---------- 1. 读入 mold2.png 并重新生成“黄金视角”整板 ----------
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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
    poly_file = f"cell_{cell_index}_polylines.npy"
    dots_file = f"cell_{cell_index}_dots.npy"

    polylines = []  # list of (N,2)
    dots = []       # list of (2,)
    current_line = []

    try:
        arr = np.load(poly_file, allow_pickle=True)
        for seg in arr:
            seg = np.array(seg, dtype=np.float32)
            if seg.size >= 2:
                polylines.append(seg)
        print(f"✅ 载入已有折线 {poly_file}, 条数 = {len(polylines)}")
    except Exception:
        print("ℹ 未找到历史折线文件，重新开始绘制")

    try:
        arr = np.load(dots_file)
        dots = [tuple(map(float, p)) for p in arr.reshape(-1, 2)]
        print(f"✅ 载入已有点集 {dots_file}, 点数 = {len(dots)}")
    except Exception:
        print("ℹ 未找到历史点文件，重新开始绘制")

    # ---------- 4.5 工具：小格坐标 -> 黄金坐标 ----------
    def local_to_golden(points: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts_local = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        pts_gold = cv2.perspectiveTransform(pts_local, Minv_cell)
        return pts_gold.reshape(-1, 2)

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
            pts = (seg * scale).astype(int)
            for i in range(len(pts) - 1):
                cv2.line(base_color, tuple(pts[i]), tuple(pts[i + 1]),
                         (0, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(base_color, tuple(pts[i]), 3, (0, 0, 255), -1)
                cv2.putText(base_color, str(idx),
                            (pts[i][0] + 3, pts[i][1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                idx += 1
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
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: Always add a dot
            dots.append((wx, wy))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, disp_w, disp_h + 30)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print(f"🧵 打开点胶路径示教窗口 (cell_index={cell_index})")

    while True:
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
            current_line = []
        elif key in (ord('c'), ord('C')):
            polylines.clear()
            current_line = []
            dots.clear()
        elif key in (ord('z'), ord('Z')):
            if mode == "line":
                if current_line:
                    current_line.pop()
                elif polylines:
                    polylines.pop()
            elif mode == "dot":
                if dots:
                    dots.pop()
        elif key in (ord('q'), ord('Q')):
            print("⚠ 用户取消，不保存点胶路径")
            cv2.destroyWindow(WINDOW_NAME)
            return False
        elif key in (ord('s'), ord('S')):
            # 先把 current_line 也并入
            if len(current_line) >= 2:
                polylines.append(np.array(current_line, dtype=np.float32))
                current_line = []

            # 1) 保存小格坐标数据（编辑用）
            np.save(poly_file, np.array(polylines, dtype=object))
            if dots:
                np.save(dots_file, np.array(dots, dtype=np.float32))
            else:
                np.save(dots_file, np.zeros((0, 2), dtype=np.float32))

            # 2) 生成黄金坐标系下的显示路径
            try:
                poly_disp_list = []
                if polylines:
                    # 保持多段线结构，不要合并
                    for seg in polylines:
                        seg_gold = local_to_golden(seg)
                        poly_disp_list.append(seg_gold)
                
                # 保存为对象数组 (允许不规则形状)
                np.save(f"disp_poly_cell{cell_index}.npy", np.array(poly_disp_list, dtype=object))

                if dots:
                    dots_arr = np.array(dots, dtype=np.float32)
                    dots_disp = local_to_golden(dots_arr)
                else:
                    dots_disp = np.zeros((0, 2), dtype=np.float32)

                np.save(f"disp_dots_cell{cell_index}.npy", dots_disp)
                print(f"✅ 已保存显示路径 disp_poly_cell{cell_index}.npy / disp_dots_cell{cell_index}.npy")
            except Exception as e:
                print("⚠ 生成显示路径失败：", e)

            print(f"✅ 已保存折线到 {poly_file}，条数 = {len(polylines)}")
            print(f"✅ 已保存点集到 {dots_file}，点数 = {len(dots)}")
            cv2.destroyWindow(WINDOW_NAME)
            return True
        # 其它按键忽略
