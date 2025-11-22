# path_teach.py
import cv2
import numpy as np
from mold_detector import detect_mold_and_cells

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
                   img_path: str = "mold.png") -> bool:
    """
    在黄金示教的某一个小格子里编辑点胶路径

    参数
    ----
    cell_index : 0~3  对应 4 个小黄框
    img_path   : 黄金示教时拍的整板图片（mold.png）

    返回
    ----
    True  : 用户按 S 保存
    False : 用户按 Q 或中途出错
    """

    # ============ 1. 读整板图 + 用黄金模板找 4 个小格子 ============
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"❌ 无法读取 {img_path} ，请确认路径和文件是否存在")
        return False

    color_img, big_box, cell_boxes, cell_centers = detect_mold_and_cells(gray)
    if not cell_boxes or big_box is None:
        print("❌ detect_mold_and_cells 未找到小黄框，请先做好黄金示教")
        return False

    if not (0 <= cell_index < len(cell_boxes)):
        print(f"❌ cell_index={cell_index} 越界，目前只有 {len(cell_boxes)} 个小格子")
        return False

    box = cell_boxes[cell_index].astype(np.float32)
    rect = order_points(box)

    # ============ 2. 对该小格子做透视矫正，得到正向小图 ============
    widthA = np.linalg.norm(rect[2] - rect[3])
    widthB = np.linalg.norm(rect[1] - rect[0])
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(rect[1] - rect[2])
    heightB = np.linalg.norm(rect[0] - rect[3])
    maxH = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray, M, (maxW, maxH))

    # -------- 缩放到合适大小（只影响显示，不影响保存坐标） --------
    max_display = 700  # 窗口最长边不超过 700 像素
    scale = min(max_display / maxW, max_display / maxH, 1.0)
    disp_w = int(maxW * scale)
    disp_h = int(maxH * scale)

    # ============ 3. 载入之前保存的路径（如果有的话） ============
    poly_file = f"disp_poly_cell{cell_index}.npy"
    dots_file = f"disp_dots_cell{cell_index}.npy"

    polylines = []      # list[np.ndarray(N,2)]
    current_line = []   # 正在画的这一条线（列表）
    dots = []           # list[(x,y)]

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

    # ============ 4. 交互绘制 ============
    mode = "line"  # "line" 或 "dot"

    def draw_canvas():
        """把 warped + 点 + 线 + 文字 画出来"""
        base = cv2.resize(warped, (disp_w, disp_h),
                          interpolation=cv2.INTER_LINEAR)
        base_color = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        # 多条线
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
            # 最后一个点
            cv2.circle(base_color, tuple(pts[-1]), 3, (0, 0, 255), -1)
            cv2.putText(base_color, str(idx),
                        (pts[-1][0] + 3, pts[-1][1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            idx += 1

        # 当前正在画的这条线
        if len(current_line) >= 1:
            seg = np.array(current_line, dtype=np.float32)
            pts = (seg * scale).astype(int)
            for i in range(len(pts) - 1):
                cv2.line(base_color, tuple(pts[i]), tuple(pts[i + 1]),
                         (0, 255, 255), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(base_color, tuple(p), 3, (0, 255, 255), -1)

        # 点胶点
        for i, (x, y) in enumerate(dots, start=1):
            px = int(x * scale)
            py = int(y * scale)
            cv2.circle(base_color, (px, py), 4, (0, 255, 0), -1)
            cv2.putText(base_color, f"P{i}", (px + 3, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 顶部提示
        head = np.zeros((30, disp_w, 3), dtype=np.uint8)
        cv2.putText(head, "L: line   R: dot   N: new line   "
                          "C: clear   S: save   Q: quit",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        canvas = np.vstack([head, base_color])

        # 底部模式提示
        cv2.putText(canvas, f"Mode: {mode}",
                    (10, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def on_mouse(event, x, y, flags, param):
        nonlocal mode, current_line

        # 头部 30 像素是黑条，点击那一块忽略
        if y < 30:
            return

        # 显示坐标 -> warped 坐标
        wx = x / scale
        wy = (y - 30) / scale

        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == "line":
                current_line.append((wx, wy))
            elif mode == "dot":
                dots.append((wx, wy))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, disp_w, disp_h + 30)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print(f"🧵 打开点胶路径示教窗口 (使用 {img_path}, 小格 index={cell_index})")

    while True:
        canvas = draw_canvas()
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('l'), ord('L')):
            mode = "line"
        elif key in (ord('r'), ord('R')):
            mode = "dot"
        elif key in (ord('n'), ord('N')):
            # 把当前线收进 polylines，开启新线
            if len(current_line) >= 2:
                polylines.append(np.array(current_line, dtype=np.float32))
            current_line = []
        elif key in (ord('c'), ord('C')):
            polylines.clear()
            current_line = []
            dots.clear()
        elif key in (ord('q'), ord('Q')):
            print("⚠ 用户取消，不保存点胶路径")
            cv2.destroyWindow(WINDOW_NAME)
            return False
        elif key in (ord('s'), ord('S')):
            # 先把 current_line 也并入
            if len(current_line) >= 2:
                polylines.append(np.array(current_line, dtype=np.float32))
                current_line = []

            # 保存折线（object 数组）
            np.save(poly_file, np.array(polylines, dtype=object))
            # 保存点
            if dots:
                np.save(dots_file, np.array(dots, dtype=np.float32))
            else:
                # 没有点就保存一个空数组
                np.save(dots_file, np.zeros((0, 2), dtype=np.float32))

            print(f"✅ 已保存折线到 {poly_file}，条数 = {len(polylines)}")
            print(f"✅ 已保存点集到 {dots_file}，点数 = {len(dots)}")
            cv2.destroyWindow(WINDOW_NAME)
            return True
        # 其它按键忽略

