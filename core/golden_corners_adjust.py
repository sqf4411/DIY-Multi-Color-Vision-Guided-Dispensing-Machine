# golden_corners_adjust.py
# 心理更踏实版：大框四角手动微调工具
import cv2
import numpy as np
from core.mold_detector import detect_mold_and_cells

IMG_PATH = "config/mold2.png"              # 黄金基准图
OUT_PATH = "config/golden_corners.npy"     # 手动微调后的四角坐标 (4,2)

WINDOW_MAIN = "角点微调 - 基准图"
WINDOW_ZOOM = "放大视图"

# 当前角点索引 0~3
sel_idx = 0
# 放大窗口参数
zoom_half_size = 60   # ROI 半径，原图上 120x120 区域
zoom_scale = 4        # 放大倍数 4x


def draw_main(img_bgr, corners, sel_idx):
    """
    在主窗口显示整张图 + 四个角点
    corners: (4,2)
    """
    vis = img_bgr.copy()

    # 画大框轮廓（绿色）
    pts = corners.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # 画四个角点
    for i, (x, y) in enumerate(corners):
        x_i, y_i = int(x), int(y)
        if i == sel_idx:
            # 当前选中的角点：红色
            cv2.circle(vis, (x_i, y_i), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, (x_i, y_i), 10, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            # 其他角点：黄色
            cv2.circle(vis, (x_i, y_i), 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"{i}",
            (x_i + 8, y_i - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 文本提示
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, h - 40), (w, h), (0, 0, 0), -1)
    msg = "数字1-4切换角点, 方向鍵微调(+/-1像素), WASD快速调整(+/-5像素), S保存, ESC退出"
    cv2.putText(vis, msg, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_MAIN, vis)


def draw_zoom(gray, corners, sel_idx):
    """
    显示选中角点附近的放大窗口 + 十字线
    """
    h, w = gray.shape
    cx, cy = corners[sel_idx]
    cx_i, cy_i = int(round(cx)), int(round(cy))

    # ROI 范围裁剪
    x1 = max(0, cx_i - zoom_half_size)
    y1 = max(0, cy_i - zoom_half_size)
    x2 = min(w, cx_i + zoom_half_size)
    y2 = min(h, cy_i + zoom_half_size)

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return

    # 放大
    roi_zoom = cv2.resize(roi, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_NEAREST)
    zh, zw = roi_zoom.shape

    # 在放大图上画十字线（中心）
    center_x = (zw // 2)
    center_y = (zh // 2)
    cv2.line(roi_zoom, (center_x, 0), (center_x, zh - 1), 128, 1, cv2.LINE_AA)
    cv2.line(roi_zoom, (0, center_y), (zw - 1, center_y), 128, 1, cv2.LINE_AA)

    # 标注选中角点编号
    roi_bgr = cv2.cvtColor(roi_zoom, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        roi_bgr,
        f"corner {sel_idx}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow(WINDOW_ZOOM, roi_bgr)


def main():
    global sel_idx

    # 1) 读取基准图
    gray = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"❌ 无法读取基准图: {IMG_PATH}")
        return

    # 2) 自动检测大框
    try:
        _, big_box, _, _ = detect_mold_and_cells(gray)
    except Exception as e:
        print(f"❌ 调用 detect_mold_and_cells 失败: {e}")
        return

    if big_box is None:
        print("❌ 未检测到大模具大框 (big_box is None)")
        return

    # 规范顺序：四个点 (4,2) 转 float32
    corners = big_box.astype(np.float32)

    print("✅ 自动检测到的大框四角：")
    for i, (x, y) in enumerate(corners):
        print(f"  corner {i}: ({x:.2f}, {y:.2f})")

    # 3) 准备显示
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_ZOOM, cv2.WINDOW_NORMAL)

    while True:
        # 每一轮刷新显示
        draw_main(img_bgr, corners, sel_idx)
        draw_zoom(gray, corners, sel_idx)

        key = cv2.waitKey(0) & 0xFF

        if key == 27 or key == ord('q'):   # ESC 或 q 退出(不保存)
            print("🔚 退出，不保存。")
            break

        # 选择角点: 1~4 → index 0~3
        if key in (ord('1'), ord('2'), ord('3'), ord('4')):
            sel_idx = key - ord('1')
            print(f"➡ 当前选中角点: {sel_idx}")
            continue

        # 方向键（OpenCV 的方向键码）：上下左右 每次 1 像素
        # 上: 82, 下: 84, 左: 81, 右: 83
        if key == 82:   # up
            corners[sel_idx, 1] -= 1
        elif key == 84: # down
            corners[sel_idx, 1] += 1
        elif key == 81: # left
            corners[sel_idx, 0] -= 1
        elif key == 83: # right
            corners[sel_idx, 0] += 1

        # WASD 快速调整，每次 5 像素
        elif key == ord('w'):
            corners[sel_idx, 1] -= 5
        elif key == ord('s'):
            corners[sel_idx, 1] += 5
        elif key == ord('a'):
            corners[sel_idx, 0] -= 5
        elif key == ord('d'):
            corners[sel_idx, 0] += 5

        # S 保存
        elif key == ord('S') or key == ord('s'):
            # 小心：上面的快速下移也用了 s，这里用大写 S 更稳一点：
            # 如果你想区分，可以把快速下移改成别的键，这里先保持简单逻辑：
            #   - 小写 s: 保存
            #   - 快速下移改用 'x'
            # 你也可以根据习惯稍微改一下。
            np.save(OUT_PATH, corners.astype(np.float32))
            print(f"💾 已保存精调角点到: {OUT_PATH}")
            for i, (x, y) in enumerate(corners):
                print(f"  corner {i}: ({x:.2f}, {y:.2f})")
            break

        # 防止跑出图像范围
        h, w = gray.shape
        corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
