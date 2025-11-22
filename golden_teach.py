import cv2
import numpy as np
from mold_detector import detect_mold_and_cells

WINDOW_NAME = "黄金示教"

slot_boxes = []
drawing = False
x1 = y1 = 0

def mouse_callback(event, x, y, flags, param):
    global drawing, x1, y1, slot_boxes, draw_img, warped

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y

        # 截取用户画的区域
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        roi = warped[ymin:ymax, xmin:xmax]

        # 自动检测内部小框
        color, box, _, _ = detect_mold_and_cells(roi)
        if box is not None:
            box = box + np.array([xmin, ymin])
            slot_boxes.append(box)
            cv2.drawContours(draw_img, [box.astype(np.int32)], 0, (0,255,255), 2)


def run_golden_teach():
    global warped, draw_img, slot_boxes, drawing, x1, y1

    # ⭐⭐ 每次黄金示教前，清空所有缓存 ⭐⭐
    slot_boxes = []
    drawing = False
    x1 = y1 = 0

    img = cv2.imread("mold.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ 无法读取 mold.png")
        return False

    # 找大模具框
    color, big_box, _, _ = detect_mold_and_cells(img)
    if big_box is None:
        print("❌ 未找到大模具")
        return False

    # --- 透视矫正 ---
    pts = big_box.astype(np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)

    W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))

    dst_pts = np.array([
        [0,0],
        [W-1,0],
        [W-1,H-1],
        [0,H-1]
    ], np.float32)

    M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (W,H))
    draw_img = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, W//2, H//2)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("👉 黄金示教开始：框选 4 个凹槽，按 s 保存，q 退出")

    while True:
        cv2.imshow(WINDOW_NAME, draw_img)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            return False

        if key == ord('s'):
            if len(slot_boxes) != 4:
                print(f"❌ 你只画了 {len(slot_boxes)} 个，需要 4 个")
                continue

            np.save("slot_templates.npy", np.array(slot_boxes))
            np.save("warp_size.npy", np.array([W, H], dtype=np.int32))

            print("✅ 已保存 slot_templates.npy 和 warp_size.npy")
            cv2.destroyAllWindows()
            return True



if __name__ == "__main__":
    run_golden_teach()
