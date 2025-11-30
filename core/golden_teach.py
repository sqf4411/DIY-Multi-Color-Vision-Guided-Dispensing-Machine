import cv2
import numpy as np
from core.mold_detector import detect_mold_and_cells

WINDOW_NAME = "黄金示教"

# Global State
slot_boxes = []
drawing = False
start_point = (0, 0)
current_point = (0, 0)
warped_img = None  # Base image (clean)

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, current_point, slot_boxes, warped_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_point = (x, y)
        
        x1, y1 = start_point
        x2, y2 = current_point
        
        # Avoid zero-size boxes
        if abs(x1 - x2) < 5 or abs(y1 - y2) < 5:
            return

        # 1. Crop user selection
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        
        # Ensure within bounds
        h, w = warped_img.shape[:2]
        xmin, xmax = max(0, xmin), min(w, xmax)
        ymin, ymax = max(0, ymin), min(h, ymax)

        roi = warped_img[ymin:ymax, xmin:xmax]

        # 2. Auto-detect inner box
        # Note: detect_mold_and_cells expects grayscale or BGR. warped_img is BGR.
        # But detect_mold_and_cells usually takes grayscale.
        # Let's check warped_img type. In run_golden_teach it's converted to BGR.
        # We should pass grayscale to detect_mold_and_cells if possible, or let it handle it.
        # Looking at mold_detector.py, it takes 'gray'.
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        try:
            _, box, _, _ = detect_mold_and_cells(roi_gray)
            if box is not None:
                # Offset back to global coordinates
                box = box + np.array([xmin, ymin])
                slot_boxes.append(box)
                print(f"✅ Added box #{len(slot_boxes)}")
            else:
                print("⚠ No inner box detected, using manual selection")
                # Fallback: Use the manual box
                manual_box = np.array([
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax]
                ], dtype=np.int32)
                slot_boxes.append(manual_box)
        except Exception as e:
            print(f"⚠ Detection error: {e}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Delete box under cursor
        # Check if (x, y) is inside any box
        for i in range(len(slot_boxes) - 1, -1, -1):
            box = slot_boxes[i]
            if cv2.pointPolygonTest(box, (x, y), False) >= 0:
                print(f"🗑 Deleted box #{i + 1}")
                slot_boxes.pop(i)
                break


def run_golden_teach():
    global warped_img, slot_boxes, drawing, start_point, current_point

    # Reset State
    slot_boxes = []
    drawing = False
    start_point = (0, 0)
    current_point = (0, 0)

    img = cv2.imread("config/mold2.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ 无法读取 mold2.png")
        return False

    # 找大模具框
    _, big_box, _, _ = detect_mold_and_cells(img)
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
    warped_gray = cv2.warpPerspective(img, M, (W,H))
    warped_img = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, W//2, H//2)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("👉 黄金示教开始：框选 4 个凹槽，按 s 保存，q 退出")

    while True:
        # 1. Copy base image
        display_img = warped_img.copy()

        # 2. Draw existing boxes
        for i, box in enumerate(slot_boxes):
            cv2.drawContours(display_img, [box.astype(np.int32)], 0, (0, 255, 0), 2)
            # Draw index
            cx, cy = int(np.mean(box[:, 0])), int(np.mean(box[:, 1]))
            cv2.putText(display_img, str(i + 1), (cx - 10, cy + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 3. Draw current dragging rectangle
        if drawing:
            cv2.rectangle(display_img, start_point, current_point, (0, 255, 255), 2)

        # 4. Draw UI Instructions (Top Layer)
        info_text = f"Boxes: {len(slot_boxes)}/4 | [S] Save | [Z] Undo | [R-Click] Delete | [Q] Quit"
        cv2.rectangle(display_img, (0, 0), (W, 40), (0, 0, 0), -1)
        cv2.putText(display_img, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, display_img)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            return False

        elif key == ord('s'):
            if len(slot_boxes) != 4:
                print(f"❌ 你只画了 {len(slot_boxes)} 个，需要 4 个")
                # Show warning on screen
                # (Optional, but print is visible in console)
                continue

            np.save("config/slot_templates.npy", np.array(slot_boxes))
            np.save("config/warp_size.npy", np.array([W, H], dtype=np.int32))

            print("✅ 已保存 slot_templates.npy 和 warp_size.npy")
            cv2.destroyAllWindows()
            return True
        
        elif key == ord('z'):
            if slot_boxes:
                slot_boxes.pop()
                print("↩ Undo last box")
            else:
                print("⚠ Nothing to undo")

if __name__ == "__main__":
    run_golden_teach()
