import cv2
import numpy as np
from core.mold_detector import detect_big_mold   # 你需要从 mold_detector 里提取一个“只识别大框”的函数

# ---------- 读取黄金模板 ----------
slot_templates = np.load("config/slot_templates.npy")     # shape (4,4,2)
M = np.load("config/mold_warp_M.npy")                     # 原图 → 拉正
Minv = np.load("config/mold_warp_Minv.npy")               # 拉正 → 原图

def map_slot_boxes_back(boxes_warped):
    """
    输入：拉正后的 4 个模板小框
    输出：映射回原图坐标的小框
    """
    mapped_boxes = []

    for box_w in boxes_warped:
        box = box_w.reshape(-1,1,2).astype(np.float32)   # (4,1,2)
        box_orig = cv2.perspectiveTransform(box, Minv)   # 映射回原图
        mapped_boxes.append(box_orig.reshape(4,2))

    return mapped_boxes


def compute_slot_centers(boxes):
    centers = []
    for box in boxes:
        cx = int(box[:,0].mean())
        cy = int(box[:,1].mean())
        centers.append((cx,cy))
    return centers


def main():
    cap = cv2.VideoCapture(0)   # 工业相机请换成你的相机接口

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. 找大绿框
        big_box, rect_pts, M_now, warped_now = detect_big_mold(gray)
        if big_box is None:
            cv2.imshow("prod", frame)
            if cv2.waitKey(10) == 27:
                break
            continue

        # 2. 用 M_now 把模板框逆映射到当前画面
        mapped_boxes = []
        for box in slot_templates:
            pts = box.reshape(-1,1,2).astype(np.float32)
            box_now = cv2.perspectiveTransform(pts, M_now)   # 用当前的 M_now 替代 M_inv
            mapped_boxes.append(box_now.reshape(4,2))

        # 3. 计算中心点
        centers = compute_slot_centers(mapped_boxes)

        # 4. 绘图
        vis = frame.copy()
        for i, box in enumerate(mapped_boxes):
            box_int = box.astype(np.int32)
            cv2.polylines(vis, [box_int], True, (0,255,255), 2)

            cx,cy = centers[i]
            cv2.drawMarker(vis, (cx,cy), (0,0,255),
                           cv2.MARKER_CROSS, 20,2)
            cv2.putText(vis, f"slot {i}", (cx+5, cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("prod", vis)

        key = cv2.waitKey(10)
        if key == 27:  # esc退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
