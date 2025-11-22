import cv2
import numpy as np

# 1. 读取你保存的那张模具图片
img = cv2.imread("mold2.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("找不到 mold2.png，请确认文件名和路径")
    exit(0)

h_img, w_img = img.shape
img_area = h_img * w_img

# 2. 自动找大模具外框
blur = cv2.GaussianBlur(img, (5, 5), 0)
_, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_big)
cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

big_mold_cnt = None
max_area = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > img_area * 0.15 and area > max_area:
        max_area = area
        big_mold_cnt = c

if big_mold_cnt is None:
    print("没找到大模具，先调整图片再试")
    exit(0)

big_rect = cv2.minAreaRect(big_mold_cnt)
box = cv2.boxPoints(big_rect).astype("float32")

# 3. 透视矫正，把大模具拉正
pts = box.copy()
s = pts.sum(axis=1)
tl = pts[np.argmin(s)]
br = pts[np.argmax(s)]
diff = np.diff(pts, axis=1)
tr = pts[np.argmin(diff)]
bl = pts[np.argmax(diff)]
rect_pts = np.array([tl, tr, br, bl], dtype="float32")

widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
maxWidth = int(max(widthA, widthB))
heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxHeight = int(max(heightA, heightB))

dst_pts = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype="float32")

M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

# 4. 在 warped 上手动框选 4 个小凹槽外框
show = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
cv2.namedWindow("标定凹槽 - 依次框选 4 个凹槽", cv2.WINDOW_NORMAL)
cv2.imshow("标定凹槽 - 依次框选 4 个凹槽", show)
cv2.resizeWindow("标定凹槽 - 依次框选 4 个凹槽", show.shape[1], show.shape[0])

print("请依次框选 4 个小方框：左上 → 右上 → 左下 → 右下，框完按 ENTER 再按 ESC 退出。")
rois = cv2.selectROIs("标定凹槽 - 依次框选 4 个凹槽", show,
                      showCrosshair=True, fromCenter=False)
cv2.destroyWindow("标定凹槽 - 依次框选 4 个凹槽")

if len(rois) != 4:
    print("需要精确框选 4 个凹槽（左上→右上→左下→右下），当前数量：", len(rois))
    exit(0)

slot_templates = []
for (x, y, w, h) in rois:
    box_w = np.array([
        [x,     y],
        [x + w, y],
        [x + w, y + h],
        [x,     y + h]
    ], dtype=np.float32)
    slot_templates.append(box_w)

slot_templates = np.array(slot_templates, dtype=np.float32)
np.save("slot_templates.npy", slot_templates)
print("✅ 标定完成，4 个小方框已保存到 slot_templates.npy")
