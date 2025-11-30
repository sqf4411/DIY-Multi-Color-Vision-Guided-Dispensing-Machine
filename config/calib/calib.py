import glob
import os

import cv2
import numpy as np

# ========= 需要你改的两个地方 =========

# 1. 标定图片所在目录（把这个路径改成你自己的）
IMAGE_DIR = r"E:\PytonProNew\P24_Camrea\P24_Camrea\calib"   # 举例：D:\hik_calib\imgs

# 2. 每个棋盘格的实际边长（毫米）
SQUARE_SIZE = 12.5  # 你量出来那个 38.4 mm，尽量写实测值


# ========= 下面一般不用改 =========

# 棋盘格内角点数量：我们打印的是 10x7 方格 -> 9x6 内角点
PATTERN_SIZE = (9, 6)  # (cols, rows)

def collect_image_paths(image_dir):
    exts = ("*.png", "*.jpg", "*.bmp", "*.jpeg", "*.tif")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths.sort()
    return paths


def main():
    img_paths = collect_image_paths(IMAGE_DIR)
    if not img_paths:
        print(f"❌ 没有在 {IMAGE_DIR} 找到标定图片")
        return

    print(f"📷 共找到 {len(img_paths)} 张标定图片")

    # 3D 棋盘角点（世界坐标）
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0],
                           0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # 单位：mm

    objpoints = []  # 3D 点
    imgpoints = []  # 2D 点

    gray_shape = None

    for i, fname in enumerate(img_paths, 1):
        img = cv2.imread(fname)
        if img is None:
            print(f"⚠️ 无法读取图片: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                 cv2.CALIB_CB_NORMALIZE_IMAGE +
                 cv2.CALIB_CB_FAST_CHECK)
        ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags)

        if ret:
            # 亚像素角点优化
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )

            objpoints.append(objp)
            imgpoints.append(corners2)

            print(f"✅ {os.path.basename(fname)} 角点检测成功")
        else:
            print(f"❌ {os.path.basename(fname)} 未找到棋盘角点")

    if not objpoints:
        print("❌ 没有任何图片检测到角点，标定失败")
        return

    print("🧮 开始相机标定 ...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    print("\n✅ 标定完成")
    print("平均重投影误差 reprojection error =", ret)
    print("\nCamera matrix (内参矩阵):\n", camera_matrix)
    print("\nDistortion coefficients (畸变参数):\n", dist_coeffs)

    # 保存结果
    np.save("camera_matrix.npy", camera_matrix)
    np.save("dist_coeffs.npy", dist_coeffs)
    print("\n💾 已保存 camera_matrix.npy 和 dist_coeffs.npy 到当前目录")

    # 额外：用第一张图做个去畸变预览
    test_img = cv2.imread(img_paths[0])
    h, w = test_img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(
        test_img, camera_matrix, dist_coeffs, None, new_camera_mtx
    )

    cv2.imshow("original", test_img)
    cv2.imshow("undistorted", undistorted)
    print("按任意键关闭预览窗口 ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
