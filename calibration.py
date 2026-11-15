import cv2
import numpy as np
import glob
import os

# ==========================================
# 1. 设置参数（根据你的棋盘格修改）
# ==========================================
# 棋盘格的内部角点数（注意：是内角点！横向9个，纵向6个）
# 如果是一张 10x7 个格子的图，内角点就是 9x6
CHESSBOARD_SIZE = (6, 6)

# 棋盘格每个格子的真实物理尺寸（单位：毫米）
# 比如你打印出来后，用尺子量一下其中一个黑格或白格的宽度，假设是 25mm
SQUARE_SIZE = 20.0

# 照片所在的文件夹路径
IMAGE_DIR = 'calibration_images/*.jpg'

# ==========================================
# 2. 准备世界坐标系中的 3D 点 (0,0,0), (25,0,0), (50,0,0) ...
# ==========================================
# 假设棋盘格在世界坐标系中的 Z=0 平面上
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
# 生成 x, y 坐标网格
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
# 乘上真实的物理尺寸
objp = objp * SQUARE_SIZE

# 存放所有图片的 3D 点和 2D 点
objpoints = []  # 真实世界中的 3D 点
imgpoints = []  # 图像平面中的 2D 点

# ==========================================
# 3. 循环读取图片，寻找角点
# ==========================================
images = glob.glob(IMAGE_DIR)
if not images:
    print(f"错误：在 {IMAGE_DIR} 下没有找到图片，请检查路径！")
    exit()
# 亚像素角点优化的终止条件（迭代30次或精度达到0.001）
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 【关键修改】加入 flags 增强参数，让 OpenCV 开启自适应阈值和归一化，更容易找到角点
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow('Find Corners', img)
        cv2.waitKey(500)
    else:
        print(f"警告：图片 {fname} 中未找到角点！")
        # 【关键修改】如果没找到，直接把这张图弹出来让你看！
        cv2.imshow('NOT FOUND - Press any key', img)
        cv2.waitKey(0)  # 按任意键继续

cv2.destroyAllWindows()

# ==========================================
# 4. 执行标定计算
# ==========================================
if len(objpoints) > 5:  # 至少需要几张图片才能算
    print("正在计算相机内参...")
    # 返回：重投影误差, 内参矩阵, 畸变系数, 旋转向量, 平移向量
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n" + "=" * 40)
    print("标定成功！结果如下：")
    print("=" * 40)

    print(f"\n重投影误差 (RMS): {ret:.4f}")
    print("（注：这个值越接近0越好，通常要求 < 0.5，如果很大说明图片拍得不好）")

    print("\n相机内参矩阵 mtx:")
    print(mtx)

    print("\n相机畸变系数 dist:")
    print(dist)

    # 提取出你最需要的焦距参数
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    print("-" * 40)
    print(f"像素焦距 fx = {fx:.2f}, fy = {fy:.2f}")
    print(f"光学中心 cx = {cx:.2f}, cy = {cy:.2f}")
    print("-" * 40)
    print("👉 把上面的 mtx 和 dist 的数值复制下来，填入你项目一的代码里！")

else:
    print("错误：成功提取角点的图片太少（少于5张），无法计算标定参数。请重新拍摄。")