import cv2
import numpy as np

# ==========================================
# 1. 填入你的标定数据（这里我用一组假数据演示）
# 你要换成你自己跑标定代码打印出来的 mtx 和 dist
# ==========================================
mtx = np.array([[2.70879900e+03 ,0.00000000e+00, 7.36518265e+02],
 [0.00000000e+00 ,2.76637227e+03, 6.37755736e+02],
 [0.00000000e+00 ,0.00000000e+00, 1.00000000e+00]])
dist = np.array([[-1.08598541 , 3.78413214 ,-0.00830719 ,-0.00852516 ,-4.84213018]])

# 假设相机距离桌面工作距离 Z = 300 mm
Z = 300.0
fx = mtx[0, 0]
fy = mtx[1, 1]


# ==========================================
# 2. 定义角点排序函数（极其重要，否则拉出来的图是扭的）
# ==========================================
def order_points(pts):
    # 创建一个 4x2 的零矩阵
    rect = np.zeros((4, 2), dtype="float32")

    # 按行求和：左上角 x+y 最小，右下角 x+y 最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 按行求差：右上角 y-x 最小（或 x-y 最大），左下角 y-x 最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# ==========================================
# 3. 主程序
# ==========================================
def main():
    # 读取图片（换成你自己的图片路径）
    img = cv2.imread('test_images/test_part7.jpg')
    if img is None:
        print("图片读取失败，请检查路径！")
        return

    # 步骤 A：去畸变（用标定的参数把图拉直）
    img_undistort = cv2.undistort(img, mtx, dist)

    # 步骤 B：预处理与找边缘
    gray = cv2.cvtColor(img_undistort, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 这里的 (50, 150) 需要根据你的图片明暗调整
    edges = cv2.Canny(blurred, 100, 150)

    #cv2.imshow("Edges", edges)  # 看看 Canny 提取出来的到底是什么鬼
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()

    # 步骤 C：找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("没找到任何轮廓！")
        return

    # 找面积最大的轮廓（假设你要测的零件是图中最大的物体）
    c = max(contours, key=cv2.contourArea)

    # 步骤 D：多边形拟合找4个角
    peri = cv2.arcLength(c, True)
    # 0.02 是逼近精度，如果拟合出来的点数不是4个，就微调这个值（比如 0.01 或 0.05）
    approx = cv2.approxPolyDP(c, 0.02* peri, True)

    if len(approx) == 4:
        print("成功找到4个角点！")
        # 把四个点排好序（左上、右上、右下、左下）
        pts_src = order_points(approx.reshape(4, 2))

        # 步骤 E：透视变换（把歪的物体拉正）
        # 假设我们想把物体拉正成一个 400x300 的标准长方形图片
        W, H = 400, 300
        pts_dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")

        # 计算变换矩阵并执行拉正
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(img_undistort, M, (W, H))

        # 步骤 F：物理尺寸计算（核心！）
        # 在拉正后的图里，我们强行规定了它的像素尺寸是 400x300
        # 但真实世界有多大？我们用 单目测距公式算一下源图中它占了多少物理尺寸
        pixel_width = np.linalg.norm(pts_src[0] - pts_src[1])  # 上边像素长度
        pixel_height = np.linalg.norm(pts_src[0] - pts_src[3])  # 左边像素长度

        # 公式：实际宽度(mm) = 像素宽 * 距离Z / fx
        real_width = (pixel_width * Z) / fx
        real_height = (pixel_height * Z) / fy

        # 步骤 G：在图片上画框和写文字
        result_img = img_undistort.copy()
        cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 2)

        text = f"W: {real_width:.1f}mm, H: {real_height:.1f}mm"
        cv2.putText(result_img, text, (int(pts_src[0][0]), int(pts_src[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 打印误差（假设你的零件真实宽度是 210mm）
        actual_real_width = 210.0  # 换成你用尺子量出来的真实宽度
        error = abs(real_width - actual_real_width) / actual_real_width * 100
        print(f"测量宽度: {real_width:.2f} mm, 真实宽度: {actual_real_width} mm, 误差: {error:.2f}%")

        # 显示结果
        cv2.imshow("Original with Measurement", result_img)
        cv2.imshow("Warped (Rectified)", warped)
        # 👇 加上这两行，把图片存到本地 👇
        cv2.imwrite('result_img.png', result_img)  # 保存带框和尺寸的结果图
        cv2.imwrite('warped_img.png', warped)  # 保存透视变换拉正后的图

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print(f"拟合失败，找到了 {len(approx)} 个点。请调整 approxPolyDP 的精度或 Canny 的阈值。")


if __name__ == '__main__':
    main()