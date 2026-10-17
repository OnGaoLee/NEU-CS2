import os
import cv2
import numpy as np
from tqdm import tqdm



def undistort(frame):
    fx = 632.167
    cx = 630.841
    fy = 631.546
    cy = 361.9
    k1, k2, p1, p2, k3 = -0.0574505, 0.0697538, -0.00105846, -0.000757105, -0.0220959

    # 相机坐标系到像素坐标系的转换矩阵
    k = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    # 畸变系数
    d = np.array([
        k1, k2, p1, p2, k3
    ])
    h, w = frame.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

if __name__ == '__main__':
    input_dir = "./rgb/2.png"
    output_dir = "./rgb/2undist.png"
    img = cv2.imread(input_dir)
    img_undist = undistort(img)
    cv2.imwrite(output_dir, img_undist)