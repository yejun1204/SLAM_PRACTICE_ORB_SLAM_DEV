"""EuRoC cam0 parameters and preprocessing helpers."""
import numpy as np
import cv2

# Image dimensions
IMG_W_ORIG, IMG_H_ORIG = 752, 480
IMG_W,      IMG_H      = 600, 350

_SX = IMG_W / IMG_W_ORIG   # 600/752
_SY = IMG_H / IMG_H_ORIG   # 350/480

# Scaled camera intrinsics (after resize)
K = np.array([[458.654 * _SX, 0.0,           367.215 * _SX],
              [0.0,           457.296 * _SY, 248.375 * _SY],
              [0.0,           0.0,           1.0          ]], dtype=np.float32)

# Distortion coefficients (k1, k2, p1, p2)
DIST = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05], dtype=np.float32)


def resize_image(img):
    """Resize image to (IMG_W, IMG_H) to match ORB-SLAM3 EuRoC.yaml Camera.newWidth/newHeight."""
    return cv2.resize(img, (IMG_W, IMG_H))


def undistort_keypoints(keypoints):
    """Undistort keypoints using scaled K and distortion coefficients.

    Mirrors Frame::UndistortKeyPoints in ORB-SLAM3:
      cv::undistortPoints(pts, pts, K, distCoef, cv::Mat(), K)
    """
    if not keypoints:
        return keypoints
    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints],
                   dtype=np.float32).reshape(-1, 1, 2)
    pts_un = cv2.undistortPoints(pts, K, DIST, None, K).reshape(-1, 2)
    return [cv2.KeyPoint(float(pts_un[i, 0]), float(pts_un[i, 1]),
                         kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for i, kp in enumerate(keypoints)]
