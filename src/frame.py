"""
Frame - ORB-SLAM3 style Frame class

Key feature: grid-based spatial index for fast GetFeaturesInArea queries.
ORB-SLAM3 divides the image into FRAME_GRID_COLS x FRAME_GRID_ROWS cells
and stores keypoint indices per cell.
"""

import cv2
import numpy as np

FRAME_GRID_ROWS = 48
FRAME_GRID_COLS = 64


class Frame:
    def __init__(self, image, keypoints, descriptors, K, dist_coeffs, timestamp=0.0, frame_id=0):
        """
        Args:
            image: grayscale image
            keypoints: list of cv2.KeyPoint (level-0 coords)
            descriptors: np.ndarray (N x 32)
            K: camera intrinsic matrix (3x3)
            dist_coeffs: distortion coefficients
        """
        self.image = image
        self.descriptors = descriptors
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.K = K
        self.dist_coeffs = dist_coeffs

        self.h, self.w = image.shape[:2]

        # Undistort keypoints
        self.keypoints_raw = keypoints
        self.keypoints = self._undistort_keypoints(keypoints, K, dist_coeffs)

        # Build spatial grid for fast area search
        self._grid_cell_w = self.w / FRAME_GRID_COLS
        self._grid_cell_h = self.h / FRAME_GRID_ROWS
        self._grid = self._assign_grid(self.keypoints)

    def _undistort_keypoints(self, keypoints, K, dist_coeffs):
        if not keypoints:
            return []
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        pts_undist = cv2.undistortPoints(pts, K, dist_coeffs, P=K).reshape(-1, 2)
        undistorted = []
        for i, kp in enumerate(keypoints):
            kp2 = cv2.KeyPoint(
                x=float(pts_undist[i, 0]),
                y=float(pts_undist[i, 1]),
                _size=kp.size,
                _angle=kp.angle,
                _response=kp.response,
                _octave=kp.octave
            )
            undistorted.append(kp2)
        return undistorted

    def _assign_grid(self, keypoints):
        grid = [[[] for _ in range(FRAME_GRID_COLS)] for _ in range(FRAME_GRID_ROWS)]
        for i, kp in enumerate(keypoints):
            col = int(kp.pt[0] / self._grid_cell_w)
            row = int(kp.pt[1] / self._grid_cell_h)
            col = max(0, min(col, FRAME_GRID_COLS - 1))
            row = max(0, min(row, FRAME_GRID_ROWS - 1))
            grid[row][col].append(i)
        return grid

    def get_features_in_area(self, x, y, r, min_level=0, max_level=0):
        """
        Return indices of keypoints within radius r of (x,y) at levels [min_level, max_level].
        ORB-SLAM3: Frame::GetFeaturesInArea
        """
        result = []

        min_col = max(0, int((x - r) / self._grid_cell_w) - 1)
        max_col = min(FRAME_GRID_COLS - 1, int((x + r) / self._grid_cell_w) + 1)
        min_row = max(0, int((y - r) / self._grid_cell_h) - 1)
        max_row = min(FRAME_GRID_ROWS - 1, int((y + r) / self._grid_cell_h) + 1)

        r2 = r * r
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                for idx in self._grid[row][col]:
                    kp = self.keypoints[idx]
                    if kp.octave < min_level or kp.octave > max_level:
                        continue
                    dx = kp.pt[0] - x
                    dy = kp.pt[1] - y
                    if dx * dx + dy * dy <= r2:
                        result.append(idx)
        return result
