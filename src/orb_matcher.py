"""
ORB Matcher - ORB-SLAM3 style

Implements SearchForInitialization: window-based matching between two frames
using Hamming distance, ratio test, and rotation consistency check.
"""

import numpy as np

TH_LOW = 50
TH_HIGH = 100
HISTO_LENGTH = 30
FRAME_GRID_COLS = 64
FRAME_GRID_ROWS = 48


def hamming_distance(a, b):
    """Compute Hamming distance between two (32,) uint8 descriptors."""
    return int(np.unpackbits(np.frombuffer(a ^ b, dtype=np.uint8)).sum())


def _compute_three_maxima(histo):
    """Return indices of the three largest histogram bins.
    Bins smaller than 10% of the largest are discarded (-1).
    """
    sizes = np.array([len(h) for h in histo])
    order = np.argsort(-sizes)
    ind1, ind2, ind3 = order[0], order[1], order[2]
    max1 = sizes[ind1]

    if sizes[ind2] < 0.1 * max1:
        ind2 = -1
        ind3 = -1
    elif sizes[ind3] < 0.1 * max1:
        ind3 = -1

    return ind1, ind2, ind3


class Frame:
    """Lightweight frame: keypoints + descriptors + spatial grid."""

    def __init__(self, keypoints, descriptors, img_w, img_h):
        """
        keypoints : list of cv2.KeyPoint (level-0 coords, undistorted)
        descriptors : np.ndarray (N, 32) uint8
        img_w, img_h : image dimensions (used for grid bounds)
        """
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.img_w = img_w
        self.img_h = img_h

        self.min_x = 0.0
        self.max_x = float(img_w)
        self.min_y = 0.0
        self.max_y = float(img_h)

        self._grid_elem_w_inv = FRAME_GRID_COLS / (self.max_x - self.min_x)
        self._grid_elem_h_inv = FRAME_GRID_ROWS / (self.max_y - self.min_y)

        self._grid = [[[] for _ in range(FRAME_GRID_ROWS)]
                      for _ in range(FRAME_GRID_COLS)]
        self._assign_features_to_grid()

    def _pos_in_grid(self, kp):
        col = round((kp.pt[0] - self.min_x) * self._grid_elem_w_inv)
        row = round((kp.pt[1] - self.min_y) * self._grid_elem_h_inv)
        if col < 0 or col >= FRAME_GRID_COLS or row < 0 or row >= FRAME_GRID_ROWS:
            return None, None
        return int(col), int(row)

    def _assign_features_to_grid(self):
        for i, kp in enumerate(self.keypoints):
            col, row = self._pos_in_grid(kp)
            if col is not None:
                self._grid[col][row].append(i)

    def get_features_in_area(self, x, y, r, min_level=-1, max_level=-1):
        """Return indices of keypoints within window [x±r, y±r] at given levels."""
        min_col = max(0, int(np.floor((x - self.min_x - r) * self._grid_elem_w_inv)))
        max_col = min(FRAME_GRID_COLS - 1,
                      int(np.ceil((x - self.min_x + r) * self._grid_elem_w_inv)))
        min_row = max(0, int(np.floor((y - self.min_y - r) * self._grid_elem_h_inv)))
        max_row = min(FRAME_GRID_ROWS - 1,
                      int(np.ceil((y - self.min_y + r) * self._grid_elem_h_inv)))

        if min_col >= FRAME_GRID_COLS or max_col < 0:
            return []
        if min_row >= FRAME_GRID_ROWS or max_row < 0:
            return []

        check_levels = (min_level > 0) or (max_level >= 0)
        indices = []
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                for idx in self._grid[col][row]:
                    kp = self.keypoints[idx]
                    if check_levels:
                        if kp.octave < min_level:
                            continue
                        if max_level >= 0 and kp.octave > max_level:
                            continue
                    if abs(kp.pt[0] - x) < r and abs(kp.pt[1] - y) < r:
                        indices.append(idx)
        return indices


def search_for_initialization(frame1, frame2,
                               prev_matched,
                               window_size=100,
                               nn_ratio=0.9,
                               check_orientation=True):
    """
    Match frame1 keypoints (level-0 only) to frame2 using window search.

    Mirrors ORB-SLAM3 ORBmatcher::SearchForInitialization.

    Args:
        frame1       : Frame
        frame2       : Frame
        prev_matched : list of (x, y) — search center in frame2 per frame1 keypoint
                       (initially frame1 keypoint positions, updated in-place on return)
        window_size  : search window half-size in pixels (default 100)
        nn_ratio     : Lowe's ratio test threshold (default 0.9)
        check_orientation : enable rotation histogram consistency check

    Returns:
        matches12 : list of int, length = len(frame1.keypoints)
                    matches12[i] = j  (frame2 index) or -1 (no match)
        n_matches : int
    """
    n1 = len(frame1.keypoints)
    n2 = len(frame2.keypoints)

    matches12 = [-1] * n1
    matches21 = [-1] * n2
    matched_dist = [float('inf')] * n2

    rot_hist = [[] for _ in range(HISTO_LENGTH)]
    factor = 1.0 / HISTO_LENGTH

    for i1, kp1 in enumerate(frame1.keypoints):
        if kp1.octave > 0:
            continue

        cx, cy = prev_matched[i1]
        candidates = frame2.get_features_in_area(cx, cy, window_size,
                                                  kp1.octave, kp1.octave)
        if not candidates:
            continue

        d1 = frame1.descriptors[i1]
        best_dist = float('inf')
        best_dist2 = float('inf')
        best_idx2 = -1

        for i2 in candidates:
            d2 = frame2.descriptors[i2]
            dist = hamming_distance(d1, d2)

            if matched_dist[i2] <= dist:
                continue

            if dist < best_dist:
                best_dist2 = best_dist
                best_dist = dist
                best_idx2 = i2
            elif dist < best_dist2:
                best_dist2 = dist

        if best_dist > TH_LOW:
            continue

        if best_dist >= nn_ratio * best_dist2:
            continue

        # Resolve conflicts: if frame2[best_idx2] was already matched
        if matches21[best_idx2] >= 0:
            matches12[matches21[best_idx2]] = -1

        matches12[i1] = best_idx2
        matches21[best_idx2] = i1
        matched_dist[best_idx2] = best_dist

        if check_orientation:
            rot = kp1.angle - frame2.keypoints[best_idx2].angle
            if rot < 0.0:
                rot += 360.0
            bin_idx = round(rot * factor)
            if bin_idx == HISTO_LENGTH:
                bin_idx = 0
            rot_hist[bin_idx].append(i1)

    if check_orientation:
        ind1, ind2, ind3 = _compute_three_maxima(rot_hist)
        valid = {ind1, ind2, ind3} - {-1}
        n_matches = 0
        for i in range(HISTO_LENGTH):
            if i not in valid:
                for i1 in rot_hist[i]:
                    if matches12[i1] >= 0:
                        matches12[i1] = -1
        n_matches = sum(1 for m in matches12 if m >= 0)
    else:
        n_matches = sum(1 for m in matches12 if m >= 0)

    # Update prev_matched with matched frame2 positions
    for i1, i2 in enumerate(matches12):
        if i2 >= 0:
            prev_matched[i1] = frame2.keypoints[i2].pt

    return matches12, n_matches
