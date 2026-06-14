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
    v = np.bitwise_xor(a, b).view(np.uint32)          # (8,) uint32
    v = v - ((v >> 1) & np.uint32(0x55555555))
    v = (v & np.uint32(0x33333333)) + ((v >> 2) & np.uint32(0x33333333))
    v = ((v + (v >> 4)) & np.uint32(0x0F0F0F0F))
    return int(((v * np.uint32(0x01010101)) >> np.uint32(24)).sum())


def hamming_distances(d1, d2s):
    """Batch Hamming distances: d1 (32,) uint8 vs d2s (M, 32) uint8 → (M,) int."""
    xor = np.bitwise_xor(d1, d2s)                      # (M, 32) uint8
    v = np.ascontiguousarray(xor).view(np.uint32)       # (M, 8) uint32
    v = v - ((v >> 1) & np.uint32(0x55555555))
    v = (v & np.uint32(0x33333333)) + ((v >> 2) & np.uint32(0x33333333))
    v = ((v + (v >> 4)) & np.uint32(0x0F0F0F0F))
    return ((v * np.uint32(0x01010101)) >> np.uint32(24)).sum(axis=1)


def _compute_three_maxima(histo):
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
    """Lightweight frame: keypoints + descriptors + precomputed numpy arrays."""

    def __init__(self, keypoints, descriptors, img_w, img_h):
        self.keypoints   = keypoints
        self.descriptors = descriptors
        self.img_w = img_w
        self.img_h = img_h

        # Precompute numpy arrays for fast vectorized access
        N = len(keypoints)
        if N > 0:
            self.kp_pts     = np.array([kp.pt     for kp in keypoints], dtype=np.float32)  # (N,2)
            self.kp_octaves = np.array([kp.octave for kp in keypoints], dtype=np.int32)    # (N,)
            self.kp_angles  = np.array([kp.angle  for kp in keypoints], dtype=np.float32)  # (N,)
        else:
            self.kp_pts     = np.empty((0, 2), dtype=np.float32)
            self.kp_octaves = np.empty(0, dtype=np.int32)
            self.kp_angles  = np.empty(0, dtype=np.float32)

        # Level-0 subset for fast window search
        l0_mask          = self.kp_octaves == 0
        self.l0_indices  = np.where(l0_mask)[0]        # (M,) original indices
        self.l0_pts      = self.kp_pts[self.l0_indices] if len(self.l0_indices) else np.empty((0,2), dtype=np.float32)

    def get_features_in_area(self, x, y, r, min_level=-1, max_level=-1):
        """Return indices of keypoints within window [x±r, y±r] at given levels.
        Uses precomputed numpy arrays instead of grid iteration.
        """
        if min_level <= 0 and (max_level < 0 or max_level == 0):
            # Level-0 only (common case for SearchForInitialization)
            if len(self.l0_indices) == 0:
                return []
            dx = np.abs(self.l0_pts[:, 0] - x)
            dy = np.abs(self.l0_pts[:, 1] - y)
            mask = (dx < r) & (dy < r)
            return self.l0_indices[mask].tolist()

        # General case: filter by octave range then position
        pts = self.kp_pts; oct = self.kp_octaves
        level_ok = np.ones(len(pts), dtype=bool)
        if min_level > 0:
            level_ok &= oct >= min_level
        if max_level >= 0:
            level_ok &= oct <= max_level
        dx = np.abs(pts[:, 0] - x)
        dy = np.abs(pts[:, 1] - y)
        mask = level_ok & (dx < r) & (dy < r)
        return np.where(mask)[0].tolist()


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

    matches12    = np.full(n1, -1, dtype=np.int32)
    matches21    = np.full(n2, -1, dtype=np.int32)
    matched_dist = np.full(n2, np.inf, dtype=np.float32)

    rot_hist = [[] for _ in range(HISTO_LENGTH)]
    factor = 1.0 / HISTO_LENGTH

    prev_arr = np.array(prev_matched, dtype=np.float32)  # (N1, 2)

    for i1 in frame1.l0_indices:
        cx, cy = prev_arr[i1, 0], prev_arr[i1, 1]

        if len(frame2.l0_indices) == 0:
            continue
        dx = np.abs(frame2.l0_pts[:, 0] - cx)
        dy = np.abs(frame2.l0_pts[:, 1] - cy)
        in_win = (dx < window_size) & (dy < window_size)
        if not in_win.any():
            continue
        cand = frame2.l0_indices[in_win]

        dists = hamming_distances(frame1.descriptors[i1], frame2.descriptors[cand])

        md = matched_dist[cand]
        valid_mask = dists < md
        if not valid_mask.any():
            continue

        vdists = dists[valid_mask]
        vcand  = cand[valid_mask]
        order  = np.argsort(vdists)

        best_dist  = int(vdists[order[0]])
        best_idx2  = int(vcand[order[0]])
        best_dist2 = int(vdists[order[1]]) if len(order) > 1 else float('inf')

        if best_dist > TH_LOW:
            continue
        if best_dist >= nn_ratio * best_dist2:
            continue

        prev_i1 = int(matches21[best_idx2])
        if prev_i1 >= 0:
            matches12[prev_i1] = -1

        matches12[i1]        = best_idx2
        matches21[best_idx2] = i1
        matched_dist[best_idx2] = best_dist

        if check_orientation:
            rot = float(frame1.kp_angles[i1]) - float(frame2.kp_angles[best_idx2])
            if rot < 0.0:
                rot += 360.0
            bin_idx = round(rot * factor)
            if bin_idx == HISTO_LENGTH:
                bin_idx = 0
            rot_hist[bin_idx].append(int(i1))

    if check_orientation:
        ind1, ind2, ind3 = _compute_three_maxima(rot_hist)
        valid_bins = {ind1, ind2, ind3} - {-1}
        for i in range(HISTO_LENGTH):
            if i not in valid_bins:
                for i1 in rot_hist[i]:
                    if matches12[i1] >= 0:
                        matches12[i1] = -1

    n_matches = int((matches12 >= 0).sum())
    matches12_list = matches12.tolist()

    for i1, i2 in enumerate(matches12_list):
        if i2 >= 0:
            prev_matched[i1] = frame2.keypoints[i2].pt

    return matches12_list, n_matches
