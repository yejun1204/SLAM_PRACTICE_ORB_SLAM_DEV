"""
Tracking — ORB-SLAM3 Tracking.cc port (monocular).

Implements:
  - MapPoint
  - search_by_projection  (ORBmatcher::SearchByProjection, Frame×LastFrame)
  - pose_optimization     (Optimizer::PoseOptimization via iterative solvePnP)
  - track_with_motion_model
"""

import numpy as np
import cv2
from src.orb_matcher import Frame, hamming_distances

TH_HIGH      = 100
SCALE_FACTOR = 1.2
N_LEVELS     = 8
SCALE_FACTORS = np.array([SCALE_FACTOR ** i for i in range(N_LEVELS)], dtype=np.float32)


# ── MapPoint ─────────────────────────────────────────────────────────────────

class MapPoint:
    """3D map point with representative descriptor."""
    def __init__(self, pos, descriptor, octave=0):
        self.pos        = np.asarray(pos,        dtype=np.float32)   # (3,) world coords
        self.descriptor = np.asarray(descriptor, dtype=np.uint8)     # (32,)
        self.octave     = int(octave)                                 # scale level


# ── SearchByProjection ────────────────────────────────────────────────────────

def search_by_projection(frame, map_points, T_cw, K, th=15):
    """
    Project map_points into current frame using T_cw, match by descriptor.
    Mirrors ORBmatcher::SearchByProjection(Frame, LastFrame, th, bMono=true).

    Args:
        frame      : Frame (orb_matcher.Frame)
        map_points : iterable of MapPoint
        T_cw       : (4,4) float64, world→camera transform
        K          : (3,3) camera intrinsics
        th         : search radius factor (15 for monocular)

    Returns:
        matches : dict {kp_idx_in_frame: MapPoint}
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    R = T_cw[:3, :3].astype(np.float64)
    t = T_cw[:3, 3].astype(np.float64)
    img_w, img_h = frame.img_w, frame.img_h

    matched_kps  = {}       # kp_idx → MapPoint
    matched_dist = {}       # kp_idx → best Hamming dist (for conflict resolution)

    for mp in map_points:
        # Project world point to camera coords
        x3dc = R @ mp.pos.astype(np.float64) + t
        if x3dc[2] <= 0:
            continue

        iz = 1.0 / x3dc[2]
        u = fx * x3dc[0] * iz + cx
        v = fy * x3dc[1] * iz + cy

        if u < 0 or u >= img_w or v < 0 or v >= img_h:
            continue

        # Scale-aware search radius (same octave ±1)
        oct = mp.octave
        radius = th * float(SCALE_FACTORS[min(oct, N_LEVELS - 1)])
        candidates = frame.get_features_in_area(u, v, radius, oct - 1, oct + 1)
        if not candidates:
            continue

        cand = np.array(candidates, dtype=np.int32)
        dists = hamming_distances(mp.descriptor, frame.descriptors[cand])

        best_i    = int(np.argmin(dists))
        best_dist = int(dists[best_i])
        best_kp   = int(cand[best_i])

        if best_dist > TH_HIGH:
            continue

        # Conflict resolution: keep lower Hamming distance per frame kp
        if best_kp in matched_dist and matched_dist[best_kp] <= best_dist:
            continue

        matched_kps[best_kp]  = mp
        matched_dist[best_kp] = best_dist

    return matched_kps


# ── PoseOptimization ──────────────────────────────────────────────────────────

def pose_optimization(T_cw_init, kps, mp_list, kp_indices, K, n_iter=4):
    """
    Refine T_cw with 3D-2D correspondences via iterative solvePnP + outlier removal.
    Mirrors Optimizer::PoseOptimization (g2o SE3 + Huber, approximated with LM).

    Args:
        T_cw_init  : (4,4) initial world→camera pose
        kps        : list of cv2.KeyPoint (current frame)
        mp_list    : list of MapPoint, same order as kp_indices
        kp_indices : list of int — which keypoints are matched
        K          : (3,3) camera intrinsics
        n_iter     : number of outlier-removal iterations (default 4)

    Returns:
        T_cw    : (4,4) float64 refined pose
        inliers : (N,) bool array
    """
    chi2_th = 5.991   # chi2 95% for 2 DOF (reprojection in x,y)
    Kd      = K.astype(np.float64)
    fx, fy  = float(Kd[0, 0]), float(Kd[1, 1])
    cx, cy  = float(Kd[0, 2]), float(Kd[1, 2])

    pts3d   = np.array([mp.pos for mp in mp_list],           dtype=np.float64)  # (N,3)
    pts2d   = np.array([kps[ki].pt for ki in kp_indices],    dtype=np.float64)  # (N,2)

    T       = T_cw_init.astype(np.float64).copy()
    inliers = np.ones(len(pts3d), dtype=bool)

    for _ in range(n_iter):
        n_in = int(inliers.sum())
        if n_in < 4:
            break

        rvec_init = cv2.Rodrigues(T[:3, :3])[0]
        tvec_init = T[:3, 3].reshape(3, 1)

        ret, rvec, tvec = cv2.solvePnP(
            pts3d[inliers], pts2d[inliers],
            Kd, None,
            rvec_init, tvec_init,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ret:
            break

        R_new = cv2.Rodrigues(rvec)[0]
        t_new = tvec.flatten()
        T[:3, :3] = R_new
        T[:3, 3]  = t_new

        # Recompute reprojection errors for all points
        x3dc  = (R_new @ pts3d.T).T + t_new          # (N,3)
        valid = x3dc[:, 2] > 0
        iz    = np.where(valid, 1.0 / (x3dc[:, 2] + 1e-10), 0.0)
        px    = fx * x3dc[:, 0] * iz + cx
        py    = fy * x3dc[:, 1] * iz + cy
        chi2  = (px - pts2d[:, 0]) ** 2 + (py - pts2d[:, 1]) ** 2

        inliers = valid & (chi2 < chi2_th)

    return T, inliers


# ── TrackWithMotionModel ──────────────────────────────────────────────────────

def track_with_motion_model(frame, map_points, T_cw_last, velocity, K, th=15):
    """
    Estimate current pose using constant-velocity prediction + map point projection.
    Mirrors Tracking::TrackWithMotionModel (monocular path).

    Args:
        frame      : Frame (current)
        map_points : list of MapPoint (from map / last frame)
        T_cw_last  : (4,4) pose of previous frame
        velocity   : (4,4) T_cur_last — relative motion estimate
        K          : (3,3) camera intrinsics
        th         : projection search radius factor (15)

    Returns:
        T_cw      : (4,4) optimized pose, or None if tracking failed
        n_inliers : int
        inlier_mps: dict {kp_idx: MapPoint} of inlier matches
    """
    # 1. Predict pose with constant velocity
    T_cw_pred = velocity @ T_cw_last

    # 2. SearchByProjection with th=15
    matches = search_by_projection(frame, map_points, T_cw_pred, K, th)

    # 3. Retry with wider window if too few matches
    if len(matches) < 20:
        matches = search_by_projection(frame, map_points, T_cw_pred, K, 2 * th)

    if len(matches) < 20:
        return None, len(matches), {}

    # 4. PoseOptimization
    kp_indices = list(matches.keys())
    mp_list    = [matches[ki] for ki in kp_indices]

    T_cw, inliers = pose_optimization(T_cw_pred, frame.keypoints, mp_list, kp_indices, K)

    n_inliers  = int(inliers.sum())
    inlier_mps = {kp_indices[i]: mp_list[i]
                  for i in range(len(kp_indices)) if inliers[i]}

    # 5. Success criterion: >= 10 inlier map point matches
    if n_inliers < 10:
        return None, n_inliers, {}

    return T_cw, n_inliers, inlier_mps
