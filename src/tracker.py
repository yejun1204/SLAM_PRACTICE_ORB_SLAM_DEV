"""
Tracking — ORB-SLAM3 Tracking.cc port (monocular).

Implements:
  - MapPoint
  - search_by_projection      (ORBmatcher::SearchByProjection)
  - pose_optimization         (Optimizer::PoseOptimization via GTSAM LM + Huber)
  - track_with_motion_model   (Tracking::TrackWithMotionModel)
  - track_local_map           (Tracking::TrackLocalMap → SearchLocalPoints + PoseOptimization)
"""

import numpy as np
import gtsam
from src.orb_matcher import Frame, hamming_distances

TH_HIGH       = 100
SCALE_FACTOR  = 1.2
N_LEVELS      = 8
SCALE_FACTORS = np.array([SCALE_FACTOR ** i for i in range(N_LEVELS)], dtype=np.float64)


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


# ── PoseOptimization (GTSAM) ─────────────────────────────────────────────────

_HUBER_DELTA = float(np.sqrt(5.991))
_CHI2_TH     = 5.991
_FIX_NOISE   = gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)
_POSE_KEY    = gtsam.symbol('x', 0)


def _T_cw_to_pose3(T_cw):
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    return gtsam.Pose3(gtsam.Rot3(R_cw.T), -R_cw.T @ t_cw)


def _pose3_to_T_cw(pose):
    R_wc = pose.rotation().matrix()
    t_wc = pose.translation()
    T = np.eye(4)
    T[:3, :3] = R_wc.T
    T[:3, 3]  = -R_wc.T @ t_wc
    return T


def pose_optimization(T_cw_init, kps, mp_list, kp_indices, K, n_iter=4):
    """
    Refine T_cw via GTSAM LM with per-octave Huber noise model.
    Mirrors Optimizer::PoseOptimization (g2o SE3 + Huber + invSigma2 per octave).

    Iters 0-1: Huber robust kernel (soft outlier down-weighting).
    Iters 2-3: no robust kernel, hard chi2 outlier removal only.

    Args:
        T_cw_init  : (4,4) initial world→camera pose
        kps        : list of cv2.KeyPoint (current frame)
        mp_list    : list of MapPoint, same order as kp_indices
        kp_indices : list of int — which keypoints are matched
        K          : (3,3) camera intrinsics
        n_iter     : outer iterations (default 4)

    Returns:
        T_cw    : (4,4) float64 refined pose
        inliers : (N,) bool array
    """
    Kd     = K.astype(np.float64)
    fx, fy = float(Kd[0, 0]), float(Kd[1, 1])
    cx, cy = float(Kd[0, 2]), float(Kd[1, 2])
    cal    = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

    pts3d  = np.array([mp.pos for mp in mp_list],        dtype=np.float64)  # (N,3)
    pts2d  = np.array([kps[ki].pt for ki in kp_indices], dtype=np.float64)  # (N,2)

    octaves    = np.clip([kps[ki].octave for ki in kp_indices], 0, N_LEVELS - 1)
    sigmas     = SCALE_FACTORS[np.array(octaves, dtype=np.int32)]   # (N,) sigma per point
    inv_sigma2 = 1.0 / sigmas ** 2

    # Pre-build per-octave noise models
    lm_params = gtsam.LevenbergMarquardtParams()
    lm_params.setMaxIterations(10)
    lm_params.setVerbosity('SILENT')

    T       = T_cw_init.astype(np.float64).copy()
    inliers = np.ones(len(pts3d), dtype=bool)

    for it in range(n_iter):
        active = np.where(inliers)[0]
        if len(active) < 4:
            break

        graph  = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        values.insert(_POSE_KEY, _T_cw_to_pose3(T))

        for i in active:
            pk = gtsam.symbol('l', int(i))
            values.insert(pk, gtsam.Point3(*pts3d[i]))

            base = gtsam.noiseModel.Isotropic.Sigma(2, float(sigmas[i]))
            if it < 2:
                noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(_HUBER_DELTA), base)
            else:
                noise = base

            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(*pts2d[i]), noise, _POSE_KEY, pk, cal))
            graph.add(gtsam.PriorFactorPoint3(pk, gtsam.Point3(*pts3d[i]), _FIX_NOISE))

        try:
            result = gtsam.LevenbergMarquardtOptimizer(graph, values, lm_params).optimize()
            T = _pose3_to_T_cw(result.atPose3(_POSE_KEY))
        except Exception:
            break

        # Outlier removal: whitened chi2 > 5.991
        R_new, t_new = T[:3, :3], T[:3, 3]
        x3dc  = (R_new @ pts3d.T).T + t_new
        valid = x3dc[:, 2] > 0
        iz    = np.where(valid, 1.0 / (x3dc[:, 2] + 1e-10), 0.0)
        px    = fx * x3dc[:, 0] * iz + cx
        py    = fy * x3dc[:, 1] * iz + cy
        chi2  = (px - pts2d[:, 0]) ** 2 + (py - pts2d[:, 1]) ** 2
        inliers = valid & (chi2 * inv_sigma2 < _CHI2_TH)

    return T, inliers


# ── TrackWithMotionModel ──────────────────────────────────────────────────────

def track_with_motion_model(frame, last_frame_mps, T_cw_last, velocity, K, th=15):
    """
    Estimate current pose using constant-velocity prediction + map point projection.
    Mirrors Tracking::TrackWithMotionModel (monocular path).

    Args:
        frame         : Frame (current)
        last_frame_mps: list of MapPoint visible/inlier in previous frame
        T_cw_last     : (4,4) pose of previous frame
        velocity      : (4,4) T_cur_last — relative motion estimate
        K             : (3,3) camera intrinsics
        th            : projection search radius factor (15)

    Returns:
        T_cw      : (4,4) optimized pose, or None if tracking failed
        n_inliers : int
        inlier_mps: dict {kp_idx: MapPoint} of inlier matches
    """
    # 1. Predict pose with constant velocity
    T_cw_pred = velocity @ T_cw_last

    # 2. SearchByProjection with th=15
    matches = search_by_projection(frame, last_frame_mps, T_cw_pred, K, th)

    # 3. Retry with wider window if too few matches
    if len(matches) < 20:
        matches = search_by_projection(frame, last_frame_mps, T_cw_pred, K, 2 * th)

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


# ── TrackLocalMap ─────────────────────────────────────────────────────────────

def track_local_map(frame, all_map_points, T_cw, existing_matches, K, th=1):
    """
    Project remaining local map points with tight window, then re-optimize pose.
    Mirrors Tracking::TrackLocalMap → SearchLocalPoints() + PoseOptimization().

    Args:
        frame            : Frame (current)
        all_map_points   : all available map points (local map)
        T_cw             : pose estimate from TrackWithMotionModel
        existing_matches : dict {kp_idx: MapPoint} already found by TrackWithMotionModel
        K                : (3,3) camera intrinsics
        th               : search radius factor (1 for monocular without IMU)

    Returns:
        T_cw      : (4,4) refined pose
        n_inliers : int
        inlier_mps: dict {kp_idx: MapPoint} (existing + new inliers)
    """
    matched_kp_set = set(existing_matches.keys())
    matched_mp_set = set(id(mp) for mp in existing_matches.values())

    # Project map points not already matched, with tight search window
    remaining  = [mp for mp in all_map_points if id(mp) not in matched_mp_set]
    new_matches = search_by_projection(frame, remaining, T_cw, K, th=th)
    new_matches = {ki: mp for ki, mp in new_matches.items()
                   if ki not in matched_kp_set}

    # Combine existing + new matches
    combined   = {**existing_matches, **new_matches}
    kp_indices = list(combined.keys())
    mp_list    = [combined[ki] for ki in kp_indices]

    if len(kp_indices) < 4:
        return T_cw, len(kp_indices), existing_matches

    T_cw_new, inliers = pose_optimization(T_cw, frame.keypoints, mp_list, kp_indices, K)

    n_inliers  = int(inliers.sum())
    inlier_mps = {kp_indices[i]: mp_list[i]
                  for i in range(len(kp_indices)) if inliers[i]}

    return T_cw_new, n_inliers, inlier_mps
