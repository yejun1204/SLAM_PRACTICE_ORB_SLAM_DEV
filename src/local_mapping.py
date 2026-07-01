"""
LocalMapping — ORB-SLAM3 LocalMapping.cc port (monocular).

Implements:
  - KeyFrame
  - need_new_keyframe   (Tracking::NeedNewKeyFrame, monocular path)
  - search_for_triangulation  (ORBmatcher::SearchForTriangulation, no BoW)
  - create_new_map_points     (LocalMapping::CreateNewMapPoints, monocular)
"""

import numpy as np
from src.orb_matcher import hamming_distances
from src.tracker import MapPoint, SCALE_FACTORS, N_LEVELS

TH_LOW      = 50
RATIO_FACTOR = 1.5 * SCALE_FACTORS[1]   # 1.5 * 1.2 = 1.8  (ratioFactor in C++)


# ── KeyFrame ──────────────────────────────────────────────────────────────────

class KeyFrame:
    _next_id = 0

    def __init__(self, frame_id, T_cw, keypoints, descriptors, map_point_matches):
        self.id               = KeyFrame._next_id
        KeyFrame._next_id    += 1
        self.frame_id         = frame_id
        self.T_cw             = T_cw.copy()
        self.keypoints        = keypoints                    # list[cv2.KeyPoint]
        self.descriptors      = np.asarray(descriptors)     # (N,32) uint8
        self.map_point_matches = dict(map_point_matches)    # {kp_idx: MapPoint}

    @property
    def camera_center(self):
        R = self.T_cw[:3, :3]
        t = self.T_cw[:3, 3]
        return -R.T @ t

    def median_scene_depth(self):
        R2 = self.T_cw[2, :3]
        t2 = self.T_cw[2, 3]
        depths = [float(R2 @ mp.pos + t2)
                  for mp in self.map_point_matches.values()]
        depths = [d for d in depths if d > 0]
        return float(np.median(depths)) if depths else 1.0


# ── NeedNewKeyFrame ───────────────────────────────────────────────────────────

def need_new_keyframe(frame_id, last_kf_frame_id, n_inliers, n_ref_matches, max_frames=20):
    """
    Monocular NeedNewKeyFrame (synchronous: LocalMapping always idle).

    n_ref_matches: tracked map points in the reference KF at insertion time
                   (frozen before CreateNewMapPoints runs — mirrors
                   TrackedMapPoints(nMinObs) which excludes newly created points
                   with only 1 observation).

    Mirrors ORB-SLAM3 monocular path:
      thRefRatio = 0.9
      c1a: frame_id >= last_kf_id + max_frames
      c1b: True (mMinFrames=0, LocalMapping idle)
      c2 : n_inliers < n_ref * 0.9 AND n_inliers > 15
      return (c1a or c1b) and c2
    """
    c1a = frame_id >= last_kf_frame_id + max_frames
    c1b = frame_id > last_kf_frame_id       # at least 1 frame gap
    c2  = (n_inliers < n_ref_matches * 0.9) and (n_inliers > 15)
    return (c1a or c1b) and c2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_fundamental(T_cw1, T_cw2, K):
    """F12: maps a point in image1 to its epipolar line in image2."""
    T12   = T_cw1 @ np.linalg.inv(T_cw2)
    R12   = T12[:3, :3]
    t12   = T12[:3, 3]
    t12x  = np.array([[     0, -t12[2],  t12[1]],
                      [ t12[2],      0, -t12[0]],
                      [-t12[1],  t12[0],      0]], dtype=np.float64)
    K_inv = np.linalg.inv(K.astype(np.float64))
    return K_inv.T @ t12x @ R12 @ K_inv


def _triangulate_point(pt1, pt2, P1, P2):
    """Single-point DLT triangulation. Returns (3,) or None."""
    A = np.array([
        pt1[0] * P1[2] - P1[0],
        pt1[1] * P1[2] - P1[1],
        pt2[0] * P2[2] - P2[0],
        pt2[1] * P2[2] - P2[1],
    ], dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-10:
        return None
    return X[:3] / X[3]


# ── SearchForTriangulation ────────────────────────────────────────────────────

def search_for_triangulation(kf1, kf2, K):
    """
    Match unmatched keypoints between two KFs using epipolar + descriptor.
    Mirrors ORBmatcher::SearchForTriangulation (no BoW — brute-force with
    epipolar constraint as the primary filter).

    Returns: list of (idx1, idx2) — indices into kf1/kf2 keypoints.
    """
    matched1   = set(kf1.map_point_matches.keys())
    matched2   = set(kf2.map_point_matches.keys())
    idx1_list  = [i for i in range(len(kf1.keypoints)) if i not in matched1]
    idx2_list  = [i for i in range(len(kf2.keypoints)) if i not in matched2]
    if not idx1_list or not idx2_list:
        return []

    F12  = _compute_fundamental(kf1.T_cw, kf2.T_cw, K)
    K64  = K.astype(np.float64)
    fx, fy, cx, cy = K64[0,0], K64[1,1], K64[0,2], K64[1,2]

    # Epipole in image 2 (projection of KF1 center)
    C1    = kf1.camera_center
    R2, t2 = kf2.T_cw[:3,:3], kf2.T_cw[:3,3]
    C1_in2 = R2 @ C1 + t2
    if C1_in2[2] > 1e-6:
        ep2 = np.array([fx * C1_in2[0] / C1_in2[2] + cx,
                        fy * C1_in2[1] / C1_in2[2] + cy])
    else:
        ep2 = None

    kps2     = kf2.keypoints
    oct2     = np.array([kps2[j].octave for j in idx2_list])
    sigmas2  = SCALE_FACTORS[np.clip(oct2, 0, N_LEVELS - 1)]
    pts2     = np.array([kps2[j].pt for j in idx2_list], dtype=np.float64)
    desc2    = kf2.descriptors[np.array(idx2_list)]       # (M2, 32)
    pts2_h   = np.column_stack([pts2, np.ones(len(pts2))])  # (M2, 3)

    # Epipole exclusion mask for image 2
    if ep2 is not None:
        ep_dists2 = ((pts2 - ep2) ** 2).sum(axis=1)           # (M2,)
        ep_thresh2 = 100.0 * SCALE_FACTORS[np.clip(oct2, 0, N_LEVELS - 1)]
        ep_ok2 = ep_dists2 > ep_thresh2                        # (M2,)
    else:
        ep_ok2 = np.ones(len(idx2_list), dtype=bool)

    kps1   = kf1.keypoints
    pts1   = np.array([kps1[i].pt for i in idx1_list], dtype=np.float64)
    desc1  = kf1.descriptors[np.array(idx1_list)]             # (M1, 32)
    pts1_h = np.column_stack([pts1, np.ones(len(pts1))])      # (M1, 3)

    # Epipolar lines in image 2 for all kp1: (M1, 3) each row [a, b, c]
    lines2      = (F12 @ pts1_h.T).T                          # (M1, 3)
    line_norms  = np.sqrt(lines2[:,0]**2 + lines2[:,1]**2 + 1e-12)  # (M1,)

    matches = []
    used2   = set()

    # Chunked loop to limit peak memory (chunk_size × M2 floats at a time)
    CHUNK = 500
    for c_start in range(0, len(idx1_list), CHUNK):
        c_end   = min(c_start + CHUNK, len(idx1_list))
        sl      = slice(c_start, c_end)

        # Epipolar distances: (M2, chunk)
        epi_num  = np.abs(pts2_h @ lines2[sl].T)              # (M2, chunk)
        epi_dist = epi_num / line_norms[None, sl]              # (M2, chunk)

        # Threshold per kp2: 3.84 * sigma  (chi2(1df, 0.95) = 3.84)
        thresh2  = (3.84 * sigmas2)[:, None]                   # (M2, 1)
        epi_ok   = (epi_dist < thresh2) & ep_ok2[:, None]     # (M2, chunk)

        for local_i in range(c_end - c_start):
            i = c_start + local_i
            cand_j = [j for j in np.where(epi_ok[:, local_i])[0]
                      if j not in used2]
            if not cand_j:
                continue

            cand_arr  = np.array(cand_j, dtype=np.int32)
            dists     = hamming_distances(desc1[i], desc2[cand_arr])
            best_local = int(np.argmin(dists))
            if dists[best_local] >= TH_LOW:
                continue

            j = int(cand_arr[best_local])
            matches.append((idx1_list[i], idx2_list[j]))
            used2.add(j)

    return matches


# ── CreateNewMapPoints ────────────────────────────────────────────────────────

def create_new_map_points(cur_kf, keyframes, K, map_points):
    """
    Triangulate new map points from cur_kf paired with recent neighbor KFs.
    Mirrors LocalMapping::CreateNewMapPoints (monocular path).

    Adds new MapPoints to map_points in-place.
    Returns number of newly created points.
    """
    K64  = K.astype(np.float64)
    fx, fy, cx, cy = K64[0,0], K64[1,1], K64[0,2], K64[1,2]
    R1, t1 = cur_kf.T_cw[:3,:3], cur_kf.T_cw[:3,3]
    Ow1    = cur_kf.camera_center
    P1     = K64 @ cur_kf.T_cw[:3]           # (3,4) projection matrix

    # Monocular: use up to 10 most recent neighbors
    nn        = min(10, len(keyframes) - 1)
    neighbors = keyframes[max(0, len(keyframes) - 1 - nn) : len(keyframes) - 1]

    n_new = 0
    for kf2 in neighbors:
        Ow2 = kf2.camera_center
        R2, t2 = kf2.T_cw[:3,:3], kf2.T_cw[:3,3]
        P2  = K64 @ kf2.T_cw[:3]

        # Skip if baseline too short relative to scene depth
        baseline = float(np.linalg.norm(Ow2 - Ow1))
        if baseline / max(cur_kf.median_scene_depth(), 1e-6) < 0.01:
            continue

        match_pairs = search_for_triangulation(cur_kf, kf2, K)

        for idx1, idx2 in match_pairs:
            kp1 = cur_kf.keypoints[idx1]
            kp2 = kf2.keypoints[idx2]

            # Parallax check (cosParallax < 0.9998 ≈ > 1.1 degrees)
            xn1  = np.array([(kp1.pt[0]-cx)/fx, (kp1.pt[1]-cy)/fy, 1.0])
            xn2  = np.array([(kp2.pt[0]-cx)/fx, (kp2.pt[1]-cy)/fy, 1.0])
            ray1 = R1.T @ xn1
            ray2 = R2.T @ xn2
            cos_par = ray1.dot(ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2) + 1e-12)
            if cos_par >= 0.9998:
                continue

            x3d = _triangulate_point(kp1.pt, kp2.pt, P1, P2)
            if x3d is None:
                continue

            # Positive depth in both cameras
            z1 = float(R1[2] @ x3d + t1[2])
            if z1 <= 0:
                continue
            z2 = float(R2[2] @ x3d + t2[2])
            if z2 <= 0:
                continue

            # Reprojection error in KF1
            sigma1 = SCALE_FACTORS[min(kp1.octave, N_LEVELS - 1)]
            x1c   = R1 @ x3d + t1
            u1    = fx * x1c[0] / x1c[2] + cx
            v1    = fy * x1c[1] / x1c[2] + cy
            if (u1 - kp1.pt[0])**2 + (v1 - kp1.pt[1])**2 > 5.991 * sigma1**2:
                continue

            # Reprojection error in KF2
            sigma2 = SCALE_FACTORS[min(kp2.octave, N_LEVELS - 1)]
            x2c   = R2 @ x3d + t2
            u2    = fx * x2c[0] / x2c[2] + cx
            v2    = fy * x2c[1] / x2c[2] + cy
            if (u2 - kp2.pt[0])**2 + (v2 - kp2.pt[1])**2 > 5.991 * sigma2**2:
                continue

            # Scale consistency
            dist1 = float(np.linalg.norm(x3d - Ow1))
            dist2 = float(np.linalg.norm(x3d - Ow2))
            if dist1 < 1e-6 or dist2 < 1e-6:
                continue
            ratio_dist   = dist2 / dist1
            ratio_octave = SCALE_FACTORS[kp1.octave] / SCALE_FACTORS[kp2.octave]
            if ratio_dist * RATIO_FACTOR < ratio_octave or ratio_dist > ratio_octave * RATIO_FACTOR:
                continue

            mp = MapPoint(pos=x3d, descriptor=cur_kf.descriptors[idx1],
                          octave=kp1.octave)
            map_points.append(mp)
            # Mark so they won't be re-triangulated
            cur_kf.map_point_matches[idx1] = mp
            kf2.map_point_matches[idx2]    = mp
            n_new += 1

    return n_new
