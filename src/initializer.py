"""
Two-view reconstruction initializer — ORB-SLAM3 TwoViewReconstruction port.

Pipeline:
  1. RANSAC: FindHomography + FindFundamental in parallel (here sequential)
  2. Model selection: RH = SH / (SH + SF), if RH > 0.5 → H else F
  3. Pose recovery:
     - From F: E = K^T F K → decompose → 4 hypotheses → CheckRT
     - From H: Faugeras decomposition → 8 hypotheses → CheckRT
  4. Triangulate inliers, check reprojection + depth + parallax
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.camera import K

SIGMA            = 1.0
SIGMA2           = SIGMA * SIGMA
MAX_ITER         = 200
MIN_PARALLAX     = 1.0
MIN_TRIANGULATED = 50


# ── Coordinate normalization ─────────────────────────────────────────────────

def _normalize(pts):
    mean = pts.mean(axis=0)
    centered = pts - mean
    mean_dev = np.abs(centered).mean(axis=0)
    sx = 1.0 / (mean_dev[0] + 1e-10)
    sy = 1.0 / (mean_dev[1] + 1e-10)
    T = np.array([[sx,  0,  -mean[0]*sx],
                  [0,   sy, -mean[1]*sy],
                  [0,   0,   1.0       ]], dtype=np.float64)
    pts_n = centered * np.array([sx, sy])
    return pts_n.astype(np.float64), T.astype(np.float64)


# ── DLT solvers ─────────────────────────────────────────────────────────────

def _compute_H21(p1, p2):
    u1, v1 = p1[:, 0], p1[:, 1]
    u2, v2 = p2[:, 0], p2[:, 1]
    z = np.zeros(len(p1)); o = np.ones(len(p1))
    A_even = np.column_stack([z,  z,  z,  -u1, -v1, -o,  v2*u1,  v2*v1,  v2])
    A_odd  = np.column_stack([u1, v1, o,   z,   z,   z, -u2*u1, -u2*v1, -u2])
    A = np.empty((2*len(p1), 9)); A[0::2] = A_even; A[1::2] = A_odd
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def _compute_F21(p1, p2):
    u1, v1 = p1[:, 0], p1[:, 1]
    u2, v2 = p2[:, 0], p2[:, 1]
    A = np.column_stack([u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, np.ones(len(p1))])
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, s, Vt2 = np.linalg.svd(F); s[2] = 0
    return U @ np.diag(s) @ Vt2


# ── Scoring (vectorized) ──────────────────────────────────────────────────────

def _check_homography(H21, H12, pts1_m, pts2_m, sigma):
    """Vectorized symmetric transfer error. pts1_m/pts2_m: (N,2) float64."""
    th = 5.991; inv_s2 = 1.0 / (sigma * sigma)
    N = len(pts1_m); ones = np.ones((N, 1))

    x1h = np.hstack([pts1_m, ones])                       # (N,3)
    x2p = (H21 @ x1h.T).T; x2p = x2p[:, :2] / x2p[:, 2:3]
    e2 = ((pts2_m - x2p)**2).sum(1) * inv_s2

    x2h = np.hstack([pts2_m, ones])                       # (N,3)
    x1p = (H12 @ x2h.T).T; x1p = x1p[:, :2] / x1p[:, 2:3]
    e1 = ((pts1_m - x1p)**2).sum(1) * inv_s2

    inliers = (e2 < th) & (e1 < th)
    score = float(np.where(inliers, (th - e2) + (th - e1), 0.0).sum())
    return score, inliers


def _check_fundamental(F21, pts1_m, pts2_m, sigma):
    """Vectorized Sampson distance. pts1_m/pts2_m: (N,2) float64."""
    th = 3.841; th_score = 5.991; inv_s2 = 1.0 / (sigma * sigma)
    N = len(pts1_m); ones = np.ones((N, 1))

    x1h = np.hstack([pts1_m, ones])                       # (N,3)
    x2h = np.hstack([pts2_m, ones])                       # (N,3)

    l2 = (F21 @ x1h.T).T                                  # (N,3)
    d1 = (x2h * l2).sum(1)**2 / (l2[:,0]**2 + l2[:,1]**2) * inv_s2

    l1 = (F21.T @ x2h.T).T                                # (N,3)
    d2 = (x1h * l1).sum(1)**2 / (l1[:,0]**2 + l1[:,1]**2) * inv_s2

    inliers = (d1 < th) & (d2 < th)
    score = float(np.where(inliers, (th_score - d1) + (th_score - d2), 0.0).sum())
    return score, inliers


# ── RANSAC wrappers ──────────────────────────────────────────────────────────

def _find_homography(kps1, kps2, matches, sets, sigma):
    pts1n, T1 = _normalize(kps1)
    pts2n, T2 = _normalize(kps2)
    T2inv = np.linalg.inv(T2)

    m1s = np.array([m[0] for m in matches])
    m2s = np.array([m[1] for m in matches])
    pts1_m = kps1[m1s].astype(np.float64)
    pts2_m = kps2[m2s].astype(np.float64)

    best_score, best_inliers, best_H = 0.0, None, None
    for idxs in sets:
        p1 = pts1n[m1s[idxs]]; p2 = pts2n[m2s[idxs]]
        Hn = _compute_H21(p1, p2)
        H21 = T2inv @ Hn @ T1
        H12 = np.linalg.inv(H21)
        s, inliers = _check_homography(H21, H12, pts1_m, pts2_m, sigma)
        if s > best_score:
            best_score, best_inliers, best_H = s, inliers, H21.astype(np.float32)

    return best_score, best_inliers, best_H


def _find_fundamental(kps1, kps2, matches, sets, sigma):
    pts1n, T1 = _normalize(kps1)
    pts2n, T2 = _normalize(kps2)

    m1s = np.array([m[0] for m in matches])
    m2s = np.array([m[1] for m in matches])
    pts1_m = kps1[m1s].astype(np.float64)
    pts2_m = kps2[m2s].astype(np.float64)

    best_score, best_inliers, best_F = 0.0, None, None
    for idxs in sets:
        p1 = pts1n[m1s[idxs]]; p2 = pts2n[m2s[idxs]]
        Fn = _compute_F21(p1, p2)
        F21 = T2.T @ Fn @ T1
        s, inliers = _check_fundamental(F21, pts1_m, pts2_m, sigma)
        if s > best_score:
            best_score, best_inliers, best_F = s, inliers, F21.astype(np.float32)

    return best_score, best_inliers, best_F


# ── Triangulation ─────────────────────────────────────────────────────────────

def _triangulate_batch(pts1, pts2, P1, P2):
    """Batch DLT triangulation. pts1/2: (M,2) float64. Returns (M,3), nan for invalid."""
    N = len(pts1)
    A = np.empty((N, 4, 4), dtype=np.float64)
    A[:, 0] = pts1[:, 0:1] * P1[2] - P1[0]
    A[:, 1] = pts1[:, 1:2] * P1[2] - P1[1]
    A[:, 2] = pts2[:, 0:1] * P2[2] - P2[0]
    A[:, 3] = pts2[:, 1:2] * P2[2] - P2[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1, :]                   # (N,4) right singular vector for smallest σ
    valid = np.abs(X[:, 3]) > 1e-10
    pts3d = np.full((N, 3), np.nan)
    pts3d[valid] = X[valid, :3] / X[valid, 3:4]
    return pts3d


def _check_rt(R, t, kps1, kps2, matches, inliers, K, th2):
    """Vectorized triangulation + depth/reprojection checks."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    Kd = K.astype(np.float64)
    Rd = R.astype(np.float64); td = t.astype(np.float64)
    P1 = np.hstack([Kd, np.zeros((3, 1))])
    P2 = Kd @ np.hstack([Rd, td.reshape(3, 1)])
    O2 = -Rd.T @ td

    n_ref = len(kps1)
    points3d = [None] * n_ref
    good = np.zeros(n_ref, dtype=bool)

    sel = inliers.nonzero()[0]
    if len(sel) == 0:
        return 0, points3d, good, 0.0

    i1s = np.array([matches[i][0] for i in sel])
    i2s = np.array([matches[i][1] for i in sel])
    pts1 = kps1[i1s].astype(np.float64)
    pts2 = kps2[i2s].astype(np.float64)

    pts3d = _triangulate_batch(pts1, pts2, P1, P2)        # (M,3)

    finite = np.all(np.isfinite(pts3d), axis=1)

    n2 = pts3d - O2
    cos_p = (pts3d * n2).sum(1) / (
        np.linalg.norm(pts3d, axis=1) * np.linalg.norm(n2, axis=1) + 1e-10)

    depth1 = pts3d[:, 2]
    pts3d_c2 = (Rd @ pts3d.T).T + td
    depth2 = pts3d_c2[:, 2]

    iz1 = 1.0 / (depth1 + 1e-10)
    err1 = (fx*pts3d[:,0]*iz1+cx - pts1[:,0])**2 + (fy*pts3d[:,1]*iz1+cy - pts1[:,1])**2

    iz2 = 1.0 / (depth2 + 1e-10)
    err2 = (fx*pts3d_c2[:,0]*iz2+cx - pts2[:,0])**2 + (fy*pts3d_c2[:,1]*iz2+cy - pts2[:,1])**2

    ok_mask = (finite
               & ((depth1 > 0) | (cos_p >= 0.99998))
               & ((depth2 > 0) | (cos_p >= 0.99998))
               & (err1 <= th2) & (err2 <= th2))

    cos_good = []; n_good = 0
    for j in ok_mask.nonzero()[0]:
        i1 = int(i1s[j]); cp = float(cos_p[j])
        points3d[i1] = pts3d[j].astype(np.float32)
        cos_good.append(cp); n_good += 1
        if cp < 0.99998:
            good[i1] = True

    if n_good > 0:
        cos_good.sort()
        parallax = float(np.degrees(np.arccos(cos_good[min(50, n_good - 1)])))
    else:
        parallax = 0.0

    return n_good, points3d, good, parallax


# ── Pose recovery ─────────────────────────────────────────────────────────────

def _decompose_e(E):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:  U = -U
    if np.linalg.det(Vt) < 0: Vt = -Vt
    t = U[:, 2]; t /= np.linalg.norm(t) + 1e-10
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    R1 = U @ W @ Vt;   R1 = R1 if np.linalg.det(R1) > 0 else -R1
    R2 = U @ W.T @ Vt; R2 = R2 if np.linalg.det(R2) > 0 else -R2
    return R1.astype(np.float32), R2.astype(np.float32), t.astype(np.float32)


def _reconstruct_f(inliers, F21, K, kps1, kps2, matches, min_parallax, min_tri):
    N = inliers.sum()
    Kd = K.astype(np.float64)
    E21 = Kd.T @ F21.astype(np.float64) @ Kd
    R1, R2, t = _decompose_e(E21)
    th2 = 4.0 * SIGMA2

    results = []
    for R, tv in [(R1, t), (R2, t), (R1, -t), (R2, -t)]:
        ng, p3d, good, par = _check_rt(R, tv, kps1, kps2, matches, inliers, K, th2)
        results.append((ng, p3d, good, par, R, tv))

    max_good = max(r[0] for r in results)
    n_min = max(int(0.9 * N), min_tri)
    n_similar = sum(1 for r in results if r[0] > 0.7 * max_good)

    if max_good < n_min or n_similar > 1:
        return False, None, None, None

    best = max(results, key=lambda r: r[0])
    ng, p3d, good, par, R, tv = best
    if par < min_parallax:
        return False, None, None, None

    return True, R, tv, p3d


def _reconstruct_h(inliers, H21, K, kps1, kps2, matches, min_parallax, min_tri):
    N = inliers.sum()
    Kd = K.astype(np.float64)
    invK = np.linalg.inv(Kd)
    A = invK @ H21.astype(np.float64) @ Kd
    U, w, Vt = np.linalg.svd(A)
    V = Vt.T; s = np.linalg.det(U) * np.linalg.det(V)
    d1, d2, d3 = w[0], w[1], w[2]

    if d1/d2 < 1.00001 or d2/d3 < 1.00001:
        return False, None, None, None

    aux1 = np.sqrt((d1**2 - d2**2) / (d1**2 - d3**2))
    aux3 = np.sqrt((d2**2 - d3**2) / (d1**2 - d3**2))
    x1s = [ aux1,  aux1, -aux1, -aux1]
    x3s = [ aux3, -aux3,  aux3, -aux3]

    Rs, ts = [], []
    aux_st = np.sqrt((d1**2 - d2**2) * (d2**2 - d3**2)) / ((d1 + d3) * d2)
    ct = (d2**2 + d1*d3) / ((d1 + d3) * d2)
    sts = [aux_st, -aux_st, -aux_st, aux_st]

    for i in range(4):
        Rp = np.array([[ct, 0, -sts[i]], [0, 1, 0], [sts[i], 0, ct]])
        R = s * U @ Rp @ Vt
        tp = np.array([x1s[i], 0, -x3s[i]]) * (d1 - d3)
        t = U @ tp; t /= np.linalg.norm(t)
        np_ = V @ np.array([x1s[i], 0, x3s[i]])
        if np_[2] < 0: np_ = -np_
        Rs.append(R.astype(np.float32)); ts.append(t.astype(np.float32))

    aux_sp = np.sqrt((d1**2 - d2**2) * (d2**2 - d3**2)) / ((d1 - d3) * d2)
    cp = (d1*d3 - d2**2) / ((d1 - d3) * d2)
    sps = [aux_sp, -aux_sp, -aux_sp, aux_sp]

    for i in range(4):
        Rp = np.array([[cp, 0, sps[i]], [0, -1, 0], [sps[i], 0, -cp]])
        R = s * U @ Rp @ Vt
        tp = np.array([x1s[i], 0, x3s[i]]) * (d1 + d3)
        t = U @ tp; t /= np.linalg.norm(t)
        Rs.append(R.astype(np.float32)); ts.append(t.astype(np.float32))

    best_good, second_good = 0, 0
    best_idx, best_par, best_p3d = -1, -1.0, None
    th2 = 4.0 * SIGMA2

    for i, (R, tv) in enumerate(zip(Rs, ts)):
        ng, p3d, good, par = _check_rt(R, tv, kps1, kps2, matches, inliers, K, th2)
        if ng > best_good:
            second_good = best_good
            best_good = ng; best_idx = i; best_par = par; best_p3d = p3d
        elif ng > second_good:
            second_good = ng

    n_min = max(int(0.9 * N), min_tri)
    if (second_good >= 0.75 * best_good or best_par < min_parallax
            or best_good < n_min or best_good < 0.9 * N):
        return False, None, None, None

    return True, Rs[best_idx], ts[best_idx], best_p3d


# ── Public API ────────────────────────────────────────────────────────────────

def reconstruct(kps1, kps2, matches12, K=K,
                sigma=SIGMA, n_iter=MAX_ITER,
                min_parallax=MIN_PARALLAX, min_tri=MIN_TRIANGULATED):
    """
    Two-view reconstruction (ORB-SLAM3 TwoViewReconstruction::Reconstruct).

    Args:
        kps1, kps2  : (N,2) float32 — pixel coordinates (level-0, undistorted)
        matches12   : list of int, len=N1; matches12[i]=j or -1
        K           : 3x3 camera intrinsic matrix
        sigma       : reprojection error std dev (default 1.0)
        n_iter      : RANSAC iterations (default 200)
        min_parallax: minimum parallax in degrees (default 1.0)
        min_tri     : minimum triangulated points (default 50)

    Returns:
        success     : bool
        R           : (3,3) rotation  frame1→frame2
        t           : (3,)  translation frame1→frame2 (unit vector)
        points3d    : list of (3,) or None, indexed by frame1 keypoint index
        triangulated: bool array, len=N1
    """
    match_pairs = [(i, j) for i, j in enumerate(matches12) if j >= 0]
    N = len(match_pairs)
    if N < 8:
        return False, None, None, None, None

    rng = np.random.default_rng(0)
    sets = np.array([rng.choice(N, 8, replace=False) for _ in range(n_iter)])  # (n_iter, 8)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fH = ex.submit(_find_homography, kps1, kps2, match_pairs, sets, sigma)
        fF = ex.submit(_find_fundamental, kps1, kps2, match_pairs, sets, sigma)
        SH, inliers_H, H21 = fH.result()
        SF, inliers_F, F21 = fF.result()

    if SH + SF == 0:
        return False, None, None, None, None

    RH = SH / (SH + SF)

    if RH > 0.50:
        ok, R, t, p3d = _reconstruct_h(inliers_H, H21, K, kps1, kps2,
                                        match_pairs, min_parallax, min_tri)
    else:
        ok, R, t, p3d = _reconstruct_f(inliers_F, F21, K, kps1, kps2,
                                        match_pairs, min_parallax, min_tri)

    if not ok:
        return False, None, None, None, None

    tri = np.array([p is not None for p in p3d], dtype=bool)
    return True, R, t, p3d, tri
