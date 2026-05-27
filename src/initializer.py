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

# ── Camera intrinsics (EuRoC cam0) ─────────────────────────────────────────
K = np.array([[458.654,   0.0,   367.215],
              [  0.0,   457.296, 248.375],
              [  0.0,     0.0,     1.0  ]], dtype=np.float32)

# ── Constants ───────────────────────────────────────────────────────────────
SIGMA       = 1.0
SIGMA2      = SIGMA * SIGMA
MAX_ITER    = 200
MIN_PARALLAX = 1.0   # degrees
MIN_TRIANGULATED = 50


# ── Coordinate normalization ─────────────────────────────────────────────────

def _normalize(pts):
    """Isotropic normalization (ORB-SLAM3 Normalize).
    pts: (N,2) float32
    Returns: (pts_norm, T) where T is 3x3 transform matrix.
    """
    mean = pts.mean(axis=0)
    centered = pts - mean
    mean_dev = np.abs(centered).mean(axis=0)
    sx = 1.0 / (mean_dev[0] + 1e-10)
    sy = 1.0 / (mean_dev[1] + 1e-10)
    T = np.array([[sx,  0,  -mean[0]*sx],
                  [0,   sy, -mean[1]*sy],
                  [0,   0,   1.0       ]], dtype=np.float64)
    pts_n = centered * np.array([sx, sy])
    return pts_n.astype(np.float32), T.astype(np.float32)


# ── DLT solvers ─────────────────────────────────────────────────────────────

def _compute_H21(p1, p2):
    """Homography from 8 normalized point pairs via DLT."""
    N = len(p1)
    A = np.zeros((2*N, 9), dtype=np.float64)
    for i, ((u1, v1), (u2, v2)) in enumerate(zip(p1, p2)):
        A[2*i]   = [0, 0, 0, -u1, -v1, -1,  v2*u1, v2*v1, v2]
        A[2*i+1] = [u1, v1, 1,  0,   0,  0, -u2*u1,-u2*v1,-u2]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return (H / H[2, 2]).astype(np.float32)


def _compute_F21(p1, p2):
    """Fundamental matrix from 8 normalized point pairs via 8-point algorithm."""
    N = len(p1)
    A = np.zeros((N, 9), dtype=np.float64)
    for i, ((u1, v1), (u2, v2)) in enumerate(zip(p1, p2)):
        A[i] = [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    # Enforce rank-2
    U, s, Vt2 = np.linalg.svd(F)
    s[2] = 0
    return (U @ np.diag(s) @ Vt2).astype(np.float32)


# ── Scoring ──────────────────────────────────────────────────────────────────

def _check_homography(H21, H12, kps1, kps2, matches, sigma):
    """Symmetric transfer error score + inlier mask."""
    th = 5.991
    inv_s2 = 1.0 / (sigma * sigma)
    score = 0.0
    inliers = np.zeros(len(matches), dtype=bool)

    for i, (i1, i2) in enumerate(matches):
        u1, v1 = kps1[i1]
        u2, v2 = kps2[i2]

        # H21 * x1 → x2
        x2 = H21 @ np.array([u1, v1, 1.0], dtype=np.float64)
        x2 /= x2[2]
        e2 = (u2 - x2[0])**2 + (v2 - x2[1])**2
        chi2 = e2 * inv_s2
        if chi2 > th:
            continue

        # H12 * x2 → x1
        x1 = H12 @ np.array([u2, v2, 1.0], dtype=np.float64)
        x1 /= x1[2]
        e1 = (u1 - x1[0])**2 + (v1 - x1[1])**2
        chi1 = e1 * inv_s2
        if chi1 > th:
            continue

        inliers[i] = True
        score += (th - chi2) + (th - chi1)

    return score, inliers


def _check_fundamental(F21, kps1, kps2, matches, sigma):
    """Sampson distance score + inlier mask."""
    th      = 3.841
    th_score = 5.991
    inv_s2  = 1.0 / (sigma * sigma)
    score   = 0.0
    inliers = np.zeros(len(matches), dtype=bool)

    for i, (i1, i2) in enumerate(matches):
        u1, v1 = kps1[i1]
        u2, v2 = kps2[i2]
        x1 = np.array([u1, v1, 1.0], dtype=np.float64)
        x2 = np.array([u2, v2, 1.0], dtype=np.float64)

        # l2 = F21 * x1
        l2 = F21 @ x1
        num2 = (x2 @ l2) ** 2
        d1 = num2 / (l2[0]**2 + l2[1]**2) * inv_s2
        if d1 > th:
            continue

        # l1 = F21^T * x2
        l1 = F21.T @ x2
        num1 = (x1 @ l1) ** 2
        d2 = num1 / (l1[0]**2 + l1[1]**2) * inv_s2
        if d2 > th:
            continue

        inliers[i] = True
        score += (th_score - d1) + (th_score - d2)

    return score, inliers


# ── RANSAC wrappers ──────────────────────────────────────────────────────────

def _find_homography(kps1, kps2, matches, sets, sigma):
    pts1n, T1 = _normalize(kps1)
    pts2n, T2 = _normalize(kps2)
    T2inv = np.linalg.inv(T2)
    best_score, best_inliers, best_H = 0.0, None, None

    for idxs in sets:
        p1 = pts1n[[matches[j][0] for j in idxs]]
        p2 = pts2n[[matches[j][1] for j in idxs]]
        Hn = _compute_H21(p1, p2)
        H21 = (T2inv @ Hn @ T1).astype(np.float32)
        H12 = np.linalg.inv(H21).astype(np.float32)
        s, inliers = _check_homography(H21, H12, kps1, kps2, matches, sigma)
        if s > best_score:
            best_score, best_inliers, best_H = s, inliers, H21

    return best_score, best_inliers, best_H


def _find_fundamental(kps1, kps2, matches, sets, sigma):
    pts1n, T1 = _normalize(kps1)
    pts2n, T2 = _normalize(kps2)
    best_score, best_inliers, best_F = 0.0, None, None

    for idxs in sets:
        p1 = pts1n[[matches[j][0] for j in idxs]]
        p2 = pts2n[[matches[j][1] for j in idxs]]
        Fn = _compute_F21(p1, p2)
        F21 = (T2.T @ Fn @ T1).astype(np.float32)
        s, inliers = _check_fundamental(F21, kps1, kps2, matches, sigma)
        if s > best_score:
            best_score, best_inliers, best_F = s, inliers, F21

    return best_score, best_inliers, best_F


# ── Triangulation ─────────────────────────────────────────────────────────────

def _triangulate(x1, x2, P1, P2):
    """DLT triangulation. x1, x2: (3,) homogeneous. Returns (3,) 3D point."""
    A = np.array([x1[0]*P1[2] - P1[0],
                  x1[1]*P1[2] - P1[1],
                  x2[0]*P2[2] - P2[0],
                  x2[1]*P2[2] - P2[1]], dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-10:
        return None
    return (X[:3] / X[3]).astype(np.float32)


def _check_rt(R, t, kps1, kps2, matches, inliers, K, th2):
    """Triangulate inlier matches, check depth + reprojection. Returns (n_good, points3d, good_mask, parallax_deg)."""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    P1 = np.hstack([K, np.zeros((3,1), dtype=np.float32)])
    P2 = (K @ np.hstack([R, t.reshape(3,1)])).astype(np.float32)

    O1 = np.zeros(3, dtype=np.float32)
    O2 = (-R.T @ t).astype(np.float32)

    n_pts = len(kps1)
    points3d = [None] * n_pts
    good = np.zeros(n_pts, dtype=bool)
    cos_parallax = []
    n_good = 0

    for i, (i1, i2) in enumerate(matches):
        if not inliers[i]:
            continue
        u1, v1 = kps1[i1]
        u2, v2 = kps2[i2]
        x1 = np.array([u1, v1, 1.0], dtype=np.float32)
        x2 = np.array([u2, v2, 1.0], dtype=np.float32)

        p3d = _triangulate(x1, x2, P1, P2)
        if p3d is None or not np.all(np.isfinite(p3d)):
            continue

        # Parallax
        n1 = p3d - O1;  n2 = p3d - O2
        d1 = np.linalg.norm(n1);  d2 = np.linalg.norm(n2)
        cos_p = float(n1 @ n2) / (d1 * d2 + 1e-10)

        # Depth in camera 1
        if p3d[2] <= 0 and cos_p < 0.99998:
            continue
        # Depth in camera 2
        p3d_c2 = R @ p3d + t
        if p3d_c2[2] <= 0 and cos_p < 0.99998:
            continue

        # Reprojection error in image 1
        iz1 = 1.0 / (p3d[2] + 1e-10)
        px1 = fx * p3d[0] * iz1 + cx
        py1 = fy * p3d[1] * iz1 + cy
        if (px1 - u1)**2 + (py1 - v1)**2 > th2:
            continue

        # Reprojection error in image 2
        iz2 = 1.0 / (p3d_c2[2] + 1e-10)
        px2 = fx * p3d_c2[0] * iz2 + cx
        py2 = fy * p3d_c2[1] * iz2 + cy
        if (px2 - u2)**2 + (py2 - v2)**2 > th2:
            continue

        cos_parallax.append(cos_p)
        points3d[i1] = p3d
        n_good += 1
        if cos_p < 0.99998:
            good[i1] = True

    if n_good > 0:
        cos_parallax.sort()
        idx = min(50, len(cos_parallax) - 1)
        parallax = float(np.degrees(np.arccos(cos_parallax[idx])))
    else:
        parallax = 0.0

    return n_good, points3d, good, parallax


# ── Pose recovery ─────────────────────────────────────────────────────────────

def _decompose_e(E):
    """Decompose Essential matrix into (R1, R2, t)."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:  U = -U
    if np.linalg.det(Vt) < 0: Vt = -Vt
    t = U[:, 2]
    t = t / (np.linalg.norm(t) + 1e-10)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=np.float64)
    R1 = U @ W @ Vt;   R1 = R1 if np.linalg.det(R1) > 0 else -R1
    R2 = U @ W.T @ Vt; R2 = R2 if np.linalg.det(R2) > 0 else -R2
    return R1.astype(np.float32), R2.astype(np.float32), t.astype(np.float32)


def _reconstruct_f(inliers, F21, K, kps1, kps2, matches, min_parallax, min_tri):
    N = inliers.sum()
    E21 = K.T @ F21 @ K
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
    invK = np.linalg.inv(K)
    A = invK @ H21 @ K
    U, w, Vt = np.linalg.svd(A)
    V = Vt.T
    s = np.linalg.det(U) * np.linalg.det(V)
    d1, d2, d3 = w[0], w[1], w[2]

    if d1/d2 < 1.00001 or d2/d3 < 1.00001:
        return False, None, None, None

    aux1 = np.sqrt((d1**2 - d2**2) / (d1**2 - d3**2))
    aux3 = np.sqrt((d2**2 - d3**2) / (d1**2 - d3**2))
    x1s = [ aux1,  aux1, -aux1, -aux1]
    x3s = [ aux3, -aux3,  aux3, -aux3]

    Rs, ts = [], []
    aux_st = np.sqrt((d1**2-d2**2)*(d2**2-d3**2)) / ((d1+d3)*d2)
    ct = (d2**2 + d1*d3) / ((d1+d3)*d2)
    sts = [aux_st, -aux_st, -aux_st, aux_st]

    for i in range(4):
        Rp = np.array([[ct, 0, -sts[i]], [0, 1, 0], [sts[i], 0, ct]], dtype=np.float64)
        R = s * U @ Rp @ Vt
        tp = np.array([x1s[i], 0, -x3s[i]], dtype=np.float64) * (d1 - d3)
        t = U @ tp; t /= np.linalg.norm(t)
        np_ = V @ np.array([x1s[i], 0, x3s[i]])
        if np_[2] < 0: np_ = -np_
        Rs.append(R.astype(np.float32)); ts.append(t.astype(np.float32))

    aux_sp = np.sqrt((d1**2-d2**2)*(d2**2-d3**2)) / ((d1-d3)*d2)
    cp = (d1*d3 - d2**2) / ((d1-d3)*d2)
    sps = [aux_sp, -aux_sp, -aux_sp, aux_sp]

    for i in range(4):
        Rp = np.array([[cp, 0, sps[i]], [0, -1, 0], [sps[i], 0, -cp]], dtype=np.float64)
        R = s * U @ Rp @ Vt
        tp = np.array([x1s[i], 0, x3s[i]], dtype=np.float64) * (d1 + d3)
        t = U @ tp; t /= np.linalg.norm(t)
        Rs.append(R.astype(np.float32)); ts.append(t.astype(np.float32))

    best_good, second_good = 0, 0
    best_idx, best_par = -1, -1.0
    best_p3d = None
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
    # Build match list
    match_pairs = [(i, j) for i, j in enumerate(matches12) if j >= 0]
    N = len(match_pairs)
    if N < 8:
        return False, None, None, None, None

    # Random 8-point sets for RANSAC
    rng = np.random.default_rng(0)
    sets = [rng.choice(N, 8, replace=False).tolist() for _ in range(n_iter)]

    # Run H and F in parallel
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
