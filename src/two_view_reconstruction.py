"""
TwoViewReconstruction - ORB-SLAM3 style monocular initialization

Implements:
1. FindHomography / FindFundamental (RANSAC + symmetric score)
2. ReconstructH / ReconstructF (4-solution disambiguation via CheckRT)
3. CheckRT (triangulation + depth + parallax + reprojection checks)
4. nsimilar check (reject ambiguous reconstructions)
"""

import cv2
import numpy as np


class TwoViewReconstruction:
    def __init__(self, K, sigma=1.0, max_iterations=200):
        self.K  = K.astype(np.float64)
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.sigma  = sigma
        self.sigma2 = sigma * sigma
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #
    def reconstruct(self, kps1, kps2, matches12):
        """
        ORB-SLAM3: TwoViewReconstruction::Reconstruct

        Args:
            kps1, kps2 : list of cv2.KeyPoint (undistorted)
            matches12  : list of length len(kps1), value = kps2 index or -1

        Returns:
            success    : bool
            T21        : (4,4) np.float64 — T_cw for frame 2 (frame 1 = identity)
            points_3d  : list of (x,y,z) or None per kps1 index
            triangulated: list of bool per kps1 index
        """
        # Build match pair list
        match_pairs = [(i, j) for i, j in enumerate(matches12) if j >= 0]
        N = len(match_pairs)
        if N < 8:
            return False, None, None, None

        # Compute H and F (parallel in C++, sequential here)
        H, inliers_H, score_H = self._find_homography(kps1, kps2, match_pairs)
        F, inliers_F, score_F = self._find_fundamental(kps1, kps2, match_pairs)

        if score_H + score_F == 0:
            return False, None, None, None

        RH = score_H / (score_H + score_F)
        min_parallax     = 1.0
        min_triangulated = 50  # ORB-SLAM3 hardcodes 50; 0.9*N_inliers computed inside

        if RH > 0.50:
            return self._reconstruct_H(kps1, kps2, match_pairs, inliers_H, H,
                                       min_parallax, min_triangulated)
        else:
            return self._reconstruct_F(kps1, kps2, match_pairs, inliers_F, F,
                                       min_parallax, min_triangulated)

    # ------------------------------------------------------------------ #
    #  FindHomography                                                      #
    # ------------------------------------------------------------------ #
    def _find_homography(self, kps1, kps2, match_pairs):
        pts1 = np.float32([kps1[i].pt for i, _ in match_pairs])
        pts2 = np.float32([kps2[j].pt for _, j in match_pairs])

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,
                                     self.sigma, confidence=0.999,
                                     maxIters=self.max_iterations)
        if H is None:
            return None, [False] * len(match_pairs), 0.0

        score, inliers = self._score_homography(H, kps1, kps2, match_pairs)
        return H, inliers, score

    def _score_homography(self, H, kps1, kps2, match_pairs):
        th     = 5.991 * self.sigma2   # chi2, 2 DOF
        H_inv  = np.linalg.inv(H)
        score  = 0.0
        inliers = []

        for i, j in match_pairs:
            p1 = np.array([kps1[i].pt[0], kps1[i].pt[1], 1.0])
            p2 = np.array([kps2[j].pt[0], kps2[j].pt[1], 1.0])

            # Forward: p1 → p2
            p2_proj = H @ p1
            p2_proj /= p2_proj[2]
            e_fwd = np.sum((p2_proj[:2] - p2[:2]) ** 2)

            # Backward: p2 → p1
            p1_proj = H_inv @ p2
            p1_proj /= p1_proj[2]
            e_bwd = np.sum((p1_proj[:2] - p1[:2]) ** 2)

            chi2_fwd = e_fwd / self.sigma2
            chi2_bwd = e_bwd / self.sigma2

            if chi2_fwd < th and chi2_bwd < th:
                inliers.append(True)
                score += (th - chi2_fwd) + (th - chi2_bwd)
            else:
                inliers.append(False)

        return score, inliers

    # ------------------------------------------------------------------ #
    #  FindFundamental                                                     #
    # ------------------------------------------------------------------ #
    def _find_fundamental(self, kps1, kps2, match_pairs):
        pts1 = np.float32([kps1[i].pt for i, _ in match_pairs])
        pts2 = np.float32([kps2[j].pt for _, j in match_pairs])

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                         self.sigma, 0.999)
        if F is None:
            return None, [False] * len(match_pairs), 0.0

        score, inliers = self._score_fundamental(F, kps1, kps2, match_pairs)
        return F, inliers, score

    def _score_fundamental(self, F, kps1, kps2, match_pairs):
        th_score = 5.991 * self.sigma2
        th_check = 3.841 * self.sigma2
        score   = 0.0
        inliers = []

        for i, j in match_pairs:
            p1 = np.array([kps1[i].pt[0], kps1[i].pt[1], 1.0])
            p2 = np.array([kps2[j].pt[0], kps2[j].pt[1], 1.0])

            l2 = F @ p1          # epipolar line in image 2
            l1 = F.T @ p2        # epipolar line in image 1

            d2 = (p2 @ l2) ** 2 / (l2[0]**2 + l2[1]**2)
            d1 = (p1 @ l1) ** 2 / (l1[0]**2 + l1[1]**2)

            chi2_1 = d1 / self.sigma2
            chi2_2 = d2 / self.sigma2

            if chi2_1 < th_check and chi2_2 < th_check:
                inliers.append(True)
                score += (th_score - chi2_1) + (th_score - chi2_2)
            else:
                inliers.append(False)

        return score, inliers

    # ------------------------------------------------------------------ #
    #  ReconstructF                                                        #
    # ------------------------------------------------------------------ #
    def _reconstruct_F(self, kps1, kps2, match_pairs, inliers, F,
                       min_parallax, min_triangulated):
        # Use findEssentialMat directly for more accurate decomposition
        pts1 = np.float32([kps1[i].pt for i, _ in match_pairs])
        pts2 = np.float32([kps2[j].pt for _, j in match_pairs])

        E, mask_E = cv2.findEssentialMat(pts1, pts2, self.K,
                                          method=cv2.RANSAC,
                                          prob=0.999,
                                          threshold=self.sigma)
        if E is None:
            return False, None, None, None

        # findEssentialMat can return (9,3) for multiple solutions — use first
        if E.shape[0] > 3:
            E = E[:3, :]

        # Recompute inliers from E
        inliers_E = [bool(mask_E[k]) if mask_E is not None else True
                     for k in range(len(match_pairs))]

        R1, R2, t = self._decompose_E(E)
        t1, t2 = t, -t

        n_good  = [0] * 4
        parallax = [0.0] * 4
        points   = [None] * 4
        triang   = [None] * 4

        for k, (R, tv) in enumerate([(R1,t1),(R2,t1),(R1,t2),(R2,t2)]):
            ng, par, pts, tri = self._check_RT(R, tv, kps1, kps2,
                                               match_pairs, inliers_E)
            n_good[k]  = ng
            parallax[k] = par
            points[k]  = pts
            triang[k]  = tri

        N_inliers = sum(inliers_E)
        return self._select_solution(n_good, parallax, points, triang,
                                     [(R1,t1),(R2,t1),(R1,t2),(R2,t2)],
                                     min_parallax, min_triangulated,
                                     N_inliers)

    # ------------------------------------------------------------------ #
    #  ReconstructH                                                        #
    # ------------------------------------------------------------------ #
    def _reconstruct_H(self, kps1, kps2, match_pairs, inliers, H,
                       min_parallax, min_triangulated):
        pts1 = np.float32([kps1[i].pt for i, _ in match_pairs])
        pts2 = np.float32([kps2[j].pt for _, j in match_pairs])

        num, Rs, ts, _ = cv2.decomposeHomographyMat(H, self.K)

        n_good = []
        parallax_list = []
        points_list   = []
        triang_list   = []
        hypotheses    = []

        for k in range(num):
            R  = Rs[k]
            tv = ts[k].ravel()
            ng, par, pts, tri = self._check_RT(R, tv, kps1, kps2,
                                               match_pairs, inliers)
            n_good.append(ng)
            parallax_list.append(par)
            points_list.append(pts)
            triang_list.append(tri)
            hypotheses.append((R, tv))

        N_inliers = sum(inliers)
        return self._select_solution(n_good, parallax_list, points_list,
                                     triang_list, hypotheses,
                                     min_parallax, min_triangulated,
                                     N_inliers)

    # ------------------------------------------------------------------ #
    #  CheckRT                                                             #
    # ------------------------------------------------------------------ #
    def _check_RT(self, R, t, kps1, kps2, match_pairs, inliers):
        """
        ORB-SLAM3: TwoViewReconstruction::CheckRT
        Triangulate inlier matches, check depth, parallax, reprojection.
        Returns (n_good, parallax_deg, points_3d list, triangulated mask)
        """
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        th2 = 4.0 * self.sigma2

        P1 = np.zeros((3, 4))
        P1[:3, :3] = self.K
        O1 = np.zeros(3)

        P2 = self.K @ np.hstack([R, t.reshape(3, 1)])
        O2 = -R.T @ t

        n1 = len(kps1)
        points_3d  = [None] * n1
        triangulated = [False] * n1
        cos_parallax_list = []
        n_good = 0

        for idx, (i, j) in enumerate(match_pairs):
            if not inliers[idx]:
                continue

            x1 = np.array([kps1[i].pt[0], kps1[i].pt[1], 1.0])
            x2 = np.array([kps2[j].pt[0], kps2[j].pt[1], 1.0])

            p3d = self._triangulate_point(x1, x2, P1, P2)

            if not all(np.isfinite(p3d)):
                continue

            # Parallax
            n1v = p3d - O1;  d1 = np.linalg.norm(n1v)
            n2v = p3d - O2;  d2 = np.linalg.norm(n2v)
            if d1 < 1e-10 or d2 < 1e-10:
                continue
            cos_par = float(n1v @ n2v) / (d1 * d2)

            # Depth check in cam1
            if p3d[2] <= 0 and cos_par < 0.99998:
                continue

            # Depth check in cam2
            p3d_c2 = R @ p3d + t
            if p3d_c2[2] <= 0 and cos_par < 0.99998:
                continue

            # Reprojection error in image 1
            inv_z1 = 1.0 / p3d[2]
            u1 = fx * p3d[0] * inv_z1 + cx
            v1 = fy * p3d[1] * inv_z1 + cy
            e1 = (u1 - kps1[i].pt[0])**2 + (v1 - kps1[i].pt[1])**2
            if e1 > th2:
                continue

            # Reprojection error in image 2
            inv_z2 = 1.0 / p3d_c2[2]
            u2 = fx * p3d_c2[0] * inv_z2 + cx
            v2 = fy * p3d_c2[1] * inv_z2 + cy
            e2 = (u2 - kps2[j].pt[0])**2 + (v2 - kps2[j].pt[1])**2
            if e2 > th2:
                continue

            cos_parallax_list.append(cos_par)
            points_3d[i]    = tuple(p3d)
            n_good += 1
            if cos_par < 0.99998:
                triangulated[i] = True

        # Parallax at 50th smallest (ORB-SLAM3 style)
        if n_good > 0:
            cos_parallax_list.sort()
            idx50 = min(50, len(cos_parallax_list) - 1)
            parallax_deg = float(np.degrees(np.arccos(cos_parallax_list[idx50])))
        else:
            parallax_deg = 0.0

        return n_good, parallax_deg, points_3d, triangulated

    # ------------------------------------------------------------------ #
    #  Select best solution (nsimilar check)                              #
    # ------------------------------------------------------------------ #
    def _select_solution(self, n_good, parallax, points, triang,
                         hypotheses, min_parallax, min_triangulated, N):
        max_good = max(n_good)
        n_min    = max(int(0.9 * N), min_triangulated)

        if max_good < n_min:
            return False, None, None, None

        # Count solutions with n_good > 0.7 * max_good
        n_similar = sum(1 for ng in n_good if ng > 0.7 * max_good)
        if n_similar > 1:
            return False, None, None, None  # Ambiguous

        # Pick best
        best = -1
        for k, (ng, par) in enumerate(zip(n_good, parallax)):
            if ng == max_good and par > min_parallax:
                best = k
                break

        if best < 0:
            return False, None, None, None

        R, t = hypotheses[best]
        T21 = np.eye(4)
        T21[:3, :3] = R
        T21[:3,  3] = t

        return True, T21, points[best], triang[best]

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _triangulate_point(self, x1, x2, P1, P2):
        """Linear triangulation (DLT)."""
        A = np.array([
            x1[0] * P1[2] - P1[0],
            x1[1] * P1[2] - P1[1],
            x2[0] * P2[2] - P2[0],
            x2[1] * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return X[:3] / X[3]

    def _decompose_E(self, E):
        """ORB-SLAM3: DecomposeE — SVD decomposition of Essential matrix."""
        U, _, Vt = np.linalg.svd(E)
        if np.linalg.det(U) < 0:
            U = -U
        if np.linalg.det(Vt) < 0:
            Vt = -Vt

        W = np.array([[0, -1, 0],
                      [1,  0, 0],
                      [0,  0, 1]], dtype=np.float64)

        R1 = U @ W   @ Vt
        R2 = U @ W.T @ Vt
        t  = U[:, 2]
        return R1, R2, t / np.linalg.norm(t)

    def _normalize(self, pts):
        """Normalize points for numerical stability (zero mean, mean dist = sqrt(2))."""
        mean = pts.mean(axis=0)
        centered = pts - mean
        mean_dist = np.sqrt((centered ** 2).sum(axis=1)).mean()
        if mean_dist < 1e-10:
            return pts, np.eye(3)
        scale = np.sqrt(2) / mean_dist
        T = np.array([[scale, 0,    -scale * mean[0]],
                      [0,    scale, -scale * mean[1]],
                      [0,    0,      1              ]])
        pts_n = (T @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :2]
        return pts_n.astype(np.float32), T
