"""
Step 3: Monocular Initialization (ORB-SLAM3 style)

Full pipeline:
1. ORB extraction (Step 1)
2. SearchForInitialization (Step 2) — frame-by-frame prev_matched update
3. TwoViewReconstruction (H/F + CheckRT + nsimilar)
4. Global BA (gtsam)
5. Scale normalization (median depth = 1.0)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import gtsam
from src.frame import Frame
from src.feature_matcher import search_for_initialization
from src.two_view_reconstruction import TwoViewReconstruction

DATA_DIR   = '/home/yejun/V1_01_easy/mav0/cam0/data'
TIMESTAMPS = '/home/yejun/ORB_SLAM3/Examples/Monocular/EuRoC_TimeStamps/V101.txt'

K    = np.array([[458.654,   0.,    367.215],
                 [  0.,    457.296, 248.375],
                 [  0.,      0.,      1.   ]])
DIST = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])

MIN_PARALLAX    = 1.0   # degrees
MIN_FEATURES    = 100   # minimum features to attempt init


def global_ba(K, T_ref, T_cur, points_3d, kps_ref, kps_cur, matches12):
    """
    Global BA after initialization — ORB-SLAM3: Optimizer::GlobalBundleAdjustemnt
    Fixes frame 1 (identity), optimizes frame 2 pose + all 3D points.
    """
    cal = gtsam.Cal3_S2(float(K[0,0]), float(K[1,1]), 0.,
                        float(K[0,2]), float(K[1,2]))
    huber  = gtsam.noiseModel.mEstimator.Huber.Create(2.0)
    noise  = gtsam.noiseModel.Robust.Create(
                 huber, gtsam.noiseModel.Isotropic.Sigma(2, 1.0))
    p_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-8)

    graph   = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    def to_gtsam_pose(T_cw):
        R = T_cw[:3,:3]; t = T_cw[:3,3]
        return gtsam.Pose3(gtsam.Rot3(R.T), gtsam.Point3(-R.T @ t))

    pose_key1 = gtsam.symbol('x', 0)
    pose_key2 = gtsam.symbol('x', 1)
    initial.insert(pose_key1, to_gtsam_pose(T_ref))
    initial.insert(pose_key2, to_gtsam_pose(T_cur))

    # Fix frame 1 (reference = identity)
    graph.add(gtsam.PriorFactorPose3(pose_key1,
                                      initial.atPose3(pose_key1),
                                      prior_noise))

    n_obs = 0
    for mp_idx, (i1, i2) in enumerate([(i, j) for i, j in enumerate(matches12) if j >= 0]):
        pt3d = points_3d[i1]
        if pt3d is None:
            continue
        lkey = gtsam.symbol('l', mp_idx)
        initial.insert(lkey, gtsam.Point3(*pt3d))

        graph.add(gtsam.GenericProjectionFactorCal3_S2(
            gtsam.Point2(*kps_ref[i1].pt), noise, pose_key1, lkey, cal))
        graph.add(gtsam.GenericProjectionFactorCal3_S2(
            gtsam.Point2(*kps_cur[i2].pt), noise, pose_key2, lkey, cal))
        n_obs += 2

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(20)
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()

    # Extract optimized pose2
    pose2  = result.atPose3(pose_key2)
    R_wc   = pose2.rotation().matrix()
    t_wc   = pose2.translation()
    T_opt  = np.eye(4)
    T_opt[:3,:3] = R_wc.T
    T_opt[:3, 3] = -R_wc.T @ t_wc

    # Extract optimized 3D points
    pts_opt = list(points_3d)
    for mp_idx, (i1, i2) in enumerate([(i, j) for i, j in enumerate(matches12) if j >= 0]):
        if points_3d[i1] is None:
            continue
        lkey = gtsam.symbol('l', mp_idx)
        try:
            pts_opt[i1] = tuple(result.atPoint3(lkey))
        except Exception:
            pass

    return T_opt, pts_opt


def scale_normalize(T_cur, points_3d):
    """
    Set median depth to 1.0 — ORB-SLAM3: CreateInitialMapMonocular scale normalization
    """
    depths = [pt[2] for pt in points_3d if pt is not None and pt[2] > 0]
    if not depths:
        return T_cur, points_3d
    median_depth = float(np.median(depths))
    if median_depth <= 0:
        return T_cur, points_3d

    inv_d = 1.0 / median_depth

    T_new = T_cur.copy()
    T_new[:3, 3] *= inv_d

    pts_new = [((p[0]*inv_d, p[1]*inv_d, p[2]*inv_d) if p is not None else None)
               for p in points_3d]
    return T_new, pts_new


def main():
    with open(TIMESTAMPS) as f:
        lines = [l.strip() for l in f if l.strip()]

    extractor   = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
    reconstruct = TwoViewReconstruction(K, sigma=1.0)

    # --- Reference frame (frame 0) ---
    img0 = cv2.imread(os.path.join(DATA_DIR, lines[0]+'.png'), cv2.IMREAD_GRAYSCALE)
    kps0, desc0 = extractor.detectAndCompute(img0, None)
    frame_ref   = Frame(img0, kps0, desc0, K, DIST, frame_id=0)
    prev_matched = [kp.pt for kp in frame_ref.keypoints]

    print(f"Reference frame 0: {len(frame_ref.keypoints)} keypoints")
    print("\nAttempting initialization...")

    T_ref = np.eye(4)   # frame 1 is the world origin

    for frame_idx in range(1, len(lines)):
        img = cv2.imread(os.path.join(DATA_DIR, lines[frame_idx]+'.png'),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        kps, desc = extractor.detectAndCompute(img, None)
        if len(kps) < MIN_FEATURES:
            continue

        frame_cur = Frame(img, kps, desc, K, DIST, frame_id=frame_idx)

        # SearchForInitialization
        matches12, n_matches, prev_matched = search_for_initialization(
            frame_ref, frame_cur, prev_matched, window_size=100
        )

        if n_matches < 100:
            continue

        # TwoViewReconstruction
        success, T21, points_3d, triangulated = reconstruct.reconstruct(
            frame_ref.keypoints, frame_cur.keypoints, matches12
        )

        n_tri = sum(1 for t in (triangulated or []) if t)
        print(f"  Frame {frame_idx:4d}: {n_matches} matches, "
              f"{'OK' if success else 'FAIL'}"
              f"{f' | {n_tri} triangulated' if success else ''}")

        if not success:
            continue

        # Global BA
        T21_ba, points_ba = global_ba(K, T_ref, T21, points_3d,
                                       frame_ref.keypoints, frame_cur.keypoints,
                                       matches12)

        # Scale normalization
        T21_final, points_final = scale_normalize(T21_ba, points_ba)

        n_final = sum(1 for p in points_final if p is not None)

        print(f"\n{'='*50}")
        print(f"INITIALIZATION SUCCESSFUL at frame {frame_idx}")
        print(f"  Triangulated points: {n_final}")
        print(f"  T21 translation: {T21_final[:3, 3]}")

        depths = [p[2] for p in points_final if p is not None]
        print(f"  Median depth (after normalization): {np.median(depths):.3f}")
        break


if __name__ == '__main__':
    main()
