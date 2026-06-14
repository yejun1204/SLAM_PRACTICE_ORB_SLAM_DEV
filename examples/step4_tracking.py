"""
Step 4: Monocular initialization + TrackWithMotionModel

Pipeline:
  1. Initialization (H/F RANSAC → pose + triangulation)  — same as step3
  2. For each subsequent frame:
       extract ORB → TrackWithMotionModel (velocity + SearchByProjection + PoseOptimization)
"""

import sys, os, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np

from src.orb_matcher import Frame, search_for_initialization
from src.initializer import reconstruct
from src.camera      import K, resize_image, undistort_keypoints
from src.tracker     import MapPoint, track_with_motion_model

DATA    = os.path.join(os.path.dirname(__file__),
                       '../data/V1_01_easy/mav0/cam0/data')
FRAMES  = sorted(os.listdir(DATA))
MAX_FRAMES = 500
ORB_BIN = os.path.join(os.path.dirname(__file__), '../cpp/orb_extractor_bin')


# ── ORB extraction ────────────────────────────────────────────────────────────

def extract(path, n_features=5000):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = resize_image(img)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f: tmp_img = f.name
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f: tmp_out = f.name
    try:
        cv2.imwrite(tmp_img, img)
        subprocess.run([ORB_BIN, tmp_img, tmp_out, str(n_features)],
                       check=True, capture_output=True)
        with open(tmp_out, 'rb') as f:
            N     = np.frombuffer(f.read(4),      dtype=np.int32)[0]
            raw   = np.frombuffer(f.read(N * 24), dtype=np.float32).reshape(N, 6)
            descs = np.frombuffer(f.read(N * 32), dtype=np.uint8).reshape(N, 32)
        octaves = raw[:, 5].view(np.int32)
        kps = [cv2.KeyPoint(float(r[0]), float(r[1]), float(r[2]),
                            float(r[3] % 360), float(r[4]), int(octaves[i]))
               for i, r in enumerate(raw)]
        kps = undistort_keypoints(kps)
    finally:
        os.unlink(tmp_img); os.unlink(tmp_out)
    return img, kps, descs


def sep(title=''):
    w = 70
    if title:
        print(f"\n{'─'*w}\n  {title}\n{'─'*w}")
    else:
        print('─' * w)


# ── Initialization (same logic as step3) ─────────────────────────────────────

def run_initialization():
    sep("Phase 1 — Monocular initialization")

    ref_idx = 0
    img_ref, kps_ref, descs_ref = extract(os.path.join(DATA, FRAMES[ref_idx]),
                                           n_features=5000)
    prev_matched = [kp.pt for kp in kps_ref]
    print(f"  Initial ref frame: {FRAMES[ref_idx][:20]}")

    for cur_idx in range(1, min(len(FRAMES), MAX_FRAMES)):
        img_cur, kps_cur, descs_cur = extract(os.path.join(DATA, FRAMES[cur_idx]),
                                               n_features=5000)
        h, w = img_ref.shape
        f1 = Frame(kps_ref, descs_ref, w, h)
        f2 = Frame(kps_cur, descs_cur, w, h)
        prev = list(prev_matched)
        matches12, n_matches = search_for_initialization(
            f1, f2, prev, window_size=100, nn_ratio=0.9, check_orientation=True)

        if n_matches < 100:
            ref_idx = cur_idx
            img_ref, kps_ref, descs_ref = img_cur, kps_cur, descs_cur
            prev_matched = [kp.pt for kp in kps_ref]
            continue

        pts_ref = np.array([kp.pt for kp in kps_ref], dtype=np.float32)
        pts_cur = np.array([kp.pt for kp in kps_cur], dtype=np.float32)
        ok, R, t, points3d, triangulated = reconstruct(pts_ref, pts_cur, matches12)

        if ok:
            n_tri = int(triangulated.sum())
            print(f"  SUCCESS  ref={ref_idx} cur={cur_idx} "
                  f"matches={n_matches} tri={n_tri}")
            return dict(ref_idx=ref_idx, cur_idx=cur_idx,
                        kps_ref=kps_ref, kps_cur=kps_cur,
                        descs_ref=descs_ref, descs_cur=descs_cur,
                        matches12=matches12, R=R, t=t,
                        points3d=points3d, triangulated=triangulated)

        prev_matched = prev

    return None


# ── Build initial map from initialization result ──────────────────────────────

def build_map(init_result):
    """Create MapPoints from triangulated initialization points."""
    R  = init_result['R']
    t  = init_result['t']
    kps_cur  = init_result['kps_cur']
    descs_cur = init_result['descs_cur']
    matches12 = init_result['matches12']
    points3d  = init_result['points3d']
    tri       = init_result['triangulated']

    # T_cw for initialization frames
    T_cw_ref = np.eye(4, dtype=np.float64)
    T_cw_cur = np.eye(4, dtype=np.float64)
    T_cw_cur[:3, :3] = R
    T_cw_cur[:3, 3]  = t

    map_points = []
    for i, (p3d, is_tri) in enumerate(zip(points3d, tri)):
        if not is_tri or p3d is None:
            continue
        j = matches12[i]
        if j < 0:
            continue
        mp = MapPoint(
            pos        = p3d,
            descriptor = descs_cur[j],
            octave     = kps_cur[j].octave
        )
        map_points.append(mp)

    return T_cw_ref, T_cw_cur, map_points


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Phase 1: Initialization
    init = run_initialization()
    if init is None:
        print("  Initialization FAILED.")
        return

    T_cw_ref, T_cw_last, map_points = build_map(init)
    print(f"  Map: {len(map_points)} points")

    # Initial velocity: identity (no prior motion estimate)
    velocity = np.eye(4, dtype=np.float64)

    sep("Phase 2 — TrackWithMotionModel")
    print(f"  {'frame':>5}  {'status':<8}  {'inliers':>7}  {'map_pts':>7}")
    sep()

    start_idx = init['cur_idx'] + 1
    n_success = 0
    n_fail    = 0

    for cur_idx in range(start_idx, min(len(FRAMES), MAX_FRAMES)):
        _, kps_cur, descs_cur = extract(os.path.join(DATA, FRAMES[cur_idx]))
        h_img, w_img = 350, 600
        frame = Frame(kps_cur, descs_cur, w_img, h_img)

        T_cw, n_inliers, inlier_mps = track_with_motion_model(
            frame, map_points, T_cw_last, velocity, K)

        if T_cw is None:
            print(f"  [{cur_idx:4d}]  FAIL     inliers={n_inliers:4d}  map={len(map_points):4d}")
            n_fail += 1
            if n_fail >= 3:
                print("  Tracking LOST (3 consecutive failures).")
                break
            continue

        n_fail = 0
        n_success += 1

        # Update velocity: V = T_cur * T_last^{-1}
        velocity = T_cw @ np.linalg.inv(T_cw_last)
        T_cw_last = T_cw

        # Extract translation for display
        t_vec = T_cw[:3, 3]
        print(f"  [{cur_idx:4d}]  OK       inliers={n_inliers:4d}  map={len(map_points):4d}  "
              f"t=[{t_vec[0]:6.3f} {t_vec[1]:6.3f} {t_vec[2]:6.3f}]")

    sep()
    print(f"  Tracked {n_success} frames successfully.")


if __name__ == '__main__':
    main()
