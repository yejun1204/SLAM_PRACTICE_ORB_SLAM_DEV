"""
Step 6: TrackWithMotionModel + TrackLocalMap + LocalMapping

Pipeline:
  1. Initialization (same as step5)
  2. Per frame:
       a. TrackWithMotionModel  (last_frame_mps, th=15)
       b. TrackLocalMap         (all map_points, th=1)
       c. NeedNewKeyFrame?
            → CreateNewMapPoints (triangulate with recent KFs)
            → new points added to map_points
"""

import sys, os, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.orb_matcher   import Frame, search_for_initialization
from src.initializer   import reconstruct, global_ba, normalize_scale
from src.camera        import K, IMG_W, IMG_H, resize_image, undistort_keypoints
from src.tracker       import MapPoint, track_with_motion_model, track_local_map
from src.local_mapping import KeyFrame, need_new_keyframe, create_new_map_points

DATA       = os.path.join(os.path.dirname(__file__), '../data/V1_01_easy/mav0/cam0/data')
FRAMES     = sorted(os.listdir(DATA))
MAX_FRAMES = 500
ORB_BIN    = os.path.join(os.path.dirname(__file__), '../cpp/orb_extractor_bin')


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


# ── Initialization ────────────────────────────────────────────────────────────

def run_initialization():
    sep("Phase 1 — Monocular initialization")
    ref_idx = 0
    img_ref, kps_ref, descs_ref = extract(os.path.join(DATA, FRAMES[ref_idx]))
    prev_matched = [kp.pt for kp in kps_ref]

    for cur_idx in range(1, min(len(FRAMES), MAX_FRAMES)):
        img_cur, kps_cur, descs_cur = extract(os.path.join(DATA, FRAMES[cur_idx]))
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
            print(f"  SUCCESS  ref={ref_idx} cur={cur_idx} "
                  f"matches={n_matches} tri={int(triangulated.sum())}")
            return dict(ref_idx=ref_idx, cur_idx=cur_idx,
                        kps_ref=kps_ref, kps_cur=kps_cur,
                        descs_ref=descs_ref, descs_cur=descs_cur,
                        matches12=matches12, R=R, t=t,
                        points3d=points3d, triangulated=triangulated)
        prev_matched = prev
    return None


def build_map(init_result):
    R         = init_result['R']
    t         = init_result['t']
    kps_ref   = init_result['kps_ref']
    kps_cur   = init_result['kps_cur']
    descs_ref = init_result['descs_ref']
    descs_cur = init_result['descs_cur']
    matches12 = init_result['matches12']
    points3d  = init_result['points3d']
    tri       = init_result['triangulated']

    R, t, points3d = global_ba(R, t, points3d, tri, matches12, kps_ref, kps_cur)
    t, points3d    = normalize_scale(t, points3d, tri)

    T_cw_ref = np.eye(4, dtype=np.float64)
    T_cw_cur = np.eye(4, dtype=np.float64)
    T_cw_cur[:3, :3] = R
    T_cw_cur[:3, 3]  = t

    map_points = []
    ref_mp_matches = {}   # kp_idx_in_ref → MapPoint
    cur_mp_matches = {}   # kp_idx_in_cur → MapPoint

    for i, (p3d, is_tri) in enumerate(zip(points3d, tri)):
        if not is_tri or p3d is None:
            continue
        j = matches12[i]
        if j < 0:
            continue
        mp = MapPoint(pos=p3d, descriptor=descs_cur[j], octave=kps_cur[j].octave)
        map_points.append(mp)
        ref_mp_matches[i] = mp
        cur_mp_matches[j] = mp

    kf_ref = KeyFrame(init_result['ref_idx'], T_cw_ref, kps_ref, descs_ref, ref_mp_matches)
    kf_cur = KeyFrame(init_result['cur_idx'], T_cw_cur, kps_cur, descs_cur, cur_mp_matches)

    return T_cw_cur, map_points, [kf_ref, kf_cur]


# ── Visualization ─────────────────────────────────────────────────────────────

def draw_reprojection(img, map_points, T_cw, inlier_mps):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])
    R = T_cw[:3,:3]; t = T_cw[:3,3]
    h, w = img.shape
    inlier_ids = {id(mp) for mp in inlier_mps.values()}
    for mp in map_points:
        x3dc = R @ mp.pos.astype(np.float64) + t
        if x3dc[2] <= 0: continue
        iz = 1.0 / x3dc[2]
        u = int(fx * x3dc[0] * iz + cx)
        v = int(fy * x3dc[1] * iz + cy)
        if 0 <= u < w and 0 <= v < h:
            color = (0, 255, 0) if id(mp) in inlier_ids else (0, 0, 200)
            cv2.circle(vis, (u, v), 3, color, -1)
    return vis


def setup_viz():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    ax2d = fig.add_subplot(121)
    ax2d.set_title('Trajectory (x-z)'); ax2d.set_xlabel('x'); ax2d.set_ylabel('z')
    ax2d.grid(True)
    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.set_title('Camera poses (3D)')
    ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
    plt.tight_layout()
    return fig, ax2d, ax3d


def update_viz(ax2d, ax3d, trajectory, poses):
    ax2d.cla()
    ax2d.set_title('Trajectory (x-z)'); ax2d.set_xlabel('x'); ax2d.set_ylabel('z')
    ax2d.grid(True)
    xs = [p[0] for p in trajectory]; zs = [p[2] for p in trajectory]
    ax2d.plot(xs, zs, 'b-', linewidth=1)
    ax2d.scatter(xs[0], zs[0], c='green', s=50, zorder=5)
    ax2d.scatter(xs[-1], zs[-1], c='red', s=50, zorder=5)
    ax2d.set_aspect('equal')

    ax3d.cla()
    ax3d.set_title('Camera poses (3D)')
    ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
    arr = np.array(trajectory)
    extent   = float(np.max(arr.max(axis=0) - arr.min(axis=0))) if len(arr) > 1 else 1.0
    axis_len = max(extent * 0.06, 1e-3)
    for C, T_cw in zip(trajectory, poses):
        R_wc = T_cw[:3,:3].T
        for i, color in enumerate(['r','g','b']):
            d = R_wc[:, i] * axis_len
            ax3d.quiver(C[0],C[1],C[2], d[0],d[1],d[2], color=color, linewidth=0.8)
    ax3d.plot(arr[:,0], arr[:,1], arr[:,2], 'b-', linewidth=0.6, alpha=0.5)
    ax3d.scatter(*trajectory[-1], c='red', s=30, zorder=5)

    mid  = (arr.max(axis=0) + arr.min(axis=0)) / 2.0
    half = max(float(np.max(arr.max(axis=0) - arr.min(axis=0))) / 2.0, 1e-3)
    ax3d.set_xlim(mid[0]-half, mid[0]+half)
    ax3d.set_ylim(mid[1]-half, mid[1]+half)
    ax3d.set_zlim(mid[2]-half, mid[2]+half)

    plt.pause(0.001)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    init = run_initialization()
    if init is None:
        print("  Initialization FAILED."); return

    T_cw_last, map_points, keyframes = build_map(init)
    print(f"  Map: {len(map_points)} points  KFs: {len(keyframes)}")

    velocity        = np.eye(4, dtype=np.float64)
    last_frame_mps  = list(keyframes[-1].map_point_matches.values())
    ref_kf          = keyframes[-1]
    last_kf_frame   = init['cur_idx']
    ref_n_matches   = 300

    sep("Phase 2 — TrackWithMotionModel + TrackLocalMap + LocalMapping")
    print(f"  {'frame':>5}  {'MM':>5}  {'LM':>5}  {'KF':>3}  {'new':>5}  {'map':>6}  C")
    sep()

    start_idx = init['cur_idx'] + 1
    n_success = n_fail = 0
    trajectory = []; poses = []

    _, ax2d, ax3d = setup_viz()
    cv2.namedWindow('Reprojection', cv2.WINDOW_NORMAL)
    print("  [any key] next frame   [ESC] quit")

    for cur_idx in range(start_idx, min(len(FRAMES), MAX_FRAMES)):
        img_cur, kps_cur, descs_cur = extract(os.path.join(DATA, FRAMES[cur_idx]))
        frame = Frame(kps_cur, descs_cur, IMG_W, IMG_H)

        # Stage 1: TrackWithMotionModel — 직전 프레임 LM 인라이어만 투영(th=15).
        # 한번 아웃라이어로 탈락한 포인트는 last_frame_mps에서 빠지므로 다시
        # 후보가 되지 않는다. step4가 전체 map_points를 쓰는 것과 다른 점이다.
        T_cw, n_mm, mm_matches = track_with_motion_model(
            frame, last_frame_mps, T_cw_last, velocity, K)

        if T_cw is None:
            print(f"  [{cur_idx:4d}]  MM FAIL  n={n_mm}")
            n_fail += 1
            if n_fail >= 3:
                print("  Tracking LOST."); break
            continue

        # Stage 2: TrackLocalMap
        T_cw, n_lm, lm_matches = track_local_map(
            frame, map_points, T_cw, mm_matches, K, th=1)

        if n_lm < 30:
            print(f"  [{cur_idx:4d}]  LM FAIL  MM={n_mm:3d}  LM={n_lm:3d}")
            n_fail += 1
            if n_fail >= 3:
                print("  Tracking LOST."); break
            continue

        n_fail = 0; n_success += 1
        velocity      = T_cw @ np.linalg.inv(T_cw_last)
        T_cw_last     = T_cw
        # LM 인라이어 전체를 다음 프레임 MM 후보로 넘긴다.
        # map_points가 늘어날수록 LM 인라이어도 늘어나므로 MM 후보도 함께 증가한다.
        last_frame_mps = list(lm_matches.values())

        # Stage 3: NeedNewKeyFrame → LocalMapping
        n_new = 0
        is_kf = need_new_keyframe(cur_idx, last_kf_frame, n_lm, ref_n_matches)
        if is_kf:
            cur_kf = KeyFrame(cur_idx, T_cw, kps_cur, descs_cur, lm_matches)
            keyframes.append(cur_kf)
            n_new = create_new_map_points(cur_kf, keyframes, K, map_points)
            ref_kf        = cur_kf
            last_kf_frame = cur_idx

        R_cw = T_cw[:3,:3]; t_cw = T_cw[:3,3]
        C = -R_cw.T @ t_cw
        trajectory.append(C); poses.append(T_cw.copy())

        kf_mark = '*' if is_kf else ' '
        print(f"  [{cur_idx:4d}]{kf_mark}  MM={n_mm:3d}  LM={n_lm:3d}  "
              f"+{n_new:4d}  map={len(map_points):5d}  "
              f"C=[{C[0]:6.3f} {C[1]:6.3f} {C[2]:6.3f}]")

        vis = draw_reprojection(img_cur, map_points, T_cw, lm_matches)
        cv2.putText(vis,
                    f"frame {cur_idx}{kf_mark}  MM={n_mm} LM={n_lm}  "
                    f"map={len(map_points)}  KFs={len(keyframes)} | ESC: quit",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv2.imshow('Reprojection', vis)
        update_viz(ax2d, ax3d, trajectory, poses)

        if cv2.waitKey(0) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    plt.ioff()
    sep()
    print(f"  Tracked {n_success} frames.  KFs: {len(keyframes)}  Map: {len(map_points)} pts.")
    plt.show()


if __name__ == '__main__':
    main()
