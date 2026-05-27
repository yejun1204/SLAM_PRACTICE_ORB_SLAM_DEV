"""
Step 3: Two-view initialization — H/F RANSAC → pose + triangulation

Simulates ORB-SLAM3 MonocularInitialization:
  - Keep a reference frame fixed while nmatches >= 100
  - Reset reference frame when matches drop below 100
  - Try H/F reconstruction until parallax is sufficient
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from src.orb_extractor import ORBExtractor
from src.orb_matcher  import Frame, search_for_initialization
from src.initializer  import reconstruct, K

DATA = os.path.join(os.path.dirname(__file__),
                    '../data/V1_01_easy/mav0/cam0/data')
FRAMES = sorted(os.listdir(DATA))
MAX_FRAMES = 500


def extract(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ext = ORBExtractor(n_features=1000, scale_factor=1.2, n_levels=8,
                       ini_th_fast=20, min_th_fast=7)
    kps, descs = ext.detect_and_compute(img)
    return img, kps, descs


def sep(title=""):
    w = 70
    if title:
        print(f"\n{'─'*w}\n  {title}\n{'─'*w}")
    else:
        print('─' * w)


def main():
    sep("Monocular initialization  (sliding reference frame)")

    # Set initial reference frame
    ref_idx = 0
    img_ref, kps_ref, descs_ref = extract(os.path.join(DATA, FRAMES[ref_idx]))
    prev_matched = [kp.pt for kp in kps_ref]
    print(f"  Initial ref frame: {FRAMES[ref_idx][:20]}")

    result = None

    for cur_idx in range(1, min(len(FRAMES), MAX_FRAMES)):
        img_cur, kps_cur, descs_cur = extract(os.path.join(DATA, FRAMES[cur_idx]))
        h, w = img_ref.shape
        f1 = Frame(kps_ref, descs_ref, w, h)
        f2 = Frame(kps_cur, descs_cur, w, h)
        prev = list(prev_matched)
        matches12, n_matches = search_for_initialization(
            f1, f2, prev, window_size=100, nn_ratio=0.9, check_orientation=True)

        if n_matches < 100:
            # Reset reference frame (ORB-SLAM3 behaviour)
            ref_idx = cur_idx
            img_ref, kps_ref, descs_ref = img_cur, kps_cur, descs_cur
            prev_matched = [kp.pt for kp in kps_ref]
            print(f"  [{cur_idx:4d}] ref reset  (matches={n_matches})")
            continue

        pts_ref = np.array([kp.pt for kp in kps_ref], dtype=np.float32)
        pts_cur = np.array([kp.pt for kp in kps_cur], dtype=np.float32)

        ok, R, t, points3d, triangulated = reconstruct(
            pts_ref, pts_cur, matches12, K=K)

        if ok:
            n_tri = int(triangulated.sum())
            print(f"  [{cur_idx:4d}] SUCCESS  ref={ref_idx} cur={cur_idx} "
                  f"matches={n_matches} tri={n_tri}")
            result = dict(ref_idx=ref_idx, cur_idx=cur_idx,
                          img_ref=img_ref, img_cur=img_cur,
                          kps_ref=kps_ref, kps_cur=kps_cur,
                          matches12=matches12, R=R, t=t,
                          points3d=points3d, triangulated=triangulated,
                          n_matches=n_matches, n_tri=n_tri)
            break

        prev_matched = prev

    if result is None:
        print("  Initialization FAILED within frame limit.")
        return

    # ── Results ──────────────────────────────────────────────────────────────
    sep("Recovered pose  T21: ref → cur")
    R, t = result['R'], result['t']
    print("  R =")
    for row in R:
        print(f"    [{row[0]:9.6f}  {row[1]:9.6f}  {row[2]:9.6f}]")
    print(f"  t = [{t[0]:9.6f}  {t[1]:9.6f}  {t[2]:9.6f}]  (unit vector)")

    sep("Triangulated map points")
    p3d_valid = [p for p in result['points3d'] if p is not None]
    depths = [float(p[2]) for p in p3d_valid]
    print(f"  Count  : {result['n_tri']}")
    print(f"  Depth  mean={np.mean(depths):.3f}  "
          f"min={np.min(depths):.3f}  max={np.max(depths):.3f}")

    # ── Visualization ─────────────────────────────────────────────────────────
    sep("Saving visualization")
    kps_ref, kps_cur = result['kps_ref'], result['kps_cur']
    matches12 = result['matches12']
    tri = result['triangulated']

    kps1_m, kps2_m, cv_m = [], [], []
    for i, j in enumerate(matches12):
        if j >= 0 and tri[i]:
            idx = len(kps1_m)
            kps1_m.append(cv2.KeyPoint(kps_ref[i].pt[0], kps_ref[i].pt[1], 5))
            kps2_m.append(cv2.KeyPoint(kps_cur[j].pt[0],  kps_cur[j].pt[1],  5))
            cv_m.append(cv2.DMatch(idx, idx, 0))

    vis = cv2.drawMatches(
        cv2.cvtColor(result['img_ref'], cv2.COLOR_GRAY2BGR), kps1_m,
        cv2.cvtColor(result['img_cur'], cv2.COLOR_GRAY2BGR), kps2_m,
        cv_m, None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.putText(vis,
                f"ref={result['ref_idx']}  cur={result['cur_idx']}  "
                f"tri={result['n_tri']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    out = os.path.join(os.path.dirname(__file__), 'init_triangulated.png')
    cv2.imwrite(out, vis)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
