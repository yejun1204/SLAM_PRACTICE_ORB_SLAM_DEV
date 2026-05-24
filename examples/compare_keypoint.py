"""
Per-octave keypoint comparison: C++ ORB-SLAM3 binary vs Python ORBExtractor.
Checks position (x, y), response, and angle agreement for each scale level.
"""

import sys, os, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from src.orb_extractor import ORBExtractor

ROOT     = os.path.join(os.path.dirname(__file__), '..')
BINARY   = os.path.join(ROOT, 'cpp', 'orb_extractor_bin')
IMAGE    = os.path.join(ROOT, 'data/V1_01_easy/mav0/cam0/data/1403715273262142976.png')
N_LEVELS = 8
POS_TOL  = 0.1   # px — two keypoints are "same position" if rounded to this


def run_cpp(image_path):
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        out_path = f.name
    try:
        subprocess.run([BINARY, image_path, out_path], check=True, capture_output=True)
        with open(out_path, 'rb') as f:
            N   = np.frombuffer(f.read(4), dtype=np.int32)[0]
            raw = np.frombuffer(f.read(N * 24), dtype=np.float32).reshape(N, 6)
            octaves = raw[:, 5].view(np.int32)
            kps = np.column_stack([raw[:, :5], octaves.astype(np.float32)])
        return kps  # (N,6): x, y, size, angle, response, octave
    finally:
        os.unlink(out_path)


def run_python(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ext = ORBExtractor(n_features=1000, scale_factor=1.2, n_levels=8,
                       ini_th_fast=20, min_th_fast=7)
    keypoints, _ = ext.detect_and_compute(img)
    return np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave]
                     for kp in keypoints], dtype=np.float32)


def match_by_position(kps_a, kps_b, tol=POS_TOL):
    pos_to_b = {}
    for i, kp in enumerate(kps_b):
        key = (round(kp[0] / tol), round(kp[1] / tol))
        pos_to_b.setdefault(key, []).append(i)
    pairs = []
    for i, kp in enumerate(kps_a):
        key = (round(kp[0] / tol), round(kp[1] / tol))
        for j in pos_to_b.get(key, []):
            pairs.append((i, j))
    return pairs


def angle_diff(a, b):
    d = abs(float(a) % 360 - float(b) % 360)
    return min(d, 360 - d)


def sep(title=""):
    width = 90
    if title:
        print(f"\n{'─'*width}\n  {title}\n{'─'*width}")
    else:
        print('─' * width)


def print_per_octave(label, kps_cpp, kps_py):
    """Print per-octave count / matched / resp_diff / ang_diff table."""
    print(f"\n  [{label}]")
    print(f"  {'oct':>3}  {'C++':>6}  {'Py':>6}  {'matched':>7}  "
          f"{'resp_diff_mean':>14}  {'resp_diff_max':>13}  "
          f"{'ang_diff_mean':>13}  {'ang_diff_max':>12}")
    sep()
    total = 0
    for lvl in range(N_LEVELS):
        c     = kps_cpp[kps_cpp[:, 5] == lvl]
        p     = kps_py [kps_py[:,  5] == lvl]
        pairs = match_by_position(c, p)
        total += len(pairs)
        if pairs:
            rd = [abs(float(c[i, 4]) - float(p[j, 4])) for i, j in pairs]
            ad = [angle_diff(c[i, 3], p[j, 3])          for i, j in pairs]
            rd_mean, rd_max = np.mean(rd), np.max(rd)
            ad_mean, ad_max = np.mean(ad), np.max(ad)
        else:
            rd_mean = rd_max = ad_mean = ad_max = float('nan')
        print(f"  {lvl:>3}  {len(c):>6}  {len(p):>6}  {len(pairs):>7}  "
              f"  {rd_mean:>13.4f}  {rd_max:>13.4f}  "
              f"  {ad_mean:>12.4f}°  {ad_max:>11.4f}°")
    sep()
    print(f"  Total matched: {total}")


def print_top_angle_mismatches(label, kps_cpp, kps_py):
    print(f"\n  [{label}]")
    for lvl in range(N_LEVELS):
        c     = kps_cpp[kps_cpp[:, 5] == lvl]
        p     = kps_py [kps_py[:,  5] == lvl]
        pairs = match_by_position(c, p)
        if not pairs:
            continue
        cases = sorted(
            [(angle_diff(c[i, 3], p[j, 3]),
              abs(float(c[i, 4]) - float(p[j, 4])),
              (float(c[i, 0]), float(c[i, 1])),
              float(c[i, 3]) % 360, float(p[j, 3]) % 360,
              float(c[i, 4]), float(p[j, 4]))
             for i, j in pairs],
            reverse=True
        )
        print(f"\n    Octave {lvl}:")
        print(f"    {'pos':>20}  {'ang_C++':>9}  {'ang_Py':>9}  "
              f"{'ang_diff':>9}  {'resp_C++':>9}  {'resp_Py':>9}  {'resp_diff':>9}")
        for adiff, rdiff, pos, ac, ap, rc, rp in cases[:5]:
            print(f"    ({pos[0]:6.1f},{pos[1]:6.1f})  "
                  f"{ac:>9.3f}  {ap:>9.3f}  {adiff:>9.3f}°  "
                  f"{rc:>9.3f}  {rp:>9.3f}  {rdiff:>9.3f}")


def to_cv_keypoints(kps_arr):
    return [cv2.KeyPoint(float(r[0]), float(r[1]), float(r[2]),
                         float(r[3] % 360), float(r[4]), int(r[5]))
            for r in kps_arr]


def save_level_images(image_path, kps_cpp, kps_py, out_path):
    img_bgr = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE),
                            cv2.COLOR_GRAY2BGR)
    rows = []
    for lvl in range(N_LEVELS):
        c = kps_cpp[kps_cpp[:, 5] == lvl]
        p = kps_py [kps_py[:,  5] == lvl]

        vis_c = cv2.drawKeypoints(img_bgr, to_cv_keypoints(c), None,
                                  color=(0, 0, 255),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        vis_p = cv2.drawKeypoints(img_bgr, to_cv_keypoints(p), None,
                                  color=(0, 255, 0),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        row = np.hstack([vis_c, vis_p])
        cv2.putText(row, f"L{lvl} C++ ({len(c)})",
                    (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(row, f"L{lvl} Python ({len(p)})",
                    (img_bgr.shape[1] + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        rows.append(row)

    cv2.imwrite(out_path, np.vstack(rows))
    print(f"  Saved: {out_path}  (Red=C++  Green=Python)")


def main():
    print(f"Image : {IMAGE}")

    kps_cpp = run_cpp(IMAGE)
    kps_py  = run_python(IMAGE)

    sep("Keypoint count")
    print(f"  C++    : {len(kps_cpp)}")
    print(f"  Python : {len(kps_py)}")

    sep("Per-octave comparison  (position / response / angle)")
    print_per_octave("C++ vs Python", kps_cpp, kps_py)

    sep("Top-5 angle mismatches per octave")
    print_top_angle_mismatches("C++ vs Python", kps_cpp, kps_py)

    sep("Saving visualization")
    out_path = os.path.join(os.path.dirname(__file__), 'keypoint_per_level.png')
    save_level_images(IMAGE, kps_cpp, kps_py, out_path)


if __name__ == '__main__':
    main()
