"""
Step 2: SearchForInitialization — C++ vs Python comparison

Runs the same two images through both C++ and Python implementations
and compares match counts and matched positions.
"""

import sys, os, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from src.orb_extractor import ORBExtractor
from src.orb_matcher import Frame, search_for_initialization

ROOT    = os.path.join(os.path.dirname(__file__), '..')
BINARY  = os.path.join(ROOT, 'cpp', 'search_init_bin')
DATA    = os.path.join(ROOT, 'data/V1_01_easy/mav0/cam0/data')
IMAGE1  = os.path.join(DATA, '1403715273262142976.png')
IMAGE2  = os.path.join(DATA, '1403715273762142976.png')


# ── C++ helpers ────────────────────────────────────────────────────────────

def run_cpp_extractor(path):
    """Run C++ ORBextractor binary, return (keypoints, descriptors)."""
    orb_bin = os.path.join(ROOT, 'cpp', 'orb_extractor_bin')
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        out = f.name
    try:
        subprocess.run([orb_bin, path, out], check=True, capture_output=True)
        with open(out, 'rb') as f:
            N    = np.frombuffer(f.read(4),      dtype=np.int32)[0]
            raw  = np.frombuffer(f.read(N * 24), dtype=np.float32).reshape(N, 6)
            desc = np.frombuffer(f.read(N * 32), dtype=np.uint8).reshape(N, 32)
        octaves = raw[:, 5].view(np.int32)
        kps = [cv2.KeyPoint(float(r[0]), float(r[1]), float(r[2]),
                            float(r[3] % 360), float(r[4]), int(octaves[i]))
               for i, r in enumerate(raw)]
        return kps, desc
    finally:
        os.unlink(out)


def run_cpp_matcher(img1_path, img2_path):
    """Run C++ search_init_bin, return (kps1, kps2, matches12, n_matches)."""
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        out = f.name
    try:
        subprocess.run([BINARY, img1_path, img2_path, out],
                       check=True, capture_output=True)
        with open(out, 'rb') as f:
            def read_kps(f):
                N   = np.frombuffer(f.read(4),      dtype=np.int32)[0]
                raw = np.frombuffer(f.read(N * 24), dtype=np.float32).reshape(N, 6)
                oct = raw[:, 5].view(np.int32)
                return [cv2.KeyPoint(float(r[0]), float(r[1]), float(r[2]),
                                     float(r[3] % 360), float(r[4]), int(oct[i]))
                        for i, r in enumerate(raw)]

            kps1 = read_kps(f)
            kps2 = read_kps(f)
            M    = np.frombuffer(f.read(4), dtype=np.int32)[0]
            pairs = np.frombuffer(f.read(M * 8), dtype=np.int32).reshape(M, 2)

        # Convert pairs to matches12 list (index i1 → i2, else -1)
        matches12 = [-1] * len(kps1)
        for i1, i2 in pairs:
            matches12[i1] = i2

        return kps1, kps2, matches12, M
    finally:
        os.unlink(out)


# ── Python helpers ─────────────────────────────────────────────────────────

def run_python_extractor(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ext = ORBExtractor(n_features=1000, scale_factor=1.2, n_levels=8,
                       ini_th_fast=20, min_th_fast=7)
    kps, descs = ext.detect_and_compute(img)
    return img, kps, descs


def run_python_matcher(kps1, descs1, kps2, descs2, img_shape):
    h, w = img_shape
    f1 = Frame(kps1, descs1, w, h)
    f2 = Frame(kps2, descs2, w, h)
    prev = [kp.pt for kp in kps1]
    matches, n = search_for_initialization(
        f1, f2, prev, window_size=100, nn_ratio=0.9, check_orientation=True)
    return matches, n


# ── Visualization ──────────────────────────────────────────────────────────

def save_match_image(img1, img2, kps1, kps2, matches12, out_path, title=""):
    kps1_cv, kps2_cv, cv_matches = [], [], []
    for i, j in enumerate(matches12):
        if j < 0:
            continue
        idx = len(kps1_cv)
        kps1_cv.append(cv2.KeyPoint(kps1[i].pt[0], kps1[i].pt[1], 5))
        kps2_cv.append(cv2.KeyPoint(kps2[j].pt[0], kps2[j].pt[1], 5))
        cv_matches.append(cv2.DMatch(idx, idx, 0))

    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    vis = cv2.drawMatches(img1_bgr, kps1_cv, img2_bgr, kps2_cv, cv_matches, None,
                          matchColor=(0, 255, 0),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if title:
        cv2.putText(vis, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    cv2.imwrite(out_path, vis)
    print(f"  Saved: {out_path}")


# ── Separator ──────────────────────────────────────────────────────────────

def sep(title=""):
    w = 70
    if title:
        print(f"\n{'─'*w}\n  {title}\n{'─'*w}")
    else:
        print('─' * w)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"Image1: {os.path.basename(IMAGE1)}")
    print(f"Image2: {os.path.basename(IMAGE2)}")

    img1 = cv2.imread(IMAGE1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(IMAGE2, cv2.IMREAD_GRAYSCALE)

    # ── C++ (extractor + matcher) ─────────────────────────────────────────
    sep("C++ (ORB-SLAM3 binary)")
    kps1_cpp, kps2_cpp, matches_cpp, n_cpp = run_cpp_matcher(IMAGE1, IMAGE2)
    l0_cpp = sum(1 for kp in kps1_cpp if kp.octave == 0)
    print(f"  Frame1 keypoints : {len(kps1_cpp)}  (level-0: {l0_cpp})")
    print(f"  Frame2 keypoints : {len(kps2_cpp)}")
    print(f"  Matches          : {n_cpp}")

    # ── Python (own keypoints + Python matcher) ───────────────────────────
    sep("Python (own keypoints + Python matcher)")
    _, kps1_py, descs1_py = run_python_extractor(IMAGE1)
    _, kps2_py, descs2_py = run_python_extractor(IMAGE2)
    l0_py = sum(1 for kp in kps1_py if kp.octave == 0)
    matches_py, n_py = run_python_matcher(kps1_py, descs1_py,
                                          kps2_py, descs2_py, img1.shape)
    print(f"  Frame1 keypoints : {len(kps1_py)}  (level-0: {l0_py})")
    print(f"  Frame2 keypoints : {len(kps2_py)}")
    print(f"  Matches          : {n_py}")

    # ── Python matcher + C++ keypoints (logic isolation) ──────────────────
    sep("Python matcher  +  C++ keypoints  (logic isolation)")
    _, descs1_cpp = run_cpp_extractor(IMAGE1)
    _, descs2_cpp = run_cpp_extractor(IMAGE2)
    matches_iso, n_iso = run_python_matcher(kps1_cpp, descs1_cpp,
                                            kps2_cpp, descs2_cpp, img1.shape)
    print(f"  Matches          : {n_iso}  "
          f"({'== C++' if n_iso == n_cpp else f'diff {n_iso - n_cpp:+d} vs C++'})")

    # ── Match pair comparison: C++ vs Python matcher (same C++ keypoints) ────
    sep("Match pair comparison  (C++ kps, C++ matcher vs Python matcher)")
    pairs_cpp = {(i, j) for i, j in enumerate(matches_cpp) if j >= 0}
    pairs_iso = {(i, j) for i, j in enumerate(matches_iso) if j >= 0}
    identical  = pairs_cpp & pairs_iso
    only_cpp   = pairs_cpp - pairs_iso
    only_py    = pairs_iso - pairs_cpp
    print(f"  Identical pairs  : {len(identical)} / {n_cpp}")
    print(f"  Only in C++      : {len(only_cpp)}")
    print(f"  Only in Python   : {len(only_py)}")

    # ── Summary ───────────────────────────────────────────────────────────
    sep("Summary")
    print(f"  {'':32s}  {'matches':>7}  {'vs C++':>8}")
    sep()
    print(f"  {'C++ (reference)':32s}  {n_cpp:>7}")
    print(f"  {'Python (own kps)':32s}  {n_py:>7}  {n_py - n_cpp:>+8d}")
    print(f"  {'Python matcher + C++ kps':32s}  {n_iso:>7}  {n_iso - n_cpp:>+8d}")

    # ── Visualization ─────────────────────────────────────────────────────
    sep("Visualization")
    out_dir = os.path.dirname(__file__)
    save_match_image(img1, img2, kps1_cpp, kps2_cpp, matches_cpp,
                     os.path.join(out_dir, 'init_matches_cpp.png'),
                     f"C++  {n_cpp} matches")
    save_match_image(img1, img2, kps1_cpp, kps2_cpp, matches_iso,
                     os.path.join(out_dir, 'init_matches_cpp_kps.png'),
                     f"Python matcher + C++ kps  {n_iso} matches")
    save_match_image(img1, img2, kps1_py, kps2_py, matches_py,
                     os.path.join(out_dir, 'init_matches_python.png'),
                     f"Python  {n_py} matches")


if __name__ == '__main__':
    main()
