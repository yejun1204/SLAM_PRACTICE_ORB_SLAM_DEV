"""
Descriptor comparison: C++ ORB-SLAM3 binary vs Python ORBExtractor.

Uses C++ keypoints directly in the Python descriptor computation,
so comparison is purely about descriptor logic (no position-matching noise).
"""

import sys, os, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from src.orb_extractor import ORBExtractor

ROOT    = os.path.join(os.path.dirname(__file__), '..')
BINARY  = os.path.join(ROOT, 'cpp', 'orb_extractor_bin')
IMAGE   = os.path.join(ROOT, 'data/V1_01_easy/mav0/cam0/data/1403715273262142976.png')
N_LEVELS = 8


def run_cpp(image_path):
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        out_path = f.name
    try:
        subprocess.run([BINARY, image_path, out_path], check=True, capture_output=True)
        with open(out_path, 'rb') as f:
            N    = np.frombuffer(f.read(4), dtype=np.int32)[0]
            raw  = np.frombuffer(f.read(N * 24), dtype=np.float32).reshape(N, 6)
            desc = np.frombuffer(f.read(N * 32), dtype=np.uint8).reshape(N, 32)
            octaves = raw[:, 5].view(np.int32)
            kps = np.column_stack([raw[:, :5], octaves.astype(np.float32)])
        return kps, desc  # kps: (N,6) x,y,size,angle,response,octave
    finally:
        os.unlink(out_path)


def compute_desc_from_cpp_kps(image_path, kps_cpp):
    """Build pyramid and compute descriptors using C++ keypoints."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ext = ORBExtractor(n_features=1000, scale_factor=1.2, n_levels=8,
                       ini_th_fast=20, min_th_fast=7)

    pyramid = ext._build_pyramid(img)

    # Group C++ keypoints by level as cv2.KeyPoint (level-0 coords)
    all_keypoints = [[] for _ in range(N_LEVELS)]
    for r in kps_cpp:
        kp = cv2.KeyPoint(float(r[0]), float(r[1]), float(r[2]),
                          float(r[3] % 360), float(r[4]), int(r[5]))
        all_keypoints[int(r[5])].append(kp)

    _, desc = ext._compute_descriptors(pyramid, all_keypoints)
    return desc  # same order as kps_cpp


def hamming(a, b):
    return int(np.unpackbits(a ^ b).sum())


def sep(title=""):
    width = 72
    if title:
        print(f"\n{'─'*width}\n  {title}\n{'─'*width}")
    else:
        print('─' * width)


def main():
    print(f"Image : {IMAGE}")

    kps_cpp, desc_cpp = run_cpp(IMAGE)
    desc_py           = compute_desc_from_cpp_kps(IMAGE, kps_cpp)

    h_dists   = np.array([hamming(desc_cpp[i], desc_py[i]) for i in range(len(kps_cpp))])
    identical = int((h_dists == 0).sum())
    N         = len(kps_cpp)

    sep("Overview")
    print(f"  Keypoints        : {N}")
    print(f"  Identical        : {identical} / {N}  ({100*identical/N:.1f}%)")
    print(f"  Hamming  mean={h_dists.mean():.3f}  "
          f"std={h_dists.std():.3f}  "
          f"median={int(np.median(h_dists))}  "
          f"max={h_dists.max()}")

    sep("Per-octave breakdown")
    print(f"  {'oct':>3}  {'total':>6}  {'identical':>9}  {'match%':>7}  "
          f"{'mean':>7}  {'median':>7}  {'max':>5}")
    sep()
    for lvl in range(N_LEVELS):
        mask = kps_cpp[:, 5] == lvl
        if not mask.any():
            continue
        hd  = h_dists[mask]
        idn = int((hd == 0).sum())
        print(f"  {lvl:>3}  {mask.sum():>6}  {idn:>9}  "
              f"{100*idn/mask.sum():>6.1f}%  "
              f"{hd.mean():>7.3f}  {int(np.median(hd)):>7}  {hd.max():>5}")

    sep("Top-10 differing descriptors  (Hamming 높은 순)")
    order = np.argsort(-h_dists)
    print(f"  {'idx':>5}  {'Hamming':>7}  {'oct':>3}  {'pos':>20}  {'angle':>8}")
    for i in order[:10]:
        if h_dists[i] == 0:
            break
        print(f"  {i:>5}  {h_dists[i]:>7}  {int(kps_cpp[i,5]):>3}  "
              f"({kps_cpp[i,0]:6.1f},{kps_cpp[i,1]:6.1f})  "
              f"{kps_cpp[i,3] % 360:>8.3f}°")


if __name__ == '__main__':
    main()
