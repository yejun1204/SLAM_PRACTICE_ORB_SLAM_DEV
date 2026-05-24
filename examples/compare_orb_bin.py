"""
Compare ORB-SLAM3 C++ binary vs Python ORBExtractor.

C++ binary : cpp/orb_extractor_bin
Python impl: src/orb_extractor.py
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


# ---- I/O ------------------------------------------------------------------

def run_cpp(image_path):
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        out_path = f.name
    try:
        subprocess.run([BINARY, image_path, out_path], check=True, capture_output=True)
        with open(out_path, 'rb') as f:
            N = np.frombuffer(f.read(4), dtype=np.int32)[0]
            kps_raw = np.frombuffer(f.read(N * 24), dtype=np.float32).reshape(N, 6)
            # last field is octave stored as float32 bits → reinterpret as int32
            octaves = kps_raw[:, 5].view(np.int32)
            kps = np.column_stack([kps_raw[:, :5], octaves.astype(np.float32)])
            desc = np.frombuffer(f.read(N * 32), dtype=np.uint8).reshape(N, 32)
        return kps, desc  # kps: (N,6) [x,y,size,angle,response,octave]
    finally:
        os.unlink(out_path)


def run_python(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ext = ORBExtractor(n_features=1000, scale_factor=1.2, n_levels=8,
                       ini_th_fast=20, min_th_fast=7)
    keypoints, desc = ext.detect_and_compute(img)
    kps = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave]
                    for kp in keypoints], dtype=np.float32)
    return kps, desc


# ---- report ---------------------------------------------------------------

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    print(f"Image: {IMAGE}")

    kps_cpp,  desc_cpp  = run_cpp(IMAGE)
    kps_py,   desc_py   = run_python(IMAGE)

    # keypoint count
    section("Keypoint count")
    print(f"  C++    : {len(kps_cpp)}")
    print(f"  Python : {len(kps_py)}")

    # per-level
    section("Per-level distribution")
    print(f"  {'Level':>5}  {'C++':>8}  {'Python':>8}")
    for lvl in range(N_LEVELS):
        nc = int((kps_cpp[:, 5] == lvl).sum())
        np_ = int((kps_py[:, 5] == lvl).sum())
        print(f"  {lvl:>5}  {nc:>8}  {np_:>8}")

    # exact position match
    section("Exact keypoint overlap (rounded to 0.1px)")
    pts_cpp = set(map(tuple, np.round(kps_cpp[:, :2], 1).tolist()))
    pts_py  = set(map(tuple, np.round(kps_py[:,  :2], 1).tolist()))
    common  = pts_cpp & pts_py
    print(f"  Shared : {len(common)}  /  C++={len(pts_cpp)}  Python={len(pts_py)}"
          f"  ({100*len(common)/max(len(pts_cpp),1):.1f}%)")

    # descriptor comparison at exact same positions
    section("Descriptor comparison at exact same keypoint positions")
    # build (rounded_pos → index) maps
    pos_to_idx_cpp = {tuple(np.round(kps_cpp[i, :2], 1).tolist()): i for i in range(len(kps_cpp))}
    pos_to_idx_py  = {tuple(np.round(kps_py[i,  :2], 1).tolist()): i for i in range(len(kps_py))}
    common_pos = set(pos_to_idx_cpp) & set(pos_to_idx_py)

    if common_pos:
        hamming_dists = []
        identical = 0
        for pos in common_pos:
            d_cpp = desc_cpp[pos_to_idx_cpp[pos]]
            d_py  = desc_py[pos_to_idx_py[pos]]
            h = int(np.unpackbits(d_cpp ^ d_py).sum())
            hamming_dists.append(h)
            if h == 0:
                identical += 1
        hamming_dists = np.array(hamming_dists)
        print(f"  Shared keypoints     : {len(common_pos)}")
        print(f"  Identical descriptors: {identical}  ({100*identical/len(common_pos):.1f}%)")
        print(f"  Hamming dist  mean={hamming_dists.mean():.2f}  "
              f"std={hamming_dists.std():.2f}  "
              f"max={hamming_dists.max()}  "
              f"median={int(np.median(hamming_dists))}")
    else:
        print("  No shared keypoint positions found")

    # position stats
    section("Position stats")
    for name, kps in [("C++", kps_cpp), ("Python", kps_py)]:
        print(f"  {name}:  x mean={kps[:,0].mean():.2f} std={kps[:,0].std():.2f}"
              f"   y mean={kps[:,1].mean():.2f} std={kps[:,1].std():.2f}")

    # orientation — normalize both to [0,360) for fair comparison
    section("Orientation stats  (both normalized to [0,360))")
    for name, kps in [("C++", kps_cpp), ("Python", kps_py)]:
        angles = kps[:, 3] % 360
        print(f"  {name}:  mean={angles.mean():.2f}  std={angles.std():.2f}"
              f"  range=[{angles.min():.1f}, {angles.max():.1f}]")

    # descriptor match rate
    section("Descriptor match rate  (BFMatcher Hamming, ratio=0.75)")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_ratio = [m for m, n in bf.knnMatch(desc_cpp, desc_py, k=2)
                     if m.distance < 0.75 * n.distance]
    nn_dists = [m[0].distance for m in bf.knnMatch(desc_cpp, desc_py, k=1)]
    print(f"  Good matches : {len(matches_ratio)}"
          f"  /  {max(len(desc_cpp), len(desc_py))}"
          f"  ({100*len(matches_ratio)/max(len(desc_cpp),1):.1f}%)")
    print(f"  Avg Hamming (nearest-neighbor): {np.mean(nn_dists):.2f}")

    # visualization
    section("Saving visualization")
    img_bgr = cv2.cvtColor(cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)

    def to_cv(kps_arr):
        return [cv2.KeyPoint(float(r[0]), float(r[1]), float(r[2]),
                             float(r[3] % 360), float(r[4]), int(r[5]))
                for r in kps_arr]

    vis_cpp = cv2.drawKeypoints(img_bgr, to_cv(kps_cpp), None,
                                color=(0, 0, 255),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    vis_py  = cv2.drawKeypoints(img_bgr, to_cv(kps_py),  None,
                                color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out = np.hstack([vis_cpp, vis_py])
    cv2.putText(out, f"C++ ({len(kps_cpp)})",  (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(out, f"Python ({len(kps_py)})", (img_bgr.shape[1]+10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out_path = os.path.join(os.path.dirname(__file__), 'orb_comparison_bin.png')
    cv2.imwrite(out_path, out)
    print(f"  Saved: {out_path}  (Red=C++  Green=Python)")


if __name__ == '__main__':
    main()
