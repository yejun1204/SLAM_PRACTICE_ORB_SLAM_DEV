"""
Step 1: ORB Feature Extraction (ORB-SLAM3 style)

Compares:
- cv2.ORB_create (standard)
- ORBExtractor (our implementation)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from src.orb_extractor import ORBExtractor

IMAGE_PATH = '/home/yejun/V1_01_easy/mav0/cam0/data/1403715273262142976.png'

N_FEATURES   = 1000
SCALE_FACTOR = 1.2
N_LEVELS     = 8
INI_TH_FAST  = 20
MIN_TH_FAST  = 7


def main():
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Image not found: {IMAGE_PATH}"
    print(f"Image: {img.shape[1]}x{img.shape[0]}")

    # --- cv2 baseline ---
    orb_cv2 = cv2.ORB_create(
        nfeatures=N_FEATURES,
        scaleFactor=SCALE_FACTOR,
        nlevels=N_LEVELS,
        edgeThreshold=19,
    )
    kps_cv2, desc_cv2 = orb_cv2.detectAndCompute(img, None)

    # --- Our extractor ---
    extractor = ORBExtractor(
        n_features=N_FEATURES,
        scale_factor=SCALE_FACTOR,
        n_levels=N_LEVELS,
        ini_th_fast=INI_TH_FAST,
        min_th_fast=MIN_TH_FAST,
    )
    kps_ours, desc_ours = extractor.detect_and_compute(img)

    # --- Compare ---
    print("\n=== Feature Count ===")
    print(f"  cv2 ORB:  {len(kps_cv2)} keypoints")
    print(f"  Ours:     {len(kps_ours)} keypoints")

    print("\n=== Level Distribution ===")
    from collections import Counter
    cv2_levels = Counter(kp.octave for kp in kps_cv2)
    our_levels  = Counter(kp.octave for kp in kps_ours)
    print(f"  {'Level':<8} {'cv2':>8} {'ours':>8} {'target':>8}")
    for lvl in range(N_LEVELS):
        print(f"  {lvl:<8} {cv2_levels.get(lvl,0):>8} {our_levels.get(lvl,0):>8} "
              f"{extractor.n_features_per_level[lvl]:>8}")

    # --- Visualize ---
    vis_cv2  = cv2.drawKeypoints(img, kps_cv2,  None, color=(0, 255, 0),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    vis_ours = cv2.drawKeypoints(img, kps_ours, None, color=(0, 200, 255),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    combined = np.hstack([vis_cv2, vis_ours])
    cv2.putText(combined, 'cv2 ORB', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, 'Ours (ORB-SLAM3 style)', (img.shape[1] + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.imshow('ORB Comparison', combined)
    print("\nPress any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
