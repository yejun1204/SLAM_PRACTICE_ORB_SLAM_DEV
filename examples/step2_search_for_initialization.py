"""
Step 2: SearchForInitialization (ORB-SLAM3 style)

Tests window-based matching between two frames:
- Level-0 keypoints only
- 100px window search with vbPrevMatched tracking
- Rotation histogram consistency filter
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from src.orb_extractor import ORBExtractor
from src.frame import Frame
from src.feature_matcher import search_for_initialization

DATA_DIR = '/home/yejun/V1_01_easy/mav0/cam0/data'
TIMESTAMPS = '/home/yejun/ORB_SLAM3/Examples/Monocular/EuRoC_TimeStamps/V101.txt'

K = np.array([[458.654,   0.,    367.215],
              [  0.,    457.296, 248.375],
              [  0.,      0.,      1.   ]])
DIST = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])


def load_image(frame_idx):
    with open(TIMESTAMPS) as f:
        lines = [l.strip() for l in f if l.strip()]
    ts = lines[frame_idx]
    path = os.path.join(DATA_DIR, ts + '.png')
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def main():
    extractor = ORBExtractor(n_features=1000)

    # Frame 0 = reference, Frame 110 = initialization candidate (same as ORB-SLAM3)
    print("Loading frames...")
    img0   = load_image(0)
    img110 = load_image(110)

    kps0,   desc0   = extractor.detect_and_compute(img0)
    kps110, desc110 = extractor.detect_and_compute(img110)

    frame0   = Frame(img0,   kps0,   desc0,   K, DIST, frame_id=0)
    frame110 = Frame(img110, kps110, desc110, K, DIST, frame_id=110)

    print(f"Frame 0:   {len(frame0.keypoints)} keypoints "
          f"(level-0: {sum(1 for kp in frame0.keypoints if kp.octave==0)})")
    print(f"Frame 110: {len(frame110.keypoints)} keypoints "
          f"(level-0: {sum(1 for kp in frame110.keypoints if kp.octave==0)})")

    # Initialize prev_matched = F0 keypoint positions
    prev_matched = [kp.pt for kp in frame0.keypoints]

    # --- Our SearchForInitialization ---
    matches12, n_matches, _ = search_for_initialization(
        frame0, frame110, prev_matched, window_size=100
    )

    # --- cv2 BFMatcher baseline ---
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(desc0, desc110, k=2)
    bf_matches = [m for m, n in raw if m.distance < 0.9 * n.distance]

    print(f"\n=== Match Count ===")
    print(f"  SearchForInitialization: {n_matches}")
    print(f"  cv2 BFMatcher (ratio):   {len(bf_matches)}")

    # Visualize
    good_matches = [cv2.DMatch(i1, i2, 0)
                    for i1, i2 in enumerate(matches12) if i2 >= 0]

    vis = cv2.drawMatches(img0, frame0.keypoints,
                          img110, frame110.keypoints,
                          good_matches[:100], None,
                          matchColor=(0, 255, 0),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.putText(vis, f'SearchForInitialization: {n_matches} matches',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('SearchForInitialization', vis)
    print("\nPress any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
