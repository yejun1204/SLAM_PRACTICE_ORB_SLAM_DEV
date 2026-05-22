"""
Feature Matcher - ORB-SLAM3 style matching functions

Implements:
- SearchForInitialization: window-based matching for monocular init
"""

import cv2
import numpy as np

TH_LOW  = 50   # ORB descriptor distance threshold (tight)
TH_HIGH = 100  # ORB descriptor distance threshold (loose)
HISTO_LENGTH = 30  # Rotation histogram bins


def _descriptor_distance(d1, d2):
    """Hamming distance between two ORB descriptors."""
    return cv2.norm(d1, d2, cv2.NORM_HAMMING)


def _compute_three_maxima(histo, n_bins):
    """
    Find the 3 most populated bins in the rotation histogram.
    ORB-SLAM3: ORBmatcher::ComputeThreeMaxima
    """
    counts = [len(histo[i]) for i in range(n_bins)]
    sorted_bins = sorted(range(n_bins), key=lambda i: counts[i], reverse=True)

    ind1 = sorted_bins[0]
    ind2 = sorted_bins[1] if len(sorted_bins) > 1 else -1
    ind3 = sorted_bins[2] if len(sorted_bins) > 2 else -1

    # Only keep bins with at least 10% of the max count
    max_count = counts[ind1]
    if max_count == 0:
        return -1, -1, -1
    if ind2 >= 0 and counts[ind2] < max_count * 0.1:
        ind2 = -1
    if ind3 >= 0 and counts[ind3] < max_count * 0.1:
        ind3 = -1

    return ind1, ind2, ind3


def search_for_initialization(frame1, frame2, prev_matched,
                               window_size=100, nn_ratio=0.9,
                               check_orientation=True):
    """
    Match level-0 keypoints for monocular initialization.
    ORB-SLAM3: ORBmatcher::SearchForInitialization

    Args:
        frame1: Frame (reference frame)
        frame2: Frame (current frame)
        prev_matched: list of (x,y) tuples — search centers for each F1 keypoint
                      (starts as F1 keypoint positions, updated after each call)
        window_size: search radius in pixels (default 100)
        nn_ratio: ratio test threshold (default 0.9)
        check_orientation: whether to apply rotation histogram filter

    Returns:
        matches12: list indexed by F1 keypoint index → F2 keypoint index (-1 if unmatched)
        n_matches: number of valid matches
        prev_matched: updated search positions (F2 matched positions)
    """
    n1 = len(frame1.keypoints)
    n2 = len(frame2.keypoints)

    matches12 = [-1] * n1           # F1 idx → F2 idx
    matches21 = [-1] * n2           # F2 idx → F1 idx (for cross-check)
    matched_dist = [float('inf')] * n2  # best distance per F2 keypoint

    # Rotation histogram
    rot_hist = [[] for _ in range(HISTO_LENGTH)]
    factor = 1.0 / HISTO_LENGTH

    n_matches = 0

    for i1, kp1 in enumerate(frame1.keypoints):
        # Only use level-0 keypoints from F1
        if kp1.octave != 0:
            continue

        cx, cy = prev_matched[i1]

        # Search in window around prev_matched position, same level (0)
        candidates = frame2.get_features_in_area(cx, cy, window_size,
                                                  min_level=0, max_level=0)
        if not candidates:
            continue

        d1 = frame1.descriptors[i1]
        best_dist  = float('inf')
        best_dist2 = float('inf')
        best_i2    = -1

        for i2 in candidates:
            d2 = frame2.descriptors[i2]
            dist = _descriptor_distance(d1, d2)

            # Cross-check: skip if F2 keypoint already has a better match
            if matched_dist[i2] <= dist:
                continue

            if dist < best_dist:
                best_dist2 = best_dist
                best_dist  = dist
                best_i2    = i2
            elif dist < best_dist2:
                best_dist2 = dist

        # Accept match: within TH_LOW and passes ratio test
        if best_dist <= TH_LOW and best_dist < nn_ratio * best_dist2:
            # Remove previous match if F2 keypoint was already matched
            if matches21[best_i2] >= 0:
                matches12[matches21[best_i2]] = -1
                n_matches -= 1

            matches12[i1]  = best_i2
            matches21[best_i2] = i1
            matched_dist[best_i2] = best_dist
            n_matches += 1

            if check_orientation:
                kp2 = frame2.keypoints[best_i2]
                rot = kp1.angle - kp2.angle
                if rot < 0:
                    rot += 360.0
                bin_idx = round(rot * factor) % HISTO_LENGTH
                rot_hist[bin_idx].append(i1)

    # Rotation consistency filter
    if check_orientation:
        ind1, ind2, ind3 = _compute_three_maxima(rot_hist, HISTO_LENGTH)
        for b in range(HISTO_LENGTH):
            if b == ind1 or b == ind2 or b == ind3:
                continue
            for i1 in rot_hist[b]:
                if matches12[i1] >= 0:
                    matches12[i1] = -1
                    n_matches -= 1

    # Update prev_matched to F2 matched positions (for next frame's window search)
    for i1, i2 in enumerate(matches12):
        if i2 >= 0:
            prev_matched[i1] = frame2.keypoints[i2].pt

    return matches12, n_matches, prev_matched
