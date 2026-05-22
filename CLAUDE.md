# SLAM_PRACTICE_NEW

ORB-SLAM3 monocular SLAM implementation in Python, built component by component to match ORB-SLAM3 behavior.

## Reference
- ORB-SLAM3 source: `/home/yejun/ORB_SLAM3`
- EuRoC dataset: `/home/yejun/V1_01_easy`

## Running
```bash
python examples/step1_orb_extraction.py
```

## Implementation Order
1. ORB feature extraction (scale pyramid)
2. SearchForInitialization (window-based matching)
3. Initialization (H/F + Global BA + scale normalization)
4. TrackWithMotionModel
5. TrackReferenceKeyFrame
6. TrackLocalMap + PoseOptimization
7. LocalMapping
8. LoopClosing
