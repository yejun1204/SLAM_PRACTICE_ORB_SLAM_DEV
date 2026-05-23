# SLAM_PRACTICE_NEW

ORB-SLAM3 monocular SLAM implementation in Python, built component by component to match ORB-SLAM3 behavior.

## Goal
ORB-SLAM3 C++ 구현(`/home/yejun/ORB_SLAM3`)의 `Examples/Monocular/mono_euroc`을 V1_01_easy 데이터셋으로 실행했을 때와 **각 단계의 로직이 동일하게 동작**하는 Python 구현을 만드는 것이 목표다. 단순히 최종 궤적 정확도를 맞추는 것이 아니라, ORB 추출 → 초기화 → 트래킹 → 매핑 → 루프클로징 각 단계에서 C++ 소스(`/home/yejun/ORB_SLAM3/src/`)의 로직을 레퍼런스로 삼아 동등한 동작을 구현하는 것이 핵심이다.

레퍼런스 실행 방법:
```bash
cd /home/yejun/ORB_SLAM3
./Examples/Monocular/mono_euroc Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml /home/yejun/V1_01_easy Examples/Monocular/EuRoC_TimeStamps/V101.txt
```

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
