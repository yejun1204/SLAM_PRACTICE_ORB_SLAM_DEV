#!/bin/bash
set -e
cd "$(dirname "$0")"

ORBSLAM_SRC=/home/yejun/ORB_SLAM3/src
ORBSLAM_INC=/home/yejun/ORB_SLAM3/include

g++ -O2 -std=c++14 \
    -I${ORBSLAM_INC} \
    $(pkg-config --cflags opencv4) \
    orb_extractor_bin.cc \
    ${ORBSLAM_SRC}/ORBextractor.cc \
    $(pkg-config --libs opencv4) \
    -o orb_extractor_bin

echo "Built: orb_extractor_bin"
