#!/bin/bash
set -e
cd "$(dirname "$0")"

ORBSLAM_SRC=/home/yejun/ORB_SLAM3/src
ORBSLAM_INC=/home/yejun/ORB_SLAM3/include

CFLAGS="-O2 -std=c++14 -I${ORBSLAM_INC} $(pkg-config --cflags opencv4)"
LIBS="${ORBSLAM_SRC}/ORBextractor.cc $(pkg-config --libs opencv4)"

g++ $CFLAGS orb_extractor_bin.cc $LIBS -o orb_extractor_bin
echo "Built: orb_extractor_bin"

g++ $CFLAGS search_init_bin.cc $LIBS -o search_init_bin
echo "Built: search_init_bin"
