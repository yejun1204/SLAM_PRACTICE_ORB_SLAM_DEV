// Usage: ./orb_extractor_bin <image_path> <output_bin_path> [n_features=1000]
//
// Output binary format:
//   [N: int32]
//   [keypoints: N x {x,y,size,angle,response: float32, octave: int32}]  = N x 24 bytes
//   [descriptors: N x 32 x uint8]                                        = N x 32 bytes

#include <cstdio>
#include <cstdint>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ORBextractor.h"

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s <image_path> <output_bin> [n_features=1000]\n", argv[0]);
        return 1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        fprintf(stderr, "Cannot load image: %s\n", argv[1]);
        return 1;
    }

    int n_features = (argc == 4) ? atoi(argv[3]) : 1000;
    ORB_SLAM3::ORBextractor extractor(n_features, 1.2f, 8, 20, 7);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<int> lapping = {0, 0};
    extractor(img, cv::Mat(), keypoints, descriptors, lapping);

    FILE* fp = fopen(argv[2], "wb");
    if (!fp) {
        fprintf(stderr, "Cannot open output: %s\n", argv[2]);
        return 1;
    }

    int32_t N = (int32_t)keypoints.size();
    fwrite(&N, sizeof(int32_t), 1, fp);

    for (auto& kp : keypoints) {
        float fields[5] = {kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response};
        int32_t octave = kp.octave;
        fwrite(fields, sizeof(float), 5, fp);
        fwrite(&octave, sizeof(int32_t), 1, fp);
    }

    if (N > 0)
        fwrite(descriptors.data, 1, N * 32, fp);

    fclose(fp);
    fprintf(stdout, "Wrote %d keypoints to %s\n", N, argv[2]);
    return 0;
}
