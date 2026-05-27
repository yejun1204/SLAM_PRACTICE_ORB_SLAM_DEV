// Usage: ./search_init_bin <image1> <image2>
//
// Extracts ORB features from two images and runs SearchForInitialization,
// printing the match count for comparison with the Python implementation.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <climits>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ORBextractor.h"

// ---- Constants (ORBmatcher) ------------------------------------------------
static const int TH_LOW       = 50;
static const int HISTO_LENGTH = 30;
static const float NN_RATIO   = 0.9f;
static const int WINDOW_SIZE  = 100;

// ---- Constants (Frame grid) ------------------------------------------------
static const int FRAME_GRID_COLS = 64;
static const int FRAME_GRID_ROWS = 48;

// ---- Hamming distance (ORBmatcher::DescriptorDistance) ---------------------
static int descriptor_distance(const cv::Mat& a, const cv::Mat& b) {
    const int* pa = a.ptr<int32_t>();
    const int* pb = b.ptr<int32_t>();
    int dist = 0;
    for (int i = 0; i < 8; i++, pa++, pb++) {
        unsigned int v = *pa ^ *pb;
        v -= ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return dist;
}

// ---- Minimal Frame (grid-based spatial lookup) -----------------------------
struct SimpleFrame {
    std::vector<cv::KeyPoint> kps;
    cv::Mat descs;
    int img_w, img_h;

    float min_x, max_x, min_y, max_y;
    float grid_w_inv, grid_h_inv;
    std::vector<std::vector<std::vector<size_t>>> grid; // [col][row]

    SimpleFrame(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                int w, int h)
        : kps(keypoints), descs(descriptors), img_w(w), img_h(h),
          min_x(0), max_x((float)w), min_y(0), max_y((float)h),
          grid(FRAME_GRID_COLS, std::vector<std::vector<size_t>>(FRAME_GRID_ROWS))
    {
        grid_w_inv = FRAME_GRID_COLS / (max_x - min_x);
        grid_h_inv = FRAME_GRID_ROWS / (max_y - min_y);
        for (size_t i = 0; i < kps.size(); i++) {
            int col = (int)round((kps[i].pt.x - min_x) * grid_w_inv);
            int row = (int)round((kps[i].pt.y - min_y) * grid_h_inv);
            if (col >= 0 && col < FRAME_GRID_COLS && row >= 0 && row < FRAME_GRID_ROWS)
                grid[col][row].push_back(i);
        }
    }

    std::vector<size_t> get_features_in_area(float x, float y, float r,
                                              int min_lvl, int max_lvl) const {
        std::vector<size_t> res;
        int c0 = std::max(0,   (int)floor((x - min_x - r) * grid_w_inv));
        int c1 = std::min(FRAME_GRID_COLS-1, (int)ceil((x - min_x + r) * grid_w_inv));
        int r0 = std::max(0,   (int)floor((y - min_y - r) * grid_h_inv));
        int r1 = std::min(FRAME_GRID_ROWS-1, (int)ceil((y - min_y + r) * grid_h_inv));
        if (c0 >= FRAME_GRID_COLS || c1 < 0 || r0 >= FRAME_GRID_ROWS || r1 < 0)
            return res;
        bool check = (min_lvl > 0) || (max_lvl >= 0);
        for (int c = c0; c <= c1; c++) {
            for (int r = r0; r <= r1; r++) {
                for (size_t idx : grid[c][r]) {
                    if (check) {
                        if (kps[idx].octave < min_lvl) continue;
                        if (max_lvl >= 0 && kps[idx].octave > max_lvl) continue;
                    }
                    float dx = kps[idx].pt.x - x;
                    float dy = kps[idx].pt.y - y;
                    if (fabs(dx) < r && fabs(dy) < r)
                        res.push_back(idx);
                }
            }
        }
        return res;
    }
};

// ---- ComputeThreeMaxima ----------------------------------------------------
static void compute_three_maxima(std::vector<int>* histo, int L,
                                  int& ind1, int& ind2, int& ind3) {
    int max1=0, max2=0, max3=0;
    ind1=ind2=ind3=-1;
    for (int i = 0; i < L; i++) {
        int s = (int)histo[i].size();
        if (s > max1) { max3=max2; max2=max1; max1=s; ind3=ind2; ind2=ind1; ind1=i; }
        else if (s > max2) { max3=max2; max2=s; ind3=ind2; ind2=i; }
        else if (s > max3) { max3=s; ind3=i; }
    }
    if (max2 < 0.1f * max1) { ind2=-1; ind3=-1; }
    else if (max3 < 0.1f * max1) { ind3=-1; }
}

// ---- SearchForInitialization -----------------------------------------------
static int search_for_initialization(const SimpleFrame& F1, const SimpleFrame& F2,
                                      std::vector<cv::Point2f>& prev_matched,
                                      std::vector<int>& matches12) {
    int nmatches = 0;
    matches12.assign(F1.kps.size(), -1);

    std::vector<int> rot_hist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++) rot_hist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    std::vector<int> matched_dist(F2.kps.size(), INT_MAX);
    std::vector<int> matches21(F2.kps.size(), -1);

    for (size_t i1 = 0; i1 < F1.kps.size(); i1++) {
        const cv::KeyPoint& kp1 = F1.kps[i1];
        if (kp1.octave > 0) continue;

        auto cands = F2.get_features_in_area(prev_matched[i1].x, prev_matched[i1].y,
                                              WINDOW_SIZE, kp1.octave, kp1.octave);
        if (cands.empty()) continue;

        cv::Mat d1 = F1.descs.row((int)i1);
        int best = INT_MAX, best2 = INT_MAX, best_idx = -1;

        for (size_t i2 : cands) {
            cv::Mat d2 = F2.descs.row((int)i2);
            int dist = descriptor_distance(d1, d2);
            if (matched_dist[i2] <= dist) continue;
            if (dist < best)  { best2=best; best=dist; best_idx=(int)i2; }
            else if (dist < best2) { best2=dist; }
        }

        if (best > TH_LOW) continue;
        if (best >= NN_RATIO * best2) continue;

        if (matches21[best_idx] >= 0) {
            matches12[matches21[best_idx]] = -1;
            nmatches--;
        }
        matches12[i1]      = best_idx;
        matches21[best_idx] = (int)i1;
        matched_dist[best_idx] = best;
        nmatches++;

        float rot = kp1.angle - F2.kps[best_idx].angle;
        if (rot < 0.0f) rot += 360.0f;
        int bin = (int)round(rot * factor);
        if (bin == HISTO_LENGTH) bin = 0;
        rot_hist[bin].push_back((int)i1);
    }

    // Rotation consistency
    int ind1, ind2, ind3;
    compute_three_maxima(rot_hist, HISTO_LENGTH, ind1, ind2, ind3);
    for (int i = 0; i < HISTO_LENGTH; i++) {
        if (i == ind1 || i == ind2 || i == ind3) continue;
        for (int i1 : rot_hist[i]) {
            if (matches12[i1] >= 0) { matches12[i1]=-1; nmatches--; }
        }
    }

    // Update prev_matched
    for (size_t i1 = 0; i1 < matches12.size(); i1++)
        if (matches12[i1] >= 0)
            prev_matched[i1] = F2.kps[matches12[i1]].pt;

    return nmatches;
}

// ---- main ------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <image1> <image2>\n", argv[0]);
        return 1;
    }

    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        fprintf(stderr, "Cannot load images\n"); return 1;
    }

    ORB_SLAM3::ORBextractor ext(1000, 1.2f, 8, 20, 7);

    std::vector<cv::KeyPoint> kps1, kps2;
    cv::Mat descs1, descs2;
    std::vector<int> lapping = {0, 0};
    ext(img1, cv::Mat(), kps1, descs1, lapping);
    ext(img2, cv::Mat(), kps2, descs2, lapping);

    printf("Frame1 keypoints: %d\n", (int)kps1.size());
    printf("Frame2 keypoints: %d\n", (int)kps2.size());
    printf("Level-0 in Frame1: %d\n",
           (int)std::count_if(kps1.begin(), kps1.end(),
                              [](const cv::KeyPoint& k){ return k.octave==0; }));

    SimpleFrame F1(kps1, descs1, img1.cols, img1.rows);
    SimpleFrame F2(kps2, descs2, img2.cols, img2.rows);

    std::vector<cv::Point2f> prev_matched;
    for (auto& kp : kps1) prev_matched.push_back(kp.pt);

    std::vector<int> matches12;
    int nmatches = search_for_initialization(F1, F2, prev_matched, matches12);

    printf("Matches: %d\n", nmatches);

    return 0;
}
