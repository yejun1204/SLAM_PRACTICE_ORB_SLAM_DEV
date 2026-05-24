"""
ORB Feature Extractor - ORB-SLAM3 style

Key differences from cv2.ORB_create:
1. Scale pyramid built manually with Gaussian blur per level
2. FAST detection per 35x35 cell (iniThFAST=20, minThFAST=7 fallback)
3. Features distributed across levels using geometric series
4. OctTree-based uniform distribution within each level
5. Keypoint coordinates scaled back to level-0 space
"""

import cv2
import numpy as np


EDGE_THRESHOLD = 19
PATCH_SIZE = 31
HALF_PATCH_SIZE = 15
CELL_SIZE = 35

# Precompute umax table for IC_Angle (ORB-SLAM3 style)
def _build_umax():
    umax = [0] * (HALF_PATCH_SIZE + 1)
    vmax = int(np.floor(HALF_PATCH_SIZE * np.sqrt(2) / 2 + 1))
    vmin = int(np.ceil(HALF_PATCH_SIZE * np.sqrt(2) / 2))
    hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE
    for v in range(vmax + 1):
        umax[v] = int(round(np.sqrt(hp2 - v * v)))
    v0 = 0
    for v in range(HALF_PATCH_SIZE, vmin - 1, -1):
        while umax[v0] == umax[v0 + 1]:
            v0 += 1
        umax[v] = v0
        v0 += 1
    return umax

_UMAX = _build_umax()

# ORB-SLAM3 bit_pattern_31_: 256 point pairs (x0,y0,x1,y1) × 256
_BIT_PATTERN_RAW = [
    8,-3, 9,5, 4,2, 7,-12, -11,9, -8,2, 7,-12, 12,-13,
    2,-13, 2,12, 1,-7, 1,6, -2,-10, -2,-4, -13,-13, -11,-8,
    -13,-3, -12,-9, 10,4, 11,9, -13,-8, -8,-9, -11,7, -9,12,
    7,7, 12,6, -4,-5, -3,0, -13,2, -12,-3, -9,0, -7,5,
    12,-6, 12,-1, -3,6, -2,12, -6,-13, -4,-8, 11,-13, 12,-8,
    4,7, 5,1, 5,-3, 10,-3, 3,-7, 6,12, -8,-7, -6,-2,
    -2,11, -1,-10, -13,12, -8,10, -7,3, -5,-3, -4,2, -3,7,
    -10,-12, -6,11, 5,-12, 6,-7, 5,-6, 7,-1, 1,0, 4,-5,
    9,11, 11,-13, 4,7, 4,12, 2,-1, 4,4, -4,-12, -2,7,
    -8,-5, -7,-10, 4,11, 9,12, 0,-8, 1,-13, -13,-2, -8,2,
    -3,-2, -2,3, -6,9, -4,-9, 8,12, 10,7, 0,9, 1,3,
    7,-5, 11,-10, -13,-6, -11,0, 10,7, 12,1, -6,-3, -6,12,
    10,-9, 12,-4, -13,8, -8,-12, -13,0, -8,-4, 3,3, 7,8,
    5,7, 10,-7, -1,7, 1,-12, 3,-10, 5,6, 2,-4, 3,-10,
    -13,0, -13,5, -13,-7, -12,12, -13,3, -11,8, -7,12, -4,7,
    6,-10, 12,8, -9,-1, -7,-6, -2,-5, 0,12, -12,5, -7,5,
    3,-10, 8,-13, -7,-7, -4,5, -3,-2, -1,-7, 2,9, 5,-11,
    -11,-13, -5,-13, -1,6, 0,-1, 5,-3, 5,2, -4,-13, -4,12,
    -9,-6, -9,6, -12,-10, -8,-4, 10,2, 12,-3, 7,12, 12,12,
    -7,-13, -6,5, -4,9, -3,4, 7,-1, 12,2, -7,6, -5,1,
    -13,11, -12,5, -3,7, -2,-6, 7,-8, 12,-7, -13,-7, -11,-12,
    1,-3, 12,12, 2,-6, 3,0, -4,3, -2,-13, -1,-13, 1,9,
    7,1, 8,-6, 1,-1, 3,12, 9,1, 12,6, -1,-9, -1,3,
    -13,-13, -10,5, 7,7, 10,12, 12,-5, 12,9, 6,3, 7,11,
    5,-13, 6,10, 2,-12, 2,3, 3,8, 4,-6, 2,6, 12,-13,
    9,-12, 10,3, -8,4, -7,9, -11,12, -4,-6, 1,12, 2,-8,
    6,-9, 7,-4, 2,3, 3,-2, 6,3, 11,0, 3,-3, 8,-8,
    7,8, 9,3, -11,-5, -6,-4, -10,11, -5,10, -5,-8, -3,12,
    -10,5, -9,0, 8,-1, 12,-6, 4,-6, 6,-11, -10,12, -8,7,
    4,-2, 6,7, -2,0, -2,12, -5,-8, -5,2, 7,-6, 10,12,
    -9,-13, -8,-8, -5,-13, -5,-2, 8,-8, 9,-13, -9,-11, -9,0,
    1,-8, 1,-2, 7,-4, 9,1, -2,1, -1,-4, 11,-6, 12,-11,
    -12,-9, -6,4, 3,7, 7,12, 5,5, 10,8, 0,-4, 2,8,
    -9,12, -5,-13, 0,7, 2,12, -1,2, 1,7, 5,11, 7,-9,
    3,5, 6,-8, -13,-4, -8,9, -5,9, -3,-3, -4,-7, -3,-12,
    6,5, 8,0, -7,6, -6,12, -13,6, -5,-2, 1,-10, 3,10,
    4,1, 8,-4, -2,-2, 2,-13, 2,-12, 12,12, -2,-13, 0,-6,
    4,1, 9,3, -6,-10, -3,-5, -3,-13, -1,1, 7,5, 12,-11,
    4,-2, 5,-7, -13,9, -9,-5, 7,1, 8,6, 7,-8, 7,6,
    -7,-4, -7,1, -8,11, -7,-8, -13,6, -12,-8, 2,4, 3,9,
    10,-5, 12,3, -6,-5, -6,7, 8,-3, 9,-8, 2,-12, 2,8,
    -11,-2, -10,3, -12,-13, -7,-9, -11,0, -10,-5, 5,-3, 11,8,
    -2,-13, -1,12, -1,-8, 0,9, -13,-11, -12,-5, -10,-2, -10,11,
    -3,9, -2,-13, 2,-3, 3,2, -9,-13, -4,0, -4,6, -3,-10,
    -4,12, -2,-7, -6,-11, -4,9, 6,-3, 6,11, -13,11, -5,5,
    11,11, 12,6, 7,-5, 12,-2, -1,12, 0,7, -4,-8, -3,-2,
    -7,1, -6,7, -13,-12, -8,-13, -7,-2, -6,-8, -8,5, -6,-9,
    -5,-1, -4,5, -13,7, -8,10, 1,5, 5,-13, 1,0, 10,-13,
    9,12, 10,-1, 5,-8, 10,-9, -1,11, 1,-13, -9,-3, -6,2,
    -1,-10, 1,12, -13,1, -8,-10, 8,-11, 10,-6, 2,-13, 3,-6,
    7,-13, 12,-9, -10,-10, -5,-7, -10,-8, -8,-13, 4,-6, 8,5,
    3,12, 8,-13, -4,2, -3,-3, 5,-13, 10,-12, 4,-13, 5,-1,
    -9,9, -4,3, 0,3, 3,-9, -12,1, -6,1, 3,2, 4,-8,
    -10,-10, -10,9, 8,-13, 12,12, -8,-12, -6,-5, 2,2, 3,7,
    10,6, 11,-8, 6,8, 8,-12, -7,10, -6,5, -3,-9, -3,9,
    -1,-13, -1,5, -3,-7, -3,4, -8,-2, -8,3, 4,2, 12,12,
    2,-5, 3,11, 6,-9, 11,-13, 3,-1, 7,12, 11,-1, 12,4,
    -3,0, -3,6, 4,-11, 4,12, 2,-4, 2,1, -10,-6, -8,1,
    -13,7, -11,1, -13,12, -11,-13, 6,0, 11,-13, 0,-1, 1,4,
    -13,3, -9,-2, -9,8, -6,-3, -13,-6, -8,-2, 5,-9, 8,10,
    2,7, 3,-9, -1,-6, -1,-1, 9,5, 11,-2, 11,-3, 12,-8,
    3,0, 3,5, -1,4, 0,10, 3,-6, 4,5, -13,0, -10,5,
    5,8, 12,11, 8,9, 9,-6, 7,-4, 8,-12, -10,4, -10,9,
    7,3, 12,4, 9,-7, 10,-2, 7,0, 12,-2, -1,-6, 0,-11,
]

# Reshape to (256, 4): each row = (x0, y0, x1, y1)
_PATTERN = np.array(_BIT_PATTERN_RAW, dtype=np.float32).reshape(256, 4)


class ORBExtractor:
    def __init__(self, n_features=1000, scale_factor=1.2, n_levels=8,
                 ini_th_fast=20, min_th_fast=7):
        self.n_features = n_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.ini_th_fast = ini_th_fast
        self.min_th_fast = min_th_fast

        # Scale factors per level
        self.scale_factors = [scale_factor ** i for i in range(n_levels)]
        self.inv_scale_factors = [1.0 / s for s in self.scale_factors]

        # Features per level: geometric series (more features at finer levels)
        factor = 1.0 / scale_factor
        n_desired = n_features * (1 - factor) / (1 - factor ** n_levels)
        self.n_features_per_level = []
        total = 0
        for level in range(n_levels - 1):
            n = round(n_desired)
            self.n_features_per_level.append(n)
            total += n
            n_desired *= factor
        self.n_features_per_level.append(max(n_features - total, 0))


    def detect_and_compute(self, image):
        """
        Extract ORB features ORB-SLAM3 style.

        Returns:
            keypoints: list of cv2.KeyPoint (coordinates in level-0 space)
            descriptors: np.ndarray (N x 32, uint8)
        """
        # Build image pyramid
        pyramid = self._build_pyramid(image)

        # Detect FAST keypoints at each level
        all_keypoints = []
        for level in range(self.n_levels):
            kps = self._detect_fast(pyramid[level], level)
            all_keypoints.append(kps)

        # Compute descriptors using cv2 ORB on the pyramid
        keypoints, descriptors = self._compute_descriptors(pyramid, all_keypoints)

        return keypoints, descriptors

    def _build_pyramid(self, image):
        # ORB-SLAM3 ComputePyramid: resize from interior (border-free) of previous level.
        # C++: mvImagePyramid[level] is a sub-rect (no border); resize reads only interior.
        # Storing padded image and resizing from it would stretch border pixels into content.
        interiors = []
        pyramid = []
        for level in range(self.n_levels):
            scale = self.inv_scale_factors[level]
            h = round(image.shape[0] * scale)
            w = round(image.shape[1] * scale)
            if level == 0:
                interior = image
                border_type = cv2.BORDER_REFLECT_101
            else:
                interior = cv2.resize(interiors[level - 1],
                                      (w, h), interpolation=cv2.INTER_LINEAR)
                # BORDER_ISOLATED: border computed from interior only (matches C++)
                border_type = cv2.BORDER_REFLECT_101 | cv2.BORDER_ISOLATED
            padded = cv2.copyMakeBorder(interior,
                                        EDGE_THRESHOLD, EDGE_THRESHOLD,
                                        EDGE_THRESHOLD, EDGE_THRESHOLD,
                                        border_type)
            interiors.append(interior)
            pyramid.append(padded)
        return pyramid

    def _detect_fast(self, img, level):
        """Detect FAST keypoints in 35x35 cells, then distribute via OctTree.
        img is the padded image (EDGE_THRESHOLD border on all sides).
        Actual image starts at (EDGE_THRESHOLD, EDGE_THRESHOLD) in padded coords.
        Detection region = actual image inset by (EDGE_THRESHOLD-3) on each side.
        → In padded coords: [2*EDGE_THRESHOLD-3, w-2*EDGE_THRESHOLD+3]
        """
        h, w = img.shape
        min_x = 2 * EDGE_THRESHOLD - 3   # = 35
        min_y = 2 * EDGE_THRESHOLD - 3
        max_x = w - 2 * EDGE_THRESHOLD + 3
        max_y = h - 2 * EDGE_THRESHOLD + 3

        n_cols = max(1, round((max_x - min_x) / CELL_SIZE))
        n_rows = max(1, round((max_y - min_y) / CELL_SIZE))
        cell_w = int(np.ceil((max_x - min_x) / n_cols))
        cell_h = int(np.ceil((max_y - min_y) / n_rows))

        candidates = []

        for r in range(n_rows):
            ini_y = min_y + r * cell_h
            end_y = min(ini_y + cell_h + 6, max_y)
            if ini_y >= max_y - 3:
                continue

            for c in range(n_cols):
                ini_x = min_x + c * cell_w
                end_x = min(ini_x + cell_w + 6, max_x)
                if ini_x >= max_x - 6:
                    continue

                cell = img[ini_y:end_y, ini_x:end_x]
                kps = cv2.FastFeatureDetector_create(
                    threshold=self.ini_th_fast, nonmaxSuppression=True
                ).detect(cell)

                if not kps:
                    kps = cv2.FastFeatureDetector_create(
                        threshold=self.min_th_fast, nonmaxSuppression=True
                    ).detect(cell)

                for kp in kps:
                    kp.pt = (kp.pt[0] + ini_x, kp.pt[1] + ini_y)
                    kp.octave = level
                    kp.size = PATCH_SIZE * self.scale_factors[level]
                    candidates.append(kp)

        # Distribute uniformly via OctTree
        n_target = self.n_features_per_level[level]
        distributed = self._distribute_oct_tree(
            candidates, min_x, max_x, min_y, max_y, n_target
        )

        # Compute IC_Angle orientation (in padded image coords)
        self._compute_orientation(img, distributed)

        # Remove border offset, then scale back to level-0 space
        scale = self.scale_factors[level]
        for kp in distributed:
            x = (kp.pt[0] - EDGE_THRESHOLD) * scale
            y = (kp.pt[1] - EDGE_THRESHOLD) * scale
            kp.pt = (x, y)

        return distributed

    def _compute_orientation(self, image, keypoints):
        """Compute IC_Angle for each keypoint (ORB-SLAM3: computeOrientation)."""
        for kp in keypoints:
            cx = int(round(kp.pt[0]))
            cy = int(round(kp.pt[1]))
            h, w = image.shape

            if (cx - HALF_PATCH_SIZE < 0 or cx + HALF_PATCH_SIZE >= w or
                    cy - HALF_PATCH_SIZE < 0 or cy + HALF_PATCH_SIZE >= h):
                kp.angle = 0.0
                continue

            # Center row (v=0)
            row0 = image[cy, cx - HALF_PATCH_SIZE: cx + HALF_PATCH_SIZE + 1].astype(np.int32)
            us = np.arange(-HALF_PATCH_SIZE, HALF_PATCH_SIZE + 1)
            m10 = int(np.dot(us, row0))
            m01 = 0

            for v in range(1, HALF_PATCH_SIZE + 1):
                d = _UMAX[v]
                row_p = image[cy + v, cx - d: cx + d + 1].astype(np.int32)
                row_m = image[cy - v, cx - d: cx + d + 1].astype(np.int32)
                us_v = np.arange(-d, d + 1)
                m01 += v * int(np.sum(row_p - row_m))
                m10 += int(np.dot(us_v, row_p + row_m))

            kp.angle = float(np.degrees(np.arctan2(m01, m10)))

    def _distribute_oct_tree(self, keypoints, min_x, max_x, min_y, max_y, n_target):
        """
        OctTree distribution: recursively subdivide until each node has 1 keypoint.
        Keeps the strongest (highest response) keypoint per node.
        """
        if not keypoints or n_target == 0:
            return []

        width = max_x - min_x
        height = max_y - min_y

        # Start with root node covering the whole region
        nodes = [{'kps': keypoints, 'x0': min_x, 'x1': max_x, 'y0': min_y, 'y1': max_y}]

        while len(nodes) < n_target:
            prev_size = len(nodes)
            new_nodes = []
            for node in nodes:
                if len(node['kps']) == 1:
                    new_nodes.append(node)
                    continue
                # Split into 4 children
                mx = (node['x0'] + node['x1']) / 2
                my = (node['y0'] + node['y1']) / 2
                children = [
                    {'x0': node['x0'], 'x1': mx,   'y0': node['y0'], 'y1': my,   'kps': []},
                    {'x0': mx,         'x1': node['x1'], 'y0': node['y0'], 'y1': my, 'kps': []},
                    {'x0': node['x0'], 'x1': mx,   'y0': my, 'y1': node['y1'],   'kps': []},
                    {'x0': mx,         'x1': node['x1'], 'y0': my, 'y1': node['y1'], 'kps': []},
                ]
                for kp in node['kps']:
                    x, y = kp.pt
                    ci = (1 if x >= mx else 0) + (2 if y >= my else 0)
                    children[ci]['kps'].append(kp)
                for child in children:
                    if child['kps']:
                        new_nodes.append(child)

            nodes = new_nodes
            if len(nodes) == prev_size:
                break  # Can't split further

        # Keep best keypoint per node
        result = []
        for node in nodes:
            best = max(node['kps'], key=lambda kp: kp.response)
            result.append(best)

        # If too many, keep top-n by response
        if len(result) > n_target:
            result.sort(key=lambda kp: kp.response, reverse=True)
            result = result[:n_target]

        return result

    def _compute_descriptors(self, pyramid, all_keypoints):
        """
        Compute ORB descriptors using ORB-SLAM3's bit_pattern_31_ (vectorized).
        """
        final_kps = []
        final_descs = []

        # Precompute pattern as (256, 2, 2): [pair_idx, point(0/1), (x,y)]
        pat = _PATTERN.reshape(256, 2, 2)  # (256, 2, 2)
        px = pat[:, :, 0]  # (256, 2)
        py = pat[:, :, 1]  # (256, 2)

        for level in range(self.n_levels):
            kps_level = all_keypoints[level]
            if not kps_level:
                continue

            working = cv2.GaussianBlur(pyramid[level], (7, 7), 2,
                                       borderType=cv2.BORDER_REFLECT_101)
            h, w = working.shape
            scale = self.inv_scale_factors[level]

            n_kps = len(kps_level)
            descs = np.zeros((n_kps, 32), dtype=np.uint8)

            # Batch all keypoints
            angles_rad = np.array([np.radians(kp.angle) for kp in kps_level], dtype=np.float32)
            a = np.cos(angles_rad)  # (N,)
            b = np.sin(angles_rad)  # (N,)

            cx = np.array([int(round(kp.pt[0] * scale + EDGE_THRESHOLD)) for kp in kps_level])
            cy = np.array([int(round(kp.pt[1] * scale + EDGE_THRESHOLD)) for kp in kps_level])

            # For each pattern pair, compute rotated offsets for all keypoints
            # px0[i], py0[i] → rotate by each keypoint's angle
            # rx = round(px*a - py*b), ry = round(px*b + py*a)
            # shape: (256, 2, N)
            px_bc = px[:, :, None]  # (256,2,1)
            py_bc = py[:, :, None]  # (256,2,1)
            a_bc  = a[None, None, :]  # (1,1,N)
            b_bc  = b[None, None, :]  # (1,1,N)

            rx = np.round(px_bc * a_bc - py_bc * b_bc).astype(np.int32)  # (256,2,N)
            ry = np.round(px_bc * b_bc + py_bc * a_bc).astype(np.int32)  # (256,2,N)

            # Absolute pixel positions
            xs = cx[None, None, :] + rx  # (256,2,N)
            ys = cy[None, None, :] + ry  # (256,2,N)

            # Clip to image bounds
            xs = np.clip(xs, 0, w - 1)
            ys = np.clip(ys, 0, h - 1)

            # Sample pixel values: (256, 2, N)
            vals = working[ys, xs].astype(np.int32)  # (256,2,N)

            # bits[i, n] = (vals[i,0,n] < vals[i,1,n])
            bits = (vals[:, 0, :] < vals[:, 1, :]).astype(np.uint8)  # (256, N)

            # Pack 8 bits → 1 byte: bits[0:8] → byte 0, etc.
            bits_T = bits.T  # (N, 256)
            for byte_i in range(32):
                byte_bits = bits_T[:, byte_i*8:(byte_i+1)*8]  # (N, 8)
                powers = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
                descs[:, byte_i] = (byte_bits * powers).sum(axis=1).astype(np.uint8)

            final_kps.extend(kps_level)
            final_descs.append(descs)

        if not final_descs:
            return [], None

        return final_kps, np.vstack(final_descs)
