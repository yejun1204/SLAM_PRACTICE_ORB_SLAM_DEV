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

        # cv2 ORB for descriptor computation (uses its own pattern)
        self._orb = cv2.ORB_create(nfeatures=n_features * 2,
                                   scaleFactor=scale_factor,
                                   nlevels=n_levels,
                                   edgeThreshold=EDGE_THRESHOLD,
                                   patchSize=PATCH_SIZE)

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
        # ORB-SLAM3: resize only (no blur). blur happens later before descriptor computation.
        # Also add EDGE_THRESHOLD border (REFLECT_101) so FAST can detect near edges.
        pyramid = []
        for level in range(self.n_levels):
            scale = self.inv_scale_factors[level]
            h = round(image.shape[0] * scale)
            w = round(image.shape[1] * scale)
            if level == 0:
                resized = image
            else:
                resized = cv2.resize(pyramid[level - 1],
                                     (w, h), interpolation=cv2.INTER_LINEAR)
            # Add border so FAST can detect corners near image edges
            padded = cv2.copyMakeBorder(resized,
                                        EDGE_THRESHOLD, EDGE_THRESHOLD,
                                        EDGE_THRESHOLD, EDGE_THRESHOLD,
                                        cv2.BORDER_REFLECT_101)
            pyramid.append(padded)
        return pyramid

    def _detect_fast(self, img, level):
        """Detect FAST keypoints in 35x35 cells, then distribute via OctTree.
        img is the padded image (EDGE_THRESHOLD border on all sides).
        """
        h, w = img.shape
        # The actual image starts at (EDGE_THRESHOLD, EDGE_THRESHOLD) inside padded img
        # Detection region excludes 3px from the inner border
        min_x = EDGE_THRESHOLD - 3
        min_y = EDGE_THRESHOLD - 3
        max_x = w - EDGE_THRESHOLD + 3
        max_y = h - EDGE_THRESHOLD + 3

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

        # Remove border offset, then scale back to level-0 space
        scale = self.scale_factors[level]
        for kp in distributed:
            x = (kp.pt[0] - EDGE_THRESHOLD) * scale
            y = (kp.pt[1] - EDGE_THRESHOLD) * scale
            kp.pt = (x, y)

        return distributed

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
        """Compute ORB descriptors using cv2 on each pyramid level."""
        final_kps = []
        final_descs = []

        for level in range(self.n_levels):
            kps_level = all_keypoints[level]
            if not kps_level:
                continue

            # Convert level-0 coords → level coords, then add border offset
            scale = self.inv_scale_factors[level]
            kps_scaled = []
            for kp in kps_level:
                kp2 = cv2.KeyPoint(
                    x=float(kp.pt[0] * scale + EDGE_THRESHOLD),
                    y=float(kp.pt[1] * scale + EDGE_THRESHOLD),
                    _size=float(kp.size * scale),
                    _angle=kp.angle,
                    _response=kp.response,
                    _octave=level
                )
                kps_scaled.append(kp2)

            # ORB-SLAM3: apply Gaussian blur on a working copy before descriptor computation
            working = cv2.GaussianBlur(pyramid[level], (7, 7), 2,
                                       borderType=cv2.BORDER_REFLECT_101)
            _, descs = self._orb.compute(working, kps_scaled)
            if descs is None or len(descs) == 0:
                continue

            # Restore level-0 coordinates
            for kp in kps_level[:len(descs)]:
                final_kps.append(kp)
            final_descs.append(descs)

        if not final_descs:
            return [], None

        descriptors = np.vstack(final_descs)
        return final_kps, descriptors
