import numpy as np
from scipy.spatial.distance import cdist

class CentroidTracker:
    def __init__(self, max_dist=50):
        self.objects = {}
        self.next_id = 0
        self.max_dist = max_dist

    def update(self, detections):
        if len(self.objects) == 0:
            for det in detections:
                self.objects[self.next_id] = det
                self.next_id += 1
            return self.objects

        if len(detections) == 0:
            return self.objects        

        obj_ids = list(self.objects.keys())
        obj_centroids = [self._centroid(bbox) for bbox in self.objects.values()]
        new_centroids = [self._centroid(bbox) for bbox in detections]

        if not obj_centroids or not new_centroids:
            return self.objects

        D = cdist(np.array(obj_centroids), np.array(new_centroids))
        rows, cols = np.where(D < self.max_dist)
        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            self.objects[obj_ids[row]] = detections[col]
            used_rows.add(row)
            used_cols.add(col)

        for i, det in enumerate(detections):
            if i not in used_cols:
                self.objects[self.next_id] = det
                self.next_id += 1
        return self.objects

    def _centroid(self, bbox):
        x1, y1, x2, y2, _ = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
