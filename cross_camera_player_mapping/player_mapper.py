# import cv2
# import os
# import gdown
# import numpy as np
# from utils.detector import PlayerDetector
# from utils.features import FeatureExtractor
# from utils.tracker import CentroidTracker
# from sklearn.metrics.pairwise import cosine_similarity

# # Download model from Google Drive if not present
# model_url = "https://drive.google.com/uc?id=1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD"
# model_path = "yolov11_model.pt"

# if not os.path.exists(model_path):
#     print("Downloading YOLOv11 model...")
#     gdown.download(model_url, model_path, quiet=False)

# def process_video(video_path, detector, extractor, tracker):
#     cap = cv2.VideoCapture(video_path)
#     features_dict = {}
#     while True:
#         ret, frame = cap.read()
#         if not ret > 20:
#             break
#         detections = detector.detect_players(frame)
#         tracked = tracker.update(detections)
#         for track_id, bbox in tracked.items():
#             x1, y1, x2, y2, _ = bbox
#             crop = frame[y1:y2, x1:x2]
#             if crop.size == 0:
#                 continue
#             feat = extractor.extract(crop)
#             features_dict.setdefault(track_id, []).append(feat)
#     for tid in features_dict:
#         features_dict[tid] = np.mean(features_dict[tid], axis=0)
#     return features_dict

# def match_players(broadcast_feats, tacticam_feats):
#     broadcast_ids = list(broadcast_feats.keys())
#     tacticam_ids = list(tacticam_feats.keys())
#     broadcast_matrix = np.array([broadcast_feats[i] for i in broadcast_ids])
#     tacticam_matrix = np.array([tacticam_feats[i] for i in tacticam_ids])
#     similarity = cosine_similarity(tacticam_matrix, broadcast_matrix)
#     matches = {}
#     for i, row in enumerate(similarity):
#         best_match = broadcast_ids[np.argmax(row)]
#         matches[tacticam_ids[i]] = best_match
#     return matches

# if __name__ == "__main__":
#     detector = PlayerDetector(model_path)
#     extractor = FeatureExtractor()
#     print("Processing broadcast video...")
#     broadcast_feats = process_video("broadcast.mp4", detector, extractor, CentroidTracker())
#     print("Processing tacticam video...")
#     tacticam_feats = process_video("tacticam.mp4", detector, extractor, CentroidTracker())
#     matches = match_players(broadcast_feats, tacticam_feats)
#     print("\nTacticam to Broadcast Player Mapping:")
#     for t_id, b_id in matches.items():
#         print(f"Tacticam ID {t_id} → Broadcast ID {b_id}")

import os
import cv2
import gdown
import numpy as np
from ultralytics import YOLO
from utils.detector import PlayerDetector
from utils.tracker import CentroidTracker
from utils.features import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

model_url = "https://drive.google.com/uc?id=1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD"
model_path = "yolov11_model.pt"

if not os.path.exists(model_path):
    print("Downloading YOLOv11 model...")
    gdown.download(model_url, model_path, quiet=False)

def draw_tracks(frame, tracked_players, color=(0, 255, 0)):
    for track_id, (x1, y1, x2, y2, _) in tracked_players.items():
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def process_video(video_path, detector, extractor, tracker, save_path):
    cap = cv2.VideoCapture(video_path)
    features_dict = {}
    out_writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > 20:
            break

        detections = detector.detect_players(frame)
        # if frame_count < 5:
        #      print(f"[Frame {frame_count}] Detections: {detections}")
        tracked = tracker.update(detections)
        draw_tracks(frame, tracked)

        for track_id, bbox in tracked.items():
            x1, y1, x2, y2, _ = bbox
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            feat = extractor.extract(crop)
            features_dict.setdefault(track_id, []).append(feat)

        if out_writer is None and frame is not None:
            h, w = frame.shape[:2]
            out_writer = cv2.VideoWriter(save_path, fourcc, 20.0, (w, h))
        if out_writer:
            out_writer.write(frame)

        frame_count += 1

    cap.release()
    if out_writer:
        out_writer.release()

    # Average features per track_id
    for tid in features_dict:
        features_dict[tid] = np.mean(features_dict[tid], axis=0)
    return features_dict

if __name__ == "__main__":
    detector = PlayerDetector(model_path, player_class_id=2)
    extractor = FeatureExtractor()
    tracker = CentroidTracker()

    print("Processing broadcast video...")
    broadcast_feats = process_video("broadcast.mp4", detector, extractor, tracker, "broadcast_out.mp4")

    print("Processing tacticam video...")
    tacticam_feats = process_video("tacticam.mp4", detector, extractor, tracker, "tacticam_out.mp4")

    print("Computing similarity matrix...")
    broadcast_ids = list(broadcast_feats.keys())
    tacticam_ids = list(tacticam_feats.keys())

    sim_matrix = np.zeros((len(tacticam_ids), len(broadcast_ids)))
    for i, tid in enumerate(tacticam_ids):
        for j, bid in enumerate(broadcast_ids):
            sim = cosine_similarity([tacticam_feats[tid]], [broadcast_feats[bid]])[0][0]
            sim_matrix[i, j] = sim

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    print("Mapped Players (Tacticam → Broadcast):")
    for r, c in zip(row_ind, col_ind):
        print(f"Tacticam ID {tacticam_ids[r]} → Broadcast ID {broadcast_ids[c]}")
