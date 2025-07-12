from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path, player_class_id=2):
        self.model = YOLO(model_path)
        self.player_class_id = player_class_id

    def detect_players(self, frame):
        results = self.model(frame)[0]
        players = []
        for box in results.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            # print(f"Detected class={cls}, conf={conf:.2f}")
            if cls == self.player_class_id and conf > 0.91:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                if w * h < 1500:
                    continue
                players.append((x1, y1, x2, y2, conf))
        return players
