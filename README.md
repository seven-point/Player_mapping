# Cross-Camera Player Mapping

This project maps player identities between two video feeds (broadcast and tacticam) using a fine-tuned YOLOv11 model. It detects players, tracks them in each video, extracts visual features, and maps player IDs across views.

## Setup

1. Create a virtual environment (optional):
    >>python -m venv venv
    >>source venv/bin/activate  # or venv\Scripts\activate on Windows

2. Install dependencies:
    >>pip install -r requirements.txt

3. Place the following in the root folder:
    - `broadcast.mp4`
    - `tacticam.mp4`
    - Your YOLOv11 model from Google Drive (or let it auto-download using gdown)

## Run

>>python player_mapper.py


## Folder Structure

cross_camera_player_mapping/

├── broadcast.mp4

├── tacticam.mp4

├── broadcast_out.mp4 (the output video with anchor boxes)

├── tacticam_out.mp4  (the output video with anchor boxes)    

├── yolov11_model.pt  (or download via Google Drive)

├── player_mapper.py

├── requirements.txt

├── utils/

│   ├── detector.py

│   ├── tracker.py

│   └── features.py


