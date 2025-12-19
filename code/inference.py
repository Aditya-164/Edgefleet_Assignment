import cv2
import pandas as pd
from ultralytics import YOLO
import os
import glob
import math

MODEL_PATH = os.path.join("..", "models", "cricket_ball_refined", "weights", "best.pt")

INPUT_FOLDER = os.path.join("..", "test_videos")
OUTPUT_VIDEO_DIR = os.path.join("..", "results")
OUTPUT_CSV_DIR = os.path.join("..", "annotations")

def process_single_video(model, video_path):
    video_name = os.path.basename(video_path)
    print(f"Processing: {video_name}...")

    output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"tracked_{video_name}")
    output_csv_path = os.path.join(OUTPUT_CSV_DIR, f"{os.path.splitext(video_name)[0]}.csv")

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    trajectory = []
    csv_data = []
    frame_idx = 0
    
    # Storing history for smoothing
    # Format: {'frame': 0, 'x': 100, 'y': 100}
    history_buffer = [] 

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. LOWER CONFIDENCE to 0.15
        results = model.predict(frame, conf=0.15, imgsz=512, classes=[0], verbose=False)

        detected = False
        cx, cy = -1, -1
        
        # 2. Find best ball
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            curr_x, curr_y = (x1 + x2)/2, (y1 + y2)/2
            if len(history_buffer) > 0:
                last_valid = history_buffer[-1]
                dist = math.sqrt((curr_x - last_valid['x'])**2 + (curr_y - last_valid['y'])**2)
                
                frame_diff = frame_idx - last_valid['frame']
                if dist < (300 * frame_diff): 
                    cx, cy = curr_x, curr_y
                    detected = True
            else:
                cx, cy = curr_x, curr_y
                detected = True

        # 3. Handle Data
        if detected:
            history_buffer.append({'frame': frame_idx, 'x': cx, 'y': cy})
            trajectory.append((int(cx), int(cy)))
            cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), -1)
        else:
            # INTERPOLATION LOGIC (Fill gaps)
            # If we missed the ball for < 5 frames, predicting where it is
            if len(history_buffer) > 1:
                last = history_buffer[-1]
                second_last = history_buffer[-2]
                
                # Check how long ago we lost it
                frames_lost = frame_idx - last['frame']
                
                if frames_lost < 5:
                    # Calculate velocity from previous 2 points
                    vx = (last['x'] - second_last['x']) / (last['frame'] - second_last['frame'])
                    vy = (last['y'] - second_last['y']) / (last['frame'] - second_last['frame'])
                    
                    # Predict current pos
                    pred_x = last['x'] + (vx * frames_lost)
                    pred_y = last['y'] + (vy * frames_lost)
                    
                    cv2.circle(frame, (int(pred_x), int(pred_y)), 4, (0, 255, 255), -1)
                    
        # Draw Trajectory
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i-1], trajectory[i], (0, 0, 255), 2)

        csv_data.append([frame_idx, round(cx, 1), round(cy, 1), 1 if detected else 0])
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    
    pd.DataFrame(csv_data, columns=['frame_index', 'x_centroid', 'y_centroid', 'visibility_flag']).to_csv(output_csv_path, index=False)
    print(f"Finished {video_name}")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Waiting for training... {MODEL_PATH} not found.")
        return
        
    model = YOLO(MODEL_PATH)
    video_files = []
    for ext in ['*.mp4', '*.mov', '*.avi']:
        video_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    for v in video_files:
        process_single_video(model, v)

if __name__ == "__main__":
    main()