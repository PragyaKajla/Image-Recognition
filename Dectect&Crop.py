import cv2
import os
import time
import torch
import numpy as np
import ultralytics
from ultralytics import YOLO


def run_model(video_path):
    model_name = "YOLOv8m-face"
    yolo_model = YOLO("yolov8m-face.pt")  # Keep the model file in the same folder

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("failed to open video")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    frame_count = 0
    faces_detected_total = 0
    processing_times = []
    total_frames_checked = 0
    CONFIDENCE_THRESHOLD = 0.4
    MOTION_THRESHOLD = 300

    output_folder = f"output_faces_{model_name.replace('-', '_')}"
    os.makedirs(output_folder, exist_ok=True)

    print(f"🚀 Running {model_name} on {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        motion_pixels = np.count_nonzero(fg_mask)
        if motion_pixels < MOTION_THRESHOLD:
            continue

        start_time = time.time()
        results = yolo_model(frame)[0]
        detections = [d for d in results.boxes.data if int(d[-1]) == 0 and float(d[4]) >= CONFIDENCE_THRESHOLD]
        end_time = time.time()

        detection_time = (end_time - start_time) * 1000
        processing_times.append(detection_time)
        total_frames_checked += 1
        faces_detected_total += len(detections)

        # Save cropped face only if one face is detected
        if len(detections) == 1:
            det = detections[0]
            x1, y1, x2, y2, conf, *_ = det.cpu().numpy()
            conf_percent = int(conf * 100)

            # Crop face
            face_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            frame_time = f"{round(timestamp, 2)}s".replace('.', '_')
            output_path = os.path.join(output_folder, f"face_{frame_time}_{conf_percent}pc.jpg")
            cv2.imwrite(output_path, face_crop)

    cap.release()
    cv2.destroyAllWindows()

    avg_time = np.mean(processing_times) if processing_times else 0
    avg_faces = faces_detected_total / total_frames_checked if total_frames_checked else 0

    return {
        "model": model_name,
        "frames_checked": total_frames_checked,
        "faces_detected": faces_detected_total,
        "avg_faces_per_frame": round(avg_faces, 2),
        "avg_time_ms": round(avg_time, 2)
    }


if __name__ == "__main__":
    video_path = "cctv.mp4"  # Replace with your actual video path
    result = run_model(video_path)
    print("\n📊 YOLOv8m Summary:")
    for k, v in result.items():
        print(f"{k}: {v}")

