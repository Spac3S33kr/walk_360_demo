import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import queue
import time

# ----------------------------
# Initialize YOLO
# ----------------------------
model = YOLO("yolov8n.pt")  # lightweight YOLOv8

# ----------------------------
# Initialize TTS with Queue
# ----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:  # Exit signal
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start speech thread
threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    """Send text to the TTS queue (non-blocking)."""
    speech_queue.put(text)

# ----------------------------
# Detection function
# ----------------------------
def detect_frame(frame):
    results = model(frame, imgsz=640, verbose=False)

    frame_alerted_labels = set()  # Track per-frame announcements

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0].item())
            label = model.names[cls]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 🔊 Speak once per label per frame
            if label not in frame_alerted_labels:
                speak(f"{label} detected ahead")
                frame_alerted_labels.add(label)
                print(f"Alert: {label}")

    return frame

# ----------------------------
# Video input
# ----------------------------
def main():
    video_path = "demoNEW.mp4"   # 👈 update this
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        exit()

    print("Processing video... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = detect_frame(frame)
        cv2.imshow("YOLO Object Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Stop the speech thread cleanly
    speech_queue.put(None)