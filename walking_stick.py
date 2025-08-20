import cv2
from ultralytics import YOLO
from gtts import gTTS
import os
import threading
import queue
import time
import tempfile
from playsound import playsound
import pygame
import streamlit as st
# ----------------------------
# Initialize YOLO
# ----------------------------
model = YOLO("yolov8n.pt")  # lightweight YOLOv8

# ----------------------------
# Initialize TTS with Queue (using gTTS instead of pyttsx3)
# ----------------------------
speech_queue = queue.Queue()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)


def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:  # exit signal
            break

        try:
            filename = "voice.mp3"
            tts = gTTS(text=text, lang="en")
            tts.save(filename)

            pygame.mixer.init()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue
            os.remove(filename)
        except Exception as e:
            print(f"TTS error: {e}")

        speech_queue.task_done()

# Start background speech thread
threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    # Create a unique temp file for each speech
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts = gTTS(text=text, lang='en')
        tts.save(tmp_file.name)
        temp_filename = tmp_file.name

    # Play the file
    

    pygame.mixer.music.load(temp_filename)
    pygame.mixer.music.play()

    # Wait until playback finishes
    while pygame.mixer.music.get_busy():
       pygame.time.Clock().tick(2)

    # Clean up
    
    try:    
        os.remove(temp_filename)
    except Exception as e:
            print(f"TTS Voice played")

# ----------------------------
# Detection function
# ----------------------------

def detect_frame(frame,alerted):
    
    results = model(frame, imgsz=640, verbose=False)
    st.image(results[0].plot(), channels="BGR", caption="YOLO Detection")
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
            print(f"Alert before if loop: {alerted}")
            if label not in frame_alerted_labels and alerted ==False:
                speak(f"{label} detected ahead")
                frame_alerted_labels.add(label)
                print(f"Alert: {label}")
                alerted = True
                print(f"Alert: {alerted}")

    return frame

# ----------------------------
# Video input
# ----------------------------
def main():
    video_path = "Demo_Vid.mp4"   
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        exit()

    print("Processing video... Press 'q' to quit.")
    alerted = False
    while not alerted :
        ret, frame = cap.read()
        if not ret:
            break
        
        output_frame = detect_frame(frame,alerted)
        alerted=True
        cv2.imshow("YOLO Object Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Stop the speech thread cleanly
    speech_queue.put(None)


if __name__ == "__main__":
    main()
