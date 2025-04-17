from ultralytics import YOLO
import cv2
import cvzone
import time
import pyttsx3
import torch
import threading

# Initialize Text-to-Speech
engine = pyttsx3.init()

def speak(text):
    threading.Thread(target=lambda: (engine.say(text), engine.runAndWait())).start()

# Constants for distance calculation
KNOWN_WIDTH = 60  # Adjust based on real object size (cm)
FOCAL_LENGTH = 800  # Adjust based on camera calibration

def calculate_distance(focal_length, known_width, width_in_frame):
    return (known_width * focal_length) / width_in_frame

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load a more accurate YOLO model and auto-detect GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO("yolov8m.pt").to(device)  # Use a more accurate model

# Class labels for YOLO
classNames = model.names

prev_frame_time = 0

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", 800, 800)

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read from webcam.")
        break

    # Preprocess the image to enhance detection accuracy
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Reduce noise
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)  # Improve contrast

    # Perform object detection with a higher confidence threshold
    results = model(img, conf=0.5)  # Only detect objects with confidence > 50%

    detected_objects = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = box.conf[0].item()  # Confidence score
            if conf < 0.5:  # Ignore low-confidence detections
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            distance = calculate_distance(FOCAL_LENGTH, KNOWN_WIDTH, w)

            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2)
            class_id = int(box.cls[0])
            class_name = classNames[class_id]
            label = f"{class_name} {distance:.2f} cm (Conf: {conf:.2f})"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detected_objects.append(f"{class_name} at {distance:.2f} cm")

    if detected_objects:
        detected_text = ', '.join(detected_objects)
        print(f"Detected: {detected_text}")
        speak(f"Detected: {detected_text}")

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User requested exit.")
        break

cap.release()
cv2.destroyAllWindows()
