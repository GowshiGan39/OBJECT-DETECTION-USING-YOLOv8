from ultralytics import YOLO
import cv2
import cvzone
import time
import pyttsx3

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

KNOWN_WIDTH = 60
FOCAL_LENGTH = 800

def calculate_distance(focal_length, known_width, width_in_frame):
    return (known_width * focal_length) / width_in_frame

video_path = "roadsafety.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

model = YOLO("yolov8n.pt")

classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer",
    "toothbrush"
]

prev_frame_time = 0

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", 800, 800)

while True:
    success, img = cap.read()
    if not success:
        print("End of video file or error reading the video.")
        break

    results = model(img, stream=True)

    detected_objects = []

    for r in results:
        boxes = r.boxes
        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            distance = calculate_distance(FOCAL_LENGTH, KNOWN_WIDTH, w)

            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2)
            class_id = int(box.cls[0])
            class_name = classNames[class_id]
            label = f"{class_name} {distance:.2f} cm"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detected_objects.append(f"{class_name} at {distance:.2f} cm")


    if detected_objects:
        detected_text = ', '.join(detected_objects)
        print(f"Detected: {detected_text}")
        speak(f"Detected: {detected_text}")

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
