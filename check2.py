import cv2
from ultralytics import YOLO
from deepface import DeepFace
import threading
import queue

# Load the YOLOv8 model
model = YOLO('./pc1.pt')

# Path to DeepFace database
db_path = "./db"

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

# Queue for storing frames
frame_queue = queue.Queue(maxsize=1)  # Use a small queue size to avoid memory overflow
output_frame = None  # Global variable to hold the processed frame for display

# Frame processing parameters
recognition_interval = 5
frame_count = 0

def capture_frames():
    """Thread function to capture frames from webcam and add them to a queue."""
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Put frame in queue
            if not frame_queue.full():
                frame_queue.put(frame)
        else:
            break

def process_frames():
    """Thread function to process frames for face detection and recognition."""
    global output_frame, frame_count
    while True:
        # Get a frame from the queue
        frame = frame_queue.get()
        if frame is None:
            continue

        # YOLOv8 detection
        results = model(frame)
        
        # Verify if detections are made
        if results and len(results[0].boxes) > 0:
            for r in results:
                for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
                    # Convert box to integer coordinates
                    x1, y1, x2, y2 = map(int, box)

                    # Only proceed if confidence is above a threshold
                    if conf > 0.8:
                        # Crop the detected face
                        face = frame[y1:y2, x1:x2]
                        
                        # Resize face for faster processing
                        face = cv2.resize(face, (224, 224))

                        # Run DeepFace recognition only every few frames
                        if frame_count % recognition_interval == 0:
                            try:
                                # Use DeepFace for recognition
                                predictions = DeepFace.find(
                                    face, db_path=db_path, enforce_detection=False,
                                    model_name='VGG-Face', anti_spoofing=True
                                )

                                if predictions and not predictions[0].empty:
                                    # Extract and format the label
                                    name = predictions[0]['identity'][0].split('/')[-2]
                                    print(f"Recognized: {name}")

                                    # Draw bounding box and label
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, name, (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            except Exception as e:
                                print(f"DeepFace error: {e}")

        # Update output frame for display
        output_frame = frame
        frame_count += 1

# Start capture and processing threads
capture_thread = threading.Thread(target=capture_frames)
processing_thread = threading.Thread(target=process_frames)

capture_thread.start()
processing_thread.start()

# Display loop
while True:
    if output_frame is not None:
        # Display the processed frame
        cv2.imshow('Webcam Face Recognition', output_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
capture_thread.join()
processing_thread.join()

