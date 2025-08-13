import cv2
from ultralytics import YOLO
import numpy

cap = cv2.VideoCapture("video_data/mineral-water.mp4")  # Open the default camera
model= YOLO("yolov8n.pt")  # Load the YOLOv8 model

unique_objects = set()  # Set to store unique object IDs

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break  # Break the loop if no frame is captured
    results = model.track(frame,classes=[39], persist=True, verbose=False)  # Perform inference on the captured frame
    annotated_frame = results[0].plot()  # Get the annotated frame
    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        for obj_id in ids:
            unique_objects.add(obj_id)
        cv2.putText(annotated_frame, f"Unique Objects: {len(unique_objects)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Object Counting", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
# Save the annotated frame if needed