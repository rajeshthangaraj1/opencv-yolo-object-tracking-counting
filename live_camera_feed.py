import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)  # Open the default camera
model= YOLO("yolov8n.pt")  # Load the YOLOv8 model

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break  # Break the loop if no frame is captured

    results = model(frame)  # Perform inference on the captured frame
    annotated_frame = results[0].plot()  # Get the annotated frame

    cv2.imshow("Live Camera Feed", annotated_frame)  # Display the annotated frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break
cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
# Save the annotated frame if needed