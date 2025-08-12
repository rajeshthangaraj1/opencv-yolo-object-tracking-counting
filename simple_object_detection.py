import cv2
from ultralytics import YOLO

model= YOLO("yolov8n.pt")  # Load the YOLOv8 model
image = cv2.imread("image_data/gym_image.jpeg")  # Read the input image
results = model(image)  # Perform inference
annotated_image = results[0].plot()  # Get the annotated image
cv2.imshow("Annotated Image", annotated_image)  # Display the annotated image
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the image window
# Save the annotated image
