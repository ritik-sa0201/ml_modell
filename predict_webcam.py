import cv2
from ultralytics import YOLO

# Set up the phone webcam as the video source
camera_index = 2  # Change to the correct index for your external phone webcam
cap = cv2.VideoCapture(camera_index)

# Check if the webcam is successfully opened
if not cap.isOpened():
    print("Error: Unable to access the external phone webcam.")
    exit(1)

# Load the YOLO model
model_path = r'.\runs\detect\train19\weights\last.pt'  # Update with your model path
model = YOLO(model_path)

threshold = 0.5  # Confidence threshold for detection

print("Press 'q' to quit.")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame from webcam.")
        break

    # Run the YOLO model on the frame
    results = model(frame)[0]

    # Process the detection results
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Draw bounding boxes
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

            # Prepare label text with confidence score
            label = f"{results.names[int(class_id)].upper()} {score:.2f}"

            # Draw the label with confidence score
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

    # Display the resulting frame with detections
    cv2.imshow("YOLO Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
