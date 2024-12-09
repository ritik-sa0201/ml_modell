import os
from ultralytics import YOLO
import cv2

# Define the path to your video
VIDEOS_DIR = r'D:\\image_rec_model1\dbl_test'  # Use raw string for Windows path
video_path = os.path.join(VIDEOS_DIR, 'test_vid1.mp4')  # Specify your video file
video_path_out = os.path.join(VIDEOS_DIR, 'test_1_out.mp4')  # Output file

# Open the video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# Check if video was successfully loaded
if not ret:
    print(f"Error: Unable to read video from {video_path}")
    exit(1)

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Path to the trained YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train19', 'weights', 'last.pt')

# Load the YOLO model
model = YOLO(model_path)  # Load a custom model

threshold = 0.5  # Confidence threshold for detection

while ret:
    # Run the model on the frame
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

    # Write the modified frame to the output video
    out.write(frame)

    # Read the next frame
    ret, frame = cap.read()

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to: {video_path_out}")
