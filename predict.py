import os
from ultralytics import YOLO
import cv2

# Define the path to your photo
photo_path = r'D:\\image_rec_model1\dbl_test\test5.jpeg'  # Path to the single image
output_path = r'D:\\image_rec_model1\dbl_test\orange_0204_out.jpeg'  # Path to save the processed image

# Path to the trained YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train19', 'weights', 'last.pt')

# Load the YOLO model
model = YOLO(model_path)  # Load a custom model

threshold = 0.5  # Confidence threshold for detection
iou_threshold = 0.2  # IoU threshold

# Read the image
frame = cv2.imread(photo_path)
if frame is None:
    print(f"Error: Unable to read image {photo_path}")
    exit(1)

# Get the original dimensions of the image
original_h, original_w, _ = frame.shape

# Resize the image to 640x640
frame_resized = cv2.resize(frame, (640, 640))

# Run the model on the resized image
results = model(frame_resized)[0]

# Function to calculate IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Compute intersection coordinates
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    # Compute intersection area
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0

    return inter_area / union_area

# Filter detections based on IoU and confidence threshold
final_boxes = []
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        # Scale bounding box coordinates back to the original image size
        x1 = int(x1 * original_w / 640)
        y1 = int(y1 * original_h / 640)
        x2 = int(x2 * original_w / 640)
        y2 = int(y2 * original_h / 640)

        # Check IoU with existing boxes
        should_add = True
        for existing_box in final_boxes:
            if calculate_iou((x1, y1, x2, y2), existing_box[:4]) > iou_threshold:
                should_add = False
                break

        if should_add:
            final_boxes.append((x1, y1, x2, y2, score, class_id))

# Draw bounding boxes and labels for remaining detections
for x1, y1, x2, y2, score, class_id in final_boxes:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    label = f"{results.names[int(class_id)].upper()} {score:.2f}"  # Label with confidence score
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )
    print(f"Detected {results.names[int(class_id)].upper()} with confidence: {score:.2f}")

# Save the processed image
cv2.imwrite(output_path, frame)
print(f"Processed image saved to: {output_path}")
