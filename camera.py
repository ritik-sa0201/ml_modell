import cv2

# Set up the camera index (0 for default, 1 or higher for external cameras)
camera_index = 2 # Change this if it doesn't detect your external camera

# Initialize the video capture with the external camera
cap = cv2.VideoCapture(camera_index)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open external camera.")
    exit()

# Set optional parameters (e.g., resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

print("Press 'q' to exit the video stream.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Display the resulting frame
    cv2.imshow("External Camera Output", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the display window
cap.release()
cv2.destroyAllWindows()
