import cv2
import mediapipe as mp

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing.
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam.
cap = cv2.VideoCapture(0)


def detect_hand_movement(landmarks, width, height):
    # Retrieve specific landmarks.
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    # Convert landmark positions to pixel values.
    rs = (int(right_shoulder.x * width), int(right_shoulder.y * height))
    re = (int(right_elbow.x * width), int(right_elbow.y * height))
    rw = (int(right_wrist.x * width), int(right_wrist.y * height))

    # Initialize movement text.
    movement = ""

    # Detect hand movements based on landmark positions.
    if rw[1] < re[1] < rs[1]:
        movement = "Hand Raised"
    elif re[1] < rw[1] and rs[1] < rw[1]:
        movement = "Hand Lowered"

    if rw[0] > re[0] > rs[0]:
        movement += " & Moved Right"
    elif rw[0] < re[0] and rs[0] > rw[0]:
        movement += " & Moved Left"

    return movement


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Draw the pose annotation on the image.
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if results.pose_landmarks:
        width, height, _ = frame.shape
        movement_text = detect_hand_movement(results.pose_landmarks.landmark, width, height)

        # Display the movement text on the screen.
        if movement_text:
            cv2.putText(frame, movement_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame.
    cv2.imshow('MediaPipe Pose', frame)

    # Press 'q' to exit the loop.
    if cv2.waitKey(5) & 0xFF == 113:  # ASCII for 'q'
        break

cap.release()
cv2.destroyAllWindows()
