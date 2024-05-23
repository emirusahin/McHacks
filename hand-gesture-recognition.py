import cv2
import mediapipe as mp


class HandGestureDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=max_num_hands,
                                         min_detection_confidence=min_detection_confidence,
                                         min_tracking_confidence=min_tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

    def is_finger_open(self, landmark_list, finger_tip_idx, finger_bottom_idx):
        if finger_tip_idx == 4:
            return self.is_thumb_open(landmark_list, finger_tip_idx, finger_bottom_idx)
        return landmark_list[finger_tip_idx].y < landmark_list[finger_bottom_idx].y

    def is_thumb_open(self, landmark_list, thumb_tip_idx=4, thumb_mcp_idx=2, reference_idx=0):
        # Thumb tip and MCP joint
        thumb_tip = landmark_list[thumb_tip_idx]
        thumb_mcp = landmark_list[thumb_mcp_idx]

        # Reference point (wrist or index finger MCP)
        reference_point = landmark_list[reference_idx]

        # For right hand, check if thumb tip is to the right of the MCP joint when viewed from the wrist/index MCP
        # For left hand, the condition would be inverted
        if reference_point.x < thumb_mcp.x:  # Assuming right hand
            return thumb_tip.x > thumb_mcp.x
        else:  # Assuming left hand
            return thumb_tip.x < thumb_mcp.x

    def count_open_fingers(self, landmarks):
        # Make it so that this function considers that thumbs are still higher than the knuckles even when closed

        thumb_tip_idx, index_tip_idx, middle_tip_idx, ring_tip_idx, pinky_tip_idx = 4, 8, 12, 16, 20
        thumb_base_idx, index_bottom_idx, middle_bottom_idx, ring_bottom_idx, pinky_bottom_idx = 1, 5, 9, 13, 17

        landmark_list = landmarks.landmark
        fingers_open = [
            self.is_finger_open(landmark_list, thumb_tip_idx, thumb_tip_idx - 2),
            self.is_finger_open(landmark_list, index_tip_idx, index_bottom_idx),
            self.is_finger_open(landmark_list, middle_tip_idx, middle_bottom_idx),
            self.is_finger_open(landmark_list, ring_tip_idx, ring_bottom_idx),
            self.is_finger_open(landmark_list, pinky_tip_idx, pinky_bottom_idx)
        ]

        return fingers_open.count(True)

    # def run(self):
    #     cap = cv2.VideoCapture(0)
    #
    #     while cap.isOpened():
    #         success, image = cap.read()
    #         if not success:
    #             print("Ignoring empty camera frame.")
    #             continue
    #
    #         image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    #         image.flags.writeable = False
    #         results = self.hands.process(image)
    #
    #         image.flags.writeable = True
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #
    #         if results.multi_hand_landmarks:
    #             for hand_landmarks in results.multi_hand_landmarks:
    #                 self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
    #                 open_fingers = self.count_open_fingers(hand_landmarks)
    #                 # gesture = 'Open Hand' if open_fingers == 5 else str(open_fingers)
    #                 if open_fingers == 5:
    #                     gesture = 'Open Hand'
    #                 elif open_fingers == 0:
    #                     gesture = 'Closed Hand'
    #                 else:
    #                     gesture = str(open_fingers)
    #                 cv2.putText(image, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #
    #         cv2.imshow('Hand Gesture Recognizer', image)
    #
    #         if cv2.waitKey(5) & 0xFF == ord('q'):
    #             break

    def run(self):
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1960)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    open_fingers = self.count_open_fingers(hand_landmarks)

                    if open_fingers == 5:
                        gesture = 'Open Hand'
                    elif open_fingers == 0:
                        gesture = 'Closed Hand'
                    else:
                        gesture = str(open_fingers)

                    # Determine the hand label (left or right)
                    hand_label = "Right" if results.multi_handedness[hand_idx].classification[
                                                0].label == 'Right' else "Left"

                    # Adjust text position based on hand label
                    text_origin_x = 10 if hand_label == "Left" else image.shape[1] - 350
                    cv2.putText(image, f"{hand_label} : {gesture}", (text_origin_x, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Hand Gesture Recognizer', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        self.hands.close()
        cap.release()
        cv2.destroyAllWindows()


# Example of usage
if __name__ == "__main__":
    detector = HandGestureDetector()
    detector.run()