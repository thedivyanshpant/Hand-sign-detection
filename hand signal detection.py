import cv2
import mediapipe as mp

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Start video capture (you can also use a webcam)
cap = cv2.VideoCapture(0)

# Define ASL gestures and their meanings
asl_gestures = {
    'A': 'Divyansh',
    'B': 'Akash',
    'C': 'Sarthak',
    'T': 'Thank You',
    'D': 'Dog',
    'E': 'Elephant'
    # Add more gestures and meanings as needed
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (Mediapipe requires RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(rgb_frame)

    # Draw landmarks and connections if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark positions
            landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in hand_landmarks.landmark]

            # Check if any gesture is detected
            detected_gesture = 'Unknown'

            # Extract thumb and index finger tips
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]

            # Calculate thumb to index finger distance
            thumb_to_index_distance = abs(thumb_tip[0] - index_tip[0]) + abs(thumb_tip[1] - index_tip[1])

            # Detect gestures using a dictionary for better readability
            gestures = {
                'A': thumb_to_index_distance < 50,
                'B': thumb_tip[1] < landmarks[6][1] and thumb_tip[1] < landmarks[10][1],
                'C': thumb_tip[1] > landmarks[8][1],
                'T': thumb_tip[0] < landmarks[5][0] and thumb_tip[1] > landmarks[5][1],
                'D': thumb_tip[1] > landmarks[5][1],
                'E': thumb_tip[0] < landmarks[9][0]
            }

            for gesture, condition in gestures.items():
                if condition:
                    detected_gesture = gesture
                    break

            # Get the meaning of the recognized ASL gesture or display 'Unknown' if not recognized
            gesture_meaning = asl_gestures.get(detected_gesture, 'Unknown')
            cv2.putText(frame, f'ASL {detected_gesture} ({gesture_meaning})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw landmarks on frame (for visualization)
            for landmark in landmarks:
                cv2.circle(frame, landmark, 5, (0, 255, 0), -1)

    else:
        # No hands detected
        cv2.putText(frame, 'No hands detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('ASL Gesture Recognition', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()