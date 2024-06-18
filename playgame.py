import cv2
import numpy as np
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Get the screen size
screen_width, screen_height = pyautogui.size()

# Initialize video capture
video = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False,
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.75,
                    model_complexity=1) as hands:

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Flip and convert the image to RGB
        image = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_image)
        
        # Convert back to BGR for OpenCV
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
                
                # Get the index fingertip position
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Convert normalized coordinates to pixel coordinates
                h, w, _ = image.shape
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                # Draw a circle at the index fingertip
                cv2.circle(image, (cx, cy), 10, (0, 255, 0), -1)
                
                # Move the mouse cursor
                screen_x = np.interp(index_finger_tip.x, [0, 1], [0, screen_width])
                screen_y = np.interp(index_finger_tip.y, [0, 1], [0, screen_height])
                pyautogui.moveTo(screen_x, screen_y)
        
        # Display the image
        cv2.imshow('Play game', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
