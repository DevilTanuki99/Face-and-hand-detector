import cv2
import numpy as np
import mediapipe as mp




# flip cam
def flip_cam(frame):
    return cv2.flip(frame, 1)

# face detector
def detect_and_draw_face(frame, face_cascade, rectangle_color=(0, 0, 255), thickness=2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50), maxSize=(250, 250))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, thickness)
        cv2.putText(frame, 'Ong Ganteng', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rectangle_color, 2, cv2.LINE_AA)
    return frame

# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_fingers(frame, hands_module):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_module.process(rgb)
    finger_positions = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=5),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            finger_tip = hand_landmarks.landmark[8]
            finger_pos = (
                int(finger_tip.x * frame.shape[1]),
                int(finger_tip.y * frame.shape[0])
            )
            finger_positions.append(finger_pos)

    return finger_positions

# Floating Button in Camera
def draw_button(frame, button_x, button_y, button_width, button_height, button_status):
    # Button color and text based on status
    if button_status == 'ON':
        button_color = (0, 255, 0)  # Green
        button_text = 'ON'
    else:
        button_color = (0, 0, 255)  # Red
        button_text = 'OFF'
    
    # Drawing button on the frame
    cv2.rectangle(
        frame, 
        (button_x, button_y), 
        (button_x + button_width, button_y + button_height), 
        button_color, 
        -1
        )
    cv2.putText(
        frame, 
        button_text, 
        (button_x + 70, button_y + button_height // 2), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (255, 255, 255), 
        2, 
        cv2.LINE_AA)
    



# main function
def main():
    cam = cv2.VideoCapture(0)


    # Window setup
    window_name = "FaceCam"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1400, 800)


    # face setup
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("File face_cascade gak ketemu!")
        return

    if not cam.isOpened():
        print("Kamera gak kebuka banh!")
        return


    # Hand setup
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9
    ) as hands:
        

        while True:
            # frame setup
            ret, frame = cam.read()
            if not ret:
                print("Frame error!")
                break

            frame = flip_cam(frame)
            frame = detect_and_draw_face(frame, face_cascade)

            # Define button position
            button_status = 'OFF'
            button_x, button_y = 300, 250
            button_width, button_height = 200, 100

            # Draw button on the frame
            draw_button(frame, button_x, button_y, button_width, button_height, button_status)

            # Detect fingers
            finger_positions = detect_fingers(frame, hands)

            # Check if finger is touching the button
            for finger in finger_positions:
                fx, fy = finger
                if button_x < fx < button_x + button_width and button_y < fy < button_y + button_height:
                    cv2.putText(frame, 
                                'Button ON', 
                                (500, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (0, 255, 0), 
                                2, 
                                cv2.LINE_AA
                                )
                    cv2.rectangle(frame, 
                                  (button_x, button_y), 
                                  (button_x + button_width, button_y + button_height), 
                                  (0, 255, 0), 
                                  -1
                                  )
                    cv2.putText(frame, 
                                'ON', 
                                (button_x + 70, button_y + button_height // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, 
                                (255, 255, 255), 
                                2, 
                                cv2.LINE_AA
                                )

            # Show frame with button
            cv2.imshow(window_name, frame)

            # Quit button
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()