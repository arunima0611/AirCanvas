import cv2
import numpy as np
import mediapipe as mp
import math
import time
# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------- KEYBOARD LAYOUT ----------------
keys = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["Z","X","C","V","B","N","M"],
    ["SPACE","BACK"]
]

key_width = 70
key_height = 70
start_x = 50
start_y = 150

typed_text = ""
last_click_time = 0
click_delay = 0.6

# ---------------- DRAW KEYBOARD ----------------
def draw_keyboard(frame):
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            x = start_x + j * (key_width + 10)
            y = start_y + i * (key_height + 10)

            cv2.rectangle(frame, (x, y), 
                          (x + key_width, y + key_height),
                          (255, 0, 0), 2)

            cv2.putText(frame, key, 
                        (x + 10, y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

# ---------------- CHECK KEY PRESS ----------------
def check_key_press(ix, iy):
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            x = start_x + j * (key_width + 10)
            y = start_y + i * (key_height + 10)

            if x < ix < x + key_width and y < iy < y + key_height:
                return key
    return None

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    draw_keyboard(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            lm = hand.landmark

            # Index finger
            ix = int(lm[8].x * w)
            iy = int(lm[8].y * h)

            # Thumb
            tx = int(lm[4].x * w)
            ty = int(lm[4].y * h)

            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)

            # Distance between thumb and index
            distance = math.hypot(tx - ix, ty - iy)

            if distance < 40:
                current_time = time.time()

                if current_time - last_click_time > click_delay:
                    key = check_key_press(ix, iy)

                    if key:
                        if key == "SPACE":
                            typed_text += " "
                        elif key == "BACK":
                            typed_text = typed_text[:-1]
                        else:
                            typed_text += key

                        last_click_time = current_time

    # ---------------- DISPLAY TEXT ----------------
    cv2.rectangle(frame, (50, 50), (1100, 120), (0, 0, 0), -1)
    cv2.putText(frame, typed_text,
                (60, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 3)

    cv2.imshow("AI Air Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()