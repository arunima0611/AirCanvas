import cv2
import numpy as np
import mediapipe as mp
import os
import time
import math

# ---------------- LOAD SLIDES ----------------
folder_path = "slides"
slide_files = sorted(os.listdir(folder_path))
current_slide = 0

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ---------------- VARIABLES ----------------
canvas = None
prev_x, prev_y = 0, 0
last_slide_time = 0
slide_delay = 1.0

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

# ---------------- MAIN LOOP ----------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    slide = cv2.imread(os.path.join(folder_path, slide_files[current_slide]))
    slide = cv2.resize(slide, (1280, 720))

    if canvas is None:
        canvas = np.zeros_like(slide)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            lm = hand.landmark
            h, w, _ = frame.shape

            ix = int(lm[8].x * w)
            iy = int(lm[8].y * h)

            thumb_up = finger_up(lm, 4, 3)
            index_up = finger_up(lm, 8, 6)
            middle_up = finger_up(lm, 12, 10)

            current_time = time.time()

            # -------- NEXT SLIDE --------
            if thumb_up and not index_up and not middle_up:
                if current_time - last_slide_time > slide_delay:
                    current_slide = min(current_slide + 1, len(slide_files) - 1)
                    canvas = np.zeros_like(slide)
                    last_slide_time = current_time

            # -------- PREVIOUS SLIDE --------
            if not thumb_up and index_up and middle_up:
                if current_time - last_slide_time > slide_delay:
                    current_slide = max(current_slide - 1, 0)
                    canvas = np.zeros_like(slide)
                    last_slide_time = current_time

            # -------- DRAW --------
            if index_up and not middle_up:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = ix, iy

                cv2.line(canvas, (prev_x, prev_y), (ix, iy), (0, 0, 255), 5)
                prev_x, prev_y = ix, iy
            else:
                prev_x, prev_y = 0, 0

    output = cv2.addWeighted(slide, 1, canvas, 1, 0)
    cv2.imshow("AI Presentation Controller", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()