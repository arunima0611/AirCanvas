import cv2
import numpy as np
import mediapipe as mp
import time

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ---------------- VARIABLES ----------------
canvas = None
prev_x, prev_y = 0, 0

draw_color = (0, 0, 0)  # black pen
brush_thickness = 5
eraser_thickness = 40

# ---------------- BUTTONS ----------------
button_height = 80

buttons = [
    ("BLACK",  (0, 0, 0),    0,   150),
    ("RED",    (0, 0, 255),  150, 300),
    ("BLUE",   (255, 0, 0),  300, 450),
    ("GREEN",  (0, 255, 0),  450, 600),
    ("ERASER", (200,200,200),600, 800),
    ("CLEAR",  (150,150,150),800, 1000),
    ("SAVE",   (0,255,255),  1000,1200),
]

def draw_buttons(frame):
    for text, color, x1, x2 in buttons:
        cv2.rectangle(frame, (x1, 0), (x2, button_height), color, -1)

        txt_color = (255,255,255)
        if text in ["CLEAR", "ERASER"]:
            txt_color = (0,0,0)

        cv2.putText(frame, text, (x1+10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    txt_color, 2)

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.ones_like(frame) * 255  # white background

    draw_buttons(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            lm = hand.landmark
            h, w, _ = frame.shape

            index_up = finger_up(lm, 8, 6)
            middle_up = finger_up(lm, 12, 10)

            ix, iy = int(lm[8].x * w), int(lm[8].y * h)

            cv2.circle(frame, (ix, iy), 8, (0, 0, 255), -1)

            # ---------------- BUTTON CLICK ----------------
            if index_up and iy < button_height:
                for text, color, x1, x2 in buttons:
                    if x1 < ix < x2:

                        if text == "CLEAR":
                            canvas = np.ones_like(frame) * 255

                        elif text == "ERASER":
                            draw_color = (255, 255, 255)

                        elif text == "SAVE":
                            cv2.imwrite(f"whiteboard_{int(time.time())}.png", canvas)
                            print("Saved!")

                        else:
                            draw_color = color

                        prev_x, prev_y = 0, 0
                        time.sleep(0.3)

            # ---------------- DRAW ----------------
            elif index_up and not middle_up:
                thickness = eraser_thickness if draw_color == (255,255,255) else brush_thickness

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = ix, iy

                cv2.line(canvas, (prev_x, prev_y), (ix, iy),
                         draw_color, thickness)

                prev_x, prev_y = ix, iy

            else:
                prev_x, prev_y = 0, 0

    # ---------------- MERGE ----------------
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("AI Whiteboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()