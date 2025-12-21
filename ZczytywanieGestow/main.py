import cv2
import numpy as np
import socket
import json
import time
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================== USTAWIENIA ==================

# Modele MediaPipe Tasks
POSE_MODEL_PATH = "pose_landmarker_lite.task"    # model pozy (ciało)
HANDS_MODEL_PATH = "hand_landmarker.task"        # model dłoni

CAM_INDEX = 0

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# Historia X prawej dłoni do wykrywania wave
RIGHT_HAND_HISTORY = 20            # ile próbek trzymamy
MIN_HISTORY_WAVE = 10              # minimalna liczba próbek
WAVE_DIRECTION_CHANGES_THRESHOLD = 3  # ile zmian kierunku min.
MIN_WAVE_AMPLITUDE = 0.15          # minimalna amplituda (max_x - min_x)

SEND_COOLDOWN = 0.5                # min odstęp czasowy dla tego samego gestu

# ================== UDP ==================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
last_sent = None
last_time = 0.0


def send_gesture(gesture: str):
    """
    Wysyła gest jako JSON po UDP do Unity:
        {"gesture": "wave"}
    Nie spamuje tym samym gestem częściej niż SEND_COOLDOWN.
    """
    global last_sent, last_time

    now = time.time()
    if gesture == last_sent and (now - last_time) < SEND_COOLDOWN:
        return

    msg = json.dumps({"gesture": gesture}).encode("utf-8")
    sock.sendto(msg, (UDP_IP, UDP_PORT))

    last_sent = gesture
    last_time = now
    print("[SEND]", gesture)


# ================== DETEKCJA GESTÓW (POSE + HANDS) ==================

right_x_history = deque(maxlen=RIGHT_HAND_HISTORY)


def reset_wave_history():
    right_x_history.clear()


def update_wave_history(x: float):
    right_x_history.append(x)


def detect_wave() -> bool:
    """
    Wykrywa wave na podstawie historii X prawej dłoni:
    - musi być wystarczająca amplituda,
    - kilka zmian kierunku (oscylacje lewo-prawo).
    """
    if len(right_x_history) < MIN_HISTORY_WAVE:
        return False

    arr = np.array(right_x_history, dtype=np.float32)

    # amplituda ruchu (0..1, bo to współrzędne znormalizowane)
    min_x = float(np.min(arr))
    max_x = float(np.max(arr))
    amplitude = max_x - min_x

    if amplitude < MIN_WAVE_AMPLITUDE:
        # ruch zbyt mały, to raczej szum
        return False

    diffs = np.diff(arr)
    signs = np.sign(diffs)

    # ignorujemy bardzo małe ruchy (0)
    non_zero = signs[signs != 0]
    if len(non_zero) < 3:
        return False

    direction_changes = np.sum(np.diff(non_zero) != 0)

    return direction_changes >= WAVE_DIRECTION_CHANGES_THRESHOLD


def detect_arms_up(landmarks) -> bool:
    """
    True, jeśli obie ręce są wyraźnie nad głową (nose).
    Indeksy landmarków dla Pose Landmarker:
      0  - NOSE
      15 - LEFT_WRIST
      16 - RIGHT_WRIST
    """
    NOSE = 0
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    nose_y = landmarks[NOSE].y
    lw_y = landmarks[LEFT_WRIST].y
    rw_y = landmarks[RIGHT_WRIST].y

    # MediaPipe: mniejsze Y = wyżej
    threshold = 0.10

    left_up = lw_y < (nose_y - threshold)
    right_up = rw_y < (nose_y - threshold)

    return left_up and right_up


def detect_thumbs_up_hands(hand_landmarks) -> bool:
    """
    Detekcja 'thumbs_up' na bazie modelu Hands (hand_landmarker):
    - kciuk wyraźnie w górę względem nadgarstka,
    - kciuk wyżej niż inne palce,
    - ruch raczej pionowy niż poziomy.
    hand_landmarks = lista 21 landmarków jednej dłoni.
    """

    # Indeksy dla Hands:
    # 0 - WRIST
    # 4 - THUMB_TIP
    # 8 - INDEX_TIP
    # 12 - MIDDLE_TIP
    # 16 - RING_TIP
    # 20 - PINKY_TIP

    wrist = hand_landmarks[0]
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    ring_tip = hand_landmarks[16]
    pinky_tip = hand_landmarks[20]

    # MediaPipe: mniejsze Y = wyżej

    # 1) Kciuk wyraźnie powyżej nadgarstka (w górę)
    dy_thumb = thumb_tip.y - wrist.y  # < 0 jeśli kciuk w górę
    thumb_up_enough = dy_thumb < -0.05

    # 2) Kciuk wyżej niż inne palce (różnica w Y)
    thumb_above_others = (
        thumb_tip.y < index_tip.y - 0.02 and
        thumb_tip.y < middle_tip.y - 0.02 and
        thumb_tip.y < ring_tip.y - 0.02 and
        thumb_tip.y < pinky_tip.y - 0.02
    )

    # 3) Kciuk bardziej "pionowo" niż poziomo (bardziej w górę niż w bok)
    dx_thumb = abs(thumb_tip.x - wrist.x)
    vertical_dominates = abs(dy_thumb) > dx_thumb * 1.2

    return thumb_up_enough and thumb_above_others and vertical_dominates


# ================== GŁÓWNY PROGRAM ==================

def main():
    # ----- Kamera -----
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Nie można otworzyć kamery o indeksie {CAM_INDEX}")
        return
    else:
        print(f"[INFO] Kamera {CAM_INDEX} otwarta poprawnie")

    # ----- Modele MediaPipe Tasks -----
    BaseOptions = python.BaseOptions
    RunningMode = vision.RunningMode

    # Pose Landmarker (ciało)
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=RunningMode.VIDEO,      # synchroniczne VIDEO
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1
    )

    pose_landmarker = PoseLandmarker.create_from_options(pose_options)

    # Hand Landmarker (dłonie)
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HANDS_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    hand_landmarker = HandLandmarker.create_from_options(hand_options)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] cap.read() zwróciło False – kończę pętlę")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Tworzymy obraz dla MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # zapewniamy rosnący timestamp (w mikrosekundach) – ważne dla filterów
        frame_idx += 1
        timestamp_us = frame_idx * 33333  # ~30 FPS, ale najważniejsze, że rośnie

        # Pose (ciało)
        pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_us)

        # Hands (dłonie)
        hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_us)

        # wybieramy prawą dłoń (dla thumbs_up)
        right_hand_landmarks = None
        if hand_results.hand_landmarks:
            # hand_landmarks: lista [ręka_0, ręka_1, ...]
            # handedness: lista [ [Category("Right"), ...], [Category("Left"), ...], ... ]
            for hand_lms, hand_handedness in zip(
                hand_results.hand_landmarks,
                hand_results.handedness
            ):
                if hand_handedness[0].category_name == "Right":
                    right_hand_landmarks = hand_lms
                    break

        gesture = "idle"

        if pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
            lm = pose_results.pose_landmarks[0]

            # prawy nadgarstek – index 16 (z POSE) – używamy do wave i debug rysunku
            rw = lm[16]
            update_wave_history(rw.x)

            # 1) arms_up – z POSE (najwyższy priorytet)
            if detect_arms_up(lm):
                gesture = "arms_up"

            # 2) thumbs_up – z HANDS (prawa dłoń)
            elif right_hand_landmarks is not None and detect_thumbs_up_hands(right_hand_landmarks):
                gesture = "thumbs_up"

            # 3) wave – z historii X prawego nadgarstka (POSE)
            elif detect_wave():
                gesture = "wave"

            # debugowy rysunek
            cx, cy = int(rw.x * w), int(rw.y * h)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(frame, gesture, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        else:
            # brak sylwetki – czyścimy historię wave i zostajemy w idle
            reset_wave_history()
            gesture = "idle"

        # wyślij gest po UDP
        send_gesture(gesture)

        cv2.imshow("Gesty – Pose + Hands (MediaPipe Tasks)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            print("[INFO] Wyjście z pętli (ESC/q)")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
