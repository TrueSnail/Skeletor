#!/usr/bin/env python3

import cv2
import uuid
import json
import socket

# --- Konfiguracja globalna ---
SOURCE = '0'
SEND_METHOD = 1  # 1=UDP, 2=JSON, 3=UDP+JSON
UDP_IP = '10.78.76.48'
UDP_PORT = 56953

# --- Próba importu mediapipe ---
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[ERROR] Brak biblioteki 'mediapipe'. Nie można wykrywać sylwetki ani dłoni.")
    MEDIAPIPE_AVAILABLE = False

# --- Otwarcie źródła wideo ---
def open_video_source(source):
    try:
        src = int(source)
    except ValueError:
        src = source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Nie można otworzyć źródła wideo: {source}")
        return None
    return cap

# --- Zapis danych do JSON ---
def save_data_json(data, filename='output.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Zapisano {len(data)} pakietów do {filename}")

# --- Mapowania ---
BONE_MAP = {
    'Head': 0,
    'Neck': ('mid', 11, 12),
    'Pelvis': ('mid', 23, 24),
    'LeftClavicle': 11,
    'RightClavicle': 12,
    'LeftElbow': 13,
    'RightElbow': 14,
    'LeftWrist': 15,
    'RightWrist': 16,
    'LeftHip': 23,
    'RightHip': 24,
    'LeftKnee': 25,
    'RightKnee': 26,
    'LeftAnkle': 27,
    'RightAnkle': 28
}
HAND_BONE_MAP = {
    'Wrist': 0, 'Thumb_CMC': 1, 'Thumb_MCP': 2, 'Thumb_IP': 3, 'Thumb_Tip': 4,
    'Index_MCP': 5, 'Index_PIP': 6, 'Index_DIP': 7, 'Index_Tip': 8,
    'Middle_MCP': 9, 'Middle_PIP': 10, 'Middle_DIP': 11, 'Middle_Tip': 12,
    'Ring_MCP': 13, 'Ring_PIP': 14, 'Ring_DIP': 15, 'Ring_Tip': 16,
    'Pinky_MCP': 17, 'Pinky_PIP': 18, 'Pinky_DIP': 19, 'Pinky_Tip': 20
}

# --- Główna funkcja ---
def main(source=SOURCE):
    if not MEDIAPIPE_AVAILABLE:
        print("[ERROR] mediapipe nie jest dostępne.")
        return

    cap = open_video_source(source)
    if cap is None:
        return

    # Inicjalizacja UDP
    sock = None
    if SEND_METHOD in (1, 3):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    window_name = 'Skeleton & Hands Detector'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print('[INFO] Rozpoczynam. Naciśnij q, aby zakończyć.')

    # Inicjalizacja MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    frames_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print('[INFO] Koniec strumienia wideo.')
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detekcja sylwetki
        results_pose = pose.process(rgb)
        pose_dict = {}
        if results_pose.pose_landmarks:
            for bone, idx in BONE_MAP.items():
                if isinstance(idx, tuple):
                    _, i1, i2 = idx
                    lm1 = results_pose.pose_landmarks.landmark[i1]
                    lm2 = results_pose.pose_landmarks.landmark[i2]
                    x = (lm1.x + lm2.x) / 2 * w
                    y = (lm1.y + lm2.y) / 2 * h
                    z = (lm1.z + lm2.z) / 2
                else:
                    lm = results_pose.pose_landmarks.landmark[idx]
                    x = lm.x * w
                    y = lm.y * h
                    z = lm.z
                pose_dict[bone] = [x, y, z]
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Detekcja dłoni
        results_hands = hands.process(rgb)
        hands_list = []
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks,
                                                  results_hands.multi_handedness):
                hand_side = handedness.classification[0].label
                hand_dict = {'hand': hand_side, 'bones': {}}
                for bone_name, idx in HAND_BONE_MAP.items():
                    lm = hand_landmarks.landmark[idx]
                    hand_dict['bones'][bone_name] = [lm.x * w, lm.y * h, lm.z]
                hands_list.append(hand_dict)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Pakiet danych
        userid = str(uuid.uuid4())
        packet = {'userid': userid, 'pose': pose_dict, 'hands': hands_list}

                # Wysyłka UDP
        if SEND_METHOD in (1, 3) and sock:
            msg = json.dumps(packet).encode('utf-8')
            try:
                sent = sock.sendto(msg, (UDP_IP, UDP_PORT))
                if sent != len(msg):
                    print(f"[WARN] Wysłano {sent} z {len(msg)} bajtów przez UDP")
            except Exception as e:
                print(f"[ERROR] Błąd podczas wysyłki UDP: {e}")

        # Zapis JSON
        if SEND_METHOD in (2, 3):
            frames_data.append(packet)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if SEND_METHOD in (2, 3):
        save_data_json(frames_data)

# --- Testy jednostkowe ---
if __name__ == '__main__':
    main()
