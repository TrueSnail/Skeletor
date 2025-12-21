import cv2
import json
import socket
import time
import uuid
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =================== USTAWIENIA ===================

CAM_INDEX = 0
MODEL_PATH = "pose_landmarker_lite.task"

NUM_POSES = 1
MIN_DET_CONF = 0.5
MIN_PRES_CONF = 0.5
MIN_TRACK_CONF = 0.5

# Kalibracja kamery (CharucoCalib.py: K, dist, image_size)
CAMERA_PARAMS_PATH = "camera_params.npz"
USE_UNDISTORT = True

# Twoje zdjęcia kalibracyjne były 640x480 -> najlepiej ustawić kamerę tak samo:
CAP_WIDTH = 640
CAP_HEIGHT = 480

# Wymiary sceny (przybliżenie) do przeliczenia px -> cm
REAL_WIDTH_CM = 100.0

# Osie / odbicia
UNITY_AXES = True   # Y w górę, Z do przodu
FLIP_X = False       # zamiana lewo/prawo (jak w lustrze)
FLIP_Z = True       # zamiana znaku Z (przód/tył)
MIRROR_PREVIEW = True
WINDOW_NAME = "Pose → JSON/UDP (cm, undistort + kalibracja)"

# Pomiary długości segmentów (opcjonalnie)
SHOW_MEASUREMENTS = True
MEASURE_CONNECTIONS = [
    (11, 13, "UpperArm_L"),
    (13, 15, "Forearm_L"),
    (12, 14, "UpperArm_R"),
    (14, 16, "Forearm_R"),
    (23, 25, "Thigh_L"),
    (25, 27, "Shin_L"),
    (24, 26, "Thigh_R"),
    (26, 28, "Shin_R"),
]

# Wyjście
OUTPUT_MODE = "udp"     # "json" / "udp" / "both"
JSON_PATH = "poses.json"
UDP_HOST = "127.0.0.1"
UDP_PORT = 4242

# =================== SZKIELET ===================

BONE_MAP = {
    'Head': 0,
    'Neck': ('mid', 11, 12),
    'Pelvis': ('mid', 23, 24),
    'LeftClavicle': 11, 'RightClavicle': 12,
    'LeftElbow': 13, 'RightElbow': 14,
    'LeftWrist': 15, 'RightWrist': 16,
    'LeftHip': 23, 'RightHip': 24,
    'LeftKnee': 25, 'RightKnee': 26,
    'LeftAnkle': 27, 'RightAnkle': 28,
}

POSE_CONNECTIONS = [
    (11,13),(13,15),(12,14),(14,16),(11,12),
    (23,24),(11,23),(12,24),(23,25),(25,27),(24,26),(26,28),
    (27,29),(29,31),(28,30),(30,32),(15,21),(16,22),
    (15,17),(16,18),(11,19),(12,20)
]

# =================== GLOBALNE KALIBRACJA ===================

K_CALIB = None      # macierz kamery z kalibracji
DIST = None         # dystorsja
CALIB_SIZE = None   # (w_calib, h_calib)

K_SCALED = None     # K przeskalowana do bieżącej rozdzielczości
NEW_K = None        # new camera matrix z getOptimalNewCameraMatrix
MAP1 = None         # mapy do remap()
MAP2 = None

# =================== POMOCNICZE ===================

def load_camera_params(path):
    global K_CALIB, DIST, CALIB_SIZE
    try:
        data = np.load(path, allow_pickle=True)
        K_CALIB = data["K"]
        DIST = data["dist"]
        if "image_size" in data:
            size = data["image_size"]
            CALIB_SIZE = (int(size[0]), int(size[1]))
        else:
            CALIB_SIZE = (CAP_WIDTH, CAP_HEIGHT)
        print(f"[INFO] Wczytano parametry kamery: {path}")
        print("K_calib:\n", K_CALIB)
        print("dist:", DIST.ravel())
        print("image_size (kalibracja):", CALIB_SIZE)
    except Exception as e:
        print(f"[WARN] Nie udało się wczytać '{path}': {e}\n       Działam bez undistort.")
        K_CALIB = None
        DIST = None
        CALIB_SIZE = None

def init_undistort_maps(frame_w, frame_h):
    """
    Tworzy K_SCALED, NEW_K, MAP1, MAP2 na podstawie
    K_CALIB, DIST i rozdzielczości kalibracji vs aktualnej.
    """
    global K_SCALED, NEW_K, MAP1, MAP2
    if K_CALIB is None or DIST is None:
        print("[INFO] Brak kalibracji – undistort wyłączony.")
        K_SCALED = None
        NEW_K = None
        MAP1 = None
        MAP2 = None
        return

    w_calib, h_calib = CALIB_SIZE
    sx = frame_w / float(w_calib)
    sy = frame_h / float(h_calib)

    # przeskalowanie K do nowej rozdzielczości
    K_SCALED = K_CALIB.copy()
    K_SCALED[0,0] *= sx
    K_SCALED[1,1] *= sy
    K_SCALED[0,2] *= sx
    K_SCALED[1,2] *= sy

    # macierz optymalna + mapy do remap
    NEW_K, _ = cv2.getOptimalNewCameraMatrix(K_SCALED, DIST, (frame_w, frame_h), alpha=0.0)
    MAP1, MAP2 = cv2.initUndistortRectifyMap(
        K_SCALED, DIST, None, NEW_K, (frame_w, frame_h), cv2.CV_16SC2
    )

    print("[INFO] Zainicjowano undistort maps.")
    print("K_scaled:\n", K_SCALED)
    print("NewK:\n", NEW_K)

def undistort_frame(frame):
    if not USE_UNDISTORT or MAP1 is None or MAP2 is None:
        return frame
    # remap jest spójny z undistortPoints(P=NEW_K)
    return cv2.remap(frame, MAP1, MAP2, interpolation=cv2.INTER_LINEAR)

def undistort_points_px(pts_px):
    """
    pts_px: lista [(u,v), ...] w pikselach (na obrazie wejściowym).
    Zwraca undistortowane punkty w pikselach, zgodne z NEW_K.
    """
    if not USE_UNDISTORT or K_SCALED is None or DIST is None or NEW_K is None or len(pts_px) == 0:
        return pts_px
    pts = np.asarray(pts_px, dtype=np.float32).reshape(-1,1,2)
    und = cv2.undistortPoints(pts, K_SCALED, DIST, P=NEW_K)  # P=NEW_K – spójne z remap
    und = und.reshape(-1,2)
    return [(float(u), float(v)) for (u,v) in und]

def midpoint3(a, b):
    return [(a[i] + b[i]) / 2.0 for i in range(3)]

def landmarks_uvz_to_cm(landmarks, w, h, cm_per_px,
                        unity_axes=True, flip_x=False, flip_z=False):
    """
    Konwersja MediaPipe landmarks (u=lm.x*w, v=lm.y*h, z=lm.z*w) → [x,y,z] w cm,
    z uwzględnieniem undistortPoints.
    """
    uv_list, z_list = [], []
    for lm in landmarks:
        u = float(lm.x) * w
        v = float(lm.y) * h
        z_rel_px = float(lm.z) * w
        uv_list.append((u, v))
        z_list.append(z_rel_px)

    uv_und = undistort_points_px(uv_list)

    pts_cm = []
    for (u, v), z_px in zip(uv_und, z_list):
        # X
        if flip_x:
            u_eff = (w - u)
        else:
            u_eff = u
        x_cm = float(u_eff * cm_per_px)

        # Y
        if unity_axes:
            y_cm = float((h - v) * cm_per_px)
        else:
            y_cm = float(v * cm_per_px)

        # Z
        z_raw = +z_px if flip_z else -z_px
        z_cm = float(z_raw * cm_per_px)

        pts_cm.append([x_cm, y_cm, z_cm])
    return pts_cm

def bones_from_landmarks(pts_cm):
    bones = {}
    for name, spec in BONE_MAP.items():
        if isinstance(spec, tuple) and spec[0] == 'mid':
            _, i, j = spec
            bones[name] = midpoint3(pts_cm[i], pts_cm[j])
        else:
            bones[name] = pts_cm[spec]
    return bones

def draw_pose_preview(img, pts_cm, cm_per_px, mirror=False):
    h, w = img.shape[:2]
    cm_to_px = 1.0 / cm_per_px

    pts_px = []
    for x_cm, y_cm, _ in pts_cm:
        x_px = x_cm * cm_to_px
        y_px = h - (y_cm * cm_to_px)
        if mirror:
            x_px = w - x_px
        pts_px.append((x_px, y_px))

    # linie
    for a, b in POSE_CONNECTIONS:
        if a < len(pts_px) and b < len(pts_px):
            pa = (int(pts_px[a][0]), int(pts_px[a][1]))
            pb = (int(pts_px[b][0]), int(pts_px[b][1]))
            cv2.line(img, pa, pb, (0, 255, 255), 2, cv2.LINE_AA)

    # punkty
    for x, y in pts_px:
        cv2.circle(img, (int(x), int(y)), 3, (0, 128, 255), -1, cv2.LINE_AA)

    # długości segmentów
    if SHOW_MEASUREMENTS:
        for a, b, name in MEASURE_CONNECTIONS:
            if a < len(pts_cm) and b < len(pts_cm):
                ax, ay, az = pts_cm[a]
                bx, by, bz = pts_cm[b]
                dist_cm = float(np.linalg.norm([bx-ax, by-ay, bz-az]))
                mx_px = (pts_px[a][0] + pts_px[b][0]) * 0.5
                my_px = (pts_px[a][1] + pts_px[b][1]) * 0.5
                label = f"{dist_cm:.1f} cm"
                cv2.putText(img, label, (int(mx_px)+1, int(my_px)+1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(img, label, (int(mx_px), int(my_px)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)

# ---------- JSON: konwersja numpy -> typy Pythona ----------

def to_jsonable(obj):
    """Rekurencyjnie zamienia numpy-typy na natywne Pythona i usuwa NaN/Inf."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.astype(float).tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj

def append_json(entry, path):
    entry = to_jsonable(entry)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)

def send_udp(entry, host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = json.dumps(to_jsonable(entry), ensure_ascii=False).encode("utf-8")
    sock.sendto(payload, (host, port))

# =================== GŁÓWNA PĘTLA ===================

def main():
    load_camera_params(CAMERA_PARAMS_PATH)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć kamery {CAM_INDEX}")

    # Ustawiamy kamerę na 640x480 (jak przy kalibracji)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    # Jedna klatka, żeby poznać realne w/h
    ok, test_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Brak pierwszej klatki z kamery.")
    h0, w0 = test_frame.shape[:2]
    print(f"[INFO] Rozdzielczosc strumienia: {w0}x{h0}")
    if USE_UNDISTORT:
        init_undistort_maps(w0, h0)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=NUM_POSES,
        min_pose_detection_confidence=MIN_DET_CONF,
        min_pose_presence_confidence=MIN_PRES_CONF,
        min_tracking_confidence=MIN_TRACK_CONF
    )

    index_to_userid = {}
    prev_time = time.time()
    fps = 0.0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Brak klatki z kamery.")
                break

            frame_proc = undistort_frame(frame_bgr)
            h, w = frame_proc.shape[:2]
            cm_per_px = REAL_WIDTH_CM / float(w)

            frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            frame_vis = frame_proc.copy()
            if MIRROR_PREVIEW:
                frame_vis = cv2.flip(frame_vis, 1)

            if result and result.pose_landmarks:
                for idx, lm_set in enumerate(result.pose_landmarks):
                    pts_cm = landmarks_uvz_to_cm(
                        lm_set, w, h, cm_per_px,
                        unity_axes=UNITY_AXES, flip_x=FLIP_X, flip_z=FLIP_Z
                    )
                    bones = bones_from_landmarks(pts_cm)

                    if idx not in index_to_userid:
                        index_to_userid[idx] = str(uuid.uuid4())
                    userid = index_to_userid[idx]

                    entry = {"userid": userid, "pose": bones}
                    if OUTPUT_MODE in ("json", "both"):
                        append_json(entry, JSON_PATH)
                    if OUTPUT_MODE in ("udp", "both"):
                        send_udp([entry], UDP_HOST, UDP_PORT)

                    draw_pose_preview(frame_vis, pts_cm, cm_per_px, mirror=MIRROR_PREVIEW)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time + 1e-9))
            prev_time = now
            cv2.putText(frame_vis,
                        f"FPS:{fps:.1f} | undistort:{bool(USE_UNDISTORT and K_CALIB is not None)} | scale:{cm_per_px:.4f} cm/px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow(WINDOW_NAME, frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
