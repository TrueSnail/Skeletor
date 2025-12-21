import os
import json
import time
import uuid
import math
import socket
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

# ============================================================
#                  CONFIG: load / create
# ============================================================

CONFIG_PATH = "config.json"

DEFAULT_CONFIG = {
    "camera": {
        "index": 0,
        "width": 640,
        "height": 480,
        "mirror_preview": False
    },
    "mediapipe": {
        "model_path": "pose_landmarker_lite.task",
        "num_poses": 2,
        "min_detection_confidence": 0.5,
        "min_presence_confidence": 0.5,
        "min_tracking_confidence": 0.7
    },
    "world": {
        "scale_world": 1,   # meters -> cm
        "flip_x": True,
        "flip_y": True,
        "flip_z": False
    },
    "output": {
        "mode": "udp",          # "udp" / "json" / "both"
        "json_path": "poses_world.json",
        "udp_host": "127.0.0.1",
        "udp_port": 4242
    },
    "preview": {
        "enabled": True,
        "window_name": "Skeleton (MediaPipe world 3D -> Unity)",
        "draw_2d_skeleton": True,
        "draw_fps": True
    },
    "smoothing": {
        "enabled_3d": True,
        "enabled_2d": False,

        # OneEuro 3D (Unity) - dobre startowe pod ~30 FPS
        "oneeuro_3d": {"min_cutoff": 1.6, "beta": 0.04, "d_cutoff": 1.0},

        # OneEuro 2D (preview) - tylko podgląd
        "oneeuro_2d": {"min_cutoff": 2.0, "beta": 0.06, "d_cutoff": 1.0}
    }
}

def deep_merge(base: dict, override: dict) -> dict:
    """Łączy dicty rekurencyjnie: override nadpisuje base."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def ensure_config(path: str, defaults: dict) -> dict:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(defaults, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Utworzono plik konfiguracyjny: {path}")
        return defaults

    try:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = deep_merge(defaults, user_cfg)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return cfg
    except Exception as e:
        print(f"[WARN] Nie mogę wczytać config.json ({e}). Używam domyślnych.")
        return defaults

CFG = ensure_config(CONFIG_PATH, DEFAULT_CONFIG)

# ============================================================
#                     Skeleton mapping
# ============================================================

BONE_MAP = {
    "Head": 0,
    "Neck": ("mid", 11, 12),
    "Pelvis": ("mid", 23, 24),
    "LeftClavicle": 11, "RightClavicle": 12,
    "LeftElbow": 13, "RightElbow": 14,
    "LeftWrist": 15, "RightWrist": 16,
    "LeftHip": 23, "RightHip": 24,
    "LeftKnee": 25, "RightKnee": 26,
    "LeftAnkle": 27, "RightAnkle": 28
}

POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
    (23, 24), (11, 23), (12, 24), (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32), (15, 21), (16, 22),
    (15, 17), (16, 18), (11, 19), (12, 20)
]

# ============================================================
#                     JSON / UDP helpers
# ============================================================

def to_jsonable(obj):
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

def send_udp_payload(payload_obj, host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = json.dumps(to_jsonable(payload_obj), ensure_ascii=False).encode("utf-8")
    sock.sendto(payload, (host, port))

def append_json_payload(payload_obj, path):
    """
    Zapis “pakietów” jako lista klatek
    """
    payload_obj = to_jsonable(payload_obj)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(payload_obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ============================================================
#                 One Euro Filter (1D + states)
# ============================================================

def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))

class OneEuro1D:
    def __init__(self, min_cutoff=1.6, beta=0.04, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0

    def filter(self, x, dt):
        x = float(x)
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dx = (x - self.x_prev) / max(dt, 1e-6)
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * self.dx_prev + (1 - a_d) * dx

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * self.x_prev + (1 - a) * x

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

# states:
# 3D: userid -> boneName -> (fx, fy, fz)
ONEEURO_3D_STATE = defaultdict(dict)

# 2D: userid -> landmark_index -> (fx, fy)
ONEEURO_2D_STATE = defaultdict(dict)

def smooth_bones_3d(userid: str, bones: dict, dt: float) -> dict:
    p = CFG["smoothing"]["oneeuro_3d"]
    out = {}
    st = ONEEURO_3D_STATE[userid]
    for name, v in bones.items():
        if name not in st:
            st[name] = (
                OneEuro1D(p["min_cutoff"], p["beta"], p["d_cutoff"]),
                OneEuro1D(p["min_cutoff"], p["beta"], p["d_cutoff"]),
                OneEuro1D(p["min_cutoff"], p["beta"], p["d_cutoff"])
            )
        fx, fy, fz = st[name]
        out[name] = [
            fx.filter(v[0], dt),
            fy.filter(v[1], dt),
            fz.filter(v[2], dt)
        ]
    return out

def smooth_landmarks_2d(userid: str, landmarks_2d, dt: float):
    p = CFG["smoothing"]["oneeuro_2d"]
    out = []
    st = ONEEURO_2D_STATE[userid]
    for i, lm in enumerate(landmarks_2d):
        if i not in st:
            st[i] = (
                OneEuro1D(p["min_cutoff"], p["beta"], p["d_cutoff"]),
                OneEuro1D(p["min_cutoff"], p["beta"], p["d_cutoff"])
            )
        fx, fy = st[i]
        out.append((fx.filter(lm.x, dt), fy.filter(lm.y, dt)))
    return out

# ============================================================
#                   Conversion + preview draw
# ============================================================

def midpoint3(a, b):
    return [(a[i] + b[i]) / 2.0 for i in range(3)]

def world_landmarks_to_unity(world_lms):
    scale = float(CFG["world"]["scale_world"])
    flip_x = bool(CFG["world"]["flip_x"])
    flip_y = bool(CFG["world"]["flip_y"])
    flip_z = bool(CFG["world"]["flip_z"])

    pts = []
    for lm in world_lms:
        x = lm.x * scale
        y = lm.y * scale
        z = lm.z * scale
        if flip_x: x = -x
        if flip_y: y = -y
        if flip_z: z = -z
        pts.append([float(x), float(y), float(z)])
    return pts

def bones_from_points(pts):
    bones = {}
    for name, spec in BONE_MAP.items():
        if isinstance(spec, tuple) and spec[0] == "mid":
            _, i, j = spec
            bones[name] = midpoint3(pts[i], pts[j])
        else:
            bones[name] = pts[spec]
    return bones

def draw_pose_2d_overlay(img_bgr, landmarks_2d):
    h, w = img_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_2d]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(img_bgr, pts[a], pts[b], (0, 255, 255), 2, cv2.LINE_AA)
    for (u, v) in pts:
        cv2.circle(img_bgr, (u, v), 3, (0, 128, 255), -1, cv2.LINE_AA)

def draw_pose_2d_overlay_xy(img_bgr, pts_xy_norm):
    h, w = img_bgr.shape[:2]
    pts = [(int(x * w), int(y * h)) for (x, y) in pts_xy_norm]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(img_bgr, pts[a], pts[b], (0, 255, 255), 2, cv2.LINE_AA)
    for (u, v) in pts:
        cv2.circle(img_bgr, (u, v), 3, (0, 128, 255), -1, cv2.LINE_AA)

# ============================================================
#                           MAIN
# ============================================================

def main():
    cam_cfg = CFG["camera"]
    mp_cfg = CFG["mediapipe"]
    out_cfg = CFG["output"]
    prev_cfg = CFG["preview"]
    sm_cfg = CFG["smoothing"]

    cap = cv2.VideoCapture(int(cam_cfg["index"]))
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć kamery {cam_cfg['index']}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_cfg["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_cfg["height"]))

    # MediaPipe Tasks
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mp_cfg["model_path"]),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=int(mp_cfg["num_poses"]),
        min_pose_detection_confidence=float(mp_cfg["min_detection_confidence"]),
        min_pose_presence_confidence=float(mp_cfg["min_presence_confidence"]),
        min_tracking_confidence=float(mp_cfg["min_tracking_confidence"])
    )

    index_to_userid = {}  # idx osoby -> GUID
    prev_ts = None
    fps = 0.0
    prev_time = time.time()

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # dt (dla OneEuro)
            now = time.time()
            dt = 1 / 30.0 if prev_ts is None else max(1e-3, now - prev_ts)
            prev_ts = now

            # MediaPipe input
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, ts_ms)

            # ---------- build entries (all persons in one packet) ----------
            entries = []

            if result and result.pose_world_landmarks:
                for idx, world_lms in enumerate(result.pose_world_landmarks):
                    if idx not in index_to_userid:
                        index_to_userid[idx] = str(uuid.uuid4())
                    userid = index_to_userid[idx]

                    pts = world_landmarks_to_unity(world_lms)
                    bones = bones_from_points(pts)

                    # smoothing 3D (Unity)
                    if sm_cfg["enabled_3d"]:
                        bones = smooth_bones_3d(userid, bones, dt)

                    entries.append({"userid": userid, "pose": bones})

            # output per frame
            if entries:
                if out_cfg["mode"] in ("udp", "both"):
                    send_udp_payload(entries, out_cfg["udp_host"], int(out_cfg["udp_port"]))
                if out_cfg["mode"] in ("json", "both"):
                    append_json_payload(entries, out_cfg["json_path"])

            # ---------- preview ----------
            if prev_cfg["enabled"]:
                frame_vis = frame_bgr.copy()
                if cam_cfg["mirror_preview"]:
                    frame_vis = cv2.flip(frame_vis, 1)

                if prev_cfg["draw_2d_skeleton"] and result and result.pose_landmarks:
                    # 2D preview
                    for idx, lm_set in enumerate(result.pose_landmarks):
                        if idx not in index_to_userid:
                            index_to_userid[idx] = str(uuid.uuid4())
                        userid = index_to_userid[idx]

                        if sm_cfg["enabled_2d"]:
                            smooth_xy = smooth_landmarks_2d(userid, lm_set, dt)
                            draw_pose_2d_overlay_xy(frame_vis, smooth_xy)
                        else:
                            draw_pose_2d_overlay(frame_vis, lm_set)

                # FPS
                t = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / (t - prev_time + 1e-9))
                prev_time = t
                if prev_cfg["draw_fps"]:
                    cv2.putText(
                        frame_vis, f"FPS: {fps:.1f}  persons:{len(entries)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

                cv2.imshow(prev_cfg["window_name"], frame_vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
