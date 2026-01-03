import json
import time
import socket
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

CONFIG_PATH = "config.json"


# ================= CONFIG =================
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "inference" not in cfg:
        cfg["inference"] = {"delegate": "CPU"}

    print("[CONFIG] Wczytano config.json")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print("-" * 60)
    return cfg


# ================= FPS =================
class FpsCounter:
    def __init__(self):
        self.last = time.time()
        self.fps = 0.0

    def tick(self):
        now = time.time()
        dt = now - self.last
        self.last = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        return self.fps


# ================= UDP =================
class UdpSender:
    def __init__(self, ip: str, port: int, cooldown_sec: float):
        self.addr = (ip, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cooldown = float(cooldown_sec)
        self.last_gesture = None
        self.last_time = 0.0

    def send(self, gesture: str):
        now = time.time()
        if gesture == self.last_gesture and (now - self.last_time) < self.cooldown:
            return
        payload = {"gesture": gesture, "timestamp": now}
        self.sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)
        self.last_gesture = gesture
        self.last_time = now


# ================= Gesture Lock =================
class GestureLock:
    def __init__(self, lock_time_sec: float):
        self.lock_time = float(lock_time_sec)
        self.last_non_idle_time = 0.0

    def apply(self, gesture_raw: str) -> str:
        now = time.time()
        if gesture_raw != "idle":
            if now - self.last_non_idle_time < self.lock_time:
                return "idle"
            self.last_non_idle_time = now
            return gesture_raw
        return "idle"


# ================= STABILIZER (EMA + TTL) =================
class HandStabilizer:
    def __init__(self, alpha: float, hold_ms: int):
        self.alpha = float(alpha)
        self.hold_ms = int(hold_ms)
        self.last = {"Left": None, "Right": None}      # list[(x,y,z)] 21
        self.last_time = {"Left": 0.0, "Right": 0.0}

    def _ema(self, prev, cur):
        a = self.alpha
        out = []
        for (px, py, pz), (cx, cy, cz) in zip(prev, cur):
            out.append((
                a * cx + (1 - a) * px,
                a * cy + (1 - a) * py,
                a * cz + (1 - a) * pz
            ))
        return out

    def update(self, label, landmarks, now):
        cur = [(float(lm.x), float(lm.y), float(lm.z)) for lm in landmarks]
        if self.last[label] is None:
            self.last[label] = cur
        else:
            self.last[label] = self._ema(self.last[label], cur)
        self.last_time[label] = now

    def get(self, label, now):
        pts = self.last[label]
        if pts is None:
            return None
        if (now - self.last_time[label]) * 1000.0 <= self.hold_ms:
            return pts
        return None


# ================= DRAW HANDS (jak w .ipynb) =================
def draw_hands(frame_bgr, hands_pts_by_label, draw_points=True):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    annotated = np.copy(rgb)
    h, w, _ = annotated.shape

    for label, pts in hands_pts_by_label.items():
        if pts is None:
            continue

        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=x, y=y, z=z) for x, y, z in pts
        ])

        solutions.drawing_utils.draw_landmarks(
            annotated,
            proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style() if draw_points else None,
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        xs = [x for x, _, _ in pts]
        ys = [y for _, y, _ in pts]
        tx = int(min(xs) * w)
        ty = max(0, int(min(ys) * h) - 10)
        cv2.putText(annotated, label, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)


# ================= WAVE DETECTOR =================
class WaveDetector:
    def __init__(self, history_len: int, min_history: int, direction_changes_threshold: float, min_amplitude_factor: float):
        self.history_len = int(history_len)
        self.min_history = int(min_history)
        self.dir_thr = float(direction_changes_threshold)
        self.amp_factor = float(min_amplitude_factor)
        self.x_history = deque(maxlen=self.history_len)

    def reset(self):
        self.x_history.clear()

    def update(self, x: float):
        self.x_history.append(float(x))

    def detect(self, hand_size_x: float) -> bool:
        if len(self.x_history) < self.min_history:
            return False

        arr = list(self.x_history)
        amplitude = max(arr) - min(arr)

        required_amp = max(0.01, self.amp_factor * max(0.02, float(hand_size_x)))
        if amplitude < required_amp:
            return False

        diffs = [arr[i + 1] - arr[i] for i in range(len(arr) - 1)]
        signs = []
        for d in diffs:
            if d > 0:
                signs.append(1)
            elif d < 0:
                signs.append(-1)

        if len(signs) < 3:
            return False

        changes = 0
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1]:
                changes += 1

        return changes >= self.dir_thr


# ================= THUMBS UP (FIST + THUMB HIGHEST) =================
def _dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def hand_scale(pts):
    a = (pts[5][0], pts[5][1])   # INDEX_MCP
    b = (pts[17][0], pts[17][1]) # PINKY_MCP
    return float(np.sqrt(_dist2(a, b)))

def finger_folded(pts, tip, pip):
    return pts[tip][1] > pts[pip][1]

def detect_thumbs_up_fist(pts, thumbs_cfg) -> bool:
    wrist = pts[0]
    thumb_tip = pts[4]
    thumb_ip = pts[3]
    thumb_mcp = pts[2]

    index_tip = pts[8]
    middle_tip = pts[12]
    ring_tip = pts[16]
    pinky_tip = pts[20]

    require_folded = bool(thumbs_cfg.get("require_other_fingers_folded", True))
    if require_folded:
        if not (finger_folded(pts, 8, 6) and finger_folded(pts, 12, 10) and finger_folded(pts, 16, 14) and finger_folded(pts, 20, 18)):
            return False

    scale = max(0.02, hand_scale(pts))
    margin_factor = float(thumbs_cfg.get("thumb_above_others_margin", 0.10))
    margin = max(0.01, margin_factor * scale)

    thumb_shape_ok = (thumb_tip[1] < thumb_ip[1]) and (thumb_ip[1] < thumb_mcp[1])

    thumb_highest = (
        thumb_tip[1] < index_tip[1] - margin and
        thumb_tip[1] < middle_tip[1] - margin and
        thumb_tip[1] < ring_tip[1] - margin and
        thumb_tip[1] < pinky_tip[1] - margin
    )

    dy = abs(thumb_tip[1] - wrist[1])
    dx = abs(thumb_tip[0] - wrist[0])
    vertical_factor = float(thumbs_cfg.get("vertical_dominates_factor", 1.2))
    vertical_dominates = dy > dx * vertical_factor

    return bool(thumb_shape_ok and thumb_highest and vertical_dominates)


# ================= ANGLE + WAVE ARM-READY CONDITION (POSE) =================
def angle_deg(a, b, c):
    """
    Kąt ABC w stopniach (wierzchołek = b). a,b,c to landmarki pose z .x .y
    """
    v1 = np.array([a.x - b.x, a.y - b.y], dtype=np.float32)
    v2 = np.array([c.x - b.x, c.y - b.y], dtype=np.float32)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0

    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def is_right_arm_ready_for_wave(pose_landmarks,
                                elbow_max_angle_deg=165.0,
                                wrist_above_hip_margin=0.03):
    """
    Warunek anty-false-positive:
    - prawa ręka uniesiona (nadgarstek wyżej niż biodro o margines)
    - łokieć zgięty (kąt w łokciu < elbow_max_angle_deg)
    Indeksy pose: RS=12, RE=14, RW=16, RH=24
    """
    RS, RE, RW, RH = 12, 14, 16, 24
    shoulder = pose_landmarks[RS]
    elbow = pose_landmarks[RE]
    wrist = pose_landmarks[RW]
    hip = pose_landmarks[RH]

    ang = angle_deg(shoulder, elbow, wrist)
    elbow_bent = ang < elbow_max_angle_deg

    arm_raised = wrist.y < (hip.y - wrist_above_hip_margin)

    return elbow_bent and arm_raised


# ================= MP TASKS BASE OPTIONS (CPU/GPU with fallback) =================
def build_base_options(model_path: str, delegate_cfg: str):
    BaseOptions = python.BaseOptions
    delegate_cfg = (delegate_cfg or "CPU").upper()

    def cpu():
        return BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU)

    if delegate_cfg == "GPU":
        try:
            return BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.GPU)
        except NotImplementedError as e:
            print(f"[DELEGATE] GPU niewspierane na Windows ({e}) -> CPU")
            return cpu()
        except Exception as e:
            print(f"[DELEGATE] Nie udało się uruchomić GPU ({e}) -> CPU")
            return cpu()

    return cpu()


class LiveHands:
    def __init__(self, cfg, stabilizer: HandStabilizer):
        self.stabilizer = stabilizer

        base = build_base_options(
            cfg["models"]["hands_model_path"],
            cfg.get("inference", {}).get("delegate", "CPU")
        )

        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        RunningMode = vision.RunningMode

        def callback(result, output_image, timestamp_ms: int):
            now = time.time()
            if result.hand_landmarks and result.handedness:
                for lms, handed in zip(result.hand_landmarks, result.handedness):
                    label = handed[0].category_name
                    if label in ("Left", "Right"):
                        self.stabilizer.update(label, lms, now)

        options = HandLandmarkerOptions(
            base_options=base,
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
            result_callback=callback
        )
        self.detector = HandLandmarker.create_from_options(options)

    def process(self, rgb_frame, timestamp_ms: int):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.detector.detect_async(mp_image, timestamp_ms)


class LivePose:
    def __init__(self, cfg):
        self.last_pose_landmarks = None

        base = build_base_options(
            cfg["models"]["pose_model_path"],
            cfg.get("inference", {}).get("delegate", "CPU")
        )

        PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        RunningMode = vision.RunningMode

        def callback(result, output_image, timestamp_ms: int):
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                self.last_pose_landmarks = result.pose_landmarks[0]
            else:
                self.last_pose_landmarks = None

        options = PoseLandmarkerOptions(
            base_options=base,
            running_mode=RunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=callback
        )
        self.detector = PoseLandmarker.create_from_options(options)

    def process(self, rgb_frame, timestamp_ms: int):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.detector.detect_async(mp_image, timestamp_ms)


# ================= ARMS UP (Pose) =================
def detect_arms_up(pose_landmarks, threshold_above_nose: float) -> bool:
    NOSE = 0
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    nose_y = pose_landmarks[NOSE].y
    lw_y = pose_landmarks[LEFT_WRIST].y
    rw_y = pose_landmarks[RIGHT_WRIST].y

    return (lw_y < (nose_y - threshold_above_nose)) and (rw_y < (nose_y - threshold_above_nose))


# ================= MAIN =================
def main():
    cfg = load_config(CONFIG_PATH)

    cam = cfg["camera"]
    perf = cfg.get("performance", {})
    inf_w = int(perf.get("inference_width", cam["width"]))
    inf_h = int(perf.get("inference_height", cam["height"]))
    process_every = max(1, int(perf.get("process_every_n_frames", 1)))
    draw_points = bool(perf.get("draw_points", True))

    stabilizer = HandStabilizer(
        alpha=float(perf.get("stabilizer_alpha", 0.55)),
        hold_ms=int(perf.get("stabilizer_hold_ms", 250))
    )

    hands = LiveHands(cfg, stabilizer)
    pose = LivePose(cfg)

    sender = UdpSender(cfg["udp"]["ip"], int(cfg["udp"]["port"]), float(cfg["send"]["cooldown_same_gesture_sec"]))
    lock = GestureLock(float(cfg["send"]["gesture_lock_time_sec"]))

    wave_cfg = cfg["wave"]
    wave = WaveDetector(
        history_len=wave_cfg["history_len"],
        min_history=wave_cfg["min_history"],
        direction_changes_threshold=wave_cfg["direction_changes_threshold"],
        min_amplitude_factor=wave_cfg["min_amplitude"]
    )
    cooldown_after_arms = float(wave_cfg.get("cooldown_after_arms_up_sec", 0.6))

    thumbs_cfg = cfg["thumbs_up"]
    arms_cfg = cfg["arms_up"]

    cap = cv2.VideoCapture(int(cam["index"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam["height"]))
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("[ERROR] Nie można otworzyć kamery.")
        return

    fps = FpsCounter()
    frame_idx = 0
    last_arms_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if cam.get("flip_horizontal", True):
            frame = cv2.flip(frame, 1)

        now = time.time()
        ts = int(now * 1000)

        frame_idx += 1
        if (frame_idx % process_every) == 0:
            inf = cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(inf, cv2.COLOR_BGR2RGB)
            hands.process(rgb, ts)
            pose.process(rgb, ts)

        left_pts = stabilizer.get("Left", now)
        right_pts = stabilizer.get("Right", now)

        out = draw_hands(frame, {"Left": left_pts, "Right": right_pts}, draw_points=draw_points)

        # ==== GESTY (priorytety) ====
        gesture_raw = "idle"

        plm = pose.last_pose_landmarks

        # 1) arms_up
        if plm is not None and detect_arms_up(plm, float(arms_cfg["threshold_above_nose"])):
            gesture_raw = "arms_up"
            last_arms_time = now

        # 2) thumbs_up
        if gesture_raw == "idle" and right_pts is not None:
            if detect_thumbs_up_fist(right_pts, thumbs_cfg):
                gesture_raw = "thumbs_up"

        # 3) wave (TYLKO gdy łokieć zgięty i ręka uniesiona)
        if gesture_raw == "idle":
            arm_ok = False
            if plm is not None:
                arm_ok = is_right_arm_ready_for_wave(
                    plm,
                    elbow_max_angle_deg=165.0,
                    wrist_above_hip_margin=0.03
                )

            if right_pts is not None and arm_ok and (now - last_arms_time) >= cooldown_after_arms:
                wave.update(right_pts[0][0])  # wrist.x
                hand_size_x = abs(right_pts[5][0] - right_pts[17][0])  # index_mcp.x - pinky_mcp.x
                if wave.detect(hand_size_x=hand_size_x):
                    gesture_raw = "wave"
            else:
                # ważne: jeśli poza ręki nie spełnia warunku, czyść historię wave
                wave.reset()

        gesture = lock.apply(gesture_raw)
        sender.send(gesture)

        cv2.putText(out, f"FPS: {fps.tick():.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(out, f"Gesture: {gesture}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Hands+Pose LIVE_STREAM -> Gestures -> UDP [q/ESC]", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
