import json
import os
import socket
import sys
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np

# --- Kinect v2 ---
try:
    from pykinect2 import PyKinectV2
    from pykinect2 import PyKinectRuntime
except ImportError:
    print("Brak pykinect2. Zainstaluj: pip install pykinect2")
    sys.exit(1)

# --- (opcjonalny) podgląd i rysowanie ROI ---
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


# -----------------------------------------
# KONFIGURACJA
# -----------------------------------------

CONFIG = {
    # Maksymalna liczba śledzonych osób (jeśli chcesz sztucznie ograniczyć)
    "max_people": 6,

    # Tryb zapisu:
    # 1 - tylko JSON do pliku
    # 2 - tylko UDP
    # 3 - JSON + UDP
    "output_mode": 3,

    # Plik JSON (kolejne pakiety będą dopisywane do PLIKU jako jedna zbiorcza tablica)
    "json_path": "kinect_packets.json",

    # UDP
    "udp_host": "127.0.0.1",
    "udp_port": 7001,

    # Częstotliwość pakietów (Hz)
    "packets_per_second": 15,

    # Strefa „wystawy” w przestrzeni 2D (głębia) – prostokąt w pikselach depth-frame (512x424).
    # Podaj (x_min, y_min, x_max, y_max). Ustaw pod Twoje stanowisko:
    "roi_rect": (170, 120, 340, 320),

    # Progi selekcji „uczestników”
    "dwell_seconds_threshold": 2.0,   # minimalny czas przebywania w ROI
    "speed_threshold_m_s": 0.25,      # uznajemy za „raczej stoi/słabo się porusza”
    "front_facing_dot_min": 0.15,     # patrzy mniej więcej w kierunku sensora (dodatni iloczyn skalarny)

    # Wypełnij dłońmi w JSON
    "include_hands": True,
}


# -----------------------------------------
# MAPOWANIE KOŚCI (Unity-like) -> złącza Kinect v2
# -----------------------------------------
# Użytkownik podał BONE_MAP z indeksami, które nie pokrywają się z Kinect v2.
# Poniżej dajemy funkcjonalny odpowiednik na bazie JointType_* (współrzędne 3D).

K = PyKinectV2

# Pomoc: skrót do JointType
JT = {
    "SpineBase": K.JointType_SpineBase,        # 0
    "SpineMid": K.JointType_SpineMid,          # 1
    "Neck": K.JointType_Neck,                  # 2
    "Head": K.JointType_Head,                  # 3
    "ShoulderLeft": K.JointType_ShoulderLeft,  # 4
    "ElbowLeft": K.JointType_ElbowLeft,        # 5
    "WristLeft": K.JointType_WristLeft,        # 6
    "HandLeft": K.JointType_HandLeft,          # 7
    "ShoulderRight": K.JointType_ShoulderRight,# 8
    "ElbowRight": K.JointType_ElbowRight,      # 9
    "WristRight": K.JointType_WristRight,      #10
    "HandRight": K.JointType_HandRight,        #11
    "HipLeft": K.JointType_HipLeft,            #12
    "KneeLeft": K.JointType_KneeLeft,          #13
    "AnkleLeft": K.JointType_AnkleLeft,        #14
    "FootLeft": K.JointType_FootLeft,          #15
    "HipRight": K.JointType_HipRight,          #16
    "KneeRight": K.JointType_KneeRight,        #17
    "AnkleRight": K.JointType_AnkleRight,      #18
    "FootRight": K.JointType_FootRight,        #19
    "SpineShoulder": K.JointType_SpineShoulder,#20
    "HandTipLeft": K.JointType_HandTipLeft,    #21
    "ThumbLeft": K.JointType_ThumbLeft,        #22
    "HandTipRight": K.JointType_HandTipRight,  #23
    "ThumbRight": K.JointType_ThumbRight,      #24
}

# Finalne kości do JSON (nazwy jak podałeś), wartości liczymy z/w oparciu o stawy Kinect:
BONE_MAP = {
    "Head": ("single", JT["Head"]),
    # „Neck” jako środek pomiędzy lewym i prawym barkiem (Kinect nie ma clavicle; SpineShoulder bywa trochę niżej).
    "Neck": ("mid", JT["ShoulderLeft"], JT["ShoulderRight"]),
    # „Pelvis” jako środek pomiędzy lewym i prawym biodrem:
    "Pelvis": ("mid", JT["HipLeft"], JT["HipRight"]),
    "LeftClavicle": ("single", JT["ShoulderLeft"]),
    "RightClavicle": ("single", JT["ShoulderRight"]),
    "LeftElbow": ("single", JT["ElbowLeft"]),
    "RightElbow": ("single", JT["ElbowRight"]),
    "LeftWrist": ("single", JT["WristLeft"]),
    "RightWrist": ("single", JT["WristRight"]),
    "LeftHip": ("single", JT["HipLeft"]),
    "RightHip": ("single", JT["HipRight"]),
    "LeftKnee": ("single", JT["KneeLeft"]),
    "RightKnee": ("single", JT["KneeRight"]),
    "LeftAnkle": ("single", JT["AnkleLeft"]),
    "RightAnkle": ("single", JT["AnkleRight"]),
}


# -----------------------------------------
# NARZĘDZIA
# -----------------------------------------

def cam_point_to_list(p) -> List[float]:
    """PyKinect CameraSpacePoint -> [x,y,z] (metry)."""
    return [float(p.x), float(p.y), float(p.z)]

def joint_ok(j):
    return j.TrackingState in (PyKinectV2.TrackingState_Inferred, PyKinectV2.TrackingState_Tracked)

def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (a + b)

def vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (b - a)

def safe_norm(v: np.ndarray) -> float:
    n = float(np.linalg.norm(v))
    return n if np.isfinite(n) else 0.0

def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# -----------------------------------------
# SELEKCJA „UCZESTNIKÓW”
# -----------------------------------------

@dataclass
class TrackState:
    last_positions: deque  # deque of (t, np.array([x,y,z])) for SpineBase
    dwell_start_in_roi: Optional[float] = None
    is_in_roi: bool = False

    def update(self, t: float, pos3d: np.ndarray, in_roi: bool):
        self.last_positions.append((t, pos3d))
        if len(self.last_positions) > 15:
            self.last_positions.popleft()

        if in_roi:
            if not self.is_in_roi:
                self.dwell_start_in_roi = t
        else:
            self.dwell_start_in_roi = None

        self.is_in_roi = in_roi

    def dwell_time(self, t_now: float) -> float:
        if self.dwell_start_in_roi is None:
            return 0.0
        return max(0.0, t_now - self.dwell_start_in_roi)

    def avg_speed(self) -> float:
        """Średnia prędkość (m/s) po torze SpineBase z krótkiego okna."""
        if len(self.last_positions) < 2:
            return 0.0
        t0, p0 = self.last_positions[0]
        t1, p1 = self.last_positions[-1]
        dt = max(1e-6, t1 - t0)
        dist = safe_norm(p1 - p0)
        return dist / dt


class ParticipantSelector:
    def __init__(self, roi_rect, config):
        self.roi = roi_rect  # (x_min,y_min,x_max,y_max) w przestrzeni depth
        self.cfg = config
        self.tracks: Dict[int, TrackState] = {}

    def _point_in_roi(self, depth_point) -> bool:
        x, y = depth_point
        x0, y0, x1, y1 = self.roi
        return (x0 <= x <= x1) and (y0 <= y <= y1)

    def decide(self, body, kinect: PyKinectRuntime.PyKinectRuntime, t_now: float) -> bool:
        """Zwraca True jeśli osoba to 'uczestnik'."""
        tid = body.TrackingId
        if tid not in self.tracks:
            self.tracks[tid] = TrackState(deque(maxlen=30))

        # użyjemy SpineBase do ROI/szybkości
        joints = body.joints
        if not joint_ok(joints[JT["SpineBase"]]):
            return False

        # 3D punkt:
        p3 = np.array([
            joints[JT["SpineBase"]].Position.x,
            joints[JT["SpineBase"]].Position.y,
            joints[JT["SpineBase"]].Position.z,
        ], dtype=np.float32)

        # rzut do depth (512x424), żeby łatwo sprawdzić ROI prostokątne
        depth_point = kinect._mapper.MapCameraPointToDepthSpace(joints[JT["SpineBase"]].Position)
        in_roi = self._point_in_roi((depth_point.x, depth_point.y))

        # zaktualizuj śledzenie
        self.tracks[tid].update(t_now, p3, in_roi)

        # heurystyka „czy patrzy w naszą stronę?” (przybliżenie: wektor od SpineMid -> Neck)
        look_dot = 0.0
        if joint_ok(joints[JT["Neck"]]) and joint_ok(joints[JT["SpineMid"]]):
            neck = np.array([
                joints[JT["Neck"]].Position.x,
                joints[JT["Neck"]].Position.y,
                joints[JT["Neck"]].Position.z,
            ], dtype=np.float32)
            spine_mid = np.array([
                joints[JT["SpineMid"]].Position.x,
                joints[JT["SpineMid"]].Position.y,
                joints[JT["SpineMid"]].Position.z,
            ], dtype=np.float32)
            # „front” sensora to ujemny kierunek osi Z kamery (patrzymy w głąb), więc bierzemy -Z jako forward kamery
            forward_cam = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            torso_up = vec(spine_mid, neck)
            if safe_norm(torso_up) > 0:
                torso_up = torso_up / safe_norm(torso_up)
                look_dot = dot(torso_up, forward_cam)  # >0: mniej więcej skierowany w sensor

        # warunki
        dwell_ok = (self.tracks[tid].dwell_time(t_now) >= self.cfg["dwell_seconds_threshold"])
        speed_ok = (self.tracks[tid].avg_speed() <= self.cfg["speed_threshold_m_s"])
        facing_ok = (look_dot >= self.cfg["front_facing_dot_min"])

        return in_roi and dwell_ok and speed_ok and facing_ok


# -----------------------------------------
# SERIALIZACJA PAKIETÓW
# -----------------------------------------

class PacketWriter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sock = None
        if cfg["output_mode"] in (2, 3):
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_addr = (cfg["udp_host"], cfg["udp_port"])

        # Inicjalizacja pliku JSON jako tablicy, jeśli nie istnieje
        if cfg["output_mode"] in (1, 3):
            if not os.path.exists(cfg["json_path"]):
                with open(cfg["json_path"], "w", encoding="utf-8") as f:
                    f.write("[]")

    def _append_json_array(self, obj):
        """Bezpieczne dopisywanie obiektu do tablicy w pliku JSON."""
        path = self.cfg["json_path"]
        with open(path, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except Exception:
                data = []
            data.append(obj)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate()

    def send(self, payload: dict):
        s = json.dumps(payload, ensure_ascii=False)
        if self.cfg["output_mode"] in (2, 3) and self.sock:
            self.sock.sendto(s.encode("utf-8"), self.udp_addr)
        if self.cfg["output_mode"] in (1, 3):
            self._append_json_array(payload)


# -----------------------------------------
# GŁÓWNA PĘTLA
# -----------------------------------------

def build_pose_dict(body, kinect: PyKinectRuntime.PyKinectRuntime) -> Dict[str, List[float]]:
    """Zwraca {BoneName: [x,y,z]} w metrach."""
    joints = body.joints
    pose = {}
    # Zbierz CameraSpace jako wektory
    cam = {}
    for name, jt in JT.items():
        j = joints[jt]
        if joint_ok(j):
            cam[name] = np.array([j.Position.x, j.Position.y, j.Position.z], dtype=np.float32)

    for bone_name, rule in BONE_MAP.items():
        kind = rule[0]
        if kind == "single":
            jt = rule[1]
            j = joints[jt]
            if joint_ok(j):
                pose[bone_name] = cam_point_to_list(j.Position)
        elif kind == "mid":
            jt_a, jt_b = rule[1], rule[2]
            ja, jb = joints[jt_a], joints[jt_b]
            if joint_ok(ja) and joint_ok(jb):
                a = np.array([ja.Position.x, ja.Position.y, ja.Position.z], dtype=np.float32)
                b = np.array([jb.Position.x, jb.Position.y, jb.Position.z], dtype=np.float32)
                m = midpoint(a, b)
                pose[bone_name] = [float(m[0]), float(m[1]), float(m[2])]
        # w razie potrzeby można dodać inne reguły (np. projekcja)

    return pose


def hand_state_to_str(state: int) -> str:
    m = {
        PyKinectV2.HandState_Unknown: "unknown",
        PyKinectV2.HandState_NotTracked: "not_tracked",
        PyKinectV2.HandState_Open: "open",
        PyKinectV2.HandState_Closed: "closed",
        PyKinectV2.HandState_Lasso: "lasso",
    }
    return m.get(state, "unknown")


def run():
    cfg = CONFIG
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_Depth)

    selector = ParticipantSelector(cfg["roi_rect"], cfg)
    writer = PacketWriter(cfg)

    frame_interval = 1.0 / max(1.0, float(cfg["packets_per_second"]))
    last_sent = 0.0

    win_name = "Kinect ROI (depth)" if HAS_CV2 else None

    try:
        while True:
            if kinect.has_new_body_frame():
                bodies = kinect.get_last_body_frame()
                t_now = time.time()

                if bodies is None:
                    continue

                users_payload = []
                participants = 0

                for i in range(0, kinect.max_body_count):
                    body = bodies.bodies[i]
                    if not body.is_tracked:
                        continue

                    # ograniczenie liczby osób, jeśli chcesz
                    if participants >= cfg["max_people"]:
                        break

                    is_participant = selector.decide(body, kinect, t_now)

                    # zbierz pozę (kości)
                    pose = build_pose_dict(body, kinect)

                    # dłonie
                    hands = []
                    if cfg["include_hands"]:
                        hands.append({
                            "side": "left",
                            "state": hand_state_to_str(body.hand_left_state)
                        })
                        hands.append({
                            "side": "right",
                            "state": hand_state_to_str(body.hand_right_state)
                        })

                    user_obj = {
                        "userid": str(body.TrackingId),  # unikalne ID z Kinecta dla danej sesji
                        "participant": bool(is_participant),
                        "pose": pose,
                        "hands": hands,
                    }
                    users_payload.append(user_obj)

                    if is_participant:
                        participants += 1

                # ogranicz tempo pakietów
                if (t_now - last_sent) >= frame_interval and users_payload:
                    packet = {
                        "guid": str(uuid.uuid4()),
                        "timestamp": int(t_now * 1000),
                        "users": users_payload
                    }
                    writer.send(packet)
                    last_sent = t_now

            # (opcjonalny) podgląd ROI na klatce głębokości
            if HAS_CV2 and kinect.has_new_depth_frame():
                depth_frame = kinect.get_last_depth_frame()
                depth_img = np.array(depth_frame, dtype=np.uint16).reshape((424, 512))
                # prosta normalizacja do 8-bit
                img8 = np.clip((depth_img / 8), 0, 255).astype(np.uint8)

                # ROI rectangle
                x0, y0, x1, y1 = cfg["roi_rect"]
                cv2.rectangle(img8, (int(x0), int(y0)), (int(x1), int(y1)), 255, 2)

                cv2.imshow(win_name, img8)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        if HAS_CV2:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        try:
            kinect.close()
        except Exception:
            pass


if __name__ == "__main__":
    run()
