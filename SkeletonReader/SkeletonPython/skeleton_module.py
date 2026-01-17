from __future__ import annotations

import os
import json
import time
import uuid
import math
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark

CONFIG_PATH = "config.json"

DEFAULT_CONFIG = {
    "camera": {
        "source": "camera",
        "video_path": "",
        "index": 0,
        "width": 640,
        "height": 480,
        "mirror_preview": False
    },
    "mediapipe": {
        "model_path": "pose_landmarker_lite.task",
        "num_poses": 1,
        "min_detection_confidence": 0.5,
        "min_presence_confidence": 0.5,
        "min_tracking_confidence": 0.7
    },
    "world": {
        "scale_world": 1,
        "flip_x": True,
        "flip_y": True,
        "flip_z": False
    },
    "output": {
        "mode": "udp",
        "udp_host": "127.0.0.1",
        "udp_port": 4242
    },
    "preview": {
        "enabled": True,
        "draw_2d_skeleton": True,
        "draw_fps": True,
        "window_name": "Skeleton Preview"
    },
    "smoothing": {
        "enabled_3d": True,
        "enabled_2d": False,
        "oneeuro_3d": {
            "min_cutoff": 1.2,
            "beta": 0.02,
            "d_cutoff": 1.0
        },
        "oneeuro_2d": {
            "min_cutoff": 2.0,
            "beta": 0.02,
            "d_cutoff": 1.0
        }
    }
}

def ensure_config(path: str) -> dict:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Utworzono domyślny {path}")
        return DEFAULT_CONFIG

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

CFG = ensure_config(CONFIG_PATH)

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
    (11,13),(13,15),(12,14),(14,16),(11,12),
    (23,24),(11,23),(12,24),(23,25),(25,27),(24,26),(26,28)
]

def midpoint2(a, b):
    bone = [(a[i] + b[i]) / 2.0 for i in range(2)]
    bone.append(0)
    return bone

def bones_from_points2d(points, bone):
    bones = []
    if bone in BONE_MAP and isinstance(BONE_MAP[bone], tuple):
        _, i, j = BONE_MAP[bone]
        bones = midpoint2(points[i], points[j])

    return bones


def midpoint3(a, b):
    return [(a[i] + b[i]) / 2.0 for i in range(3)]

def bones_from_points(points):
    bones = {}
    for name, spec in BONE_MAP.items():
        if isinstance(spec, tuple):
            _, i, j = spec
            bones[name] = midpoint3(points[i], points[j])
        else:
            bones[name] = points[spec]
    return bones

def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))

class OneEuro1D:
    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0

    def filter(self, x, dt):
        if self.x_prev is None:
            self.x_prev = x
            return x

        dx = (x - self.x_prev) / dt
        dx_hat = self.dx_prev + (dx - self.dx_prev) * _alpha(self.d_cutoff, dt)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self.x_prev + (x - self.x_prev) * _alpha(cutoff, dt)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

class SkeletonTracker:

    def __init__(self):
        mp_cfg = CFG["mediapipe"]

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=mp_cfg["model_path"]),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=mp_cfg["num_poses"],
            min_pose_detection_confidence=mp_cfg["min_detection_confidence"],
            min_pose_presence_confidence=mp_cfg["min_presence_confidence"],
            min_tracking_confidence=mp_cfg["min_tracking_confidence"]
        )

        self.landmarker = PoseLandmarker.create_from_options(options)
        self.index_to_userid = {}
        self.prev_ts = None
        self.filters_3d = defaultdict(dict)

        self.user_offsets = {}

    def close(self):
        self.landmarker.close()

    def process_frame(self, frame_bgr, draw_overlay=False):
        now = time.time()
        dt = 1/30 if self.prev_ts is None else max(1e-3, now - self.prev_ts)
        self.prev_ts = now

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts_ms = int(time.time() * 1000)

        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        entries = []
        if result and result.pose_world_landmarks:
            for idx, (local_lms, world_lms) in enumerate(zip(result.pose_landmarks, result.pose_world_landmarks)):
                if idx not in self.index_to_userid:
                    self.index_to_userid[idx] = str(uuid.uuid4())
                userid = self.index_to_userid[idx]

                pts = []
                for (lm_local, lm_world) in zip(local_lms, world_lms):
                    x,y,z = [0,0,0]

                    if CFG["world"]["flip_x"]: x = -lm_world.x
                    if CFG["world"]["flip_y"]: y = -lm_world.y
                    if CFG["world"]["flip_z"]: z = -lm_world.z

                    pts.append([
                        x * CFG["world"]["scale_world"],
                        y * CFG["world"]["scale_world"],
                        z * CFG["world"]["scale_world"]
                    ])

                bones = bones_from_points(pts)


                bones["Pelvis"] = midpoint2([local_lms[BONE_MAP["Pelvis"][1]].x, local_lms[BONE_MAP["Pelvis"][1]].y, 0],
                                            [local_lms[BONE_MAP["Pelvis"][2]].x, local_lms[BONE_MAP["Pelvis"][2]].y, 0])


                if CFG["smoothing"]["enabled_3d"]:
                    p = CFG["smoothing"]["oneeuro_3d"]
                    st = self.filters_3d[userid]
                    for k, v in bones.items():
                        if k not in st:
                            st[k] = (
                                OneEuro1D(p["min_cutoff"], p["beta"], p["d_cutoff"]),
                                OneEuro1D(p["min_cutoff"], p["beta"], p["d_cutoff"]),
                                OneEuro1D(p["min_cutoff"], p["beta"], p["d_cutoff"])
                            )
                        fx, fy, fz = st[k]
                        bones[k] = [
                            fx.filter(v[0], dt),
                            fy.filter(v[1], dt),
                            fz.filter(v[2], dt)
                        ]

                '''PELIVS_Z_OFFSET = 3.6
                if "Pelvis" in bones and PELIVS_Z_OFFSET != 0:
                    if userid not in self.user_offsets:
                        self.user_offsets[userid] = len(self.user_offsets) * PELIVS_Z_OFFSET
                    bones["Pelvis"][2] = (bones["Pelvis"][0] * 1000) + self.user_offsets[userid]'''


                bones["Pelvis"][2] = bones["Pelvis"][0]*10
                bones["Pelvis"][0] = world_lms[10].x - world_lms[11].x


                entries.append({"userid": userid, "pose": bones})

        frame_vis = None
        if draw_overlay and CFG["preview"]["enabled"]:
            frame_vis = frame_bgr.copy()
            if result and result.pose_landmarks:
                h, w = frame_vis.shape[:2]
                for lm_set in result.pose_landmarks:
                    pts = [(int(lm.x*w), int(lm.y*h)) for lm in lm_set]
                    for a, b in POSE_CONNECTIONS:
                        if a < len(pts) and b < len(pts):
                            cv2.line(frame_vis, pts[a], pts[b], (0,255,255), 2)

        return entries, frame_vis
