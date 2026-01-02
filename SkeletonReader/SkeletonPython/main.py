import cv2
import json
import socket
import time
from pathlib import Path

from skeleton_module import SkeletonTracker, CFG


# ===== JSON frames: Data (string) + DelayMs =====
_last_json_ts = None

def append_json_frame_with_delay(entries, path: str):
    global _last_json_ts

    now = time.time()
    if _last_json_ts is None:
        delay_ms = 0
    else:
        delay_ms = int((now - _last_json_ts) * 1000)
    _last_json_ts = now

    record = {
        "Data": json.dumps(entries, ensure_ascii=False),
        "DelayMs": delay_ms
    }

    p = Path(path)
    try:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        else:
            data = []
    except json.JSONDecodeError:
        data = []

    data.append(record)

    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def send_udp_payload(payload_obj, host: str, port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
    sock.sendto(payload, (host, port))


def main():
    cam_cfg = CFG.get("camera", {})
    mp_cfg = CFG.get("mediapipe", {})
    out_cfg = CFG.get("output", {})
    prev_cfg = CFG.get("preview", {})

    # Multi-person limit
    num_poses = int(mp_cfg.get("num_poses", 1))

    # Output
    output_mode = str(out_cfg.get("mode", "udp")).lower()
    json_path = str(out_cfg.get("json_path", "poses_frames.json"))
    udp_host = str(out_cfg.get("udp_host", "127.0.0.1"))
    udp_port = int(out_cfg.get("udp_port", 5005))

    # Preview
    preview_enabled = bool(prev_cfg.get("enabled", True))
    window_name = str(prev_cfg.get("window_name", "Skeleton preview"))
    mirror_preview = bool(cam_cfg.get("mirror_preview", False))

    # Input source: camera or video
    source = str(cam_cfg.get("source", "camera")).lower()
    video_path = str(cam_cfg.get("video_path", ""))

    # Open input
    if source == "video":
        if not video_path:
            raise RuntimeError('camera.source="video" ale camera.video_path jest pusty')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Nie można otworzyć pliku video: {video_path}")
        # UWAGA: nie wymuszamy width/height na pliku - bierzemy natywne
        print(f"[INFO] Źródło: video: {video_path}")
    else:
        cam_index = int(cam_cfg.get("index", 0))
        cam_w = int(cam_cfg.get("width", 640))
        cam_h = int(cam_cfg.get("height", 480))
        cap = cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        if not cap.isOpened():
            raise RuntimeError(f"Nie można otworzyć kamery index={cam_index}")
        print(f"[INFO] Źródło: kamera index={cam_index} {cam_w}x{cam_h}")

    tracker = SkeletonTracker()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # koniec pliku video albo błąd kamery
                break

            entries, frame_vis = tracker.process_frame(frame, draw_overlay=preview_enabled)

            # bezpieczeństwo: nie więcej niż num_poses
            if entries and len(entries) > num_poses:
                entries = entries[:num_poses]

            # UDP
            if entries and output_mode in ("udp", "both"):
                send_udp_payload(entries, udp_host, udp_port)

            # JSON (Data string + DelayMs)
            if entries and output_mode in ("json", "both"):
                append_json_frame_with_delay(entries, json_path)

            # preview
            if preview_enabled and frame_vis is not None:
                if mirror_preview:
                    frame_vis = cv2.flip(frame_vis, 1)
                cv2.imshow(window_name, frame_vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                # przy video bez okna też daj możliwość przerwania
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
