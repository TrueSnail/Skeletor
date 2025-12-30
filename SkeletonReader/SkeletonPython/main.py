import cv2
import json
import socket
from pathlib import Path

from skeleton_module import SkeletonTracker, CFG


# ---------- I/O helpers ----------

def send_udp_payload(payload_obj, host: str, port: int):
    """Wysyła JSON (np. listę osób) jako jeden pakiet UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
    sock.sendto(payload, (host, port))


def append_json_frame(frame_entries, path: str):
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

    data.append(frame_entries)

    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------- main ----------

def main():
    cam_cfg = CFG.get("camera", {})
    mp_cfg = CFG.get("mediapipe", {})
    out_cfg = CFG.get("output", {})
    prev_cfg = CFG.get("preview", {})

    cam_index = int(cam_cfg.get("index", 0))
    cam_w = int(cam_cfg.get("width", 640))
    cam_h = int(cam_cfg.get("height", 480))
    mirror_preview = bool(cam_cfg.get("mirror_preview", False))

    # Multi-person limit
    num_poses = int(mp_cfg.get("num_poses", 1))

    output_mode = str(out_cfg.get("mode", "udp")).lower()
    json_path = str(out_cfg.get("json_path", "poses_frames.json"))
    udp_host = str(out_cfg.get("udp_host", "127.0.0.1"))
    udp_port = int(out_cfg.get("udp_port", 5005))

    preview_enabled = bool(prev_cfg.get("enabled", True))
    window_name = str(prev_cfg.get("window_name", "Skeleton preview"))

    # Kamera otwierana tylko tutaj
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć kamery index={cam_index}")

    # Tracker korzysta z configu z modułu (model, smoothing, flipy itp.)
    tracker = SkeletonTracker()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # entries: lista osób (każda ma userid + pose)
            entries, frame_vis = tracker.process_frame(frame, draw_overlay=preview_enabled)

            if entries and len(entries) > num_poses:
                entries = entries[:num_poses]

            # --- UDP (jedna paczka na klatkę) ---
            if entries and output_mode in ("udp", "both"):
                send_udp_payload(entries, udp_host, udp_port)

            # --- JSON (zapis klatkowy) ---
            if entries and output_mode in ("json", "both"):
                append_json_frame(entries, json_path)

            # --- preview ---
            if preview_enabled and frame_vis is not None:
                if mirror_preview:
                    frame_vis = cv2.flip(frame_vis, 1)
                cv2.imshow(window_name, frame_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
