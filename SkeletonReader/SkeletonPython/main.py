import cv2
import json
import socket

from skeleton_module import SkeletonTracker, CFG  # CFG pochodzi z config.json (tworzy się automatycznie)

def send_udp_payload(payload_obj, host: str, port: int):
    """Wysyła JSON (np. listę osób) jako jeden pakiet UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
    sock.sendto(payload, (host, port))

def main():
    # --- config (z modułu) ---
    cam_cfg = CFG.get("camera", {})
    out_cfg = CFG.get("output", {})
    prev_cfg = CFG.get("preview", {})

    cam_index = int(cam_cfg.get("index", 0))
    cam_w = int(cam_cfg.get("width", 640))
    cam_h = int(cam_cfg.get("height", 480))
    mirror_preview = bool(cam_cfg.get("mirror_preview", False))

    output_mode = str(out_cfg.get("mode", "udp")).lower()
    udp_host = str(out_cfg.get("udp_host", "127.0.0.1"))
    udp_port = int(out_cfg.get("udp_port", 4242))

    preview_enabled = bool(prev_cfg.get("enabled", True))
    window_name = str(prev_cfg.get("window_name", "Skeleton preview"))

    # --- kamera otwierana TYLKO TU ---
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć kamery index={cam_index}")

    tracker = SkeletonTracker()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # przetwarzanie klatki (moduł)
            entries, frame_vis = tracker.process_frame(
                frame,
                draw_overlay=preview_enabled
            )

            # --- UDP: wysyłka listy osób w jednej paczce ---
            if entries and output_mode in ("udp", "both"):
                send_udp_payload(entries, udp_host, udp_port)

            # --- podgląd ---
            if preview_enabled and frame_vis is not None:
                if mirror_preview:
                    frame_vis = cv2.flip(frame_vis, 1)

                cv2.imshow(window_name, frame_vis)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # nawet bez preview warto mieć możliwość wyjścia
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
