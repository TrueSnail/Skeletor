# File: CharucoCalib.py
# -----------------------------------------------------------
# Automatyczna kalibracja kamery z użyciem ChArUco (bez klikania).
# Działanie:
#  1) Uruchom skrypt – złap planszę ChArUco w kadr;
#  2) Skrypt sam zbierze TARGET_FRAMES „dobrych” klatek (różne pozycje/odległości);
#  3) Po zebraniu wykona kalibrację i zapisze camera_params.npz;
#  4) Opcjonalnie pokaże krótki podgląd undistort.
# -----------------------------------------------------------

import os
import glob
import time
import numpy as np
import cv2

# =============== USTAWIENIA ===============

# Kamera
CAM_INDEX = 0

# Parametry planszy (MUSZĄ odpowiadać wydrukowi)
SQUARES_X = 8
SQUARES_Y = 11
SQUARE_LENGTH_MM = 12.0
MARKER_LENGTH_MM = 9.0   # typowo 75% długości pola, ale zależy od tablicy
DICT_NAME = "DICT_4X4_50"

# Docelowa liczba klatek do kalibracji i limity
TARGET_FRAMES = 30            # ile dobrych ujęć zebrać
MAX_DURATION_SEC = 180        # maks. czas zbierania
MIN_CORNERS = 20              # minimalna liczba narożników ChArUco na klatkę
MIN_TIME_BETWEEN_S = 0.5      # minimalny odstęp czasowy między zapisami
MIN_MOVE_PX = 25.0            # minimalne „przesunięcie” planszy względem poprzedniego zapisu
MIN_AREA_CHANGE = 0.10        # minimalna zmiana pola (10%) względem ostatniego zapisu

# Ścieżki/wyjścia
CAP_DIR = "calibration_images_auto"
PARAMS_NPZ = "camera_params.npz"
SHOW_UNDISTORT_PREVIEW = True
UNDISTORT_PREVIEW_SEC = 6

# Rozmiar generowanej grafiki planszy (opcjonalnie: do wydruku)
BOARD_PNG = "charuco_board.png"
BOARD_IMG_SIZE = (600, 900)  # (szer, wys) px

# =========================================


def _get_aruco():
    try:
        from cv2 import aruco
    except Exception as e:
        raise RuntimeError(
            "cv2.aruco niedostępne. Zainstaluj: pip install opencv-contrib-python"
        ) from e
    return aruco


def make_charuco_board():
    aruco = _get_aruco()
    dictionary = getattr(aruco, DICT_NAME)
    dictionary = aruco.getPredefinedDictionary(dictionary)
    board = aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        squareLength=SQUARE_LENGTH_MM,
        markerLength=MARKER_LENGTH_MM,
        dictionary=dictionary
    )
    return board, dictionary


def save_board_png(path=BOARD_PNG):
    board, _ = make_charuco_board()
    img = board.draw(BOARD_IMG_SIZE)
    cv2.imwrite(path, img)
    print(f"[OK] Zapisano planszę ChArUco: {path}  (wydrukuj w 100% skali)")


def _detect_charuco(gray, board, dictionary):
    aruco = _get_aruco()
    corners, ids, _ = aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) == 0:
        return None, None, None, None

    aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None)

    ok, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
        markerCorners=corners, markerIds=ids, image=gray, board=board)
    if not ok or ch_corners is None or ch_ids is None:
        return corners, ids, None, None
    return corners, ids, ch_corners, ch_ids


def _charuco_stats(ch_corners):
    """
    Zwraca proste miary różnorodności:
    - centroid (cx, cy)
    - pole bbox (area_px)
    """
    pts = ch_corners.reshape(-1, 2)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    x0, y0 = np.min(pts[:, 0]), np.min(pts[:, 1])
    x1, y1 = np.max(pts[:, 0]), np.max(pts[:, 1])
    area = float((x1 - x0) * (y1 - y0))
    return (cx, cy), area


def calibrate_from_folder(image_glob, save_npz=PARAMS_NPZ):
    board, dictionary = make_charuco_board()
    images = sorted(glob.glob(image_glob))
    if not images:
        raise RuntimeError(f"Brak zdjęć do kalibracji w {image_glob}")

    all_cc = []
    all_ids = []
    image_size = None
    used = 0

    for path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Nie można wczytać: {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        _, _, ch_corners, ch_ids = _detect_charuco(gray, board, dictionary)
        if ch_corners is not None and ch_ids is not None and len(ch_corners) >= MIN_CORNERS:
            all_cc.append(ch_corners)
            all_ids.append(ch_ids)
            used += 1
        else:
            print(f"[INFO] Pominięto (za mało narożników): {path}")

    if used < max(5, int(TARGET_FRAMES * 0.5)):
        raise RuntimeError(f"Za mało użytecznych klatek: {used} (min {max(5, int(TARGET_FRAMES*0.5))})")

    aruco = _get_aruco()
    rms, K, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=all_cc,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    np.savez(save_npz,
             K=K, dist=dist, rms=rms, image_size=np.array(image_size),
             board_cfg=np.array([SQUARES_X, SQUARES_Y, SQUARE_LENGTH_MM, MARKER_LENGTH_MM], dtype=float),
             dict_name=DICT_NAME)
    print("[OK] Kalibracja zakończona.")
    print(f"  RMS: {rms:.4f}")
    print(f"  K =\n{K}")
    print(f"  dist = {dist.ravel()}")
    print(f"[OK] Zapisano: {save_npz}")


def undistort_preview(npz_path=PARAMS_NPZ, cam_index=CAM_INDEX, seconds=UNDISTORT_PREVIEW_SEC):
    if not os.path.exists(npz_path):
        print(f"[INFO] Brak {npz_path} – pomijam podgląd undistort.")
        return
    data = np.load(npz_path, allow_pickle=True)
    K = data["K"]
    dist = data["dist"]

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("[WARN] Nie można otworzyć kamery do podglądu undistort.")
        return

    t0 = time.time()
    print("[INFO] Podgląd undistort...")
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        und = cv2.undistort(frame, K, dist)
        vis = np.hstack([frame, und])
        cv2.putText(vis, "ORIGINAL | UNDISTORTED",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Undistort preview", vis)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break
    cap.release()
    cv2.destroyWindow("Undistort preview")


def auto_capture_and_calibrate():
    """
    Automatyczne zbieranie klatek:
    - zapisuje tylko wtedy, gdy:
        * wykryto >= MIN_CORNERS narożników,
        * minęło MIN_TIME_BETWEEN_S od ostatniego zapisu,
        * centroid przesunął się o >= MIN_MOVE_PX LUB
          pole bbox różni się o >= MIN_AREA_CHANGE.
    - kończy po zebraniu TARGET_FRAMES lub po MAX_DURATION_SEC.
    Następnie uruchamia kalibrację i (opcjonalnie) undistort preview.
    """
    os.makedirs(CAP_DIR, exist_ok=True)
    # wyczyść poprzednie (opcjonalnie – skomentuj jeśli chcesz zachować)
    for p in glob.glob(os.path.join(CAP_DIR, "*.png")):
        try:
            os.remove(p)
        except:
            pass

    board, dictionary = make_charuco_board()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć kamery {CAM_INDEX}")

    saved = 0
    last_time = 0.0
    last_centroid = None
    last_area = None
    t_start = time.time()

    print("[INFO] Start automatycznego zbierania klatek...")
    print(f"       Cel: {TARGET_FRAMES} ujęć, limit czasu: {MAX_DURATION_SEC} s")
    print("       Trzymaj planszę w różnych miejscach i odległościach kadru.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Brak klatki z kamery.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, ch_corners, ch_ids = _detect_charuco(gray, board, dictionary)

        status_txt = "Szukam planszy..."
        ready_to_save = False

        if ch_corners is not None and ch_ids is not None and len(ch_corners) >= MIN_CORNERS:
            (cx, cy), area = _charuco_stats(ch_corners)

            # odstęp czasowy
            enough_time = (time.time() - last_time) >= MIN_TIME_BETWEEN_S

            # zróżnicowanie pozycji/rozmiaru
            moved_ok = True
            area_ok = True
            if last_centroid is not None:
                moved_ok = (np.hypot(cx - last_centroid[0], cy - last_centroid[1]) >= MIN_MOVE_PX)
            if last_area is not None:
                # względna zmiana pola
                area_change = abs(area - last_area) / max(1.0, last_area)
                area_ok = (area_change >= MIN_AREA_CHANGE)

            ready_to_save = enough_time and (moved_ok or area_ok)
            status_txt = f"ChArUco OK: {len(ch_corners)} narożników | move:{moved_ok} areaΔ:{area_ok}"

            # narysuj wykrytą planszę (dla podglądu)
            cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), 2)
            x0y0 = np.min(ch_corners.reshape(-1,2), axis=0).astype(int)
            x1y1 = np.max(ch_corners.reshape(-1,2), axis=0).astype(int)
            cv2.rectangle(frame, tuple(x0y0), tuple(x1y1), (0, 255, 255), 2)

            if ready_to_save:
                fname = os.path.join(CAP_DIR, f"calib_{saved:03d}.png")
                cv2.imwrite(fname, frame)
                saved += 1
                last_time = time.time()
                last_centroid = (cx, cy)
                last_area = area
                status_txt = f"[ZAPISANO] {fname}  ({saved}/{TARGET_FRAMES})"
        else:
            # brak wykrycia / za mało narożników
            pass

        # overlay
        elapsed = time.time() - t_start
        cv2.putText(frame, f"{status_txt}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        cv2.putText(frame, f"Zebrano: {saved}/{TARGET_FRAMES} | Czas: {int(elapsed)}s/{MAX_DURATION_SEC}s",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        cv2.imshow("Auto ChArUco Capture", frame)

        # warunki zakończenia zbierania
        if saved >= TARGET_FRAMES or elapsed >= MAX_DURATION_SEC:
            break

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyWindow("Auto ChArUco Capture")

    print(f"[INFO] Zebrano {saved} / {TARGET_FRAMES} klatek. Start kalibracji...")
    calibrate_from_folder(os.path.join(CAP_DIR, "*.png"), PARAMS_NPZ)

    if SHOW_UNDISTORT_PREVIEW:
        undistort_preview(PARAMS_NPZ, CAM_INDEX, UNDISTORT_PREVIEW_SEC)

    print("[OK] Gotowe.")


def main():
    # (opcjonalnie) zapisz plik z planszą do wydruku
    if not os.path.exists(BOARD_PNG):
        try:
            save_board_png(BOARD_PNG)
        except Exception as e:
            print("[WARN] Nie zapisano planszy (to nie blokuje dalszych kroków):", e)

    # auto-capture + calibrate
    auto_capture_and_calibrate()


if __name__ == "__main__":
    # krótka diagnostyka aruco
    try:
        from cv2 import aruco  # noqa: F401
    except Exception as e:
        print("[ERR] cv2.aruco niedostępne. Zainstaluj: pip install opencv-contrib-python")
        raise
    main()
