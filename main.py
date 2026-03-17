import cv2
import sys
import os
import time

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from face_detection import FaceDetector
from face_recognition_module import FaceRecognizer
from attendance_manager import AttendanceManager
from anti_spoofing import LivenessDetector

# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR       = os.path.join(os.path.dirname(__file__), 'data', 'raw')
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), 'embeddings.pkl')
ATTENDANCE_FILE = os.path.join(os.path.dirname(__file__), 'attendance_log.csv')

# UI colour palette (BGR)
COLOR_LIVE    = (0,   220,  80)   # vivid green
COLOR_UNKNOWN = (0,    60, 220)   # blue-ish
COLOR_SPOOF   = (0,   165, 255)   # orange
COLOR_INFO    = (220, 220, 220)   # light grey
COLOR_FPS     = (0,   255, 255)   # cyan
COLOR_STATUS  = (80,  255, 180)   # mint green
COLOR_BANNER  = (40,   40,  40)   # dark header bg (drawn as rectangle)


def draw_banner(frame: "cv2.Mat", text: str, fps: float) -> None:
    """Draw a dark top banner with system title and FPS."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 38), COLOR_BANNER, -1)
    cv2.putText(frame, text, (8, 26), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (w - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_FPS, 1, cv2.LINE_AA)


def draw_face_overlay(frame, x, y, w, h, label: str, distance: float,
                      is_live: bool, liveness_msg: str, status_msg: str,
                      color: tuple) -> None:
    """
    Draw a polished bounding box and informational text for one detected face.
    Replaces the old plain-rectangle draw calls.
    """
    # Corner-bracket style box (more modern than full rectangle)
    thickness = 2
    corner_len = max(12, min(w, h) // 5)

    # Top-left
    cv2.line(frame, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + corner_len), color, thickness)
    # Top-right
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness)
    # Bottom-left
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness)
    # Bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

    # Semi-transparent background strip for the name label
    text_bg_top = max(0, y - 32)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, text_bg_top), (x + w, y), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Name + distance
    name_text = f"{label}  d={distance:.2f}"
    cv2.putText(frame, name_text, (x + 4, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Liveness line below box
    cv2.putText(frame, liveness_msg, (x + 4, y + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_INFO, 1, cv2.LINE_AA)

    # Status line (Marked Present / Already Marked / Spoof…)
    if status_msg:
        cv2.putText(frame, status_msg, (x + 4, y + h + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_STATUS, 1, cv2.LINE_AA)


def ensure_data_dir(path: str) -> None:
    if not os.path.exists(path):
        print(f"[INFO] Data directory '{path}' not found – creating it.")
        os.makedirs(path)
        print("[INFO] Directory created. Run 'python src/data_collection.py' to add users.")


def main():
    print("=" * 55)
    print("  Smart Attendance System  –  initialising …")
    print("=" * 55)

    # ── Component init ────────────────────────────────────────────────────
    detector      = FaceDetector(method='haar')
    recognizer    = FaceRecognizer(embeddings_path=EMBEDDINGS_PATH)
    attendance_mgr = AttendanceManager(file_path=ATTENDANCE_FILE)
    liveness_det  = LivenessDetector(threshold=60.0)   # instantiate ONCE

    # ── Training / embedding check ────────────────────────────────────────
    if not os.path.exists(EMBEDDINGS_PATH) or len(recognizer.known_names) == 0:
        print("[WARN] Embeddings missing or empty. Starting training …")
        ensure_data_dir(DATA_DIR)
        recognizer.train(DATA_DIR)
        recognizer.load_embeddings()

    # ── Webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    # Slightly higher resolution for better recognition
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    print("\n[READY] System active. Press 'q' to quit, 'r' to retrain.\n")

    marked_flash: dict[str, float] = {}   # label → timestamp for brief flash effect
    FLASH_DURATION = 2.5                  # seconds

    fps    = 0.0
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed – retrying …")
            continue

        # ── FPS ─────────────────────────────────────────────────────────
        t_now = time.time()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        # ── Face detection ───────────────────────────────────────────────
        faces = detector.detect(frame)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size == 0:
                continue

            # ── Liveness check ─────────────────────────────────────────
            is_live_flag, liveness_msg = liveness_det.is_live(face_roi)

            # ── Recognition ────────────────────────────────────────────
            face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            name_label, distance = recognizer.recognize(face_roi_rgb)

            # ── Attendance & UI state ──────────────────────────────────
            status_msg = ""
            color      = COLOR_UNKNOWN

            if name_label != "Unknown":
                if is_live_flag:
                    color = COLOR_LIVE
                    try:
                        parts  = name_label.split('_')
                        s_id   = parts[0] if len(parts) >= 2 else "N/A"
                        s_name = '_'.join(parts[1:]) if len(parts) >= 2 else name_label

                        success, msg = attendance_mgr.mark_attendance(s_name, s_id)
                        status_msg   = msg

                        if success:
                            marked_flash[name_label] = time.time()
                            print(f"[ATTENDANCE] {s_name} (ID {s_id}) marked at {time.strftime('%H:%M:%S')}")

                    except Exception as exc:
                        print(f"[ERROR] Parsing label: {exc}")
                else:
                    color      = COLOR_SPOOF
                    status_msg = "Spoof Detected"
            else:
                liveness_msg = ""   # Don't clutter for unknowns

            # Flash highlight border on fresh mark
            if name_label in marked_flash:
                elapsed = time.time() - marked_flash[name_label]
                if elapsed < FLASH_DURATION:
                    # Pulse alpha based on time
                    alpha   = max(0.0, 1.0 - elapsed / FLASH_DURATION)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x, y), (x + w, y + h),
                                  COLOR_STATUS, int(4 * alpha) + 1)
                    cv2.addWeighted(overlay, alpha * 0.4, frame, 1 - alpha * 0.4, 0, frame)
                else:
                    del marked_flash[name_label]

            # ── Draw overlay ────────────────────────────────────────────
            draw_face_overlay(frame, x, y, w, h,
                              name_label, distance,
                              is_live_flag, liveness_msg,
                              status_msg, color)

        # ── Top banner ──────────────────────────────────────────────────
        today_count = attendance_mgr.get_summary()['unique_students_today']
        banner_text = (f"Smart Attendance System   |  "
                       f"Today: {today_count} student(s)  |  "
                       f"Faces: {len(faces)}")
        draw_banner(frame, banner_text, fps)

        # ── Key-help footer ─────────────────────────────────────────────
        h_frame = frame.shape[0]
        cv2.putText(frame, "q: quit   r: retrain",
                    (8, h_frame - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (120, 120, 120), 1, cv2.LINE_AA)

        cv2.imshow('Smart Attendance System', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\n[INFO] Retraining …")
            recognizer.train(DATA_DIR)
            recognizer.load_embeddings()
            attendance_mgr._load_todays_cache()
            print("[INFO] Retrain complete.\n")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Session ended.")
    summary = attendance_mgr.get_summary()
    print(f"[INFO] Students marked today: {summary['unique_students_today']}")
    print(f"[INFO] Total records in log : {summary['total_records']}")


if __name__ == "__main__":
    main()
