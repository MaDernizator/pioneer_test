import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import cv2

try:
    from pioneer_sdk import Pioneer, Camera
except Exception:
    Pioneer = None
    Camera = None


@dataclass
class MarkerInfo:
    cx: int
    cy: int


class ArucoTrackerThread(threading.Thread):
    def __init__(self, dict_name=cv2.aruco.DICT_4X4_50):
        super().__init__(daemon=True)
        self.running = True
        self.lock = threading.Lock()

        self.camera = Camera() if Camera is not None else None
        self.latest_frame = None
        self.markers: Dict[int, MarkerInfo] = {}

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def run(self):
        if self.camera is None:
            print("[ERROR] Camera is unavailable (pioneer_sdk import failed).")
            return

        while self.running:
            try:
                frame = self.camera.get_cv_frame()

                # 1) Защита от пустых кадров (None, пустой numpy, 0x0)
                if frame is None:
                    time.sleep(0.01)
                    continue
                if not hasattr(frame, "size") or frame.size == 0:
                    time.sleep(0.01)
                    continue

                corners, ids, _ = self.aruco_detector.detectMarkers(frame)

                markers_local: Dict[int, MarkerInfo] = {}

                if ids is not None and len(corners) > 0:
                    ids_flat = ids.flatten().tolist()
                    for i, marker_id in enumerate(ids_flat):
                        pts = corners[i][0]
                        cx = int((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4.0)
                        cy = int((pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4.0)
                        markers_local[int(marker_id)] = MarkerInfo(cx=cx, cy=cy)

                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    for mid, mi in markers_local.items():
                        cv2.circle(frame, (mi.cx, mi.cy), 5, (0, 0, 255), -1)
                        cv2.putText(frame, str(mid), (mi.cx + 8, mi.cy - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                with self.lock:
                    self.latest_frame = frame
                    self.markers = markers_local

            except cv2.error as e:
                # 2) На всякий случай: не роняем поток из-за OpenCV
                print(f"[Video] OpenCV error (ignored): {e}")
                time.sleep(0.02)
                continue
            except Exception as e:
                # 3) И любые другие ошибки тоже не должны убивать поток
                print(f"[Video] Unexpected error (ignored): {e}")
                time.sleep(0.05)
                continue

    def stop(self):
        self.running = False


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def choose_next_marker(markers: Dict[int, MarkerInfo], visited: Set[int]) -> Optional[int]:
    candidates = [mid for mid in markers.keys() if mid not in visited]
    return min(candidates) if candidates else None


class DroneCommander:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.p = None
        if not dry_run:
            if Pioneer is None:
                raise RuntimeError("pioneer_sdk недоступен, но dry_run=False")
            self.p = Pioneer()

    def arm_takeoff_to_height(self, z: float):
        if self.dry_run:
            print(f"[DRY] arm(), takeoff(), go_to_local_point(z={z})")
            return

        self.p.arm()
        self.p.takeoff()
        self.p.go_to_local_point(x=0, y=0, z=z, yaw=0)
        while not self.p.point_reached():
            time.sleep(0.1)

    def set_manual_speed_body_fixed(self, vx: float, vy: float, vz: float, yaw_rate: float):
        if self.dry_run:
            print(f"[DRY] set_manual_speed_body_fixed(vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}, yaw_rate={yaw_rate:.3f})")
            return
        self.p.set_manual_speed_body_fixed(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)

    def land_and_close(self):
        if self.dry_run:
            print("[DRY] land(), close_connection()")
            return
        self.p.land()
        self.p.close_connection()
        del self.p


if __name__ == "__main__":
    # =========================
    # РЕЖИМЫ
    # =========================
    DRY_RUN = True   # True: безполётный режим

    # =========================
    # НАСТРОЙКИ
    # =========================
    TAKEOFF_HEIGHT = 1.5
    CENTER_TOL_PX = 40
    LOOP_DT = 0.05

    # x вправо (+), y вперёд (+)
    V_MAX_X = 0.35
    V_MAX_Z = 0.25
    V_FORWARD = 0.15

    KX = 0.35
    KZ = 0.25

    SHOW_WINDOW = True

    # Цвета для точки в центре (BGR)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    tracker = ArucoTrackerThread()
    tracker.start()

    cmd = DroneCommander(dry_run=DRY_RUN)

    visited: Set[int] = set()
    last_target: Optional[int] = None

    # Чтобы держать зелёную точку 3 секунды
    hold_green_until = 0.0

    try:
        if not DRY_RUN:
            print("[INFO] Real flight mode")
            cmd.arm_takeoff_to_height(TAKEOFF_HEIGHT)
            print(f"[INFO] Hover reached: z={TAKEOFF_HEIGHT:.2f}m")
        else:
            print("[INFO] DRY_RUN enabled: no arming/takeoff, no commands sent.")

        while True:
            with tracker.lock:
                frame = tracker.latest_frame
                markers = dict(tracker.markers)

            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            cx0, cy0 = int(w / 2), int(h / 2)

            # По умолчанию точка красная
            center_color = RED

            # Если мы в “3 секунды зелёного” — держим зелёную точку независимо от маркеров
            if time.time() < hold_green_until:
                center_color = GREEN

            # Управляющая логика
            if not markers:
                cmd.set_manual_speed_body_fixed(0, 0, 0, 0)
            else:
                target_id = choose_next_marker(markers, visited)

                if target_id is None:
                    cmd.set_manual_speed_body_fixed(0, 0, 0, 0)
                else:
                    if target_id != last_target:
                        print(f"[INFO] New target marker: {target_id}")
                        last_target = target_id

                    mi = markers[target_id]
                    err_x = mi.cx - (w / 2.0)
                    err_y = mi.cy - (h / 2.0)

                    centered = (abs(err_x) <= CENTER_TOL_PX and abs(err_y) <= CENTER_TOL_PX)

                    # Если маркер в центре И он ещё не посещён — точка зелёная
                    if centered and (target_id not in visited) and (time.time() >= hold_green_until):
                        center_color = GREEN

                        # Останов
                        cmd.set_manual_speed_body_fixed(0, 0, 0, 0)

                        # Отмечаем посещение и 3 секунды держим зелёную
                        visited.add(target_id)
                        print(f"[INFO] Marker {target_id} centered. Visited: {sorted(list(visited))}")
                        hold_green_until = time.time() + 3.0

                    else:
                        # Если сейчас идёт “зелёная пауза” — просто стоим
                        if time.time() < hold_green_until:
                            cmd.set_manual_speed_body_fixed(0, 0, 0, 0)
                        else:
                            # Наведение (точка остаётся красной)
                            err_x_norm = err_x / (w / 2.0)
                            err_y_norm = err_y / (h / 2.0)

                            vx = clamp(KX * err_x_norm, -V_MAX_X, V_MAX_X)  # x вправо(+)
                            vz = clamp(-KZ * err_y_norm, -V_MAX_Z, V_MAX_Z) # при необходимости инвертируй знак
                            vy = V_FORWARD                                 # y вперёд(+)

                            cmd.set_manual_speed_body_fixed(vx=vx, vy=vy, vz=vz, yaw_rate=0)

            # Отрисовка
            if SHOW_WINDOW:
                cv2.circle(frame, (cx0, cy0), 7, center_color, -1)

                # ESC для выхода
                cv2.imshow("aruco_nav", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("[INFO] ESC pressed, exiting.")
                    break

            time.sleep(LOOP_DT)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt -> exiting.")
    finally:
        tracker.stop()
        tracker.join(timeout=2.0)
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        if not DRY_RUN:
            cmd.land_and_close()