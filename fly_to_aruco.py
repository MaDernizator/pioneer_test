import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Set

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

                # защита от пустого кадра
                if frame is None or (hasattr(frame, "size") and frame.size == 0):
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
                print(f"[Video] OpenCV error (ignored): {e}")
                time.sleep(0.02)
            except Exception as e:
                print(f"[Video] Unexpected error (ignored): {e}")
                time.sleep(0.05)

    def stop(self):
        self.running = False


def choose_next_marker(markers: Dict[int, MarkerInfo], visited: Set[int]) -> Optional[int]:
    candidates = [mid for mid in markers.keys() if mid not in visited]
    return min(candidates) if candidates else None


class DroneCommander:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.p = None
        if not dry_run:
            if Pioneer is None:
                raise RuntimeError("pioneer_sdk недоступен, но DRY_RUN=False")
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
    DRY_RUN = True   # True: безполётный режим (не шлём команды)

    # =========================
    # НАСТРОЙКИ
    # =========================
    TAKEOFF_HEIGHT = 1.5

    CENTER_TOL_PX = 40   # допуск по центру
    LOOP_DT = 0.05       # частота команд

    SPEED = 0.25          # м/с, модуль скорости в плоскости XY (регулируемая константа)

    SHOW_WINDOW = True

    # Цвета точки в центре (BGR)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    tracker = ArucoTrackerThread()
    tracker.start()

    cmd = DroneCommander(dry_run=DRY_RUN)

    visited: Set[int] = set()
    last_target: Optional[int] = None

    # держать зелёную точку 3 секунды после “попадания”
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

            if frame is None or (hasattr(frame, "size") and frame.size == 0):
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            cx0, cy0 = int(w / 2), int(h / 2)

            # По умолчанию точка красная
            center_color = RED

            # Если идёт “зелёная пауза 3с” — точка зелёная и команд нет
            if time.time() < hold_green_until:
                center_color = GREEN
                cmd.set_manual_speed_body_fixed(0, 0, 0, 0)
            else:
                # Нет маркеров — стоим
                if not markers:
                    cmd.set_manual_speed_body_fixed(0, 0, 0, 0)
                else:
                    target_id = choose_next_marker(markers, visited)

                    # Все видимые посещены — стоим
                    if target_id is None:
                        cmd.set_manual_speed_body_fixed(0, 0, 0, 0)
                    else:
                        if target_id != last_target:
                            print(f"[INFO] New target marker: {target_id}")
                            last_target = target_id

                        mi = markers[target_id]

                        # Ошибки в пикселях относительно центра кадра
                        dx_px = mi.cx - cx0   # + если маркер правее
                        dy_px = mi.cy - cy0   # + если маркер ниже

                        centered = (abs(dx_px) <= CENTER_TOL_PX and abs(dy_px) <= CENTER_TOL_PX)

                        if centered and (target_id not in visited):
                            # Над маркером -> зелёная точка
                            center_color = GREEN

                            # Останов + отметить посещение + 3 секунды зелёный режим
                            cmd.set_manual_speed_body_fixed(0, 0, 0, 0)
                            visited.add(target_id)
                            print(f"[INFO] Marker {target_id} centered. Visited: {sorted(list(visited))}")
                            hold_green_until = time.time() + 3.0
                        else:
                            # Нужно смещаться -> красная точка (по умолчанию)
                            # Вектор направления в XY:
                            # x вправо, y вперёд (как ты указал, совпадает для камеры и дрона)
                            dx = dx_px / (w / 2.0)   # примерно [-1..1]
                            dy = dy_px / (h / 2.0)   # примерно [-1..1]

                            norm = (dx * dx + dy * dy) ** 0.5

                            if norm < 1e-6:
                                vx = 0.0
                                vy = 0.0
                            else:
                                # Нормируем направление и задаём модуль SPEED
                                vx = SPEED * (dx / norm)
                                vy = SPEED * (dy / norm)

                            # vz и yaw всегда 0
                            cmd.set_manual_speed_body_fixed(vx=vx, vy=vy, vz=0, yaw_rate=0)

            # Отрисовка (точка центра и инфо)
            if SHOW_WINDOW:
                cv2.circle(frame, (cx0, cy0), 7, center_color, -1)

                # небольшая диагностика
                txt = f"Visited: {len(visited)}"
                cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if last_target is not None:
                    cv2.putText(frame, f"Target: {last_target}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("aruco_nav", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
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