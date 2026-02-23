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

from pixel_projector import pixel_to_drone_xy  # <-- единая функция


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
    """
    DRY_RUN=True:
      - НЕ arm/takeoff/land
      - подключаемся (чтобы читать датчик высоты/позицию)
      - go_to_local_point только печатаем
    DRY_RUN=False:
      - обычный полёт
    """
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        if Pioneer is None:
            raise RuntimeError("pioneer_sdk недоступен")
        self.p = Pioneer()

    def arm_takeoff_to_height(self, z: float):
        if self.dry_run:
            print(f"[DRY] Skip arm/takeoff, target_z={z}")
            return
        self.p.arm()
        self.p.takeoff()
        self.p.go_to_local_point(x=0, y=0, z=z, yaw=0)
        while not self.p.point_reached():
            time.sleep(0.1)

    def go_to_local_point(self, x: float, y: float, z: float, yaw: float = 0.0):
        if self.dry_run:
            print(f"[DRY] go_to_local_point(x={x:+.2f}, y={y:+.2f}, z={z:+.2f}, yaw=0.00)")
            return
        self.p.go_to_local_point(x=x, y=y, z=z, yaw=0.0)

    def point_reached(self) -> bool:
        if self.dry_run:
            return False
        return self.p.point_reached()

    def get_pos_lps_xy(self) -> Tuple[float, float]:
        if self.dry_run:
            try:
                pos = self.p.get_pos_lps()
                return float(pos[0]), float(pos[1])
            except Exception:
                return 0.0, 0.0
        pos = self.p.get_pos_lps()
        return float(pos[0]), float(pos[1])

    def get_alt_m(self) -> float:
        # по примечанию: всегда в метрах
        try:
            return float(self.p.get_dist_sensor_data())
        except Exception:
            return 1.0

    def land_and_close(self):
        if not self.dry_run:
            self.p.land()
        self.p.close_connection()
        del self.p


class NavigatorThread(threading.Thread):
    """
    Поток навигации:
      - принимает новую цель
      - отправляет go_to_local_point ровно 1 раз на цель
      - ждёт point_reached
      - пока не достигнуто — игнорирует любые новые цели
    yaw всегда 0
    """
    def __init__(self, cmd: DroneCommander, reach_timeout: float = 15.0, poll_dt: float = 0.1):
        super().__init__(daemon=True)
        self.cmd = cmd
        self.reach_timeout = reach_timeout
        self.poll_dt = poll_dt

        self._cv = threading.Condition()
        self._running = True

        self._goal: Optional[Tuple[float, float, float]] = None  # x,y,z
        self._has_goal = False

        self._state = "IDLE"  # IDLE, MOVING, REACHED, TIMEOUT

    def stop(self):
        with self._cv:
            self._running = False
            self._cv.notify()

    def get_state(self) -> str:
        with self._cv:
            return self._state

    def submit_goal_if_idle(self, x: float, y: float, z: float) -> bool:
        with self._cv:
            if self._state == "MOVING":
                return False
            self._goal = (x, y, z)
            self._has_goal = True
            self._state = "IDLE"
            self._cv.notify()
            return True

    def run(self):
        while True:
            with self._cv:
                while self._running and not self._has_goal:
                    self._cv.wait(timeout=0.5)
                if not self._running:
                    return

                goal = self._goal
                self._has_goal = False
                self._state = "MOVING"

            # 1) отправляем команду (yaw=0)
            try:
                x, y, z = goal
                self.cmd.go_to_local_point(x, y, z, yaw=0.0)
            except Exception as e:
                print(f"[NAV] go_to_local_point error: {e}")

            # 2) ждём достижения
            start = time.time()
            reached = False
            while time.time() - start <= self.reach_timeout:
                try:
                    if self.cmd.point_reached():
                        reached = True
                        break
                except Exception as e:
                    print(f"[NAV] point_reached error: {e}")
                time.sleep(self.poll_dt)

            with self._cv:
                self._state = "REACHED" if reached else "TIMEOUT"


if __name__ == "__main__":
    DRY_RUN = True  # True: без arm/takeoff/land и go_to только печать

    TAKEOFF_HEIGHT = 1.0
    CENTER_TOL_PX = 40
    CENTER_TOL_M = 0.15
    LOOP_DT = 0.02

    MARKER_HOLD_TIME = 3.0

    REACH_TIMEOUT = 15.0
    REACH_POLL_DT = 0.1

    GOAL_TRY_HZ = 5.0
    GOAL_TRY_DT = 1.0 / GOAL_TRY_HZ
    next_goal_try_t = time.monotonic()

    SHOW_WINDOW = True
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    tracker = ArucoTrackerThread()
    tracker.start()

    cmd = DroneCommander(dry_run=DRY_RUN)
    nav = NavigatorThread(cmd=cmd, reach_timeout=REACH_TIMEOUT, poll_dt=REACH_POLL_DT)
    nav.start()

    visited: Set[int] = set()
    last_target: Optional[int] = None
    hold_until = 0.0
    current_state = "SEARCH"

    locked_goal: Optional[Tuple[float, float, float]] = None  # x,y,z

    try:
        if not DRY_RUN:
            print("[INFO] Real flight mode")
            cmd.arm_takeoff_to_height(TAKEOFF_HEIGHT)
            print(f"[INFO] Hover reached: z={TAKEOFF_HEIGHT:.2f}m")
        else:
            print("[INFO] DRY_RUN: no arm/takeoff/land, go_to printed only.")

        while True:
            with tracker.lock:
                frame = tracker.latest_frame
                markers = dict(tracker.markers)

            if frame is None or (hasattr(frame, "size") and frame.size == 0):
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            cx0, cy0 = int(w / 2), int(h / 2)
            center_color = RED

            drone_x, drone_y = cmd.get_pos_lps_xy()
            drone_alt = cmd.get_alt_m()

            nav_state = nav.get_state()

            if locked_goal is not None and nav_state in ("REACHED", "TIMEOUT"):
                locked_goal = None

            if time.time() < hold_until:
                center_color = GREEN
                current_state = "HOVERING"
            else:
                if locked_goal is not None and nav_state == "MOVING":
                    current_state = "FLYING_LOCKED"
                    center_color = RED
                else:
                    if not markers:
                        current_state = "SEARCH"
                        center_color = RED
                    else:
                        target_id = choose_next_marker(markers, visited)

                        if target_id is None:
                            current_state = "SEARCH"
                            center_color = RED
                        else:
                            if target_id != last_target:
                                print(f"[INFO] New target marker: {target_id}")
                                last_target = target_id

                            mi = markers[target_id]
                            dx_px = mi.cx - cx0
                            dy_px = mi.cy - cy0
                            centered_px = (abs(dx_px) <= CENTER_TOL_PX and abs(dy_px) <= CENTER_TOL_PX)

                            # ---- КЛЮЧЕВОЕ: координаты маркера (dx,dy) в метрах относительно центра дрона ----
                            dx_m, dy_m = pixel_to_drone_xy(mi.cx, mi.cy, drone_alt)
                            centered_m = (abs(dx_m) <= CENTER_TOL_M and abs(dy_m) <= CENTER_TOL_M)

                            if centered_px and centered_m and target_id not in visited:
                                visited.add(target_id)
                                hold_until = time.time() + MARKER_HOLD_TIME
                                current_state = "HOVERING"
                                center_color = GREEN
                                print(f"[INFO] Marker {target_id} centered. Visited: {sorted(list(visited))}")
                            else:
                                current_state = "FLYING"
                                center_color = RED

                                target_x = drone_x + dx_m
                                target_y = drone_y + dy_m
                                goal = (target_x, target_y, TAKEOFF_HEIGHT)

                                now = time.monotonic()
                                if now >= next_goal_try_t:
                                    accepted = nav.submit_goal_if_idle(*goal)
                                    if accepted:
                                        locked_goal = goal
                                    next_goal_try_t = now + GOAL_TRY_DT

            if SHOW_WINDOW:
                cv2.circle(frame, (cx0, cy0), 7, center_color, -1)

                cv2.putText(frame, f"Visited: {len(visited)} | State: {current_state} | NAV: {nav_state}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                if last_target is not None:
                    cv2.putText(frame, f"Target: {last_target}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                cv2.putText(frame, f"Pos: ({drone_x:.2f}, {drone_y:.2f}) alt={drone_alt:.2f}m yaw=0",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                if locked_goal is not None:
                    gx, gy, gz = locked_goal
                    cv2.putText(frame, f"Locked goal: ({gx:+.2f},{gy:+.2f}) z={gz:.2f} yaw=0",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                if last_target is not None and last_target in markers:
                    mi = markers[last_target]
                    dx_m, dy_m = pixel_to_drone_xy(mi.cx, mi.cy, drone_alt)
                    cv2.putText(frame, f"Marker rel: dx={dx_m:+.2f} dy={dy_m:+.2f}",
                                (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                cv2.imshow("aruco_nav_go_to_point_locked", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("[INFO] ESC pressed, exiting.")
                    break

            time.sleep(LOOP_DT)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt -> exiting.")
    finally:
        nav.stop()
        tracker.stop()
        nav.join(timeout=2.0)
        tracker.join(timeout=2.0)
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        cmd.land_and_close()