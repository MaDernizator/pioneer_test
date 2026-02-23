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

        # Пауза камеры (чтобы не мешать взлёту)
        self._pause_lock = threading.Lock()
        self._paused = False

    def set_paused(self, paused: bool):
        with self._pause_lock:
            self._paused = paused

    def is_paused(self) -> bool:
        with self._pause_lock:
            return self._paused

    def run(self):
        if self.camera is None:
            print("[ERROR] Camera is unavailable (pioneer_sdk import failed).")
            return

        while self.running:
            if self.is_paused():
                time.sleep(0.02)
                continue

            try:
                frame = self.camera.get_cv_frame()

                # защита от пустого кадра
                if frame is None or (hasattr(frame, "size") and frame.size == 0):
                    time.sleep(0.005)
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
                time.sleep(0.01)
            except Exception as e:
                print(f"[Video] Unexpected error (ignored): {e}")
                time.sleep(0.02)

    def stop(self):
        self.running = False


def choose_next_marker(markers: Dict[int, MarkerInfo], visited: Set[int]) -> Optional[int]:
    candidates = [mid for mid in markers.keys() if mid not in visited]
    return min(candidates) if candidates else None


class DroneCommander:
    def __init__(self, connect_only: bool):
        self.connect_only = connect_only
        if Pioneer is None:
            raise RuntimeError("pioneer_sdk недоступен")
        self.p = Pioneer()

    def arm_takeoff_to_height(self, z: float,
                              arm_to_takeoff_delay: float,
                              takeoff_to_goto_delay: float):
        """
        Встроили задержки между командами.
        """
        if self.connect_only:
            print(f"[CONNECT_ONLY] Skip arm/takeoff/go_to_local_point(z={z})")
            return

        self.p.arm()
        time.sleep(arm_to_takeoff_delay)

        self.p.takeoff()
        time.sleep(takeoff_to_goto_delay)

        self.p.go_to_local_point(x=0, y=0, z=z, yaw=0)
        while not self.p.point_reached():
            time.sleep(0.1)

    def send_manual_speed_body_fixed(self, vx: float, vy: float, vz: float, yaw_rate: float):
        self.p.set_manual_speed_body_fixed(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)

    def land_and_close(self):
        if not self.connect_only:
            self.p.land()
        self.p.close_connection()
        del self.p


class CommandSenderThread(threading.Thread):
    def __init__(self, cmd: DroneCommander, hz: float):
        super().__init__(daemon=True)
        self.cmd = cmd
        self.dt = 1.0 / hz
        self.running = True

        self.lock = threading.Lock()
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_rate = 0.0

        self.send_enabled = True
        self.enable_lock = threading.Lock()

        self._t_last_print = time.monotonic()
        self._last_send_ms = 0.0
        self._max_send_ms = 0.0

    def set_send_enabled(self, enabled: bool):
        with self.enable_lock:
            self.send_enabled = enabled

    def set_target(self, vx: float, vy: float, vz: float, yaw_rate: float):
        with self.lock:
            self.vx, self.vy, self.vz, self.yaw_rate = vx, vy, vz, yaw_rate

    def run(self):
        next_t = time.monotonic()
        while self.running:
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.002, next_t - now))
                continue
            next_t = now + self.dt

            with self.enable_lock:
                enabled = self.send_enabled
            if not enabled:
                continue

            with self.lock:
                vx, vy, vz, yaw_rate = self.vx, self.vy, self.vz, self.yaw_rate

            t0 = time.monotonic()
            try:
                self.cmd.send_manual_speed_body_fixed(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)
            except Exception as e:
                print(f"[CMD] send error: {e}")
            t1 = time.monotonic()

            send_ms = (t1 - t0) * 1000.0
            self._last_send_ms = send_ms
            if send_ms > self._max_send_ms:
                self._max_send_ms = send_ms

            if t1 - self._t_last_print > 1.0:
                print(f"[CMD] send: last={self._last_send_ms:.1f} ms, max={self._max_send_ms:.1f} ms")
                self._t_last_print = t1

    def stop(self):
        self.running = False


if __name__ == "__main__":
    CONNECT_ONLY = True  # если False — реальный взлёт

    TAKEOFF_HEIGHT = 1.0
    CENTER_TOL_PX = 40
    SPEED = 0.25
    CMD_HZ = 10.0

    # КЛЮЧЕВОЕ: “тишина” вокруг взлёта
    ARM_TO_TAKEOFF_DELAY_SEC = 1.0       # пауза после arm() перед takeoff()
    TAKEOFF_TO_GOTO_DELAY_SEC = 2.0      # пауза после takeoff() перед go_to_local_point()
    POST_TAKEOFF_SILENCE_SEC = 4.0       # после достижения высоты ещё 4 секунды без manual_speed
    PAUSE_CAMERA_DURING_TAKEOFF = True   # вот это часто решает проблему “сносит на взлёте”

    SHOW_WINDOW = True
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    tracker = ArucoTrackerThread()
    tracker.start()

    cmd = DroneCommander(connect_only=CONNECT_ONLY)
    sender = CommandSenderThread(cmd=cmd, hz=CMD_HZ)
    sender.start()

    visited: Set[int] = set()
    last_target: Optional[int] = None
    hold_green_until = 0.0
    pending_color = RED

    try:
        if not CONNECT_ONLY:
            print("[INFO] Real flight mode (takeoff)")

            # 1) Запретить manual_speed
            sender.set_target(0.0, 0.0, 0.0, 0.0)
            sender.set_send_enabled(False)

            # 2) (Опционально) Поставить камеру на паузу
            if PAUSE_CAMERA_DURING_TAKEOFF:
                tracker.set_paused(True)
                print("[INFO] Camera paused during takeoff sequence.")

            # 3) Взлёт и набор высоты + задержки между командами
            cmd.arm_takeoff_to_height(
                TAKEOFF_HEIGHT,
                arm_to_takeoff_delay=ARM_TO_TAKEOFF_DELAY_SEC,
                takeoff_to_goto_delay=TAKEOFF_TO_GOTO_DELAY_SEC
            )
            print(f"[INFO] Hover reached: z={TAKEOFF_HEIGHT:.2f}m")

            # 4) Ещё 4 секунды тишины (без manual_speed)
            print(f"[INFO] Post-takeoff silence: {POST_TAKEOFF_SILENCE_SEC:.1f}s (no manual_speed)")
            time.sleep(POST_TAKEOFF_SILENCE_SEC)

            # 5) Возобновить камеру
            if PAUSE_CAMERA_DURING_TAKEOFF:
                tracker.set_paused(False)
                print("[INFO] Camera resumed.")

            # 6) Разрешить manual_speed
            sender.set_send_enabled(True)
            print("[INFO] Manual speed enabled.")
        else:
            print("[INFO] CONNECT_ONLY: no arm/takeoff/land, but manual speed commands are sent.")

        while True:
            with tracker.lock:
                frame = tracker.latest_frame
                markers = dict(tracker.markers)

            if frame is None or (hasattr(frame, "size") and frame.size == 0):
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            cx0, cy0 = int(w / 2), int(h / 2)

            vx, vy = 0.0, 0.0
            pending_color = RED

            if time.time() < hold_green_until:
                pending_color = GREEN
                vx, vy = 0.0, 0.0
            else:
                if not markers:
                    vx, vy = 0.0, 0.0
                else:
                    target_id = choose_next_marker(markers, visited)
                    if target_id is None:
                        vx, vy = 0.0, 0.0
                    else:
                        if target_id != last_target:
                            print(f"[INFO] New target marker: {target_id}")
                            last_target = target_id

                        mi = markers[target_id]
                        dx_px = mi.cx - cx0
                        dy_px = mi.cy - cy0

                        centered = (abs(dx_px) <= CENTER_TOL_PX and abs(dy_px) <= CENTER_TOL_PX)
                        if centered and (target_id not in visited):
                            visited.add(target_id)
                            print(f"[INFO] Marker {target_id} centered. Visited: {sorted(list(visited))}")
                            hold_green_until = time.time() + 3.0
                            pending_color = GREEN
                            vx, vy = 0.0, 0.0
                        else:
                            dx = dx_px / (w / 2.0)
                            dy = dy_px / (h / 2.0)
                            norm = (dx * dx + dy * dy) ** 0.5
                            if norm < 1e-6:
                                vx, vy = 0.0, 0.0
                            else:
                                vx = SPEED * (dx / norm)
                                vy = SPEED * (dy / norm)
                            # если нужна инверсия Y — раскомментируй
                            # vy = -vy

            sender.set_target(vx=vx, vy=vy, vz=0.0, yaw_rate=0.0)

            if SHOW_WINDOW:
                cv2.circle(frame, (cx0, cy0), 7, pending_color, -1)
                cv2.putText(frame, f"Visited: {len(visited)}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if last_target is not None:
                    cv2.putText(frame, f"Target: {last_target}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("aruco_nav", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("[INFO] ESC pressed, exiting.")
                    break

            time.sleep(0.005)

    finally:
        sender.stop()
        tracker.stop()
        sender.join(timeout=2.0)
        tracker.join(timeout=2.0)
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        cmd.land_and_close()