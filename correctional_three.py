import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2

try:
    from pioneer_sdk import Pioneer, Camera
except Exception:
    Pioneer = None
    Camera = None

from pixel_projector import pixel_to_drone_xy
from marker_map import MARKER_COORDS


HEIGHT = 1.6

DRY_RUN = False
YAW = 0.0
MUST_EXIST = True

CENTER_TOL_PX = 40
CENTER_TOL_M = 0.15
CORRECTION_GAIN = 0.9
MAX_CORRECTION_TRIES = 8
CORRECTION_PRE_SLEEP = 1.0   # пауза перед каждой коррекцией

MARKER_CHECK_TRIES = 20
MARKER_CHECK_DT = 0.1

REACH_TIMEOUT = 15.0
REACH_POLL_DT = 0.1

GOAL_TRY_HZ = 5.0
GOAL_TRY_DT = 1.0 / GOAL_TRY_HZ

CAM_FPS = 30.0
CAM_DT = 1.0 / CAM_FPS

LOOP_DT = 0.02

SHOW_WINDOW = True
WINDOW_NAME = "camera_debug"

RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Маршрут:
# (marker_id, marker_index, z, hold_time, use_correction)
ROUTE = [
    (27, 0, HEIGHT, 0.0, False),
    (7, 0, HEIGHT, 5.0, True),
    (12, 0, HEIGHT, 0.0, True),
    (25, 0, HEIGHT, 5.0, True),
    (11, 0, HEIGHT, 0.0, True),
    (17, 0, HEIGHT, 0.0, True),
    (15, 0, HEIGHT, 0.0, True),
    (15, 1, HEIGHT - 1, 5.0, True),
    (6, 0, HEIGHT - 1, 0.0, True),
    (6, 1, HEIGHT, 0.0, True),
]


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
        self.status_text = ""
        self.status_color = WHITE
        self.window_fps = 0.0

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def set_status(self, text: str, color=WHITE):
        with self.lock:
            self.status_text = text
            self.status_color = color

    def get_snapshot(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            markers = dict(self.markers)
        return frame, markers

    def run(self):
        if self.camera is None:
            print("[ERROR] Camera is unavailable (pioneer_sdk import failed).")
            return

        last_frame_t = time.monotonic()
        fps_t0 = time.monotonic()
        fps_count = 0

        while self.running:
            try:
                now = time.monotonic()
                dt = now - last_frame_t
                if dt < CAM_DT:
                    time.sleep(CAM_DT - dt)
                last_frame_t = time.monotonic()

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
                        cv2.circle(frame, (mi.cx, mi.cy), 5, RED, -1)
                        cv2.putText(
                            frame,
                            str(mid),
                            (mi.cx + 8, mi.cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            RED,
                            2,
                        )

                h, w = frame.shape[:2]
                cx0, cy0 = w // 2, h // 2
                cv2.circle(frame, (cx0, cy0), 7, WHITE, 2)

                fps_count += 1
                t_now = time.monotonic()
                if t_now - fps_t0 >= 1.0:
                    self.window_fps = fps_count / (t_now - fps_t0)
                    fps_t0 = t_now
                    fps_count = 0

                with self.lock:
                    status_text = self.status_text
                    status_color = self.status_color
                    self.latest_frame = frame.copy()
                    self.markers = markers_local
                    fps_value = self.window_fps

                if SHOW_WINDOW:
                    if status_text:
                        cv2.putText(
                            frame,
                            status_text,
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            status_color,
                            2,
                        )

                    cv2.putText(
                        frame,
                        f"FPS: {fps_value:.1f}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        WHITE,
                        2,
                    )

                    cv2.imshow(WINDOW_NAME, frame)
                    cv2.waitKey(1)

            except cv2.error as e:
                print(f"[Video] OpenCV error (ignored): {e}")
                time.sleep(0.02)
            except Exception as e:
                print(f"[Video] Unexpected error (ignored): {e}")
                time.sleep(0.05)

    def stop(self):
        self.running = False


class DroneCommander:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        if Pioneer is None:
            raise RuntimeError("pioneer_sdk недоступен")
        self.p = Pioneer()

    def arm_takeoff(self):
        if self.dry_run:
            print("[DRY] Skip arm/takeoff")
            return
        self.p.arm()
        self.p.takeoff()

    def go_to_local_point(self, x: float, y: float, z: float, yaw: float = 0.0):
        if self.dry_run:
            print(f"[DRY] go_to_local_point(x={x:+.2f}, y={y:+.2f}, z={z:+.2f}, yaw={yaw:+.2f})")
            return
        self.p.go_to_local_point(x=x, y=y, z=z, yaw=yaw)

    def point_reached(self) -> bool:
        if self.dry_run:
            return False
        return self.p.point_reached()

    def get_pos_lps_xy(self) -> Tuple[float, float]:
        try:
            pos = self.p.get_local_position_lps(get_last_received=True)
            return float(pos[0]), float(pos[1])
        except Exception:
            return 0.0, 0.0

    def get_alt_m(self) -> float:
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
    def __init__(self, cmd: DroneCommander, reach_timeout: float = 15.0, poll_dt: float = 0.1):
        super().__init__(daemon=True)
        self.cmd = cmd
        self.reach_timeout = reach_timeout
        self.poll_dt = poll_dt

        self._cv = threading.Condition()
        self._running = True

        self._goal: Optional[Tuple[float, float, float]] = None
        self._has_goal = False
        self._state = "IDLE"

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

            try:
                x, y, z = goal
                self.cmd.go_to_local_point(x, y, z, yaw=YAW)
            except Exception as e:
                print(f"[NAV] go_to_local_point error: {e}")

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


def get_marker_coords(marker_id: int, marker_index: int):
    found = [m for m in MARKER_COORDS if m["marker_id"] == marker_id]

    if marker_index < 0 or marker_index >= len(found):
        raise ValueError(
            f"Marker ({marker_id}, index={marker_index}) not found. "
            f"Available count: {len(found)}"
        )

    marker = found[marker_index]
    return float(marker["x"]), float(marker["y"])


def wait_until_reached(cmd: DroneCommander):
    while not cmd.point_reached():
        time.sleep(0.1)


def wait_marker_visible(tracker: ArucoTrackerThread, marker_id: int) -> bool:
    for _ in range(MARKER_CHECK_TRIES):
        _, markers = tracker.get_snapshot()
        if marker_id in markers:
            return True
        time.sleep(MARKER_CHECK_DT)
    return False


def center_over_marker(
    tracker: ArucoTrackerThread,
    cmd: DroneCommander,
    nav: NavigatorThread,
    marker_id: int,
    target_z: float,
) -> bool:
    locked_goal: Optional[Tuple[float, float, float]] = None
    next_goal_try_t = time.monotonic()
    tries_sent = 0

    while True:
        frame, markers = tracker.get_snapshot()

        if frame is None or (hasattr(frame, "size") and frame.size == 0):
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        cx0, cy0 = int(w / 2), int(h / 2)

        drone_x, drone_y = cmd.get_pos_lps_xy()
        drone_alt = cmd.get_alt_m()
        nav_state = nav.get_state()

        if locked_goal is not None and nav_state in ("REACHED", "TIMEOUT"):
            locked_goal = None
            if nav_state == "TIMEOUT":
                tracker.set_status(f"CENTER TIMEOUT {marker_id}", RED)
                print(f"[WARN] Correction timeout for marker {marker_id}")
                return False

        if locked_goal is not None and nav_state == "MOVING":
            tracker.set_status(f"CENTERING_LOCKED({marker_id})", RED)
            time.sleep(LOOP_DT)
            continue

        if marker_id not in markers:
            tracker.set_status(f"SEARCH_TARGET({marker_id})", RED)
            time.sleep(LOOP_DT)
            continue

        mi = markers[marker_id]
        dx_px = mi.cx - cx0
        dy_px = mi.cy - cy0
        centered_px = abs(dx_px) <= CENTER_TOL_PX and abs(dy_px) <= CENTER_TOL_PX

        dx_m, dy_m = pixel_to_drone_xy(mi.cx, mi.cy, drone_alt)
        centered_m = abs(dx_m) <= CENTER_TOL_M and abs(dy_m) <= CENTER_TOL_M

        raw_dx_m = dx_m
        raw_dy_m = dy_m

        dx_m *= CORRECTION_GAIN
        dy_m *= CORRECTION_GAIN

        if centered_px and centered_m:
            tracker.set_status(f"CENTERED({marker_id})", GREEN)
            print(f"[INFO] Marker {marker_id} centered")
            return True

        target_x = drone_x + dx_m
        target_y = drone_y + dy_m
        goal = (target_x, target_y, target_z)

        now = time.monotonic()
        if now >= next_goal_try_t and tries_sent < MAX_CORRECTION_TRIES:
            tracker.set_status(
                f"STABILIZE({marker_id}) {CORRECTION_PRE_SLEEP:.1f}s",
                WHITE,
            )
            time.sleep(CORRECTION_PRE_SLEEP)

            # после паузы ещё раз проверяем, не занялся ли навигатор чем-то
            if nav.get_state() == "MOVING":
                next_goal_try_t = time.monotonic() + GOAL_TRY_DT
                time.sleep(LOOP_DT)
                continue

            accepted = nav.submit_goal_if_idle(*goal)
            if accepted:
                locked_goal = goal
                tries_sent += 1
                print(
                    f"[INFO] Correction {tries_sent}: "
                    f"dx={raw_dx_m:+.2f}, dy={raw_dy_m:+.2f}"
                )
            next_goal_try_t = time.monotonic() + GOAL_TRY_DT

        if tries_sent >= MAX_CORRECTION_TRIES and locked_goal is None:
            tracker.set_status(f"CENTER FAILED({marker_id})", RED)
            print(f"[WARN] Marker {marker_id} was not centered exactly")
            return False

        tracker.set_status(
            f"CENTERING({marker_id}) dx={raw_dx_m:+.2f} dy={raw_dy_m:+.2f}",
            RED,
        )
        time.sleep(LOOP_DT)


def go_to_marker(
    cmd: DroneCommander,
    tracker: ArucoTrackerThread,
    nav: NavigatorThread,
    marker_id: int,
    marker_index: int,
    z: float,
    hold_time: float,
    use_correction: bool,
    coord_shift: Tuple[float, float],
) -> Tuple[float, float]:
    base_x, base_y = get_marker_coords(marker_id, marker_index)

    shift_x, shift_y = coord_shift
    target_x = base_x + shift_x
    target_y = base_y + shift_y
    target_z = z

    tracker.set_status(f"FLY TO {marker_id}[{marker_index}]", WHITE)
    print(
        f"[INFO] Fly to marker {marker_id}[{marker_index}]: "
        f"x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}"
    )

    cmd.go_to_local_point(x=target_x, y=target_y, z=target_z, yaw=YAW)
    wait_until_reached(cmd)

    if MUST_EXIST:
        tracker.set_status(f"CHECK MARKER {marker_id}", WHITE)
        print(f"[INFO] Check marker {marker_id} presence")
        if not wait_marker_visible(tracker, marker_id):
            raise RuntimeError(f"Marker {marker_id} not found. Flight stopped.")

    if use_correction:
        print(f"[INFO] Correction enabled for marker {marker_id}")
        centered = center_over_marker(
            tracker=tracker,
            cmd=cmd,
            nav=nav,
            marker_id=marker_id,
            target_z=target_z,
        )

        if centered:
            real_x, real_y = cmd.get_pos_lps_xy()

            error_x = real_x - target_x
            error_y = real_y - target_y

            shift_x += error_x
            shift_y += error_y

            print(
                f"[INFO] Marker {marker_id}[{marker_index}] "
                f"real=({real_x:+.2f}, {real_y:+.2f}) "
                f"planned=({target_x:+.2f}, {target_y:+.2f})"
            )
            print(f"[INFO] New global shift: dx={shift_x:+.2f}, dy={shift_y:+.2f}")

    tracker.set_status(f"HOLD {marker_id}[{marker_index}] {hold_time:.1f}s", GREEN)
    print(f"[INFO] Hold {hold_time:.1f} sec at marker {marker_id}[{marker_index}]")
    time.sleep(hold_time)

    return shift_x, shift_y


if __name__ == "__main__":
    tracker = ArucoTrackerThread()
    tracker.start()

    cmd = DroneCommander(dry_run=DRY_RUN)
    nav = NavigatorThread(cmd=cmd, reach_timeout=REACH_TIMEOUT, poll_dt=REACH_POLL_DT)
    nav.start()

    coord_shift = (0.0, 0.0)

    try:
        if not DRY_RUN:
            print("[INFO] Arm")
            cmd.arm_takeoff()
            time.sleep(3)
        else:
            print("[INFO] DRY_RUN mode")

        for marker_id, marker_index, z, hold_time, use_correction in ROUTE:
            if DRY_RUN:
                base_x, base_y = get_marker_coords(marker_id, marker_index)
                sx, sy = coord_shift

                tracker.set_status(f"DRY {marker_id}[{marker_index}]", WHITE)
                print(
                    f"[DRY] go_to_local_point("
                    f"x={base_x + sx:.2f}, y={base_y + sy:.2f}, z={z:.2f}, yaw={YAW:.2f})"
                )
                print(f"[DRY] marker={marker_id}[{marker_index}]")
                print(f"[DRY] use_correction={use_correction}, must_exist={MUST_EXIST}")
                print(f"[DRY] hold {hold_time:.1f} sec")
                time.sleep(max(1.0, hold_time))
            else:
                coord_shift = go_to_marker(
                    cmd=cmd,
                    tracker=tracker,
                    nav=nav,
                    marker_id=marker_id,
                    marker_index=marker_index,
                    z=z,
                    hold_time=hold_time,
                    use_correction=use_correction,
                    coord_shift=coord_shift,
                )

        tracker.set_status("MISSION COMPLETED", GREEN)
        print("[INFO] Mission completed")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")

    except Exception as e:
        tracker.set_status(f"ERROR: {e}", RED)
        print(f"[ERROR] {e}")

    finally:
        nav.stop()
        tracker.stop()
        nav.join(timeout=2.0)
        tracker.join(timeout=2.0)

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        cmd.land_and_close()