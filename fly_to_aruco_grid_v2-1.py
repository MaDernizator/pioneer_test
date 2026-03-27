import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple, List

import cv2

try:
    from pioneer_sdk import Pioneer, Camera
except Exception:
    Pioneer = None
    Camera = None

from pixel_projector import pixel_to_drone_xy  # <-- единая функция


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

DRY_RUN = True

TAKEOFF_HEIGHT = 2.0
CENTER_TOL_PX = 40
CENTER_TOL_M = 0.15
LOOP_DT = 0.02

# ------------------ КАРТА МАРКЕРОВ 3x3 ------------------
#
#   Вид сверху (камера дрона смотрит вниз):
#
#        col=0    col=1    col=2
#        ЛЕВО    ЦЕНТР    ПРАВО
#
#   row=0  +-------+-------+-------+   ДАЛЬНЯЯ (от дрона)
#          |       |       |       |
#   row=1  +-------+-------+-------+   СРЕДНЯЯ
#          |       |       |       |
#   row=2  +-------+-------+-------+   БЛИЖНЯЯ (к дрону)
#
#   Дрон стартует ещё ближе, за row=2
#
# Заполни реальные ID маркеров:

MARKER_GRID: Dict[Tuple[int, int], int] = {
    (0, 0): 10,    # дальний левый
    (0, 1): 11,    # дальний центр
    (0, 2): 3,    # дальний правый
    (1, 0): 12,    # средний левый
    (1, 1): 2,    # средний центр
    (1, 2): 13,    # средний правый
    (2, 0): 1,    # ближний левый
    (2, 1): 14,    # ближний центр
    (2, 2): 15,    # ближний правый
}

# Расстояние между соседними маркерами (метры)
GRID_SPACING = 0.7

# ------------------ ПЛАН МИССИИ ------------------
MARKER_PLAN: List[Tuple[int, float]] = [
    (1, 2.0),
    (2, 3.0),
    (3, 2.0),
]

# ------------------ ПАРАМЕТРЫ НАВИГАЦИИ ------------------
REACH_TIMEOUT = 15.0
REACH_POLL_DT = 0.1
GOAL_TRY_HZ = 5.0
GOAL_TRY_DT = 1.0 / GOAL_TRY_HZ

SHOW_WINDOW = True
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)


# =============================================================================
# КЛАССЫ (без изменений из твоего кода)
# =============================================================================

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
                        cv2.circle(frame, (mi.cx, mi.cy), 5, RED, -1)
                        cv2.putText(frame, str(mid), (mi.cx + 8, mi.cy - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

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


class DroneCommander:
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
                pos = self.p.get_local_position_lps()
                return float(pos[0]), float(pos[1])
            except Exception:
                return 0.0, 0.0
        pos = self.p.get_local_position_lps()
        return float(pos[0]), float(pos[1])

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
                self.cmd.go_to_local_point(x, y, z, yaw=0.0)
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


# =============================================================================
# КАРТА МАРКЕРОВ
# =============================================================================

class MarkerMap:
    """
    Хранит LPS-координаты маркеров.
    После калибровки по одному маркеру вычисляет позиции остальных по сетке.
    """
    def __init__(self, grid: Dict[Tuple[int, int], int], spacing: float):
        self.grid = grid
        self.spacing = spacing
        
        # marker_id -> (row, col)
        self.id_to_cell: Dict[int, Tuple[int, int]] = {
            mid: cell for cell, mid in grid.items()
        }
        
        # marker_id -> (lps_x, lps_y)
        self.positions: Dict[int, Tuple[float, float]] = {}
        
        self.is_calibrated = False
        self.lock = threading.Lock()

    def get_cell(self, marker_id: int) -> Optional[Tuple[int, int]]:
        return self.id_to_cell.get(marker_id)

    def calibrate_from_marker(self, marker_id: int, marker_lps_x: float, marker_lps_y: float):
        """
        Калибровка: зная LPS-позицию одного маркера, вычисляем все остальные.
        
        Используем ту же систему координат, что и pixel_to_drone_xy:
        - dx_m прибавляется к drone_x
        - dy_m прибавляется к drone_y
        
        Нужно определить, как row/col соотносятся с dx/dy.
        Это зависит от ориентации дрона и сетки.
        """
        ref_cell = self.get_cell(marker_id)
        if ref_cell is None:
            print(f"[MAP] Marker {marker_id} not in grid!")
            return
        
        ref_row, ref_col = ref_cell
        
        with self.lock:
            self.positions.clear()
            
            for (row, col), mid in self.grid.items():
                # Смещение в ячейках относительно референсного маркера
                d_row = row - ref_row
                d_col = col - ref_col
                
                # Преобразование в метры
                # ВАЖНО: эти знаки нужно подобрать под твою систему!
                # Предположение:
                #   - row увеличивается "назад" (к дрону) -> это -X (дрон смотрит вперёд = +X)
                #   - col увеличивается "вправо" -> это -Y (если +Y влево)
                #
                # Если дрон полетел не туда - поменяй знаки здесь:
                dx = -d_row * self.spacing  # row↑ = назад = -X
                dy = -d_col * self.spacing  # col↑ = вправо = -Y
                
                self.positions[mid] = (marker_lps_x + dx, marker_lps_y + dy)
            
            self.is_calibrated = True
        
        print(f"[MAP] Calibrated from marker {marker_id} at ({marker_lps_x:.2f}, {marker_lps_y:.2f})")
        self._print_map()

    def update_position(self, marker_id: int, lps_x: float, lps_y: float):
        """Обновить позицию маркера (уточнение при визуальном контакте)"""
        with self.lock:
            if marker_id in self.id_to_cell:
                self.positions[marker_id] = (lps_x, lps_y)

    def get_position(self, marker_id: int) -> Optional[Tuple[float, float]]:
        with self.lock:
            return self.positions.get(marker_id)

    def get_all_positions(self) -> Dict[int, Tuple[float, float]]:
        with self.lock:
            return dict(self.positions)

    def _print_map(self):
        print("[MAP] Marker positions:")
        for (row, col), mid in sorted(self.grid.items()):
            pos = self.positions.get(mid)
            if pos:
                print(f"  [{row},{col}] ID={mid}: ({pos[0]:+.2f}, {pos[1]:+.2f})")


# =============================================================================
# ОСНОВНОЙ ЦИКЛ
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ARUCO GRID NAVIGATION")
    print("=" * 60)
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"Grid spacing: {GRID_SPACING}m")
    print(f"Plan: {MARKER_PLAN}")
    print("=" * 60)

    tracker = ArucoTrackerThread()
    tracker.start()

    cmd = DroneCommander(dry_run=DRY_RUN)
    nav = NavigatorThread(cmd=cmd, reach_timeout=REACH_TIMEOUT, poll_dt=REACH_POLL_DT)
    nav.start()

    marker_map = MarkerMap(MARKER_GRID, GRID_SPACING)

    visited: Set[int] = set()
    hold_until = 0.0
    current_state = "INIT"
    locked_goal: Optional[Tuple[float, float, float]] = None
    next_goal_try_t = time.monotonic()

    plan_idx = 0

    try:
        if not DRY_RUN:
            print("[INFO] Real flight mode")
            cmd.arm_takeoff_to_height(TAKEOFF_HEIGHT)
            print(f"[INFO] Hover reached: z={TAKEOFF_HEIGHT:.2f}m")
        else:
            print("[INFO] DRY_RUN mode")

        while True:
            with tracker.lock:
                frame = tracker.latest_frame
                markers = dict(tracker.markers)

            if frame is None or (hasattr(frame, "size") and frame.size == 0):
                time.sleep(0.01)
                continue

            # Миссия завершена
            if plan_idx >= len(MARKER_PLAN):
                current_state = "DONE"
                if SHOW_WINDOW:
                    cv2.putText(frame, "MISSION COMPLETE",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
                    cv2.imshow("aruco_grid_nav", frame)
                    cv2.waitKey(1)
                print("[INFO] Mission completed!")
                break

            h, w = frame.shape[:2]
            cx0, cy0 = int(w / 2), int(h / 2)
            center_color = RED

            drone_x, drone_y = cmd.get_pos_lps_xy()
            drone_alt = cmd.get_alt_m()
            nav_state = nav.get_state()

            if locked_goal is not None and nav_state in ("REACHED", "TIMEOUT"):
                locked_goal = None

            target_id, hold_time = MARKER_PLAN[plan_idx]

            # =========== КАЛИБРОВКА ===========
            # Если карта не откалибрована - ищем любой маркер из сетки
            if not marker_map.is_calibrated:
                current_state = "CALIBRATING"
                center_color = YELLOW

                for mid, mi in markers.items():
                    if mid in marker_map.id_to_cell:
                        # Нашли маркер! Вычисляем его LPS-позицию
                        dx_m, dy_m = pixel_to_drone_xy(mi.cx, mi.cy, drone_alt)
                        marker_lps_x = drone_x + dx_m
                        marker_lps_y = drone_y + dy_m

                        marker_map.calibrate_from_marker(mid, marker_lps_x, marker_lps_y)
                        break

            # =========== РЕЖИМ ЗАВИСАНИЯ ===========
            elif time.time() < hold_until:
                center_color = GREEN
                current_state = f"HOLD({target_id})"

            # =========== НАВИГАЦИЯ ===========
            else:
                # Если только что закончили hold - переходим к следующему
                if hold_until > 0 and time.time() >= hold_until:
                    plan_idx += 1
                    hold_until = 0.0
                    print(f"[INFO] Hold complete, next target")
                    continue

                # Летим к цели
                if locked_goal is not None and nav_state == "MOVING":
                    current_state = f"FLYING({target_id})"
                    center_color = CYAN

                # Маркер виден - визуальная навигация (как в твоём коде)
                elif target_id in markers:
                    mi = markers[target_id]
                    dx_px = mi.cx - cx0
                    dy_px = mi.cy - cy0
                    centered_px = (abs(dx_px) <= CENTER_TOL_PX and abs(dy_px) <= CENTER_TOL_PX)

                    dx_m, dy_m = pixel_to_drone_xy(mi.cx, mi.cy, drone_alt)
                    centered_m = (abs(dx_m) <= CENTER_TOL_M and abs(dy_m) <= CENTER_TOL_M)

                    # Обновляем позицию в карте
                    marker_map.update_position(target_id, drone_x + dx_m, drone_y + dy_m)

                    if centered_px and centered_m:
                        visited.add(target_id)
                        hold_until = time.time() + hold_time
                        current_state = f"HOLD({target_id})"
                        center_color = GREEN
                        print(f"[INFO] Marker {target_id} centered -> hold {hold_time:.1f}s")
                    else:
                        current_state = f"VISUAL({target_id})"
                        center_color = CYAN

                        # Точно как в твоём коде:
                        target_x = drone_x + dx_m
                        target_y = drone_y + dy_m
                        goal = (target_x, target_y, TAKEOFF_HEIGHT)

                        now = time.monotonic()
                        if now >= next_goal_try_t:
                            accepted = nav.submit_goal_if_idle(*goal)
                            if accepted:
                                locked_goal = goal
                            next_goal_try_t = now + GOAL_TRY_DT

                # Маркер НЕ виден - летим по карте
                else:
                    target_pos = marker_map.get_position(target_id)

                    if target_pos is None:
                        current_state = f"NO_POS({target_id})"
                        center_color = RED
                    else:
                        current_state = f"BLIND({target_id})"
                        center_color = YELLOW

                        goal = (target_pos[0], target_pos[1], TAKEOFF_HEIGHT)

                        now = time.monotonic()
                        if now >= next_goal_try_t:
                            accepted = nav.submit_goal_if_idle(*goal)
                            if accepted:
                                locked_goal = goal
                                print(f"[NAV] Blind flight to marker {target_id} at ({target_pos[0]:.2f}, {target_pos[1]:.2f})")
                            next_goal_try_t = now + GOAL_TRY_DT

            # =========== ОТРИСОВКА ===========
            if SHOW_WINDOW:
                cv2.circle(frame, (cx0, cy0), 7, center_color, -1)

                cv2.putText(frame, f"Plan: {plan_idx+1}/{len(MARKER_PLAN)} | State: {current_state} | NAV: {nav_state}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                cv2.putText(frame, f"Target: {target_id} hold={hold_time:.1f}s",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                cv2.putText(frame, f"Pos: ({drone_x:.2f}, {drone_y:.2f}) alt={drone_alt:.2f}m",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                if locked_goal is not None:
                    gx, gy, gz = locked_goal
                    cv2.putText(frame, f"Goal: ({gx:+.2f},{gy:+.2f})",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)

                # Относительное смещение до целевого маркера
                if target_id in markers:
                    mi = markers[target_id]
                    dx_m, dy_m = pixel_to_drone_xy(mi.cx, mi.cy, drone_alt)
                    cv2.putText(frame, f"Marker rel: dx={dx_m:+.2f} dy={dy_m:+.2f}",
                                (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                # Мини-карта позиций
                y_off = 150
                cv2.putText(frame, "Map:", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y_off += 18
                for mid, pos in sorted(marker_map.get_all_positions().items()):
                    color = GREEN if mid in markers else (150, 150, 150)
                    prefix = ">" if mid == target_id else " "
                    cv2.putText(frame, f"{prefix}{mid}: ({pos[0]:+.2f},{pos[1]:+.2f})",
                                (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_off += 16

                cv2.imshow("aruco_grid_nav", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("[INFO] ESC pressed")
                    break

            time.sleep(LOOP_DT)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt")
    except Exception as e:
        print(e)
    finally:
        nav.stop()
        tracker.stop()
        nav.join(timeout=2.0)
        tracker.join(timeout=2.0)
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        cmd.land_and_close()
