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

from pixel_projector import pixel_to_drone_xy


# =============================================================================
# КОНФИГУРАЦИЯ - НАСТРОЙ ПОД СВОЮ УСТАНОВКУ
# =============================================================================

# Режим без реального полёта (True = только печать команд)
DRY_RUN = True

# Высота полёта
TAKEOFF_HEIGHT = 2.0

# Допуски центрирования
CENTER_TOL_PX = 40      # пикселей
CENTER_TOL_M = 0.15     # метров

# Частота основного цикла
LOOP_DT = 0.02

# ------------------ КАРТА МАРКЕРОВ 3x3 ------------------
# Заполни ID маркеров в сетке:
#
#   ДАЛЬНЯЯ СТОРОНА (row=0)
#   +-------+-------+-------+
#   | (0,0) | (0,1) | (0,2) |
#   +-------+-------+-------+
#   | (1,0) | (1,1) | (1,2) |   СРЕДНЯЯ (row=1)
#   +-------+-------+-------+
#   | (2,0) | (2,1) | (2,2) |
#   +-------+-------+-------+
#   БЛИЖНЯЯ К ДРОНУ (row=2)
#
#   col=0    col=1    col=2
#   ЛЕВО    ЦЕНТР    ПРАВО
#
# Формат: (row, col): marker_id
# Если маркера нет в ячейке - не добавляй или поставь None

MARKER_GRID: Dict[Tuple[int, int], Optional[int]] = {
    (0, 0): 1,    # дальний левый
    (0, 1): 5,    # дальний центр
    (0, 2): 7,    # дальний правый
    (1, 0): 0,    # средний левый
    (1, 1): 4,    # средний центр
    (1, 2): 6,    # средний правый
    (2, 0): 8,    # ближний левый
    (2, 1): 2,    # ближний центр
    (2, 2): 3,    # ближний правый
}

# Расстояние между соседними маркерами (метры)
GRID_SPACING_X = 0.7   # по X (вперёд-назад, между рядами)
GRID_SPACING_Y = 0.7   # по Y (влево-вправо, между колонками)

# ------------------ ПЛАН МИССИИ ------------------
# Последовательность: (marker_id, hold_time_seconds)
MARKER_PLAN: List[Tuple[int, float]] = [
    (1, 2.0),   # дальний левый
    (5, 3.0),   # средний центр (через один по диагонали)
    (9, 2.0),   # ближний правый
]

# ------------------ ПАРАМЕТРЫ ПОИСКА ПЕРВОГО МАРКЕРА ------------------
# После взлёта дрон не видит маркеры - нужно лететь вперёд
INITIAL_SEARCH_STEP_X = 0.3   # шаг вперёд по X
INITIAL_SEARCH_STEP_Y = 0.0   # шаг по Y (0 = строго вперёд)
INITIAL_SEARCH_MAX_STEPS = 15
INITIAL_SEARCH_PAUSE = 1.5    # пауза после каждого шага

# ------------------ ПАРАМЕТРЫ НАВИГАЦИИ ------------------
REACH_TIMEOUT = 15.0
REACH_POLL_DT = 0.1
GOAL_TRY_HZ = 5.0
GOAL_TRY_DT = 1.0 / GOAL_TRY_HZ

# Отображение
SHOW_WINDOW = True
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)


# =============================================================================
# КЛАССЫ
# =============================================================================

@dataclass
class MarkerInfo:
    """Информация о маркере из камеры (пиксели)"""
    cx: int
    cy: int


class MarkerMap:
    """
    Карта маркеров с предзаданной сеткой.
    После калибровки по одному маркеру вычисляет позиции всех остальных.
    """
    def __init__(self, 
                 grid: Dict[Tuple[int, int], Optional[int]],
                 spacing_x: float,
                 spacing_y: float):
        self.grid = grid
        self.spacing_x = spacing_x
        self.spacing_y = spacing_y
        
        # Обратный словарь: marker_id -> (row, col)
        self.id_to_cell: Dict[int, Tuple[int, int]] = {}
        for cell, mid in grid.items():
            if mid is not None:
                self.id_to_cell[mid] = cell
        
        # Калиброванные LPS-координаты (после первого обнаружения)
        self.calibrated_positions: Dict[int, Tuple[float, float]] = {}
        self.is_calibrated = False
        self.reference_id: Optional[int] = None
        self.reference_lps: Optional[Tuple[float, float]] = None
        
        self.lock = threading.Lock()
    
    def get_cell(self, marker_id: int) -> Optional[Tuple[int, int]]:
        """Получить ячейку (row, col) по ID маркера"""
        return self.id_to_cell.get(marker_id)
    
    def get_marker_id(self, row: int, col: int) -> Optional[int]:
        """Получить ID маркера по ячейке"""
        return self.grid.get((row, col))
    
    def calibrate(self, marker_id: int, lps_x: float, lps_y: float):
        """
        Калибровка карты по одному маркеру.
        После этого все остальные позиции вычисляются автоматически.
        """
        cell = self.get_cell(marker_id)
        if cell is None:
            print(f"[MAP] Warning: marker {marker_id} not in grid, ignoring calibration")
            return
        
        with self.lock:
            self.reference_id = marker_id
            self.reference_lps = (lps_x, lps_y)
            ref_row, ref_col = cell
            
            # Вычисляем позиции всех маркеров относительно референсного
            self.calibrated_positions.clear()
            for (row, col), mid in self.grid.items():
                if mid is None:
                    continue
                
                # Смещение в ячейках относительно референсного
                d_row = row - ref_row  # положительное = дальше от дрона
                d_col = col - ref_col  # положительное = правее
                
                # Преобразуем в метры
                # row увеличивается "от дрона" -> это +X
                # col увеличивается "вправо" -> это -Y (если +Y влево)
                # ВАЖНО: подкорректируй знаки под свою систему координат!
                dx = -d_row * self.spacing_x  # минус потому что row=0 дальше
                dy = -d_col * self.spacing_y  # минус если +Y влево
                
                marker_x = lps_x + dx
                marker_y = lps_y + dy
                self.calibrated_positions[mid] = (marker_x, marker_y)
            
            self.is_calibrated = True
            print(f"[MAP] Calibrated from marker {marker_id} at ({lps_x:.2f}, {lps_y:.2f})")
            print(f"[MAP] Reference cell: row={ref_row}, col={ref_col}")
            self._print_all_positions()
    
    def update_marker_position(self, marker_id: int, lps_x: float, lps_y: float):
        """Обновить позицию маркера (уточнение после визуального обнаружения)"""
        with self.lock:
            if marker_id in self.id_to_cell:
                self.calibrated_positions[marker_id] = (lps_x, lps_y)
    
    def get_position(self, marker_id: int) -> Optional[Tuple[float, float]]:
        """Получить LPS-позицию маркера"""
        with self.lock:
            return self.calibrated_positions.get(marker_id)
    
    def get_all_positions(self) -> Dict[int, Tuple[float, float]]:
        """Получить все известные позиции"""
        with self.lock:
            return dict(self.calibrated_positions)
    
    def _print_all_positions(self):
        """Вывести все позиции в консоль"""
        print("[MAP] All marker positions:")
        for (row, col), mid in sorted(self.grid.items()):
            if mid is None:
                continue
            pos = self.calibrated_positions.get(mid)
            if pos:
                print(f"  [{row},{col}] ID={mid}: ({pos[0]:.2f}, {pos[1]:.2f})")
            else:
                print(f"  [{row},{col}] ID={mid}: (unknown)")


class ArucoTrackerThread(threading.Thread):
    """Поток захвата видео и детекции ArUco маркеров"""
    
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
    """Обёртка над Pioneer SDK с поддержкой DRY_RUN"""
    
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
            print(f"[DRY] go_to_local_point(x={x:+.2f}, y={y:+.2f}, z={z:+.2f}, yaw={yaw:.2f})")
            return
        self.p.go_to_local_point(x=x, y=y, z=z, yaw=yaw)

    def point_reached(self) -> bool:
        if self.dry_run:
            return False
        return self.p.point_reached()

    def get_pos_lps_xy(self) -> Tuple[float, float]:
        try:
            pos = self.p.get_pos_lps()
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
    """Поток навигации с очередью целей"""
    
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
        """Отправить цель, только если навигатор свободен"""
        with self._cv:
            if self._state == "MOVING":
                return False
            self._goal = (x, y, z)
            self._has_goal = True
            self._state = "IDLE"
            self._cv.notify()
            return True

    def submit_goal_force(self, x: float, y: float, z: float):
        """Принудительно задать новую цель"""
        with self._cv:
            self._goal = (x, y, z)
            self._has_goal = True
            self._state = "IDLE"
            self._cv.notify()

    def wait_until_done(self, timeout: float = None) -> str:
        """Ждать завершения текущего перемещения"""
        start = time.time()
        while True:
            state = self.get_state()
            if state in ("REACHED", "TIMEOUT", "IDLE"):
                return state
            if timeout and (time.time() - start > timeout):
                return "TIMEOUT"
            time.sleep(0.1)

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
                self.cmd.go_to_local_point(y, x, z, yaw=0.0)
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
# ФУНКЦИИ МИССИИ
# =============================================================================

def initial_search(cmd: DroneCommander, nav: NavigatorThread, 
                   tracker: ArucoTrackerThread, marker_map: MarkerMap) -> bool:
    """
    Начальный поиск: летим вперёд пока не увидим любой маркер из сетки.
    После обнаружения калибруем карту.
    Возвращает True если калибровка успешна.
    """
    print("[SEARCH] Starting initial search for any grid marker...")
    
    drone_x, drone_y = cmd.get_pos_lps_xy()
    search_x, search_y = drone_x, drone_y
    
    for step in range(INITIAL_SEARCH_MAX_STEPS):
        # Проверяем текущий кадр
        with tracker.lock:
            markers = dict(tracker.markers)
        
        drone_x, drone_y = cmd.get_pos_lps_xy()
        drone_alt = cmd.get_alt_m()
        
        # Ищем любой маркер из нашей сетки
        for mid, mi in markers.items():
            if mid in marker_map.id_to_cell:
                # Нашли! Вычисляем его LPS-позицию и калибруем карту
                dx_m, dy_m = pixel_to_drone_xy(mi.cx, mi.cy, drone_alt)
                marker_lps_x = drone_x + dx_m
                marker_lps_y = drone_y + dy_m
                
                print(f"[SEARCH] Found marker {mid} at pixel ({mi.cx}, {mi.cy})")
                print(f"[SEARCH] Drone at ({drone_x:.2f}, {drone_y:.2f}), offset ({dx_m:.2f}, {dy_m:.2f})")
                
                marker_map.calibrate(mid, marker_lps_x, marker_lps_y)
                return True
        
        # Не нашли - делаем шаг вперёд
        search_x += INITIAL_SEARCH_STEP_X
        search_y += INITIAL_SEARCH_STEP_Y
        
        print(f"[SEARCH] Step {step+1}/{INITIAL_SEARCH_MAX_STEPS}: flying to ({search_x:.2f}, {search_y:.2f})")
        
        nav.submit_goal_force(search_x, search_y, TAKEOFF_HEIGHT)
        nav.wait_until_done(timeout=REACH_TIMEOUT + 2)
        
        # Пауза для стабилизации
        time.sleep(INITIAL_SEARCH_PAUSE)
    
    print("[SEARCH] Failed to find any marker!")
    return False


def navigate_to_marker(cmd: DroneCommander, nav: NavigatorThread, 
                       tracker: ArucoTrackerThread, marker_map: MarkerMap,
                       target_id: int, hold_time: float) -> bool:
    """
    Навигация к конкретному маркеру:
    1. Если маркер виден - визуальная навигация
    2. Если не виден - летим по карте, затем довыравниваемся визуально
    
    Возвращает True если успешно зависли над маркером.
    """
    print(f"[NAV] Navigating to marker {target_id}...")
    
    next_goal_try_t = time.monotonic()
    locked_goal: Optional[Tuple[float, float, float]] = None
    hold_until = 0.0
    
    # Сначала летим к предсказанной позиции (если маркер не виден)
    blind_flight_done = False
    
    while True:
        with tracker.lock:
            frame = tracker.latest_frame
            markers = dict(tracker.markers)
        
        if frame is None:
            time.sleep(0.01)
            continue
        
        h, w = frame.shape[:2]
        cx0, cy0 = int(w / 2), int(h / 2)
        
        drone_x, drone_y = cmd.get_pos_lps_xy()
        drone_alt = cmd.get_alt_m()
        nav_state = nav.get_state()
        
        # Обновляем позиции всех видимых маркеров в карте
        for mid, mi in markers.items():
            if mid in marker_map.id_to_cell:
                dx_m, dy_m = pixel_to_drone_xy(mi.cx, mi.cy, drone_alt)
                marker_map.update_marker_position(mid, drone_x + dx_m, drone_y + dy_m)
        
        if locked_goal is not None and nav_state in ("REACHED", "TIMEOUT"):
            locked_goal = None
        
        # Режим зависания (удержание над маркером)
        if time.time() < hold_until:
            current_state = f"HOLD({target_id})"
            center_color = GREEN
            
            _draw_frame(frame, cx0, cy0, center_color, current_state, nav_state,
                       target_id, hold_time, drone_x, drone_y, drone_alt,
                       locked_goal, marker_map, markers)
            
            time.sleep(LOOP_DT)
            continue
        elif hold_until > 0:
            # Зависание завершено
            print(f"[NAV] Hold complete for marker {target_id}")
            return True
        
        # Маркер виден - визуальная навигация
        if target_id in markers:
            mi = markers[target_id]
            dx_px = mi.cx - cx0
            dy_px = mi.cy - cy0
            centered_px = (abs(dx_px) <= CENTER_TOL_PX and abs(dy_px) <= CENTER_TOL_PX)
            
            dx_m, dy_m = pixel_to_drone_xy(mi.cx, mi.cy, drone_alt)
            centered_m = (abs(dx_m) <= CENTER_TOL_M and abs(dy_m) <= CENTER_TOL_M)
            
            if centered_px and centered_m:
                hold_until = time.time() + hold_time
                print(f"[NAV] Marker {target_id} centered! Holding for {hold_time:.1f}s")
                current_state = f"HOLD({target_id})"
                center_color = GREEN
            else:
                current_state = f"VISUAL({target_id})"
                center_color = CYAN
                
                target_x = drone_x + dx_m
                target_y = drone_y + dy_m
                goal = (target_x, target_y, TAKEOFF_HEIGHT)
                
                now = time.monotonic()
                if now >= next_goal_try_t:
                    accepted = nav.submit_goal_if_idle(*goal)
                    if accepted:
                        locked_goal = goal
                    next_goal_try_t = now + GOAL_TRY_DT
        
        # Маркер не виден - летим по карте
        else:
            target_pos = marker_map.get_position(target_id)
            
            if target_pos is None:
                print(f"[NAV] ERROR: marker {target_id} not in map!")
                return False
            
            current_state = f"BLIND({target_id})"
            center_color = YELLOW
            
            # Проверяем, долетели ли мы примерно до цели
            dist_to_target = ((drone_x - target_pos[0])**2 + (drone_y - target_pos[1])**2)**0.5
            
            if dist_to_target < 0.3 and blind_flight_done:
                # Мы рядом, но не видим маркер - проблема
                current_state = f"SEARCH_LOCAL({target_id})"
                center_color = RED
            
            goal = (target_pos[0], target_pos[1], TAKEOFF_HEIGHT)
            
            now = time.monotonic()
            if now >= next_goal_try_t:
                accepted = nav.submit_goal_if_idle(*goal)
                if accepted:
                    locked_goal = goal
                    blind_flight_done = True
                    print(f"[NAV] Flying blind to ({target_pos[0]:.2f}, {target_pos[1]:.2f})")
                next_goal_try_t = now + GOAL_TRY_DT
        
        _draw_frame(frame, cx0, cy0, center_color, current_state, nav_state,
                   target_id, hold_time, drone_x, drone_y, drone_alt,
                   locked_goal, marker_map, markers)
        
        time.sleep(LOOP_DT)
        
        # Проверка выхода по ESC
        if SHOW_WINDOW:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return False


def _draw_frame(frame, cx0, cy0, center_color, state, nav_state,
                target_id, hold_time, drone_x, drone_y, drone_alt,
                locked_goal, marker_map, visible_markers):
    """Отрисовка информации на кадре"""
    if not SHOW_WINDOW:
        return
    
    cv2.circle(frame, (cx0, cy0), 7, center_color, -1)
    
    y = 25
    cv2.putText(frame, f"State: {state} | NAV: {nav_state}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y += 25
    
    cv2.putText(frame, f"Target: {target_id} | Hold: {hold_time:.1f}s",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y += 25
    
    cv2.putText(frame, f"Drone: ({drone_x:.2f}, {drone_y:.2f}) alt={drone_alt:.2f}m",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y += 25
    
    if locked_goal:
        gx, gy, gz = locked_goal
        cv2.putText(frame, f"Goal: ({gx:.2f}, {gy:.2f})",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)
        y += 25
    
    # Показываем позиции маркеров из карты
    all_pos = marker_map.get_all_positions()
    cv2.putText(frame, "Map:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 18
    
    for mid, (mx, my) in sorted(all_pos.items()):
        visible = mid in visible_markers
        color = GREEN if visible else (150, 150, 150)
        prefix = "*" if mid == target_id else " "
        cv2.putText(frame, f"{prefix}M{mid}: ({mx:.2f},{my:.2f})",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 16
    
    cv2.imshow("aruco_nav_grid", frame)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("ARUCO GRID NAVIGATION")
    print("=" * 60)
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"Grid spacing: X={GRID_SPACING_X}m, Y={GRID_SPACING_Y}m")
    print(f"Mission plan: {MARKER_PLAN}")
    print()
    print("Marker grid layout:")
    print("  [row,col] -> ID")
    for (r, c), mid in sorted(MARKER_GRID.items()):
        print(f"  [{r},{c}] -> {mid}")
    print("=" * 60)
    
    # Создаём карту маркеров
    marker_map = MarkerMap(MARKER_GRID, GRID_SPACING_X, GRID_SPACING_Y)
    
    # Запускаем потоки
    tracker = ArucoTrackerThread()
    tracker.start()
    
    cmd = DroneCommander(dry_run=DRY_RUN)
    nav = NavigatorThread(cmd=cmd, reach_timeout=REACH_TIMEOUT, poll_dt=REACH_POLL_DT)
    nav.start()
    
    try:
        # Взлёт
        if not DRY_RUN:
            print("[FLIGHT] Arming and taking off...")
            cmd.arm_takeoff_to_height(TAKEOFF_HEIGHT)
            print(f"[FLIGHT] Hovering at {TAKEOFF_HEIGHT}m")
        else:
            print("[DRY] Skipping arm/takeoff")
        
        # Ждём запуска камеры
        time.sleep(1.0)
        
        # Начальный поиск и калибровка
        if not initial_search(cmd, nav, tracker, marker_map):
            print("[ERROR] Initial search failed!")
            return
        
        print("\n[MISSION] Starting mission plan...")
        
        # Выполняем план миссии
        for idx, (target_id, hold_time) in enumerate(MARKER_PLAN):
            print(f"\n{'='*40}")
            print(f"[MISSION] Step {idx+1}/{len(MARKER_PLAN)}: Marker {target_id}, hold {hold_time}s")
            print(f"{'='*40}")
            
            success = navigate_to_marker(cmd, nav, tracker, marker_map, target_id, hold_time)
            
            if success:
                print(f"[MISSION] ✓ Marker {target_id} complete")
            else:
                print(f"[MISSION] ✗ Marker {target_id} failed")
        
        print("\n" + "=" * 60)
        print("[MISSION] COMPLETE!")
        print("=" * 60)
        
        # Финальная карта
        print("\nFinal marker positions:")
        for mid, (mx, my) in sorted(marker_map.get_all_positions().items()):
            cell = marker_map.get_cell(mid)
            print(f"  Marker {mid} [{cell}]: ({mx:.2f}, {my:.2f})")
        
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        nav.stop()
        tracker.stop()
        nav.join(timeout=2.0)
        tracker.join(timeout=2.0)
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        cmd.land_and_close()
        print("[INFO] Shutdown complete")


if __name__ == "__main__":
    main()
