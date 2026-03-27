import math
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

try:
    from pioneer_sdk import Pioneer
except Exception:
    Pioneer = None


# ============================================================
# НАСТРОЙКИ МИССИИ
# ============================================================

DRY_RUN = False          # True -> без arm/takeoff/land, команды только печатаются
TAKEOFF_HEIGHT = 2.4     # рабочая высота полета
YAW = 0.0                # yaw фиксируем как в вашем коде
LOOP_DT = 0.05

# Координаты маркеров в локальной системе координат LPS, метры.
# Формат:
#     marker_id: (x, y)
MARKERS: Dict[int, Tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (1.2, 0.0),
    2: (1.2, 1.0),
    3: (0.0, 1.0),
    4: (-0.8, 0.5),
}

# Маршрут: в каком порядке пролетать маркеры
# Здесь можно просто переставлять ID.
ROUTE: List[int] = [0, 1, 2, 3, 4, 0]

# Если нужно зависать над конкретными маркерами:
# marker_id: время зависания в секундах
HOLD_TIMES: Dict[int, float] = {
    0: 1.0,
    1: 1.0,
    2: 2.0,
    3: 1.0,
    4: 1.0,
}

# Таймаут ожидания достижения точки
REACH_TIMEOUT = 20.0
REACH_POLL_DT = 0.1

# Дополнительная проверка "мы реально близко к цели"
XY_TOLERANCE_M = 0.20
Z_TOLERANCE_M = 0.25

# Если True — после миссии дрон вернется в точку (0, 0, TAKEOFF_HEIGHT)
RETURN_HOME_BEFORE_LAND = True


# ============================================================
# ДАННЫЕ
# ============================================================

@dataclass
class MarkerPoint:
    x: float
    y: float


# ============================================================
# КОМАНДИР ДРОНА
# ============================================================

class DroneCommander:
    """
    DRY_RUN=True:
      - НЕ arm/takeoff/land
      - создаем объект Pioneer, чтобы можно было читать телеметрию
      - go_to_local_point только печатаем

    DRY_RUN=False:
      - обычный полет
    """
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        if Pioneer is None:
            raise RuntimeError("pioneer_sdk недоступен")
        self.p = Pioneer()

    def arm_takeoff_to_height(self, z: float):
        if self.dry_run:
            print(f"[DRY] Skip arm/takeoff, target_z={z:.2f}")
            return

        print("[INFO] Arm")
        self.p.arm()

        print("[INFO] Takeoff")
        self.p.takeoff()

        # Как и у вас — подъем в 2 этапа, но через обычное деление / 2
        z_half = z / 2.0
        print(f"[INFO] Climb to half height: {z_half:.2f} m")
        self.p.go_to_local_point(x=0.0, y=0.0, z=z_half, yaw=YAW)
        while not self.p.point_reached():
            time.sleep(0.1)

        print(f"[INFO] Climb to work height: {z:.2f} m")
        self.p.go_to_local_point(x=0.0, y=0.0, z=z, yaw=YAW)
        while not self.p.point_reached():
            time.sleep(0.1)

    def go_to_local_point(self, x: float, y: float, z: float, yaw: float = 0.0):
        if self.dry_run:
            print(f"[DRY] go_to_local_point(x={x:+.2f}, y={y:+.2f}, z={z:+.2f}, yaw={yaw:+.2f})")
            return

        self.p.go_to_local_point(x=x, y=y, z=z, yaw=yaw)

    def point_reached(self) -> bool:
        if self.dry_run:
            return False
        return self.p.point_reached()

    def get_pos_lps_xyz(self) -> Tuple[float, float, float]:
        try:
            pos = self.p.get_local_position_lps(get_last_received=True)
            if pos is None:
                return 0.0, 0.0, 0.0
            return float(pos[0]), float(pos[1]), float(pos[2])
        except Exception:
            return 0.0, 0.0, 0.0

    def get_alt_m(self) -> float:
        try:
            alt = self.p.get_dist_sensor_data(get_last_received=True)
            if alt is None:
                return 0.0
            return float(alt)
        except Exception:
            return 0.0

    def connected(self) -> bool:
        try:
            return bool(self.p.connected())
        except Exception:
            return False

    def land_and_close(self):
        try:
            if not self.dry_run:
                print("[INFO] Land")
                self.p.land()
        finally:
            try:
                self.p.close_connection()
            except Exception:
                pass
            del self.p


# ============================================================
# НАВИГАТОР
# ============================================================

class NavigatorThread(threading.Thread):
    """
    Поток навигации:
      - принимает цель
      - отправляет go_to_local_point ровно 1 раз на цель
      - ждёт достижения:
          * либо point_reached()
          * либо фактическое попадание в допуск по позиции
      - пока не достигнуто — новые цели не принимает
    """
    def __init__(
        self,
        cmd: DroneCommander,
        reach_timeout: float = 15.0,
        poll_dt: float = 0.1,
        xy_tol: float = 0.20,
        z_tol: float = 0.25,
    ):
        super().__init__(daemon=True)
        self.cmd = cmd
        self.reach_timeout = reach_timeout
        self.poll_dt = poll_dt
        self.xy_tol = xy_tol
        self.z_tol = z_tol

        self._cv = threading.Condition()
        self._running = True

        self._goal: Optional[Tuple[float, float, float]] = None
        self._has_goal = False

        self._state = "IDLE"      # IDLE, MOVING, REACHED, TIMEOUT, ERROR
        self._last_error = ""

    def stop(self):
        with self._cv:
            self._running = False
            self._cv.notify()

    def get_state(self) -> str:
        with self._cv:
            return self._state

    def get_last_error(self) -> str:
        with self._cv:
            return self._last_error

    def submit_goal_if_idle(self, x: float, y: float, z: float) -> bool:
        with self._cv:
            if self._state == "MOVING":
                return False
            self._goal = (x, y, z)
            self._has_goal = True
            self._state = "IDLE"
            self._last_error = ""
            self._cv.notify()
            return True

    def _is_close_enough(self, target_x: float, target_y: float, target_z: float) -> bool:
        cur_x, cur_y, cur_z = self.cmd.get_pos_lps_xyz()
        dx = target_x - cur_x
        dy = target_y - cur_y
        dz = target_z - cur_z
        xy_dist = math.hypot(dx, dy)
        return xy_dist <= self.xy_tol and abs(dz) <= self.z_tol

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
                self._last_error = ""

            try:
                x, y, z = goal
                self.cmd.go_to_local_point(x, y, z, yaw=YAW)
            except Exception as e:
                with self._cv:
                    self._state = "ERROR"
                    self._last_error = f"go_to_local_point error: {e}"
                continue

            start = time.time()
            reached = False

            while time.time() - start <= self.reach_timeout:
                try:
                    # point_reached() может быть достаточным, но добавляем и проверку координат
                    if self.cmd.point_reached() or self._is_close_enough(x, y, z):
                        reached = True
                        break
                except Exception as e:
                    with self._cv:
                        self._state = "ERROR"
                        self._last_error = f"reach check error: {e}"
                    break

                time.sleep(self.poll_dt)

            with self._cv:
                if self._state == "ERROR":
                    pass
                else:
                    self._state = "REACHED" if reached else "TIMEOUT"


# ============================================================
# ПРОВЕРКА МИССИИ
# ============================================================

def build_marker_points(raw_markers: Dict[int, Tuple[float, float]]) -> Dict[int, MarkerPoint]:
    result: Dict[int, MarkerPoint] = {}
    for marker_id, coords in raw_markers.items():
        if not isinstance(marker_id, int):
            raise ValueError(f"Marker id must be int, got: {marker_id!r}")
        if not isinstance(coords, (tuple, list)) or len(coords) != 2:
            raise ValueError(f"Marker {marker_id}: coords must be (x, y)")
        x, y = coords
        result[marker_id] = MarkerPoint(float(x), float(y))
    return result


def validate_route(marker_points: Dict[int, MarkerPoint], route: List[int]):
    if not route:
        raise ValueError("ROUTE is empty")
    for marker_id in route:
        if marker_id not in marker_points:
            raise ValueError(f"Marker {marker_id} is used in ROUTE but not present in MARKERS")


# ============================================================
# ОСНОВНОЙ СЦЕНАРИЙ
# ============================================================

if __name__ == "__main__":
    marker_points = build_marker_points(MARKERS)
    validate_route(marker_points, ROUTE)

    cmd = DroneCommander(dry_run=DRY_RUN)
    nav = NavigatorThread(
        cmd=cmd,
        reach_timeout=REACH_TIMEOUT,
        poll_dt=REACH_POLL_DT,
        xy_tol=XY_TOLERANCE_M,
        z_tol=Z_TOLERANCE_M,
    )
    nav.start()

    route_idx = 0
    active_goal: Optional[Tuple[float, float, float]] = None
    hold_until = 0.0

    try:
        if not DRY_RUN:
            print("[INFO] Real flight mode")
            cmd.arm_takeoff_to_height(TAKEOFF_HEIGHT)
            print(f"[INFO] Hover reached: z={TAKEOFF_HEIGHT:.2f} m")
        else:
            print("[INFO] DRY_RUN mode")

        while True:
            if route_idx >= len(ROUTE):
                print("[INFO] Mission completed: route finished.")
                break

            marker_id = ROUTE[route_idx]
            mp = marker_points[marker_id]
            hold_time = float(HOLD_TIMES.get(marker_id, 0.0))

            nav_state = nav.get_state()

            # Если была активная цель и навигатор закончил обработку
            if active_goal is not None and nav_state in ("REACHED", "TIMEOUT", "ERROR"):
                if nav_state == "REACHED":
                    print(f"[INFO] Marker {marker_id} reached: x={mp.x:+.2f}, y={mp.y:+.2f}")
                    if hold_time > 0:
                        hold_until = time.time() + hold_time
                        print(f"[INFO] Hold over marker {marker_id}: {hold_time:.1f} s")
                        while time.time() < hold_until:
                            time.sleep(0.05)

                    route_idx += 1

                elif nav_state == "TIMEOUT":
                    print(f"[WARN] Timeout while flying to marker {marker_id}")
                    raise RuntimeError(f"Timeout while flying to marker {marker_id}")

                elif nav_state == "ERROR":
                    raise RuntimeError(nav.get_last_error())

                active_goal = None
                time.sleep(LOOP_DT)
                continue

            # Если цели сейчас нет — отправляем следующую
            if active_goal is None:
                goal = (mp.x, mp.y, TAKEOFF_HEIGHT)
                accepted = nav.submit_goal_if_idle(*goal)
                if accepted:
                    active_goal = goal
                    print(
                        f"[INFO] Route step {route_idx + 1}/{len(ROUTE)} -> "
                        f"marker {marker_id}: "
                        f"x={goal[0]:+.2f}, y={goal[1]:+.2f}, z={goal[2]:+.2f}"
                    )

            # Отладочная телеметрия
            cur_x, cur_y, cur_z = cmd.get_pos_lps_xyz()
            print(
                f"[STATE] route_idx={route_idx}/{len(ROUTE)} "
                f"target={marker_id} nav={nav_state} "
                f"pos=({cur_x:+.2f}, {cur_y:+.2f}, {cur_z:+.2f})"
            )

            time.sleep(LOOP_DT)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt -> exiting.")
    finally:
        try:
            # Возврат в home перед посадкой
            if RETURN_HOME_BEFORE_LAND:
                print("[INFO] Return to home before landing")
                cmd.go_to_local_point(0.0, 0.0, TAKEOFF_HEIGHT, yaw=YAW)

                if not DRY_RUN:
                    t0 = time.time()
                    while time.time() - t0 <= REACH_TIMEOUT:
                        if cmd.point_reached():
                            break
                        time.sleep(0.1)

        except Exception as e:
            print(f"[WARN] Return-home failed: {e}")

        nav.stop()
        nav.join(timeout=2.0)
        cmd.land_and_close()