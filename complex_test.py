# pioneer_minimal_test.py
from pioneer_sdk import Pioneer
import time
from typing import Optional, Tuple

# =========================
# Настройки
# =========================
DRONE_IP = None
MAVLINK_PORT = None
CONNECTION_METHOD = None

TARGET_Z_AFTER_TAKEOFF = None  # например 1.0 если хочешь довести до 1м; None если не нужно
YAW = 0

# =========================
# Переключатели тестов
# =========================
TEST_ARM_TAKEOFF = True
TEST_GET_COORDS_ONCE = True

TEST_MANUAL_SPEED_WORLD = False
TEST_MANUAL_SPEED_BODY_FIXED = False
TEST_GOTO_LOCAL_POINT = True
TEST_GOTO_BODY_FIXED = False

TEST_LAND_AT_END = True


def get_coords(p: Pioneer) -> Optional[Tuple[float, float, float]]:
    arr = p.get_local_position_lps()
    if not arr:
        return None
    return float(arr[0]), float(arr[1]), float(arr[2])


def wait_point_reached(p: Pioneer, timeout_s: float = 15.0, poll_s: float = 0.1) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.point_reached():
            return True
        time.sleep(poll_s)
    return False


def manual_speed_for(p: Pioneer, *, body_fixed: bool, vx: float, vy: float, vz: float, yaw_rate: float,
                     duration_s: float, send_period_s: float = 0.05) -> None:
    t0 = time.time()
    while time.time() - t0 < duration_s:
        if body_fixed:
            p.set_manual_speed_body_fixed(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)
        else:
            p.set_manual_speed(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)
        time.sleep(send_period_s)


def connect_pioneer() -> Pioneer:
    kwargs = {}
    if DRONE_IP is not None:
        kwargs["ip"] = DRONE_IP
    if MAVLINK_PORT is not None:
        kwargs["mavlink_port"] = MAVLINK_PORT
    if CONNECTION_METHOD is not None:
        kwargs["connection_method"] = CONNECTION_METHOD
    return Pioneer(**kwargs)


def main():
    p = connect_pioneer()

    try:
        # ---- ARM + TAKEOFF (takeoff сам поднимает ~0.5м) ----
        if TEST_ARM_TAKEOFF:
            print("[TEST] arm + takeoff (~0.5m auto)")
            p.arm()
            p.takeoff()

            # Дать дрону стабилизироваться
            time.sleep(2.0)

            # (опционально) довести до конкретной высоты после takeoff
            if TARGET_Z_AFTER_TAKEOFF is not None:
                c = get_coords(p)
                if c is not None:
                    x, y, z = c
                    print(f"  current z after takeoff: {z:.2f} -> target z: {TARGET_Z_AFTER_TAKEOFF:.2f}")
                    p.go_to_local_point(x=x, y=y, z=TARGET_Z_AFTER_TAKEOFF, yaw=YAW)
                    ok = wait_point_reached(p, timeout_s=20)
                    print(f"  reached target z: {ok}")
                else:
                    print("  coords not available, skip TARGET_Z_AFTER_TAKEOFF")

        # ---- Координаты ----
        if TEST_GET_COORDS_ONCE:
            print("[TEST] get_local_position_lps (once)")
            print("  coords:", get_coords(p))

        # ==========================================================
        # 4 ОСНОВНЫЕ ФУНКЦИИ ПЕРЕДВИЖЕНИЯ
        # ==========================================================

        if TEST_MANUAL_SPEED_WORLD:
            print("[TEST] set_manual_speed")
            manual_speed_for(p, body_fixed=False, vx=0, vy=1, vz=0, yaw_rate=0, duration_s=2.0)
            time.sleep(1.0)

        if TEST_MANUAL_SPEED_BODY_FIXED:
            print("[TEST] set_manual_speed_body_fixed")
            manual_speed_for(p, body_fixed=True, vx=1, vy=0, vz=0, yaw_rate=0, duration_s=2.0)
            time.sleep(1.0)

        if TEST_GOTO_LOCAL_POINT:
            print("[TEST] go_to_local_point: (0,1,current_z)")
            c = get_coords(p)
            z = c[2] if c else 0.5  # если координат нет, оставляем дефолт
            p.go_to_local_point(x=0, y=1, z=z, yaw=YAW)
            print("  reached:", wait_point_reached(p, timeout_s=20))

        if TEST_GOTO_BODY_FIXED:
            print("[TEST] go_to_local_point_body_fixed: forward 1m")
            p.go_to_local_point_body_fixed(x=1, y=0, z=0, yaw=0)
            print("  reached:", wait_point_reached(p, timeout_s=20))

        # ---- Посадка ----
        if TEST_LAND_AT_END:
            print("[TEST] land")
            p.land()

    except KeyboardInterrupt:
        print("\n[CTRL+C] land")
        try:
            p.land()
        except Exception:
            pass
    finally:
        try:
            p.close_connection()
        except Exception:
            pass


if __name__ == "__main__":
    main()
