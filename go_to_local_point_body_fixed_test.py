# test_body_fixed_axes.py
# Тест осей для go_to_local_point_body_fixed() на Pioneer (LPS)
#
# Идея:
# 1) взлет -> зависание
# 2) делаем небольшие смещения: +X, -X, +Y, -Y, +Z, -Z
# 3) после каждого шага печатаем LPS координаты ДО/ПОСЛЕ и дельту
#
# Безопасность:
# - ставь SMALL_STEP = 0.3..0.5 м
# - выполняй в просторной зоне / в симуляторе
# - z используем аккуратно (вверх/вниз)

import time
import math
from pioneer_sdk import Pioneer

SMALL_STEP = 0.5      # метры
HOVER_Z = 1.2         # метры (для выхода на стабильную высоту)
WAIT_TIMEOUT = 12.0   # секунд на достижение точки
PAUSE_AFTER = 0.7     # пауза после достижения точки


def now():
    return time.strftime("%H:%M:%S")


def safe_get_pos(p: Pioneer):
    # В старом SDK метод: get_local_position_lps(get_last_received=True/False)
    # В SDK2: get_local_position_lps() -> tuple|None
    try:
        pos = p.get_local_position_lps()
        return pos
    except TypeError:
        # если сигнатура со старым флагом
        return p.get_local_position_lps(get_last_received=True)


def safe_get_yaw(p: Pioneer):
    # SDK2 имеет get_local_yaw_lps(); в старом SDK может отсутствовать
    if hasattr(p, "get_local_yaw_lps"):
        return p.get_local_yaw_lps()
    return None


def wait_point(p: Pioneer, timeout_s: float = WAIT_TIMEOUT):
    t0 = time.time()
    # В старом SDK: point_reached() сбрасывает флаг после чтения
    while time.time() - t0 < timeout_s:
        try:
            if p.point_reached():
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def log_step(title: str, before, after, yaw_before=None, yaw_after=None):
    def fmt(v):
        if v is None:
            return "None"
        return f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"

    delta = None
    if before is not None and after is not None:
        delta = (after[0] - before[0], after[1] - before[1], after[2] - before[2])

    print(f"\n[{now()}] === {title} ===")
    print(f"  pos before: {fmt(before)}")
    print(f"  pos after : {fmt(after)}")
    if delta is not None:
        print(f"  delta     : ({delta[0]:+.3f}, {delta[1]:+.3f}, {delta[2]:+.3f})")
    if yaw_before is not None or yaw_after is not None:
        print(f"  yaw before: {yaw_before}")
        print(f"  yaw after : {yaw_after}")


def send_body_step(p: Pioneer, x, y, z, yaw_value):
    ok = p.go_to_local_point_body_fixed(x=x, y=y, z=z, yaw=yaw_value)
    if not ok:
        print(f"[{now()}] WARN: go_to_local_point_body_fixed returned False (x={x},y={y},z={z},yaw={yaw_value})")
        return False
    reached = wait_point(p)
    if not reached:
        print(f"[{now()}] WARN: timeout waiting POINT_REACHED (x={x},y={y},z={z})")
    time.sleep(PAUSE_AFTER)
    return reached


def main():
    p = Pioneer(logger=True, log_connection=True)

    airborne = False
    try:
        print(f"[{now()}] Connecting...")

        # (опционально) если SDK2 — можно проверить, что активен LPS
        if hasattr(p, "get_nav_system"):
            nav = p.get_nav_system(update=True)
            print(f"[{now()}] Nav system: {nav}")
        if hasattr(p, "get_nav_status_lps"):
            st = p.get_nav_status_lps()
            print(f"[{now()}] LPS status: {st}")

        print(f"[{now()}] ARM")
        p.arm()

        print(f"[{now()}] TAKEOFF")
        p.takeoff()
        airborne = True

        # Взлетели — выходим на “точку зависания” в локальной СК
        # (в примерах обычно z положительное для подъема)
        print(f"[{now()}] Go hover: (0,0,{HOVER_Z}) in LOCAL frame")
        p.go_to_local_point(x=0, y=0, z=HOVER_Z, yaw=0)
        wait_point(p)
        time.sleep(PAUSE_AFTER)

        # ---- ВАЖНО ПРО YAW ----
        # SDK2: yaw в градусах. Старый SDK: yaw в радианах. :contentReference[oaicite:2]{index=2}
        # Чтобы не ловить “повороты” во время теста, ставим yaw = 0.
        # Если у тебя старый SDK — 0 рад тоже нормально.
        yaw_cmd = 0

        steps = [
            ("BODY +X", +SMALL_STEP, 0.0, 0.0),
            ("BODY -X", -SMALL_STEP, 0.0, 0.0),
            ("BODY +Y", 0.0, +SMALL_STEP, 0.0),
            ("BODY -Y", 0.0, -SMALL_STEP, 0.0),
            ("BODY +Z", 0.0, 0.0, +min(0.3, SMALL_STEP)),  # аккуратнее по Z
            ("BODY -Z", 0.0, 0.0, -min(0.3, SMALL_STEP)),
        ]

        print("\nСейчас пойдут шаги body_fixed. Смотри на delta — это и есть ‘как оси работают’ в твоём случае.\n")

        for name, x, y, z in steps:
            b = safe_get_pos(p)
            yb = safe_get_yaw(p)
            print(f"[{now()}] CMD {name}: x={x:+.2f} y={y:+.2f} z={z:+.2f} yaw={yaw_cmd}")
            send_body_step(p, x, y, z, yaw_cmd)
            a = safe_get_pos(p)
            ya = safe_get_yaw(p)
            log_step(name, b, a, yb, ya)

        print(f"\n[{now()}] Done. Landing...")
        p.land()

    finally:
        try:
            if airborne:
                time.sleep(1.0)
        except Exception:
            pass
        try:
            p.close_connection()
        except Exception:
            pass


if __name__ == "__main__":
    main()
