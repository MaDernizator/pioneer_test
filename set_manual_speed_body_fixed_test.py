import time

from pioneer_sdk import Pioneer

# Для Windows — чтение клавиш без Enter
try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False


SPEED_XY = 0.30   # м/с (лучше 0.2–0.4 для тестов)
SPEED_Z  = 0.30   # м/с
YAW_RATE = 0.60   # рад/с (визуально, без телеметрии yaw)

PULSE_SEC = 1.0   # длительность импульса
HZ = 20           # частота отправки manual-speed команд


def get_pos(mini: Pioneer):
    """Вернёт (x,y,z) или None если LPS/телеметрия недоступны."""
    try:
        p = mini.get_local_position_lps()
        if not p:
            return None
        return float(p[0]), float(p[1]), float(p[2])
    except Exception:
        return None


def fmt_pos(p):
    if p is None:
        return "None"
    return f"x={p[0]:+.3f}, y={p[1]:+.3f}, z={p[2]:+.3f}"


def sub(a, b):
    if a is None or b is None:
        return None
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def fmt_delta(d):
    if d is None:
        return "Δ=None (нет LPS/телеметрии)"
    return f"Δx={d[0]:+.3f}, Δy={d[1]:+.3f}, Δz={d[2]:+.3f}"


def stop(mini: Pioneer, sec=0.2):
    """Явно остановить ручные скорости (несколько пакетов нуля)."""
    n = max(1, int(HZ * sec))
    for _ in range(n):
        mini.set_manual_speed_body_fixed(vx=0, vy=0, vz=0, yaw_rate=0)
        time.sleep(1 / HZ)


def pulse(mini: Pioneer, vx, vy, vz, yaw_rate, duration=PULSE_SEC):
    """Импульс manual speed + отчёт по смещению."""
    p0 = get_pos(mini)
    t0 = time.time()

    n = max(1, int(HZ * duration))
    for _ in range(n):
        mini.set_manual_speed_body_fixed(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)
        time.sleep(1 / HZ)

    stop(mini, sec=0.3)

    t1 = time.time()
    p1 = get_pos(mini)
    d = sub(p0, p1)

    print(f"  pos0: {fmt_pos(p0)}")
    print(f"  pos1: {fmt_pos(p1)}")
    print(f"  {fmt_delta(d)}  (dt={(t1 - t0):.2f}s)")
    print("-" * 60)


def read_key_blocking():
    """Блокирующее чтение 1 символа."""
    if HAS_MSVCRT:
        ch = msvcrt.getch()
        # стрелки/функциональные дают префикс b'\x00' или b'\xe0'
        if ch in (b"\x00", b"\xe0"):
            _ = msvcrt.getch()
            return None
        try:
            return ch.decode("utf-8", errors="ignore")
        except Exception:
            return None
    else:
        # fallback: через Enter
        s = input("> ").strip()
        return s[:1] if s else None


def print_help():
    print(r"""
Управление (импульс ~1с):
  i  -> +vx      (тест оси vx)
  k  -> -vx
  l  -> +vy      (тест оси vy)
  j  -> -vy
  u  -> +vz      (вверх)
  o  -> -vz      (вниз)
  q  -> yaw_rate - (поворот влево)
  e  -> yaw_rate + (поворот вправо)

  t  -> arm + takeoff + подняться на ~1.2м (через go_to_local_point)
  g  -> удержание (послать 0-скорости)
  a  -> авто-тест: +vx, +vy, +vz, yaw+
  n  -> land
  x  -> выход (попробует приземлиться)

Важно:
- тестируй в безопасной зоне, скорости маленькие;
- для Δx/Δy/Δz нужен LPS (get_local_position_lps).
""")


def main():
    mini = Pioneer()

    airborne = False
    print_help()
    print("[INFO] Connected. Нажми 't' для взлёта.")

    try:
        while True:
            ch = read_key_blocking()
            if not ch:
                continue

            ch = ch.lower()

            if ch == "t" and not airborne:
                print("[INFO] ARM...")
                mini.arm()
                time.sleep(1.0)
                print("[INFO] TAKEOFF...")
                mini.takeoff()
                time.sleep(2.0)

                # Подняться на ~1.2м (локальная система)
                print("[INFO] Go hover z=1.2...")
                mini.go_to_local_point(x=0, y=0, z=1.2, yaw=0)
                while not mini.point_reached():
                    time.sleep(0.1)

                airborne = True
                stop(mini)
                print("[INFO] Airborne. Делай импульсные тесты.")

            elif ch == "n":
                print("[INFO] LAND...")
                stop(mini)
                mini.land()
                airborne = False

            elif ch == "g":
                print("[INFO] STOP/HOLD (нулевые скорости).")
                stop(mini, sec=0.5)

            elif ch == "a":
                if not airborne:
                    print("[WARN] Сначала взлети (t).")
                    continue
                print("[AUTO] +vx")
                pulse(mini, +SPEED_XY, 0, 0, 0)
                print("[AUTO] +vy")
                pulse(mini, 0, +SPEED_XY, 0, 0)
                print("[AUTO] +vz")
                pulse(mini, 0, 0, +SPEED_Z, 0)
                print("[AUTO] yaw+ (визуально)")
                pulse(mini, 0, 0, 0, +YAW_RATE)

            elif ch == "x":
                print("[INFO] Exit.")
                break

            else:
                if not airborne:
                    print("[WARN] Сначала взлети (t).")
                    continue

                if ch == "i":
                    print("[TEST] +vx")
                    pulse(mini, +SPEED_XY, 0, 0, 0)
                elif ch == "k":
                    print("[TEST] -vx")
                    pulse(mini, -SPEED_XY, 0, 0, 0)
                elif ch == "l":
                    print("[TEST] +vy")
                    pulse(mini, 0, +SPEED_XY, 0, 0)
                elif ch == "j":
                    print("[TEST] -vy")
                    pulse(mini, 0, -SPEED_XY, 0, 0)
                elif ch == "u":
                    print("[TEST] +vz (up)")
                    pulse(mini, 0, 0, +SPEED_Z, 0)
                elif ch == "o":
                    print("[TEST] -vz (down)")
                    pulse(mini, 0, 0, -SPEED_Z, 0)
                elif ch == "q":
                    print("[TEST] yaw- (left)")
                    pulse(mini, 0, 0, 0, -YAW_RATE)
                elif ch == "e":
                    print("[TEST] yaw+ (right)")
                    pulse(mini, 0, 0, 0, +YAW_RATE)
                else:
                    print("[INFO] неизвестная клавиша. Жми 'a' (авто) или см. help выше.")

    finally:
        try:
            if airborne:
                print("[INFO] Landing before close...")
                stop(mini)
                mini.land()
        except Exception:
            pass

        try:
            mini.close_connection()
        except Exception:
            pass

        print("[INFO] Done.")


if __name__ == "__main__":
    main()
