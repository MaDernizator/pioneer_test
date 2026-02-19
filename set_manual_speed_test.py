import time
from pioneer_sdk import Pioneer

HZ = 20.0
DT = 1.0 / HZ

def wait_local_pos(p: Pioneer, timeout=2.0):
    """Ждём пока начнут приходить локальные координаты."""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        pos = p.get_local_position_lps(get_last_received=True)
        if pos is not None:
            last = pos
            return last
        time.sleep(0.05)
    return last

def send_for(p: Pioneer, seconds: float, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
    """Шлём set_manual_speed с заданной частотой."""
    t0 = time.time()
    ok_cnt = 0
    total = 0
    while time.time() - t0 < seconds:
        total += 1
        if p.set_manual_speed(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate):
            ok_cnt += 1
        time.sleep(DT)
    return ok_cnt, total

def log_step(p: Pioneer, name: str, seconds: float, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
    before = wait_local_pos(p, timeout=1.0)
    if before is None:
        print("[WARN] Нет локальных координат (get_local_position_lps вернул None).")
        before = [float("nan"), float("nan"), float("nan")]

    ok, total = send_for(p, seconds, vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)
    time.sleep(0.3)  # дать фильтрам/оценке положения “досчитать”
    after = wait_local_pos(p, timeout=1.0)
    if after is None:
        after = [float("nan"), float("nan"), float("nan")]

    dx = after[0] - before[0]
    dy = after[1] - before[1]
    dz = after[2] - before[2]

    print(f"\n=== {name} ===")
    print(f"cmd: vx={vx:+.2f}, vy={vy:+.2f}, vz={vz:+.2f}, yaw_rate={yaw_rate:+.2f} rad/s, t={seconds:.1f}s")
    print(f"pos before: x={before[0]:+.3f}, y={before[1]:+.3f}, z={before[2]:+.3f}")
    print(f"pos after : x={after[0]:+.3f}, y={after[1]:+.3f}, z={after[2]:+.3f}")
    print(f"delta     : dx={dx:+.3f}, dy={dy:+.3f}, dz={dz:+.3f}")
    print(f"send ok   : {ok}/{total} ({(ok/total*100.0 if total else 0):.1f}%)")
    return dx, dy, dz

if __name__ == "__main__":
    # Подключение — подставь свои параметры, если нужно
    p = Pioneer()  # например: Pioneer(ip="192.168.4.1", mavlink_port=8001, connection_method="udpout")

    print("ПРОВЕРКА set_manual_speed(): короткие импульсы по осям.")
    print("ВАЖНО: делай в безопасной зоне, на высоте ~1–1.5м, без людей рядом.\n")

    # Небольшой обратный отсчёт (успеть отойти)
    for i in range(3, 0, -1):
        print(f"Старт через {i}...")
        time.sleep(1)

    try:
        print("ARM...")
        if not p.arm():
            print("Не удалось arm()")
            raise SystemExit(1)

        time.sleep(0.5)

        print("TAKEOFF...")
        if not p.takeoff():
            print("Не удалось takeoff()")
            raise SystemExit(1)

        time.sleep(3.0)  # стабилизироваться после взлёта

        # Параметры теста (м/с и рад/с)
        V = 0.3          # горизонтальная скорость
        VZ = 0.25        # вертикальная скорость
        T = 1.2          # длительность импульса

        # 1) Проверка осей скорости
        log_step(p, "TEST +VX", T, vx=+V, vy=0,  vz=0,  yaw_rate=0)
        time.sleep(2.0)

        log_step(p, "TEST -VX", T, vx=-V, vy=0,  vz=0,  yaw_rate=0)
        time.sleep(2.0)

        log_step(p, "TEST +VY", T, vx=0,  vy=+V, vz=0,  yaw_rate=0)
        time.sleep(2.0)

        log_step(p, "TEST -VY", T, vx=0,  vy=-V, vz=0,  yaw_rate=0)
        time.sleep(2.0)

        log_step(p, "TEST +VZ", T, vx=0,  vy=0,  vz=+VZ, yaw_rate=0)
        time.sleep(2.0)

        log_step(p, "TEST -VZ", T, vx=0,  vy=0,  vz=-VZ, yaw_rate=0)
        time.sleep(2.0)

        # 2) Проверка рысканья (yaw_rate)
        YR = 0.6  # rad/s (~34°/s)
        log_step(p, "TEST +YAW_RATE", 1.5, vx=0, vy=0, vz=0, yaw_rate=+YR)
        time.sleep(2.0)
        log_step(p, "TEST -YAW_RATE", 1.5, vx=0, vy=0, vz=0, yaw_rate=-YR)
        time.sleep(2.0)

        print("\nГотово. Если хочешь — по логам можно сразу вывести 'что есть X/Y/Z'.")
        print("Садимся...")

    finally:
        try:
            p.land()
        except Exception:
            pass
        time.sleep(2.0)
        try:
            p.disarm()
        except Exception:
            pass
