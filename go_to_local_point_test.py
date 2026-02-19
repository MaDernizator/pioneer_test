import time
from pioneer_sdk import Pioneer


def wait_reached(p: Pioneer, timeout_s: float = 15.0, poll_s: float = 0.1) -> bool:
    """Ждём point_reached() с таймаутом."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.point_reached():
            return True
        time.sleep(poll_s)
    return False


def get_pos(p: Pioneer):
    """Читаем позицию (x,y,z) из LPS/локальной системы."""
    pos = p.get_local_position_lps()
    if not pos:
        return None
    # Обычно это [x, y, z]
    return float(pos[0]), float(pos[1]), float(pos[2])


def print_pos(label: str, p: Pioneer):
    pos = get_pos(p)
    if pos is None:
        print(f"{label}: position = (no data)")
        return None
    print(f"{label}: x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}")
    return pos


def goto_and_log(p: Pioneer, x: float, y: float, z: float, yaw: float,
                 name: str, timeout_s: float = 15.0):
    print(f"\n>>> {name}: go_to_local_point(x={x}, y={y}, z={z}, yaw={yaw})")
    before = print_pos("BEFORE", p)

    p.go_to_local_point(x=x, y=y, z=z, yaw=yaw)
    ok = wait_reached(p, timeout_s=timeout_s)

    after = print_pos("AFTER ", p)
    if before and after:
        dx = after[0] - before[0]
        dy = after[1] - before[1]
        dz = after[2] - before[2]
        print(f"DELTA : dx={dx:.3f}  dy={dy:.3f}  dz={dz:.3f}")

    print("RESULT:", "REACHED" if ok else f"TIMEOUT ({timeout_s}s)")
    return ok


def main():
    # Если ты подключаешься нестандартно (udpout/udpin, ip, port) — впиши параметры тут
    p = Pioneer()

    STEP = 0.5   # шаг по оси, м (поставь 0.3 для совсем безопасного старта)
    H = 1.0      # высота, м
    YAW = 0.0    # yaw в радианах (в примерах обычно 0)

    print("=== go_to_local_point AXIS TEST ===")
    print("План:")
    print("1) взлёт -> точка (0,0,H)")
    print("2) шаг +X -> возврат")
    print("3) шаг +Y -> возврат")
    print("4) шаг -X -> возврат")
    print("5) шаг -Y -> возврат")
    print("\nНаблюдай физически: куда полетит дрон на +X и +Y.\n")

    try:
        print_pos("START", p)

        print("\nARM...")
        p.arm()
        time.sleep(0.5)

        print("TAKEOFF...")
        p.takeoff()
        time.sleep(1.0)

        # Подняться и “зафиксироваться” в нуле
        goto_and_log(p, 0.0, 0.0, H, YAW, "HOME (0,0,H)", timeout_s=20.0)

        input("\nENTER -> выполнить шаг +X (наблюдай направление движения) ")
        goto_and_log(p, +STEP, 0.0, H, YAW, f"+X ({STEP},0,H)")
        goto_and_log(p, 0.0, 0.0, H, YAW, "BACK HOME")

        input("\nENTER -> выполнить шаг +Y (наблюдай направление движения) ")
        goto_and_log(p, 0.0, +STEP, H, YAW, f"+Y (0,{STEP},H)")
        goto_and_log(p, 0.0, 0.0, H, YAW, "BACK HOME")

        input("\nENTER -> выполнить шаг -X (наблюдай направление движения) ")
        goto_and_log(p, -STEP, 0.0, H, YAW, f"-X (-{STEP},0,H)")
        goto_and_log(p, 0.0, 0.0, H, YAW, "BACK HOME")

        input("\nENTER -> выполнить шаг -Y (наблюдай направление движения) ")
        goto_and_log(p, 0.0, -STEP, H, YAW, f"-Y (0,-{STEP},H)")
        goto_and_log(p, 0.0, 0.0, H, YAW, "BACK HOME")

        print("\nLAND...")
        p.land()
        print("DONE.")

    except KeyboardInterrupt:
        print("\nCTRL+C -> LAND...")
        try:
            p.land()
        except Exception:
            pass


if __name__ == "__main__":
    main()
