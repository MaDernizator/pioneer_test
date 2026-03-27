import time
from pioneer_sdk import Pioneer


DRY_RUN = False
YAW = 0.0

# marker_id: (x, y, z, hold_time)
MARKERS = {
    3: (0.0,  0.0, 1.5, 1.0),
    2: (1.0,  0.0, 2.0, 5.0),
    0: (1.0,  1.0, 2.0, 1.0),
    4: (0.0,  1.0, 1.8, 1.0),
    7: (-1.0, 1.0, 2.2, 5.0),
    5: (-1.0, 0.0, 1.7, 1.0),
    1: (0.0, -1.0, 1.5, 1.0),
}

# Порядок облёта
ROUTE = [3, 2, 0, 4, 7, 5, 1]


def go_to_marker(drone: Pioneer, marker_id: int):
    x, y, z, hold_time = MARKERS[marker_id]

    print(f"[INFO] Fly to marker {marker_id}: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    drone.go_to_local_point(x=x, y=y, z=z, yaw=YAW)

    while not drone.point_reached():
        time.sleep(0.1)

    print(f"[INFO] Hold {hold_time:.1f} sec at marker {marker_id}")
    time.sleep(hold_time)


if __name__ == "__main__":
    drone = Pioneer()

    try:
        if not DRY_RUN:
            print("[INFO] Arm")
            drone.arm()

            print("[INFO] Takeoff")
            drone.takeoff()
            time.sleep(3)
        else:
            print("[INFO] DRY_RUN mode")

        for marker_id in ROUTE:
            if marker_id not in MARKERS:
                raise ValueError(f"Marker {marker_id} not found in MARKERS")

            x, y, z, hold_time = MARKERS[marker_id]

            if DRY_RUN:
                print(
                    f"[DRY] go_to_local_point("
                    f"x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={YAW:.2f})"
                )
                print(f"[DRY] hold {hold_time:.1f} sec")
                time.sleep(1)
            else:
                go_to_marker(drone, marker_id)

        print("[INFO] Mission completed")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")

    finally:
        if not DRY_RUN:
            print("[INFO] Land")
            drone.land()

        drone.close_connection()