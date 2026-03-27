import time

from pioneer_sdk import Pioneer


DRY_RUN = False
TAKEOFF_HEIGHT = 2.0
YAW = 0.0

# Координаты маркеров: marker_id -> (x, y)
MARKERS = {
    3: (0.0, 0.0),
    2: (1.0, 0.0),
    0: (1.0, 1.0),
    4: (0.0, 1.0),
    7: (-1.0, 1.0),
    5: (-1.0, 0.0),
    1: (0.0, -1.0),
}

# Маршрут: в каком порядке пролетать маркеры
ROUTE = [3, 2, 0, 4, 7, 5, 1]


def go_to_marker(drone: Pioneer, marker_id: int, z: float):
    x, y = MARKERS[marker_id]
    print(f"[INFO] Fly to marker {marker_id}: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    drone.go_to_local_point(x=x, y=y, z=z, yaw=YAW)

    while not drone.point_reached():
        time.sleep(0.1)


if __name__ == "__main__":
    drone = Pioneer()

    try:
        if not DRY_RUN:
            print("[INFO] Arm")
            drone.arm()

            print("[INFO] Takeoff")
            drone.takeoff()
            time.sleep(3)

            drone.go_to_local_point(x=0, y=0, z=TAKEOFF_HEIGHT, yaw=YAW)
            while not drone.point_reached():
                time.sleep(0.1)
        else:
            print("[INFO] DRY_RUN mode")

        for marker_id in ROUTE:
            if marker_id not in MARKERS:
                raise ValueError(f"Marker {marker_id} not found in MARKERS")

            if DRY_RUN:
                x, y = MARKERS[marker_id]
                print(f"[DRY] go_to_local_point(x={x:.2f}, y={y:.2f}, z={TAKEOFF_HEIGHT:.2f}, yaw={YAW:.2f})")
                time.sleep(1)
            else:
                go_to_marker(drone, marker_id, TAKEOFF_HEIGHT)

        print("[INFO] Mission completed")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")

    finally:
        if not DRY_RUN:
            print("[INFO] Land")
            drone.land()

        drone.close_connection()