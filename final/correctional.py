import time
import cv2

from pioneer_sdk import Pioneer, Camera
from pixel_projector import pixel_to_drone_xy


DRY_RUN = False
YAW = 0.0

CENTER_TOL_PX = 40
CENTER_TOL_M = 0.15
CORRECTION_GAIN = 0.9
MAX_CORRECTION_TRIES = 8

SHOW_WINDOW = True

# marker_id: (x, y, z, hold_time, use_correction)
MARKERS = {
    3: (0.0,  0.0, 1.5, 1.0, False),
    2: (1.0,  0.0, 2.0, 5.0, True),
    0: (1.0,  1.0, 2.0, 1.0, False),
    4: (0.0,  1.0, 1.8, 1.0, True),
    7: (-1.0, 1.0, 2.2, 5.0, True),
    5: (-1.0, 0.0, 1.7, 1.0, False),
    1: (0.0, -1.0, 1.5, 1.0, False),
}

ROUTE = [3, 2, 0, 4, 7, 5, 1]


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def wait_until_reached(drone: Pioneer):
    while not drone.point_reached():
        time.sleep(0.1)


def get_altitude(drone: Pioneer) -> float:
    try:
        return float(drone.get_dist_sensor_data())
    except Exception:
        return 1.0


def get_xy(drone: Pioneer):
    pos = drone.get_local_position_lps(get_last_received=True)
    return float(pos[0]), float(pos[1])


def detect_marker(frame, target_id):
    corners, ids, _ = aruco_detector.detectMarkers(frame)

    if ids is None or len(corners) == 0:
        return None

    ids_flat = ids.flatten().tolist()
    for i, marker_id in enumerate(ids_flat):
        if int(marker_id) == target_id:
            pts = corners[i][0]
            cx = int((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4.0)
            cy = int((pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4.0)
            return cx, cy, corners, ids

    return None


def correct_over_marker(drone: Pioneer, camera: Camera, marker_id: int, target_z: float):
    for attempt in range(MAX_CORRECTION_TRIES):
        frame = camera.get_cv_frame()
        if frame is None or frame.size == 0:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2

        result = detect_marker(frame, marker_id)
        if result is None:
            print(f"[INFO] Marker {marker_id} not visible, correction skipped")
            return

        cx, cy, corners, ids = result
        dx_px = cx - cx0
        dy_px = cy - cy0

        alt = get_altitude(drone)
        dx_m, dy_m = pixel_to_drone_xy(cx, cy, alt)

        centered_px = abs(dx_px) <= CENTER_TOL_PX and abs(dy_px) <= CENTER_TOL_PX
        centered_m = abs(dx_m) <= CENTER_TOL_M and abs(dy_m) <= CENTER_TOL_M

        if SHOW_WINDOW:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.circle(frame, (cx0, cy0), 6, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(frame, f"id={marker_id} dx={dx_m:+.2f} dy={dy_m:+.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("marker_correction", frame)
            cv2.waitKey(1)

        if centered_px and centered_m:
            print(f"[INFO] Marker {marker_id} centered")
            return

        drone_x, drone_y = get_xy(drone)
        target_x = drone_x + dx_m * CORRECTION_GAIN
        target_y = drone_y + dy_m * CORRECTION_GAIN

        print(f"[INFO] Correction {attempt + 1}: dx={dx_m:+.2f}, dy={dy_m:+.2f}")
        drone.go_to_local_point(x=target_x, y=target_y, z=target_z, yaw=YAW)
        wait_until_reached(drone)

    print(f"[WARN] Marker {marker_id} was not centered exactly")


def go_to_marker(drone: Pioneer, camera: Camera, marker_id: int):
    x, y, z, hold_time, use_correction = MARKERS[marker_id]

    print(f"[INFO] Fly to marker {marker_id}: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    drone.go_to_local_point(x=x, y=y, z=z, yaw=YAW)
    wait_until_reached(drone)

    if use_correction:
        print(f"[INFO] Correction enabled for marker {marker_id}")
        correct_over_marker(drone, camera, marker_id, z)

    print(f"[INFO] Hold {hold_time:.1f} sec at marker {marker_id}")
    time.sleep(hold_time)


if __name__ == "__main__":
    drone = Pioneer()
    camera = Camera()

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

            x, y, z, hold_time, use_correction = MARKERS[marker_id]

            if DRY_RUN:
                print(
                    f"[DRY] go_to_local_point("
                    f"x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={YAW:.2f})"
                )
                if use_correction:
                    print(f"[DRY] correction enabled for marker {marker_id}")
                print(f"[DRY] hold {hold_time:.1f} sec")
                time.sleep(1)
            else:
                go_to_marker(drone, camera, marker_id)

        print("[INFO] Mission completed")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")

    finally:
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        if not DRY_RUN:
            print("[INFO] Land")
            drone.land()

        drone.close_connection()