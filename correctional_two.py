import time
import cv2

from pioneer_sdk import Pioneer, Camera
from pixel_projector import pixel_to_drone_xy
from marker_map import MARKER_COORDS

HEIGHT = 1.8

DRY_RUN = False
YAW = 0.0
MUST_EXIST = True

CENTER_TOL_PX = 40
CENTER_TOL_M = 0.15
CORRECTION_GAIN = 0.9
MAX_CORRECTION_TRIES = 8
MARKER_CHECK_TRIES = 20
MARKER_CHECK_DT = 0.1

SHOW_WINDOW = True

# Маршрут:
# (marker_id, marker_index, z, hold_time, use_correction)
#
# marker_index = какой по счёту маркер с таким id брать:
#   0 -> первый
#   1 -> второй
#   2 -> третий
ROUTE = [
    (27, 0, HEIGHT, 0.0, False),
    (7, 0, HEIGHT, 5.0, False),
    (12, 0, HEIGHT, 0.0, False),
    (25, 0, HEIGHT, 5.0, False),
    (11, 0, HEIGHT, 0.0, False),
    (17, 0, HEIGHT, 0, False),
    (15, 0, HEIGHT, 0, False),
    (15, 1, HEIGHT - 1, 5, False),
    (6, 0, HEIGHT - 1, 0, False),
    (6, 1, HEIGHT, 0, False),
]

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


def get_marker_coords(marker_id: int, marker_index: int):
    found = [m for m in MARKER_COORDS if m["marker_id"] == marker_id]

    if marker_index < 0 or marker_index >= len(found):
        raise ValueError(
            f"Marker ({marker_id}, index={marker_index}) not found. "
            f"Available count: {len(found)}"
        )

    marker = found[marker_index]
    return float(marker["x"]), float(marker["y"])


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


def wait_marker_visible(camera: Camera, marker_id: int):
    for _ in range(MARKER_CHECK_TRIES):
        frame = camera.get_cv_frame()
        if frame is None or frame.size == 0:
            time.sleep(MARKER_CHECK_DT)
            continue

        if detect_marker(frame, marker_id) is not None:
            return True

        time.sleep(MARKER_CHECK_DT)

    return False


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
            return False

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
            cv2.putText(
                frame,
                f"id={marker_id} dx={dx_m:+.2f} dy={dy_m:+.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.imshow("marker_correction", frame)
            cv2.waitKey(1)

        if centered_px and centered_m:
            print(f"[INFO] Marker {marker_id} centered")
            return True

        drone_x, drone_y = get_xy(drone)
        target_x = drone_x + dx_m * CORRECTION_GAIN
        target_y = drone_y + dy_m * CORRECTION_GAIN

        print(f"[INFO] Correction {attempt + 1}: dx={dx_m:+.2f}, dy={dy_m:+.2f}")
        drone.go_to_local_point(x=target_x, y=target_y, z=target_z, yaw=YAW)
        wait_until_reached(drone)

    print(f"[WARN] Marker {marker_id} was not centered exactly")
    return False


def go_to_marker(drone: Pioneer, camera: Camera, marker_id: int, marker_index: int,
                 z: float, hold_time: float, use_correction: bool, coord_shift):
    base_x, base_y = get_marker_coords(marker_id, marker_index)

    shift_x, shift_y = coord_shift
    target_x = base_x + shift_x
    target_y = base_y + shift_y
    target_z = z

    print(
        f"[INFO] Fly to marker {marker_id}[{marker_index}]: "
        f"x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}"
    )
    drone.go_to_local_point(x=target_x, y=target_y, z=target_z, yaw=YAW)
    wait_until_reached(drone)

    if MUST_EXIST:
        print(f"[INFO] Check marker {marker_id} presence")
        if not wait_marker_visible(camera, marker_id):
            raise RuntimeError(f"Marker {marker_id} not found. Flight stopped.")

    if use_correction:
        print(f"[INFO] Correction enabled for marker {marker_id}")
        centered = correct_over_marker(drone, camera, marker_id, target_z)

        if centered:
            real_x, real_y = get_xy(drone)

            error_x = real_x - target_x
            error_y = real_y - target_y

            shift_x += error_x
            shift_y += error_y

            print(
                f"[INFO] Marker {marker_id}[{marker_index}] "
                f"real=({real_x:+.2f}, {real_y:+.2f}) "
                f"planned=({target_x:+.2f}, {target_y:+.2f})"
            )
            print(f"[INFO] New global shift: dx={shift_x:+.2f}, dy={shift_y:+.2f}")

    print(f"[INFO] Hold {hold_time:.1f} sec at marker {marker_id}[{marker_index}]")
    time.sleep(hold_time)

    return shift_x, shift_y


if __name__ == "__main__":
    drone = Pioneer()
    camera = Camera()

    coord_shift = (0.0, 0.0)

    try:
        if not DRY_RUN:
            print("[INFO] Arm")
            drone.arm()

            print("[INFO] Takeoff")
            drone.takeoff()
            time.sleep(3)
        else:
            print("[INFO] DRY_RUN mode")

        for marker_id, marker_index, z, hold_time, use_correction in ROUTE:
            if DRY_RUN:
                base_x, base_y = get_marker_coords(marker_id, marker_index)
                sx, sy = coord_shift

                print(
                    f"[DRY] go_to_local_point("
                    f"x={base_x + sx:.2f}, y={base_y + sy:.2f}, z={z:.2f}, yaw={YAW:.2f})"
                )
                print(f"[DRY] marker={marker_id}[{marker_index}]")
                print(f"[DRY] use_correction={use_correction}, must_exist={MUST_EXIST}")
                print(f"[DRY] hold {hold_time:.1f} sec")
                time.sleep(1)
            else:
                coord_shift = go_to_marker(
                    drone=drone,
                    camera=camera,
                    marker_id=marker_id,
                    marker_index=marker_index,
                    z=z,
                    hold_time=hold_time,
                    use_correction=use_correction,
                    coord_shift=coord_shift,
                )

        print("[INFO] Mission completed")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        if not DRY_RUN:
            print("[INFO] Land")
            drone.land()

        drone.close_connection()
