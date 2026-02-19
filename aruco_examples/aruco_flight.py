import cv2
from pioneer_sdk import Pioneer, Camera
import numpy as np
import threading
import time
from collections import deque


def load_coefficients(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("mtx").mat()
    dist_coeffs = cv_file.getNode("dist").mat()
    cv_file.release()
    return camera_matrix, dist_coeffs


class VideoProcessingThread(threading.Thread):
    def __init__(self, camera_matrix, dist_coeffs):
        super().__init__()
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.running = True
        self.latest_coordinates = None
        self.x_center = None
        self.frame = None

        self.frame_times = deque(maxlen=30)
        self.last_fps_log_time = time.time()

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.size_of_marker = 0.1
        self.points_of_marker = np.array([
            (self.size_of_marker / 2, -self.size_of_marker / 2, 0),
            (-self.size_of_marker / 2, -self.size_of_marker / 2, 0),
            (-self.size_of_marker / 2, self.size_of_marker / 2, 0),
            (self.size_of_marker / 2, self.size_of_marker / 2, 0),
        ])

        self.camera = Camera()

    def run(self):
        while self.running:
            try:
                frame = self.camera.get_cv_frame()
                corners, ids, _ = self.aruco_detector.detectMarkers(frame)

                if ids is not None and len(corners) > 0:
                    x_center = int(sum([p[0] for p in corners[0][0]]) / 4)
                    y_center = int(sum([p[1] for p in corners[0][0]]) / 4)
                    dot_size = 5
                    frame[y_center - dot_size:y_center + dot_size, x_center - dot_size:x_center + dot_size] = [0, 0, 255]
                    cv2.aruco.drawDetectedMarkers(frame, corners)

                    success, rvecs, tvecs = cv2.solvePnP(
                        self.points_of_marker, corners[0], self.camera_matrix, self.dist_coeffs
                    )
                    if success:
                        self.latest_coordinates = [tvecs.item(0), tvecs.item(1), tvecs.item(2)]
                        self.x_center = x_center
                    else:
                        self.latest_coordinates = None
                        self.x_center = None
                else:
                    self.latest_coordinates = None
                    self.x_center = None

                self.frame = frame.copy()

                now = time.time()
                self.frame_times.append(now)
                if now - self.last_fps_log_time > 2.0 and len(self.frame_times) > 1:
                    fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
                    print(f"[Video Thread] FPS: {fps:.2f}")
                    self.last_fps_log_time = now

            except cv2.error as e:
                print(f"[Video Thread] OpenCV error: {e}")
                continue

    def stop(self):
        self.running = False


if __name__ == "__main__":
    camera_matrix, dist_coeffs = load_coefficients("data.yml")
    video_thread = VideoProcessingThread(camera_matrix, dist_coeffs)
    video_thread.start()

    mini = Pioneer()
    send_manual_speed = False
    airborne = False

    try:
        while True:
            coordinates = video_thread.latest_coordinates
            x_center = video_thread.x_center
            frame = video_thread.frame

            v_y = 0
            yaw_rate = 0

            if airborne and coordinates is not None and x_center is not None:
                distance = np.linalg.norm(coordinates)
                print(f"Distance: {distance:.2f} m")

                if distance > 1.5:
                    v_y = 0.4
                elif distance < 1:
                    v_y = -0.4

                if x_center < frame.shape[1] / 3:
                    yaw_rate = -0.4
                elif x_center > frame.shape[1] * 2 / 3:
                    yaw_rate = 0.4

            # Если маркер виден и есть команды движения
            if airborne and (v_y != 0 or yaw_rate != 0):
                mini.set_manual_speed_body_fixed(vx=0, vy=v_y, vz=0, yaw_rate=yaw_rate)
                send_manual_speed = True

            # Если маркер потерян, и до этого коптер двигался — остановка
            elif airborne and send_manual_speed:
                print("[INFO] Marker lost. Stopping drone.")
                mini.go_to_local_point_body_fixed(x=0, y=0, z=0, yaw=0)
                send_manual_speed = False

            if frame is not None:
                cv2.imshow("marker_detection", frame)

            key = cv2.waitKey(1)
            if key == 27:  # Esc
                break
            elif key == 32 and not airborne:  # Space
                print("[INFO] Takeoff initiated.")
                mini.arm()
                mini.takeoff()
                mini.go_to_local_point(x=0, y=0, z=1.5, yaw=0)
                while not mini.point_reached():
                    time.sleep(0.1)
                airborne = True
                print("[INFO] Drone reached hover point.")

    finally:
        print("Landing...")
        video_thread.stop()
        video_thread.join()
        cv2.destroyAllWindows()
        if airborne:
            mini.land()
        mini.close_connection()
        del mini
