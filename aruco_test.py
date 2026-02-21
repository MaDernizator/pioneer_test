import cv2
from pioneer_sdk import Pioneer, Camera
import numpy as np
import threading
import time


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
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.camera = Camera()

    def run(self):
        while self.running:
            frame = self.camera.get_cv_frame()
            corners, ids, _ = self.aruco_detector.detectMarkers(frame)
            if ids is not None and len(corners) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners)

            cv2.imshow('Marker Detection', frame)
            cv2.waitKey(1)

    def stop(self):
        self.running = False


if __name__ == "__main__":
    camera_matrix, dist_coeffs = load_coefficients("data.yml")
    video_thread = VideoProcessingThread(camera_matrix, dist_coeffs)
    video_thread.start()

    mini = Pioneer()

    try:
        # Взлет на 1 метр вверх
        mini.arm()
        mini.takeoff()
        mini.go_to_local_point(x=0, y=0, z=1, yaw=0)
        while not mini.point_reached():
            time.sleep(0.1)

        # Первый полет на 1 метр вперед
        mini.go_to_local_point(x=0, y=0, z=1, yaw=0)
        while not mini.point_reached():
            time.sleep(0.1)
        time.sleep(3)  # Ждем 3 секунды

        # Второй полет на 1 метр вперед
        mini.go_to_local_point(x=0, y=0, z=2, yaw=0)
        while not mini.point_reached():
            time.sleep(0.1)

        # Посадка
        mini.land()

    except KeyboardInterrupt:
        print("Прерывание, приземляем дрон!")
        mini.land()

    finally:
        print("Завершаем работу...")
        video_thread.stop()
        video_thread.join()
        cv2.destroyAllWindows()
        mini.close_connection()
        del mini