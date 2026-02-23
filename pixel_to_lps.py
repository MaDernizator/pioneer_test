import math
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

try:
    from pioneer_sdk import Pioneer, Camera
except Exception:
    Pioneer = None
    Camera = None


# ---------- ВСПОМОГАТЕЛЬНОЕ: авто-конверсия дистанционного датчика в метры ----------
def dist_to_meters(dist_raw: float) -> float:
    """
    get_dist_sensor_data() в разных сборках может отдавать:
      - метры (например 1.23)
      - сантиметры (например 123)
      - миллиметры (например 1230)
    Без уточнения от прошивки — делаем разумную авто-эвристику.
    """
    if dist_raw is None:
        return float("nan")

    d = float(dist_raw)
    return d



# ---------- ГОТОВАЯ ФУНКЦИЯ: пиксель -> смещение маркера относительно ЦЕНТРА ДРОНА ----------
def pixel_to_drone_xy_mtx(
    u: float,
    v: float,
    drone_alt_m: float,       # высота камеры над землей (м)
    mtx: np.ndarray,
    dist: np.ndarray,
    cam_offset_x: float = 0.0,      # м, камера смещена вправо (+X)
    cam_offset_y: float = 0.075,    # м, камера смещена вперед (+Y)
    img_y_to_forward: float = -1.0  # +1: вниз кадра = вперед, -1: вниз кадра = назад
) -> tuple[float, float]:
    """
    Возвращает (dx, dy) в МЕТРАХ — положение точки на земле (под пикселем u,v)
    в системе координат ДРОНА (body-fixed):
      dx > 0  => маркер справа от центра дрона
      dy > 0  => маркер впереди центра дрона

    Важно:
    - камера смотрит строго вниз (pitch=roll=0)
    - оси камеры согласованы с осями дрона (x вправо)
    - img_y_to_forward выбирается по факту ориентации изображения
    - cam_offset_* учитывает смещение камеры относительно центра дрона
      (из-за этого при "дрон ровно над маркером" будет dx≈0, dy≈0)
    """

    # 1) Undistort -> нормализованные координаты (xn, yn)
    pts = np.array([[[float(u), float(v)]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, mtx, dist)  # (1,1,2)
    xn = float(und[0, 0, 0])
    yn = float(und[0, 0, 1])

    # 2) Пересечение луча с землей на расстоянии alt
    # Смещение точки относительно камеры:
    dx_cam = xn * drone_alt_m              # вправо
    dy_img = yn * drone_alt_m              # вниз по изображению

    # 3) Вниз по изображению -> вперед по дрону (знак)
    dy_cam_forward = img_y_to_forward * dy_img

    # 4) Перевод из "относительно камеры" в "относительно центра дрона"
    # если маркер под центром дрона, то относительно камеры он будет примерно (-offset),
    # после добавления offset получим 0.
    dx_drone = dx_cam + cam_offset_x
    dy_drone = dy_cam_forward + cam_offset_y

    return dx_drone, dy_drone


# ---------- Детектор ArUco в отдельном потоке ----------
@dataclass
class MarkerInfo:
    cx: int
    cy: int


class ArucoTrackerThread(threading.Thread):
    def __init__(self, dict_name=cv2.aruco.DICT_4X4_50):
        super().__init__(daemon=True)
        self.running = True
        self.lock = threading.Lock()

        self.camera = Camera() if Camera is not None else None
        self.latest_frame = None
        self.markers: Dict[int, MarkerInfo] = {}

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def run(self):
        if self.camera is None:
            print("[ERROR] pioneer_sdk Camera is unavailable.")
            return

        while self.running:
            try:
                frame = self.camera.get_cv_frame()
                if frame is None or (hasattr(frame, "size") and frame.size == 0):
                    time.sleep(0.01)
                    continue

                corners, ids, _ = self.aruco_detector.detectMarkers(frame)
                markers_local: Dict[int, MarkerInfo] = {}

                if ids is not None and len(corners) > 0:
                    ids_flat = ids.flatten().tolist()
                    for i, marker_id in enumerate(ids_flat):
                        pts = corners[i][0]
                        cx = int((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4.0)
                        cy = int((pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4.0)
                        markers_local[int(marker_id)] = MarkerInfo(cx=cx, cy=cy)

                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    for mid, mi in markers_local.items():
                        cv2.circle(frame, (mi.cx, mi.cy), 5, (0, 0, 255), -1)
                        cv2.putText(frame, str(mid), (mi.cx + 8, mi.cy - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                with self.lock:
                    self.latest_frame = frame
                    self.markers = markers_local

            except cv2.error as e:
                print(f"[Video] OpenCV error (ignored): {e}")
                time.sleep(0.02)
            except Exception as e:
                print(f"[Video] Unexpected error (ignored): {e}")
                time.sleep(0.05)

    def stop(self):
        self.running = False


def choose_min_marker(markers: Dict[int, MarkerInfo]) -> Optional[int]:
    return min(markers.keys()) if markers else None


if __name__ == "__main__":
    if Pioneer is None or Camera is None:
        raise RuntimeError("pioneer_sdk недоступен (нужны Pioneer и Camera).")

    # --- mtx/dist из твоего YAML ---
    mtx = np.array([
        [454.88173547887061, 0.0, 234.11271394422656],
        [0.0, 455.74484968656606, 156.1650882167196],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    dist = np.array([[
        -0.043406693911448635,
        -1.3597118731615976,
        0.013900515070715498,
        0.00038657616343558767,
        13.741457803021884
    ]], dtype=np.float64)

    # Камера смещена на 7.5 см вперед от центра дрона
    CAM_OFFSET_X = 0.0
    CAM_OFFSET_Y = 0.075

    # По твоим прошлым тестам очень вероятно, что вниз по кадру = "назад" => -1
    IMG_Y_TO_FORWARD = -1.0  # если окажется наоборот — поставь +1.0

    # Настройки частоты опроса датчика (чтобы не грузить канал)
    ALT_HZ = 10.0
    ALT_DT = 1.0 / ALT_HZ
    next_alt_t = time.monotonic()
    alt_m = 1.0
    alt_raw = None

    # Подключение к дрону (без arm/takeoff, просто телеметрия)
    p = Pioneer()

    tracker = ArucoTrackerThread()
    tracker.start()

    print("[TEST] ESC to exit. Using get_dist_sensor_data() as current altitude.")
    try:
        while True:
            # --- обновляем высоту по датчику не чаще ALT_HZ ---
            now = time.monotonic()
            if now >= next_alt_t:
                try:
                    alt_raw = p.get_dist_sensor_data()
                    alt_m = dist_to_meters(alt_raw)
                except Exception as e:
                    # если чтение датчика временно упало — оставляем прошлую alt_m
                    print(f"[ALT] read error: {e}")
                next_alt_t = now + ALT_DT

            with tracker.lock:
                frame = tracker.latest_frame
                markers = dict(tracker.markers)

            if frame is None or (hasattr(frame, "size") and frame.size == 0):
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            cx0, cy0 = w // 2, h // 2

            # Точка центра кадра
            cv2.circle(frame, (cx0, cy0), 7, (255, 255, 0), -1)

            # Координаты маркера (относительно ЦЕНТРА ДРОНА)
            mid = choose_min_marker(markers)
            if mid is not None and not (math.isnan(alt_m) or alt_m <= 0.01):
                mi = markers[mid]
                dx_m, dy_m = pixel_to_drone_xy_mtx(
                    u=mi.cx, v=mi.cy,
                    drone_alt_m=alt_m,
                    mtx=mtx, dist=dist,
                    cam_offset_x=CAM_OFFSET_X,
                    cam_offset_y=CAM_OFFSET_Y,
                    img_y_to_forward=IMG_Y_TO_FORWARD
                )
                marker_txt = f"Marker {mid} rel(drone): dx={dx_m:+.2f}m dy={dy_m:+.2f}m"
            elif mid is not None:
                marker_txt = f"Marker {mid} rel(drone): alt invalid"
            else:
                marker_txt = "No markers"

            # Отображаем высоту и marker coords
            cv2.putText(frame, f"Frame: {w}x{h}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"Dist raw: {alt_raw}  alt_m: {alt_m:.2f} m", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"cam_off: ({CAM_OFFSET_X:+.3f},{CAM_OFFSET_Y:+.3f}) imgY2F={IMG_Y_TO_FORWARD:+.0f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, marker_txt, (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Подсказка: когда дрон “ровно над маркером”, dx/dy должны быть около 0
            cv2.putText(frame, "Goal: when drone over marker -> dx~0 dy~0",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow("pixel_to_lps_test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            time.sleep(0.005)

    finally:
        tracker.stop()
        tracker.join(timeout=2.0)
        cv2.destroyAllWindows()
        p.close_connection()
        del p