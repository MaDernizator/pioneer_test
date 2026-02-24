import cv2
import numpy as np

# ---------------- Настройки проекции (под твою установку) ----------------
# камера смещена на 7.5 см вперёд от центра дрона
CAM_OFFSET_X = 0.0
CAM_OFFSET_Y = 0.075

# знак: +1 если "вниз по кадру" = "вперёд", -1 если "вниз по кадру" = "назад"
# у тебя раньше чаще получалось, что нужно -1
IMG_Y_TO_FORWARD = -1.0

# путь к калибровке
CALIB_YML_PATH = "data.yml"

# ---------------- Внутреннее: загрузка калибровки 1 раз ----------------
_MTX = None
_DIST = None


def _load_calibration_once():
    global _MTX, _DIST
    if _MTX is not None and _DIST is not None:
        return

    fs = cv2.FileStorage(CALIB_YML_PATH, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {CALIB_YML_PATH}")

    mtx_node = fs.getNode("mtx")
    dist_node = fs.getNode("dist")
    if mtx_node.empty() or dist_node.empty():
        fs.release()
        raise ValueError(f"'{CALIB_YML_PATH}' must contain nodes 'mtx' and 'dist'")

    _MTX = mtx_node.mat()
    _DIST = dist_node.mat()
    fs.release()

    if _MTX is None or _DIST is None:
        raise ValueError(f"Failed to read mtx/dist from '{CALIB_YML_PATH}'")

    _MTX = np.asarray(_MTX, dtype=np.float64)
    _DIST = np.asarray(_DIST, dtype=np.float64)


# Грузим калибровку сразу при импорте модуля (один раз)
_load_calibration_once()


def pixel_to_drone_xy(u: float, v: float, drone_alt_m: float) -> tuple[float, float]:
    """
    ЕДИНАЯ согласованная функция для всех программ проекта.

    Возвращает (dx, dy) в метрах — положение точки на земле под пикселем (u,v)
    в СК ДРОНА (body-fixed):
      dx > 0 => справа от центра дрона
      dy > 0 => впереди центра дрона

    Важно:
    - камера смотрит строго вниз
    - высота drone_alt_m — текущая высота камеры над землёй, метры
    - учитывается смещение камеры CAM_OFFSET_*
    """
    alt = float(drone_alt_m)
    if alt <= 0.01:
        return 0.0, 0.0

    # 1) undistort -> нормализованные координаты (xn, yn)
    pts = np.array([[[float(u), float(v)]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, _MTX, _DIST)  # (1,1,2)
    xn = float(und[0, 0, 0])
    yn = float(und[0, 0, 1])

    # 2) пересечение луча с землей на расстоянии alt
    dx_cam = xn * alt          # вправо
    dy_img = yn * alt          # вниз по изображению

    # 3) вниз по изображению -> вперед по дрону (знак)
    dy_forward = IMG_Y_TO_FORWARD * dy_img

    # 4) учёт смещения камеры относительно центра дрона
    dx_drone = dx_cam + CAM_OFFSET_X
    dy_drone = dy_forward + CAM_OFFSET_Y

    return dx_drone, dy_drone