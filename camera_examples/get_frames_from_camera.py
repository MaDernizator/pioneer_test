import cv2
import numpy as np
from pioneer_sdk import Camera

# Создаем объект управления камерой
camera = Camera()

if __name__ == "__main__":
    while True:
        frame = camera.get_cv_frame()   # Получаем изображение с камеры
        if frame is not None:           # Проверяем, что изображение получено
            cv2.imshow("video", frame)  # Отображаем кадр в окне с названием "video"

        if cv2.waitKey(1) == 27:  # Выход из цикла при нажатии клавиши ESC (код 27)
            break

    cv2.destroyAllWindows()       # Закрываем все открытые окна OpenCV
