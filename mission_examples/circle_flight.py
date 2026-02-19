from pioneer_sdk import Pioneer, Camera
import cv2
import math
import numpy as np

# Настройки полёта
flight_height = 1.0           # Высота в метрах
circle_radius = 0.6           # Радиус окружности
number_of_points = 10         # Кол-во точек на окружности
angle_step = 360 / number_of_points

# Начальные параметры
current_angle = 0
first_point = True
mission_complete = False

# Функция расчёта координат на окружности
def get_point_on_circle(angle_degrees, radius):
    angle_radians = math.radians(angle_degrees)
    x = radius * math.cos(angle_radians)
    y = radius * math.sin(angle_radians)
    return x, y

# Функция вывода видео с камеры дрона
def show_camera_frame():
        frame = camera.get_cv_frame()
        if frame is not None:
            cv2.imshow("pioneer_camera", frame)
            cv2.waitKey(1)

if __name__ == "__main__":
    print("Старт полёта по окружности")

    # Инициализация дрона и камеры
    drone = Pioneer()
    camera = Camera()

    try:
        drone.arm()
        drone.takeoff()

        while True:
            show_camera_frame()

            # Полет к следующей точке
            if drone.point_reached() or first_point:
                first_point = False

                if current_angle >= 360:
                    mission_complete = True
                else:
                    x, y = get_point_on_circle(current_angle, circle_radius)
                    yaw = math.radians(current_angle)

                    drone.go_to_local_point(x=x, y=y, z=flight_height, yaw=yaw)
                    current_angle += angle_step

            # Завершение миссии при нажатии ESC (27)
            if cv2.waitKey(1) == 27 or mission_complete:
                print("Миссия завершена или нажата клавиша ESC")
                drone.land()
                break

    except KeyboardInterrupt:
        print("Принудительное завершение программы")
        drone.land()

    finally:
        # Всегда выполняется при завершении
        print("Посадка дрона и завершение программы")
        cv2.destroyAllWindows()
        drone.land()
        drone.close_connection()
