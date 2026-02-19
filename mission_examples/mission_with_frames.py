from pioneer_sdk import Pioneer,Camera
import cv2
import math

# Инициализация дрона
mini = Pioneer()
camera = Camera()

# Функция перевода радианов в градусы
def degrees_to_radians(deg):
    return math.radians(deg)
    
# Функция вывода видео с камеры дрона
def show_camera():
        frame = camera.get_cv_frame()
        if frame is not None:
            cv2.imshow("pioneer_camera", frame)
            cv2.waitKey(1)

# Функция взлета на высоту 1 метр
def go_to_start_point():
    mini.arm()
    mini.takeoff()
    mini.go_to_local_point(x=0, y=0, z=1, yaw=0)  # Подъем на 1 метр
    while not mini.point_reached():
        show_camera()
        

# Функция полета по точкам
def fly_through_points(points):
    for point in points:
        mini.go_to_local_point(x=point['x'], y=point['y'], z=point['z'], yaw=degrees_to_radians(point['yaw']))
        while not mini.point_reached():
            show_camera()

# Массив точек
waypoints = [
    {'x': 1, 'y': 0, 'z': 0.7, 'yaw': 0},    # Точка 1
    {'x': 1, 'y': 1, 'z': 0.7, 'yaw': 0},    # Точка 2
    {'x': 0, 'y': 1, 'z': 0.7, 'yaw': 0},    # Точка 3
    {'x': 0, 'y': 0, 'z': 0.7, 'yaw': 0},    # Возврат к начальной точке
]

try:
    go_to_start_point()            # Взлет на 1 м
    fly_through_points(waypoints)  # Полет по точкам
    mini.land()                    # Посадка
except KeyboardInterrupt:
    print('Остановка программы, производится посадка')
    mini.land()
finally:
    mini.land()  