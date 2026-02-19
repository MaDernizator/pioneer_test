__author__ = "Peter Bozhko"
__version__ = "1.0.0"
__maintainer__ = "Peter Bozhko"
__email__ = "p.bozhko@geoscan.ru"

import time
import threading
from pioneer_sdk import Pioneer

# IP-адреса дронов
mini_ip = ["192.168.137.62", "192.168.137.245"]

# Словари для хранения объектов
mini = {}  # объекты Pioneer
threads = {}  # объекты потоков для каждого дрона
point_reached = {}  # флаг долёта до точки от каждого дрона
tasks = {}  # задание (точка) для каждого дрона
event = threading.Event()  # Ивент для синхронизации полёта
event_finish = threading.Event()  # Ивент окончания задания

logger = False  # логгирования действий

# Координаты точек маршрута

coords = [[0, 1, 1],
          [0, 0, 1],
          [0, 1, 1]]


#  Функция для логгирования сообщений в консоль
def log(msg):
    if not logger:
        return
    print(f"Logger: {msg}")


# Функция ожидания соединения со всеми дронами
def wait_connection():
    while True:
        ret = True
        for ip in mini_ip:
            if not mini[ip].connected():
                ret = False
        if ret:
            return
        time.sleep(0.5)


#  Функция ожидания долёта всех коптеров до точки
def wait_task():
    global point_reached
    while True:
        count = 0
        for ip in mini_ip:
            if point_reached[ip]:
                count += 1
        log(f"CHECK POINT {count} FROM {len(point_reached)}")
        if count == len(point_reached):
            return
        time.sleep(0.3)


# Функция для перемещения дронов в начальную точку
def go_to_start_point(drone_ip):
    point_reached[drone_ip] = False  # Выставляем флаг долёта до точки в False
    event.wait()  # Ожидаение команды на старт
    log(f"[{drone_ip}] ARM")
    mini[drone_ip].arm()
    log(f"[{drone_ip}] TAKE_OFF")
    mini[drone_ip].takeoff()
    log(f"[{drone_ip}] GO_TO_START_POINT")
    point = tasks[drone_ip]  # Берём точку из словаря tasks
    log(f"[{drone_ip}] GO_TO {point}")

    mini[drone_ip].go_to_local_point(x=point[0], y=point[1], z=point[2], yaw=0)
    while not mini[drone_ip].point_reached():
        pass
    point_reached[drone_ip] = True  # После прилёта в точку отмечаемся в словаре point_reached
    log(f"[{drone_ip}] POINT_REACHED ")
    run_mission_flight(drone_ip)  # Запуск основного задания


# Функция для выполнения полетного задания по заданным точкам
def run_mission_flight(drone_ip):
    while True:
        event.wait()  # Ожидания разрешения на полёт
        point_reached[drone_ip] = False  # Выставляем флаг долёта до точки в False
        if event_finish.is_set():  # Проверка флага конца миссии
            break
        point = tasks[drone_ip]  # Получаем следующую точку
        log(f"[{drone_ip}] GO_TO {point}")

        mini[drone_ip].go_to_local_point(x=point[0], y=point[1], z=point[2], yaw=0)
        while not mini[drone_ip].point_reached():
            pass
        point_reached[drone_ip] = True  # После прилёта в точку отмечаемся в словаре point_reached
    mini[drone_ip].land()  # После окончания миссии выполняется посадка


# Функция для экстренной посадки всех дронов
def all_land():
    for ip in mini_ip:
        mini[ip].land()
    return


#  Основная часть программы
def program():
    for ip in mini_ip:  # Создание отдельных потоков выполнения для каждого дрона
        print("Thread " + ip)
        threads[ip] = threading.Thread(target=go_to_start_point, args=[ip])
    for thread in threads.values():  # Запуск созданных потоков
        thread.start()
    input("ARM? ")  # Ожидание команды на взлёт от пользователя
    for ip in mini_ip:
        tasks[ip] = [0, 0, 1]  # Установка стартовой позиции для каждого коптера
    event.set()
    event.clear()
    wait_task()
    input("START MISSION? ")  # Ожидание команды на старт миссии от пользователя
    for coord in coords:
        for ip in mini_ip:  # Установка следующей точки для каждого коптера
            tasks[ip] = coord
        event.set()  # Флаг разрешения на полёт (синхронизация)
        event.clear()
        time.sleep(0.3)
        wait_task()  # Ожидание прилёта всех дронов
    event_finish.set()  # Флаг окончания полёта
    event.set()


if __name__ == '__main__':
    # # Инициализация объектов Pioneer для каждого дрона
    for ip in mini_ip:
        mini[ip] = Pioneer(ip=ip, name=ip, logger=False)
    wait_connection()  # Ожидание соединения со всеми дронами
    try:
        program()  # Запуск программы
    except BaseException:  # В случае ошибки или остановки пользователем посадка дронов
        all_land()
