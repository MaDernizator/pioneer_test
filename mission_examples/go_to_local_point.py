from pioneer_sdk import Pioneer
import time

# Инициализация дрона
pioneer = Pioneer()

try:
    # Взлет на 1 метр
    pioneer.arm()
    pioneer.takeoff()
    pioneer.go_to_local_point(x=0, y=0, z=1, yaw=0)
    while not pioneer.point_reached():
        time.sleep(0.1)

    # Полет в первую точку
    pioneer.go_to_local_point(x=0, y=1, z=1, yaw=0)
    while not pioneer.point_reached():
        time.sleep(0.1)

    # Полет во вторую точку
    pioneer.go_to_local_point(x=1, y=1, z=1, yaw=0)
    while not pioneer.point_reached():
        time.sleep(0.1)

    # Посадка
    pioneer.land()

except KeyboardInterrupt:
    print("Остановка программы, производится посадка")
    pioneer.land()
    
