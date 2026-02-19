from pioneer_sdk import Pioneer
import time

# Инициализация дрона
pioneer = Pioneer(ip='192.168.137.4', mavlink_port=5656) 

try:
    # Взлет на 1 метр
    pioneer.arm()
    pioneer.takeoff()
    pioneer.go_to_local_point(x=0, y=0, z=1, yaw=0)
    while not pioneer.point_reached():
        time.sleep(0.1)

    # Посадка
    pioneer.land()
    time.sleep(3)
    pioneer.disarm()

except KeyboardInterrupt:
    print("Остановка программы, производится посадка")
    pioneer.land()
    time.sleep(3)
    pioneer.disarm()
