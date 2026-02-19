from pioneer_sdk import Pioneer  
import time  

if __name__ == "__main__":
    print("start")                    # Выводим сообщение о начале выполнения программы
    pioneer_mini = Pioneer()          # Создаем объект для управления дроном

    try:
        pioneer_mini.arm()            # Включаем моторы дрона
        pioneer_mini.takeoff()        # Взлет
        time.sleep(1)                 # Ждем 1 секунду после взлета

        # Дрон летит вперед (по оси Y) в течение 2 секунд со скоростью 1 м/с
        t = time.time()
        while True:                                                      # Отправка команды set_manual_speed всегда осуществляется в бесконечном цикле
            pioneer_mini.set_manual_speed(vx=0, vy=1, vz=0, yaw_rate=0)  # Устанавливаем скорость движения по оси y
            time.sleep(0.05)                                             # Небольшая задержка
            if time.time() - t > 2:                                      # Проверяем, прошло ли 2 секунды
                break                                                    # Если прошло - выход из цикла

        time.sleep(2)                                                    # Ждем 2 секунды перед следующим движением

        # Дрон летит вправо (по оси X) в течение 2 секунд со скоростью 1 м/с
        t = time.time()                                                  # Запоминаем текущее время
        while True:
            pioneer_mini.set_manual_speed(vx=1, vy=0, vz=0, yaw_rate=0)  # Устанавливаем скорость движения по оси x
            time.sleep(0.05)
            if time.time() - t > 2:
                break

        time.sleep(2)                                                    # Ждем 2 секунды перед посадкой

        pioneer_mini.land()                                              # Осуществляем посадку

    except KeyboardInterrupt:
        print("Остановка программы, производится посадка")
        pioneer_mini.land()                                              # При прерывании программы дрон автоматически садится

    finally:
        pioneer_mini.close_connection()                                  # Закрываем соединение с дроном
        del pioneer_mini                                                 # Удаляем объект дрона
