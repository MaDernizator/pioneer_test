from pioneer_sdk import Pioneer  
import time  

# Определяем временной интервал обновления данных
delta_time = 0.1  

# Инициализируем переменные для цвета светодиодов (RGB)
r = 0
g = 0
b = 0

# Определяем пороговые значения высоты для смены цвета
height_1 = 0.25  
height_2 = 0.5   
height_3 = 0.75  

#  <------- height_1 ------- height_2 ------- height_3 ------->  
#   красный          зеленый           синий           белый  

if __name__ == "__main__":                     # Проверяем, что скрипт запущен как основной файл
    pioneer_mini = Pioneer(logger=False)       # Создаем объект управления дроном, отключая логирование
    curr_time = time.time()                    # Фиксируем текущее время

    try:
        while True:                                             # Запускаем бесконечный цикл
            if time.time() - curr_time > delta_time:            # Проверяем, прошло ли заданное время (delta_time)
                tof_data = pioneer_mini.get_dist_sensor_data()  # Получаем данные с датчика расстояния
                print(tof_data)                                 # Выводим данные в консоль для мониторинга
                
                if tof_data is not None:                        # Проверяем, что данные успешно получены с датчика
                    
                    # Определяем цвет светодиодов в зависимости от высоты
                    if tof_data <= height_1:
                        r = 0.1  # Красный
                        g = 0
                        b = 0
                    elif height_1 < tof_data <= height_2:
                        r = 0
                        g = 0.1  # Зеленый
                        b = 0
                    elif height_2 < tof_data <= height_3:
                        r = 0
                        g = 0
                        b = 0.1  # Синий
                    elif tof_data >= height_3:
                        r = 0.1
                        g = 0.1
                        b = 0.1  # Белый
                    
                    # Управляем светодиодами дрона
                    pioneer_mini.led_control(led_id=255, r=r, g=g, b=b)
                    
                    # Обновляем время последнего обновления
                    curr_time = time.time()
    
    except KeyboardInterrupt:
        print("Остановка программы. Выключаем светодиоды...")
        pioneer_mini.led_control(led_id=255, r=0, g=0, b=0)  # Выключаем светодиоды

    finally:
        pioneer_mini.close_connection()                      # Закрываем соединение с дроном
