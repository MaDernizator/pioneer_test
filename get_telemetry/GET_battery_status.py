from pioneer_sdk import Pioneer  
import time  

# Создаем объект mini класса Pioneer
mini = Pioneer()

# Бесконечный цикл для постоянного опроса состояния батареи
while True:
    # Получаем текущий вольтаж батареи
    battery_voltage = mini.get_battery_status()

    # Проверяем, получены ли данные (не None)
    if battery_voltage is not None:
        # Выводим вольтаж батареи в консоль без округления
        print("Напряжение батареи:", battery_voltage, "В")

        # Менее 3.6 В – красный (низкий заряд)
        if battery_voltage < 3.6:
            mini.led_control(led_id=255, r=0.1, g=0, b=0)  # Красный цвет
        # От 3.6 В до 4.0 В – желтый (средний заряд)
        elif battery_voltage < 4.0:
            mini.led_control(led_id=255, r=0.1, g=0.1, b=0)  # Желтый цвет (красный + зеленый)
        # От 4.0 В до 4.2 В – зеленый (хороший заряд)
        elif battery_voltage <= 4.2:
            mini.led_control(led_id=255, r=0, g=0.1, b=0)  # Зеленый цвет

    # Задержка для уменьшения частоты запросов
    time.sleep(2)
