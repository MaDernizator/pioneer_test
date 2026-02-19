from pioneer_sdk import Pioneer 
import time  

# Создаем объект mini кдасса Pioneer
mini = Pioneer()

# Бесконечный цикл для постоянного опроса позиции дрона
while True:
    # Получаем текущие координаты дрона (метод возвращает координаты в виде списка)
    array_of_coordinates = mini.get_local_position_lps()

    # Проверяем, что координаты были получены (не пустой список или None)
    if array_of_coordinates:
        # Выводим координаты x, y, z в консоль
        print(f'x={array_of_coordinates[0]} , y={array_of_coordinates[1]}, z={array_of_coordinates[2]}')

        # Пример вывода координаты x отдельно 
        # print(f'x={array_of_coordinates[0]}')

    # Задержка между итерациями
    time.sleep(2)
