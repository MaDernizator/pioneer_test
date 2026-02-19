from pioneer_sdk import Pioneer

if __name__ == "__main__":    
    pioneer_mini = Pioneer()  # Создаем объект управления дроном

    while True:           # Запускаем бесконечный цикл для обработки команд пользователя
        cmd = input()     # Считываем ввод пользователя
        
        if cmd == "start":
            pioneer_mini.lua_script_control("Start")  # Отправляем команду на запуск Lua-скрипта на дроне
        
        elif cmd == "stop":
            pioneer_mini.lua_script_control("Stop")   # Отправляем команду на остановку Lua-скрипта
