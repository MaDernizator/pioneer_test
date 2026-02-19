from pioneer_sdk import VideoStream

if __name__ == "__main__":  
    stream = VideoStream()      # Создаем объект потока видео

    while True:                 # Бесконечный цикл, ожидающий команд
        cmd = input()           # Ожидаем ввода команды от пользователя в консоль

        if cmd == "start":      # Если введена команда 'start'
            stream.start()      # Запускаем видеопоток
        
        elif cmd == "stop":     # Если введена команда 'stop'
            stream.stop()       # Останавливаем видеопоток
