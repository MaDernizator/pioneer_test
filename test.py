from pioneer_sdk import Pioneer, Camera
import cv2
import time

if __name__ == "__main__":
    print(
        """
    1 -- arm (завести моторы)
    2 -- disarm (остановить моторы)
    ESC -- выход
        """
    )

    pioneer_mini = Pioneer()
    camera = Camera()

    try:
        while True:
            frame = camera.get_cv_frame()
            if frame is not None:
                cv2.imshow("pioneer_camera_stream", frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                print("esc pressed")
                cv2.destroyAllWindows()
                pioneer_mini.disarm()
                break
            elif key == ord("1"):
                pioneer_mini.arm()
            elif key == ord("2"):
                pioneer_mini.disarm()

            time.sleep(0.02)

    finally:
        time.sleep(1)
        pioneer_mini.disarm()
        pioneer_mini.close_connection()
        del pioneer_mini
