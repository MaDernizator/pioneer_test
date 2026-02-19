import cv2
from pioneer_sdk import Pioneer
from pioneer_sdk import Camera


def main():
    cam = Camera()
    pioneer = Pioneer()

    while True:
        frame = cam.get_cv_frame()
        if frame is not None:
            cv2.imshow('Pioneer', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
