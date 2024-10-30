import cv2
import numpy as np 

cam_left = cv2.VideoCapture(1)  # Cámara izquierda
cam_right = cv2.VideoCapture(2)  # Cámara derecha

if not (cam_left.isOpened() and cam_right.isOpened()):
    print("No se pudo abrir una o ambas cámaras.")
    exit()

while True:
    ret_left, frame_left = cam_left.read()
    ret_right, frame_right = cam_right.read()

    if not ret_left or not ret_right:
        print("Error capturando imágenes.")
        break

    # Muestra ambas imágenes
    cv2.imshow("Cámara Izquierda", frame_left)
    cv2.imshow("Cámara Derecha", frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_left.release()
cam_right.release()
cv2.destroyAllWindows()