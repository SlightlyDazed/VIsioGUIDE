import cv2

# Abrir ambas cámaras
cap1 = cv2.VideoCapture(0)  # Cámara 1
cap2 = cv2.VideoCapture(2)  # Cámara 2

num1 = 0
num2 = 0

while cap1.isOpened() and cap2.isOpened():
    # Leer de ambas cámaras
    succes1, img1 = cap1.read()
    succes2, img2 = cap2.read()

    if not succes1 or not succes2:
        print("Error al acceder a las cámaras")
        break

    k = cv2.waitKey(5)

    if k == ord('q'):
        break
    elif k == ord('s'):  # Guardar imagen de ambas cámaras cuando se presiona 's'
        # Guardar imagen de la cámara 1
        cv2.imwrite('C:/Users/Alex/Documents/BehindBlueEyes/img_camera1_' + str(num1) + '.png', img1)
        print("Imagen de la cámara 1 guardada!")
        num1 += 1
        
        # Guardar imagen de la cámara 2
        cv2.imwrite('C:/Users/Alex/Documents/BehindBlueEyes/img_camera2_' + str(num2) + '.png', img2)
        print("Imagen de la cámara 2 guardada!")
        num2 += 1

    # Mostrar ambas imágenes
    cv2.imshow('Cámara 1', img1)
    cv2.imshow('Cámara 2', img2)

# Liberar ambas cámaras y destruir las ventanas
cap1.release()
cap2.release()
cv2.destroyAllWindows()
