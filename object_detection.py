import cv2
import numpy as np
import mss

# Configurar la captura de pantalla
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1000}  # Ajusta el tamaño según tu pantalla
sct = mss.mss()

# Obtener la posición y tamaño de la ventana
window_name = "YOLOv8 Inference"
window_handle = cv2.namedWindow(window_name)
window_x, window_y, window_w, window_h = cv2.getWindowImageRect(window_name)

# Ajustar la región de captura para excluir la ventana
monitor["top"] = window_y + window_h
monitor["height"] = 1000 - window_h

while True:
    # Capturar la pantalla
    img = sct.grab(monitor)
    frame = np.array(img)

    # Convertir de BGRA a BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Mostrar el frame
    cv2.imshow(window_name, frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cv2.destroyAllWindows()