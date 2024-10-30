import os
import numpy as np
import mediapipe as mp
import tensorflow as tf
import cv2
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Directorio de las imágenes
image_dir = r"C:\Users\Alex\Documents\BehindBlueEyes\captured_data"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Cargar datos previos si existen
if os.path.exists(os.path.join(image_dir, 'captured_data.pickle')):
    with open(os.path.join(image_dir, 'captured_data.pickle'), 'rb') as f:
        saved_data = pickle.load(f)
        data = saved_data['data']
        labels = saved_data['labels']
        capture_count = len(data)  # Continuar el conteo desde donde se dejó
        print(f"Datos cargados. Se han capturado {capture_count} gestos previamente.")
else:
    data = []
    labels = []
    capture_count = 0

def capture_gesture():
    global capture_count, data, labels
    # Abrir la cámara
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir el fotograma de BGR a RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar el fotograma con MediaPipe
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar los landmarks en el fotograma original
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Mostrar la imagen con los landmarks dibujados
                cv2.imshow('Hand Gesture Capture', frame)
                
                # Esperar una tecla de entrada
                key = cv2.waitKey(10)
                if key & 0xFF == ord('c'):  # 'c' para capturar el fotograma
                    # Extraer el vector de características
                    data_aux = []
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x)
                        data_aux.append(landmark.y)
                    data_aux = np.array(data_aux)
                    
                    # Pedir la etiqueta al usuario
                    label = input("Introduce la letra o número que representa este gesto: ")
                    
                    # Guardar la imagen del gesto y los vectores
                    img_name = os.path.join(image_dir, f"{label}_{capture_count}.jpg")
                    cv2.imwrite(img_name, frame)
                    
                    # Guardar el vector y la etiqueta
                    data.append(data_aux)
                    labels.append(label)
                    
                    capture_count += 1

                    # Cerrar la cámara después de capturar el gesto
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                elif key & 0xFF == ord('q'):  # 'q' para salir
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Captura de gestos detenida.")
                    return

while True:
    capture_gesture()
    continue_capture = input("¿Deseas capturar otro gesto? (s/n): ")
    if continue_capture.lower() != 's':
        break

# Guardar los vectores y las etiquetas en un archivo pickle
with open(os.path.join(image_dir, 'captured_data.pickle'), 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Datos capturados y guardados exitosamente. Se capturaron {capture_count} gestos.")
