import os
import numpy as np
import cv2
import pickle
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Directorio de las imágenes y archivo de modelo
image_dir = r"C:\Users\Alex\Documents\BehindBlueEyes\captured_data"
model_file = os.path.join(image_dir, 'gesture_knn_model.pickle')

# Cargar el modelo KNN guardado
if os.path.exists(model_file):
    with open(model_file, 'rb') as f:
        knn = pickle.load(f)
    print("Modelo KNN cargado correctamente.")
else:
    print("Modelo no encontrado. Asegúrate de haber entrenado y guardado un modelo.")
    exit()

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
            
            # Extraer el vector de características del gesto
            data_aux = []
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)
            data_aux = np.array(data_aux).reshape(1, -1)
            
            # Realizar la predicción del gesto
            prediction = knn.predict(data_aux)
            predicted_label = prediction[0]
            
            # Mostrar la predicción en la imagen
            cv2.putText(frame, f'Prediccion: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Mostrar la imagen en vivo con la predicción
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Salir si se presiona 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
