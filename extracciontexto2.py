import cv2
import numpy as np
from doctr.models import ocr_predictor
import time  # Importa time

class TextExtractor:
    def __init__(self):
        # Crear un predictor de OCR usando Doctr
        self.predictor = ocr_predictor()

    def capture_images(self, num_images):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return []

        captured_images = []

        print(f"Presiona 'c' para capturar {num_images} imágenes, 'q' para salir.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer la imagen.")
                break
            
            cv2.imshow('Captura de Imagen', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                print(f"Capturando {num_images} imágenes...")
                for i in range(num_images):
                    ret, frame = cap.read()
                    if ret:
                        captured_images.append(frame)
                        print(f"Imagen {i + 1} capturada.")
                        time.sleep(0.1)  # Pausa para asegurar que se tomen las imágenes
                break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        return captured_images

    def extract_text_from_images(self, images):
        combined_text = ""

        for image in images:
            # Convertir la imagen a RGB si está en BGR
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Asegúrate de que la imagen sea un array 3D (altura, ancho, canales)
            if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
                print("Error: La imagen debe ser multi-canal.")
                continue

            # Usar el predictor para extraer texto de la imagen
            text = self.predictor(image_rgb)
            combined_text += f"{text} "
        
        return combined_text.strip()

if __name__ == "__main__":
    num_images = 10  # Define cuántas imágenes deseas capturar
    extractor = TextExtractor()
    images = extractor.capture_images(num_images)
    
    if images:
        extracted_text = extractor.extract_text_from_images(images)
        print("Texto extraído:")
        print(extracted_text)
