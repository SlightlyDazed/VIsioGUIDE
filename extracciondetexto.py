import requests
import ollama
import cv2
import keras_ocr
import re  # Asegúrate de importar la biblioteca re

class TextExtractor:
    def __init__(self):
        # Crear un pipeline de OCR
        self.pipeline = keras_ocr.pipeline.Pipeline()

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
                    ret, img = cap.read()
                    if ret:
                        captured_images.append(img)
                        print(f"Imagen {i + 1} capturada.")
                    else:
                        print("Error: No se pudo leer la imagen.")
                break  # Salir del bucle después de capturar las imágenes
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        return captured_images

    def extract_text_from_images(self, images):
        combined_text = ""

        for image in images:
            prediction_groups = self.pipeline.recognize([image])
            for text, _ in prediction_groups[0]:  # La primera imagen en el grupo
                combined_text += f"{text} "
        
        return combined_text.strip()

def improve_text_with_ollama(text):
    conversation_history = [{"role": "user", "content": text}]
    
    # Enviar el texto a OLLAMA y obtener la respuesta
    stream = ollama.chat(model="llama3.2:3b", messages=conversation_history, stream=True)
    
    improved_text = ""
    buffer = ""
    
    for chunk in stream:
        content = chunk["message"]["content"]
        buffer += content
        sentences = re.split(r'(?<=[.!?])\s+|,', buffer)
        for sentence in sentences[:-1]:
            clean_sentence = sentence.strip()
            if clean_sentence:
                improved_text += clean_sentence + " "
        buffer = sentences[-1]
    
    if buffer.strip():
        improved_text += buffer.strip()

    return improved_text.strip()

if __name__ == "__main__":
    num_images = 10  # Define cuántas imágenes deseas capturar
    extractor = TextExtractor()
    images = extractor.capture_images(num_images)
    
    if images:
        extracted_text = extractor.extract_text_from_images(images)
        print("Texto extraído:")
        print(extracted_text)
        
        improved_text = improve_text_with_ollama(extracted_text)
        print("Texto mejorado:")
        print(improved_text)
