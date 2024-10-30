import requests
import ollama
import pyttsx3
import threading
import queue
import re
import speech_recognition as sr
import time
from urllib.parse import quote
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import deque

# API Keys y configuraciones
NEWS_API_KEY = "9c842d5f956b453e8a5294b6fe986c4d"
MODEL_PATH = r"C:\Users\Alex\Documents\BehindBlueEyes\captured_data\gesture_knn_model.pickle"

# Palabras clave para detectar intenciones
NEWS_KEYWORDS = ['noticia', 'noticias', 'acontecimiento', 'acontecimientos', 'suceso', 'sucesos', 'actualidad']
WIKI_KEYWORDS = ['información', 'informacion', 'que significa', 'que es', 'quien es', 'quién es', 'qué es', 'definición', 'definicion', 'explica']
SIGN_KEYWORDS = ['traduce señas', 'lenguaje de señas', 'traducir señas', 'interpreta señas', 'reconoce señas']

# Colas y eventos
speech_queue = queue.Queue()
is_speaking = threading.Event()

class SignLanguageTranslator:
    def __init__(self, model_path):
        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configuración para estabilización de predicciones
        self.prediction_queue = deque(maxlen=10)  # Mantiene las últimas 10 predicciones
        self.stability_threshold = 7  # Número mínimo de predicciones iguales para considerarla estable
        self.last_spoken_prediction = None
        self.frames_without_detection = 0
        self.max_frames_without_detection = 10  # Resetear después de 10 frames sin detección
        
        # Cargar el modelo KNN
        try:
            with open(model_path, 'rb') as f:
                self.knn = pickle.load(f)
            print("Modelo KNN cargado correctamente.")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.knn = None
        
        self.exit_translation = False  # New flag to control exit

    def get_stable_prediction(self):
        if len(self.prediction_queue) < self.stability_threshold:
            return None
            
        prediction_counts = {}
        for pred in self.prediction_queue:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
        most_common_prediction = max(prediction_counts.items(), key=lambda x: x[1])
        
        if most_common_prediction[1] >= self.stability_threshold:
            return most_common_prediction[0]
        return None

    def start_translation(self):
        if self.knn is None:
            return "Error: No se pudo cargar el modelo de reconocimiento de señas."

        cap = cv2.VideoCapture(0)
        print("Iniciando traducción de lenguaje de señas. Presiona 'q' para salir.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            if results.multi_hand_landmarks:
                self.frames_without_detection = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    data_aux = []
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x)
                        data_aux.append(landmark.y)
                    data_aux = np.array(data_aux).reshape(1, -1)
                    
                    prediction = self.knn.predict(data_aux)[0]
                    self.prediction_queue.append(prediction)
                    
                    stable_prediction = self.get_stable_prediction()
                    text_color = (0, 255, 0) if stable_prediction else (0, 165, 255)
                    
                    cv2.putText(frame, f'Prediccion: {prediction}', 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, text_color, 2, cv2.LINE_AA)
                    
                    if stable_prediction and stable_prediction != self.last_spoken_prediction:
                        print(f"Letra detectada: {stable_prediction}")
                        speech_queue.put(f"Letra {stable_prediction}")
                        self.last_spoken_prediction = stable_prediction
                        cv2.putText(frame, f'Estable: {stable_prediction}', 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                self.frames_without_detection += 1
                if self.frames_without_detection >= self.max_frames_without_detection:
                    self.prediction_queue.clear()
                    self.last_spoken_prediction = None
                    self.frames_without_detection = 0
            
            cv2.imshow('Reconocimiento de Señas', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return "Traducción de señas finalizada."

def search_wikipedia(query):
    search_url = f"https://es.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(query)}&format=json&utf8=1"
    try:
        search_response = requests.get(search_url)
        search_data = search_response.json()
        
        if not search_data.get('query', {}).get('search'):
            return f"No se encontraron resultados en Wikipedia para '{query}'"
        
        page_title = search_data['query']['search'][0]['title']
        summary_url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{quote(page_title)}"
        summary_response = requests.get(summary_url)
        
        if summary_response.status_code == 200:
            summary_data = summary_response.json()
            return summary_data.get("extract", "No se pudo obtener el resumen del artículo.")
        else:
            return f"Error al obtener el resumen: código {summary_response.status_code}"
            
    except Exception as e:
        return f"Error al buscar en Wikipedia: {str(e)}"

def get_news(query):
    encoded_query = quote(query)
    url = f"https://newsapi.org/v2/everything?q={encoded_query}&language=es&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            if articles:
                return "\n".join([f"• {article['title']}" for article in articles[:3]])
            else:
                return "No se encontraron noticias sobre este tema."
        else:
            return f"Error al obtener noticias: código {response.status_code}"
    except Exception as e:
        return f"Error al conectar con News API: {str(e)}"

def detect_intent(text):
    text = text.lower()
    
    # Detectar intención de traducción de señas
    for keyword in SIGN_KEYWORDS:
        if keyword in text:
            return 'sign', None
    
    # Buscar palabras clave de noticias
    for keyword in NEWS_KEYWORDS:
        if keyword in text:
            search_term = text.replace(keyword, '').strip()
            search_term = re.sub(r'^(sobre|acerca de|del|de la|las|los|el|en|por)\s+', '', search_term)
            return 'news', search_term
            
    # Buscar palabras clave de Wikipedia
    for keyword in WIKI_KEYWORDS:
        if keyword in text:
            search_term = text.replace(keyword, '').strip()
            search_term = re.sub(r'^(sobre|acerca de|del|de la|las|los|el|en|por)\s+', '', search_term)
            return 'wiki', search_term
            
    return None, None

def speak_text():
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate + 50)

    while True:
        text = speech_queue.get()
        if text == "FIN":
            break
        is_speaking.set()
        engine.say(text)
        engine.runAndWait()
        is_speaking.clear()

def listen_to_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.pause_threshold = 1.0
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Di algo: ")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="es-ES")
        print(f"Tú: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("No pude entender lo que dijiste.")
        return ""
    except sr.RequestError:
        print("Error con el servicio de reconocimiento de voz.")
        return ""

def chat_with_ollama():
    print("Iniciando chat de voz...")
    conversation_history = []
    speech_thread = threading.Thread(target=speak_text)
    speech_thread.start()
    
    # Inicializar el traductor de señas
    sign_translator = SignLanguageTranslator(MODEL_PATH)
    
    presentation = "Hola, soy tu asistente de navegacion basado en intelignecia artifcial,puedo ayudarte en multiples tareas. Di 'salir' cuando quieras terminar."
    print("Bot:", presentation)
    speech_queue.put(presentation)
    
    while True:
        while not speech_queue.empty() or is_speaking.is_set():
            time.sleep(0.5)

        user_input = listen_to_microphone()
        
        if not user_input:
            continue

        if "salir" in user_input:
            farewell = "Hasta luego, espero haberte sido de ayuda."
            print("Bot:", farewell)
            speech_queue.put(farewell)
            while not speech_queue.empty() or is_speaking.is_set():
                time.sleep(0.5)
            speech_queue.put("FIN")
            break

        # Detectar intención del usuario
        intent, search_term = detect_intent(user_input)
        
        if intent == 'sign':
            print("Iniciando traducción de lenguaje de señas...")
            result = sign_translator.start_translation()
            print(result)
            speech_queue.put(result)
            continue
        
        elif intent == 'news' and search_term:
            print("Buscando noticias sobre:", search_term)
            news = get_news(search_term)
            print("Noticias:\n", news)
            speech_queue.put(news)
            continue
            
        elif intent == 'wiki' and search_term:
            print("Buscando información sobre:", search_term)
            wiki_info = search_wikipedia(search_term)
            print("Wikipedia:", wiki_info)
            speech_queue.put(wiki_info)
            continue

        # OLLAMA response para otras preguntas
        conversation_history.append({"role": "user", "content": user_input})
        stream = ollama.chat(model="llama3.2:3b", messages=conversation_history, stream=True)
        print("Bot:")
        bot_response = ""
        buffer = ""
        
        for chunk in stream:
            content = chunk["message"]["content"]
            buffer += content
            sentences = re.split(r'(?<=[.!?])\s+|,', buffer)
            for sentence in sentences[:-1]:
                clean_sentence = sentence.strip()
                if clean_sentence:
                    print(clean_sentence)
                    bot_response += clean_sentence + " "
                    speech_queue.put(clean_sentence)
            buffer = sentences[-1]
        
        if buffer.strip():
            print(buffer.strip())
            bot_response += buffer.strip()
            speech_queue.put(buffer.strip())
        
        conversation_history.append({"role": "assistant", "content": bot_response.strip()})

    speech_thread.join()

if __name__ == "__main__":
    chat_with_ollama()