import ollama
import pyttsx3
import threading
import queue
import re
import speech_recognition as sr
import time

# Crear una cola para manejar la lectura de frases
speech_queue = queue.Queue()
# Variable para controlar si está hablando
is_speaking = threading.Event()

def speak_text():
    engine = pyttsx3.init()
    
    # Ajusta la velocidad de lectura
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate + 50)

    while True:
        text = speech_queue.get()
        if text == "FIN":
            break
        # Indicar que está hablando
        is_speaking.set()
        # Decir el texto
        engine.say(text)
        engine.runAndWait()
        # Indicar que terminó de hablar
        is_speaking.clear()

def listen_to_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.pause_threshold = 1.0
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("Di algo: ")
        audio = recognizer.listen(source)
        
    try:
        # Usar el reconocimiento de voz de Google para convertir el audio a texto
        text = recognizer.recognize_google(audio, language="es-ES")
        print(f"Tú: {text}")
        return text
    except sr.UnknownValueError:
        print("No pude entender lo que dijiste.")
        return ""
    except sr.RequestError:
        print("Error con el servicio de reconocimiento de voz.")
        return ""

def chat_with_ollama():
    print("Iniciando chat de voz...")

    # Lista para almacenar el historial de la conversación
    conversation_history = []

    # Iniciar el hilo de pyttsx3 para la lectura
    speech_thread = threading.Thread(target=speak_text)
    speech_thread.start()

    # Mensaje de presentación
    presentation = "Hola, soy tu asistente virtual. Estoy aquí para ayudarte y conversar contigo. Puedes decir 'salir' en cualquier momento para terminar la conversación."
    print("Bot:", presentation)
    speech_queue.put(presentation)

    # Esperar a que termine de decir el mensaje de presentación
    while not speech_queue.empty() or is_speaking.is_set():
        time.sleep(0.5)

    while True:
        # Esperar hasta que pyttsx3 haya terminado y la cola esté vacía antes de permitir la entrada
        while not speech_queue.empty() or is_speaking.is_set():
            time.sleep(0.5)

        # Escuchar entrada de voz del usuario
        user_input = listen_to_microphone()
        
        # Verifica si el usuario dijo "salir" para finalizar la conversación
        if "salir" in user_input.lower():
            farewell = "No te vayas, no me dejes, te lo pido, quédate conmigo, un ratito más"
            print("Bot:", farewell)
            speech_queue.put(farewell)
            # Esperar a que termine de decir el mensaje de despedida
            while not speech_queue.empty() or is_speaking.is_set():
                time.sleep(0.5)
            speech_queue.put("FIN")
            break

        # Ajustar la longitud de la respuesta
        if any(keyword in user_input.lower() for keyword in ["más detalles", "explícame más", "cuéntame más"]):
            # Respuesta completa
            conversation_history.append({"role": "user", "content": user_input})
        else:
            # Respuesta breve por defecto
            conversation_history.append({"role": "user", "content": user_input + " (por favor, responde de forma breve)"})

        # Envía la conversación completa para recibir la respuesta en partes (chunks)
        stream = ollama.chat(model="llama3.2:3b", messages=conversation_history, stream=True)

        # Procesa cada fragmento por frases y muestra una frase por línea
        print("Bot:")
        bot_response = ""
        buffer = ""

        for chunk in stream:
            content = chunk["message"]["content"]
            buffer += content
            sentences = re.split(r'(?<=[.!?])\s+|,', buffer)
            for sentence in sentences[:-1]:
                clean_sentence = sentence.strip()
                if clean_sentence:  # Solo procesar si la frase no está vacía
                    print(clean_sentence)
                    bot_response += clean_sentence + " "
                    speech_queue.put(clean_sentence)
            buffer = sentences[-1]

        # Procesar cualquier frase restante en el buffer
        if buffer.strip():
            print(buffer.strip())
            bot_response += buffer.strip()
            speech_queue.put(buffer.strip())

        print()  # Nueva línea después de completar la respuesta

        # Agrega la respuesta completa del bot al historial de la conversación
        conversation_history.append({"role": "assistant", "content": bot_response.strip()})

    # Espera a que el hilo de pyttsx3 termine antes de finalizar
    speech_thread.join()

if __name__ == "__main__":
    chat_with_ollama()
