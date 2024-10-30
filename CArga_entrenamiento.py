import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Directorio de las imágenes y archivo de datos
image_dir = r"C:\Users\Alex\Documents\BehindBlueEyes\captured_data"
data_file = os.path.join(image_dir, 'captured_data.pickle')

# Cargar datos previos
if os.path.exists(data_file):
    with open(data_file, 'rb') as f:
        saved_data = pickle.load(f)
        data = saved_data['data']
        labels = saved_data['labels']
        print(f"Datos cargados. Se han capturado {len(data)} gestos previamente.")
else:
    print("No hay datos guardados.")
    data = []
    labels = []

# Convertir las listas a numpy arrays
data = np.array(data)
labels = np.array(labels)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Crear el modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Entrenar el modelo
knn.fit(X_train, y_train)

# Evaluar el modelo
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy*100:.2f}%")

# Mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.title('Matriz de Confusión')
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')
plt.show()

# Mostrar el reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Guardar el modelo entrenado
model_file = os.path.join(image_dir, 'gesture_knn_model.pickle')
with open(model_file, 'wb') as f:
    pickle.dump(knn, f)

print(f"Modelo KNN guardado en {model_file}")
