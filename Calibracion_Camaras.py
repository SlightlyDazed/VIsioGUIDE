import cv2
import dlib
import numpy as np

# Cargar el detector de caras y el predictor de puntos faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Número de imágenes a procesar
num_images = 25  # Usa el número de imágenes que has tomado

# Directorio donde están guardadas las imágenes
image_folder = "C:/Users/Alex/Documents/BehindBlueEyes/"

# Listas para almacenar los puntos detectados
img_points_camera1 = []
img_points_camera2 = []

# Lista para almacenar las imágenes
images_camera1 = []
images_camera2 = []

# Procesar imágenes de ambas cámaras
for i in range(num_images):
    # Cargar las imágenes de ambas cámaras
    img_camera1 = cv2.imread(f"{image_folder}img_camera1_{i}.png")
    img_camera2 = cv2.imread(f"{image_folder}img_camera2_{i}.png")
    
    if img_camera1 is None or img_camera2 is None:
        print(f"Error cargando las imágenes {i}")
        continue

    # Convertir a escala de grises
    gray_camera1 = cv2.cvtColor(img_camera1, cv2.COLOR_BGR2GRAY)
    gray_camera2 = cv2.cvtColor(img_camera2, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en ambas cámaras
    faces_camera1 = detector(gray_camera1)
    faces_camera2 = detector(gray_camera2)

    # Verificar si ambas cámaras detectan caras
    if len(faces_camera1) > 0 and len(faces_camera2) > 0:
        # Procesar las caras detectadas en la cámara 1
        landmarks_camera1 = predictor(gray_camera1, faces_camera1[0])
        points_camera1 = np.array([[landmarks_camera1.part(n).x, landmarks_camera1.part(n).y] for n in range(68)], dtype=np.float32)
        img_points_camera1.append(points_camera1)

        # Procesar las caras detectadas en la cámara 2
        landmarks_camera2 = predictor(gray_camera2, faces_camera2[0])
        points_camera2 = np.array([[landmarks_camera2.part(n).x, landmarks_camera2.part(n).y] for n in range(68)], dtype=np.float32)
        img_points_camera2.append(points_camera2)

        # Almacenar las imágenes correspondientes
        images_camera1.append(img_camera1)
        images_camera2.append(img_camera2)

        # Dibujar los puntos clave en ambas imágenes
        for n in range(68):
            x1, y1 = landmarks_camera1.part(n).x, landmarks_camera1.part(n).y
            x2, y2 = landmarks_camera2.part(n).x, landmarks_camera2.part(n).y
            cv2.circle(img_camera1, (x1, y1), 2, (255, 0, 0), -1)
            cv2.circle(img_camera2, (x2, y2), 2, (0, 255, 0), -1)

        # Mostrar las imágenes con los puntos clave detectados
        cv2.imshow(f"Landmarks Camera 1 - Image {i}", img_camera1)
        cv2.imshow(f"Landmarks Camera 2 - Image {i}", img_camera2)
    else:
        print(f"Rostros no detectados en ambas cámaras para la imagen {i}, omitiendo...")

    # Esperar un tiempo antes de pasar a la siguiente imagen
    cv2.waitKey(500)

cv2.destroyAllWindows()

# Verifica el contenido y formato de los puntos detectados
print("img_points_camera1:")
for points in img_points_camera1:
    print(points.shape)

print("img_points_camera2:")
for points in img_points_camera2:
    print(points.shape)

# Guarda los puntos detectados
np.save('img_points_camera1.npy', img_points_camera1)
np.save('img_points_camera2.npy', img_points_camera2)

# Cargar los puntos detectados para verificación
loaded_img_points_camera1 = np.load('img_points_camera1.npy', allow_pickle=True)
loaded_img_points_camera2 = np.load('img_points_camera2.npy', allow_pickle=True)

print("Datos cargados img_points_camera1:")
for points in loaded_img_points_camera1:
    print(points)

print("Datos cargados img_points_camera2:")
for points in loaded_img_points_camera2:
    print(points)

# Calibración estéreo
# Es importante que el tamaño de la imagen coincida con las dimensiones reales
image_size = (images_camera1[0].shape[1], images_camera1[0].shape[0])

# Crear los puntos de objeto (en este caso, usando los 68 puntos de referencia facial)
obj_points = [np.zeros((68, 3), dtype=np.float32) for _ in range(len(img_points_camera1))]

# Llenar los puntos de objeto con coordenadas arbitrarias, ya que no tienes un tablero físico
for obj_pts in obj_points:
    obj_pts[:, :2] = np.array([[i, j] for i in range(68) for j in range(1)])

# Convertir los puntos de imagen en arrays de NumPy
img_points_camera1 = np.array(img_points_camera1, dtype=np.float32)
img_points_camera2 = np.array(img_points_camera2, dtype=np.float32)

# Definir matrices de cámara y coeficientes de distorsión iniciales
mtx1 = np.eye(3, dtype=np.float32)  # Matriz de cámara para la cámara 1
mtx2 = np.eye(3, dtype=np.float32)  # Matriz de cámara para la cámara 2
dist1 = np.zeros(5, dtype=np.float32)  # Coeficientes de distorsión para la cámara 1
dist2 = np.zeros(5, dtype=np.float32)  # Coeficientes de distorsión para la cámara 2

# Realizar la calibración estéreo
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_camera1, img_points_camera2, 
    mtx1, dist1, mtx2, dist2, 
    image_size, criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)

print("Calibración estéreo completa.")
print("Matriz de cámara 1:\n", mtx1)
print("Matriz de cámara 2:\n", mtx2)
print("Distorsión cámara 1:\n", dist1)
print("Distorsión cámara 2:\n", dist2)
print("Matriz de rotación:\n", R)
print("Vector de traslación:\n", T)
print("Matriz esencial:\n", E)
print("Matriz fundamental:\n", F)
