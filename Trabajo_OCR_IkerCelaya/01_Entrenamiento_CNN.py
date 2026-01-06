import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import time

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
IMG_SIZE = 64

def cargar_dataset_jerarquico():
    imagenes = []
    etiquetas = []
    nombres_clases = []

    print(f"--- ESCANEANDO ESTRUCTURA EN: {DATASET_PATH} ---")
    
    # Recorremos TODO el árbol de carpetas
    for root, dirs, files in os.walk(DATASET_PATH):
        # Buscamos archivos .png
        fotos = [f for f in files if f.lower().endswith('.png')]
        
        if fotos:
            # El nombre de la clase es la ÚLTIMA carpeta (ej: 'A', 'a' o '0')
            nombre_clase = os.path.basename(root)
            
            # Si la clase no está en nuestra lista, la añadimos
            if nombre_clase not in nombres_clases:
                nombres_clases.append(nombre_clase)
            
            idx_clase = nombres_clases.index(nombre_clase)
            print(f"Cargando {len(fotos)} fotos de la clase: {nombre_clase}")
            
            for archivo in fotos:
                ruta_img = os.path.join(root, archivo)
                img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    imagenes.append(img)
                    etiquetas.append(idx_clase)
                    
    return np.array(imagenes), np.array(etiquetas), nombres_clases

# --- PROCESO DE ENTRENAMIENTO ---
try:
    start_time = time.time()
    
    # Cargar datos
    X, y, nombres_clases = cargar_dataset_jerarquico()
    
    if len(X) == 0:
        raise Exception("No se encontraron imágenes .png. Verifica que la carpeta 'dataset' sea correcta.")

    print(f"\n✅ Dataset listo: {len(X)} imágenes en {len(nombres_clases)} categorías.")

    # Normalizar (0 a 1) y preparar dimensiones
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definición de la Red Neuronal (CNN)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(nombres_clases), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("\n🚀 Iniciando entrenamiento...")
    history = model.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test), batch_size=32)

    # Guardar modelo y diccionario de clases
    os.makedirs(os.path.join(BASE_DIR, 'modelos'), exist_ok=True)
    model.save(os.path.join(BASE_DIR, 'modelos', 'ocr_manual_64.h5'))
    with open(os.path.join(BASE_DIR, 'modelos', 'clases.txt'), 'w') as f:
        f.write("\n".join(nombres_clases))

    # Guardar métricas de rendimiento para el informe
    duracion = time.time() - start_time
    os.makedirs(os.path.join(BASE_DIR, 'resultados'), exist_ok=True)
    with open(os.path.join(BASE_DIR, 'resultados', 'rendimiento.txt'), 'w') as f:
        f.write(f"Tiempo entrenamiento: {duracion:.2f}s\n")
        f.write(f"Precisión final: {history.history['accuracy'][-1]*100:.2f}%\n")

    print(f"\n✨ ÉXITO ✨")
    print(f"Modelo guardado en 'modelos/ocr_manual_64.h5'. Tiempo: {duracion:.2f}s")

except Exception as e:
    print(f"❌ ERROR: {e}")