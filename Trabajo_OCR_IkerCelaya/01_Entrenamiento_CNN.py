import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Configuración
DATASET_PATH = 'ruta_a_tu_dataset' # Carpeta con subcarpetas por clase
IMG_SIZE = 32 # Tamaño estándar para procesar

def load_data():
    images = []
    labels = []
    classes = sorted(os.listdir(DATASET_PATH))
    
    for idx, label in enumerate(classes):
        folder = os.path.join(DATASET_PATH, label)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(idx)
    
    return np.array(images), np.array(labels), classes

# 1. Cargar y normalizar
X, y, class_names = load_data()
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Arquitectura de la CNN (Mínimo 3 capas para precisión)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Entrenar y guardar
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save('modelo_ocr_manual.h5')
print("Modelo guardado con éxito.")