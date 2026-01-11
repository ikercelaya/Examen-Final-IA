import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
IMG_SIZE = 64
EPOCHS = 100 # Le damos tiempo a la red para que refine

def preprocesar_limpio(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Binarización agresiva: Blanco puro sobre Negro puro
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def cargar_datos():
    X, y, clases = [], [], []
    for root, _, files in os.walk(DATASET_PATH):
        fotos = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if fotos:
            clase = os.path.basename(root)
            if clase not in clases: clases.append(clase)
            idx = clases.index(clase)
            for f in fotos:
                proc = preprocesar_limpio(os.path.join(root, f))
                if proc is not None:
                    X.append(proc)
                    y.append(idx)
    return np.array(X), np.array(y), clases

# CARGA
X, y, nombres_clases = cargar_datos()
X = X.reshape(-1, 64, 64, 1).astype('float32') / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.1,
    fill_mode='nearest'
)

# MODELO CON REGULARIZACIÓN (Evita que el accuracy se estanque en 0.68)
model = models.Sequential([
    layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(64,64,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(nombres_clases), activation='softmax')
])

# COMPILACIÓN
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks inteligentes
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.00001)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# ENTRENAMIENTO
print(f"🚀 Iniciando entrenamiento con {len(nombres_clases)} clases. Objetivo: > 0.85")
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=EPOCHS, validation_data=(X_test, y_test),
          callbacks=[reduce_lr, early_stop])

# GUARDADO
os.makedirs(os.path.join(BASE_DIR, 'modelos'), exist_ok=True)
model.save(os.path.join(BASE_DIR, 'modelos', 'ocr_final_iker.h5'))
with open(os.path.join(BASE_DIR, 'modelos', 'clases.txt'), 'w') as f: f.write("\n".join(nombres_clases))
print("\n✅ ENTRENAMIENTO FINALIZADO. Prueba ahora el Script 02.")