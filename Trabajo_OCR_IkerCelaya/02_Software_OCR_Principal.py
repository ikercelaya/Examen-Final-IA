import cv2
import numpy as np
import tensorflow as tf
import os
from modulos_vision import detectar_qr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelos', 'ocr_final_iker.h5')
CLASSES_PATH = os.path.join(BASE_DIR, 'modelos', 'clases.txt')

def normalizar_para_ia(roi):
    IMG_SIZE = 64
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    h, w = roi.shape
    f = 32 / max(h, w) # Factor 32 para evitar r -> F
    nh, nw = int(h * f), int(w * f)
    res = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    y_off = (IMG_SIZE - nh) // 2
    x_off = (IMG_SIZE - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = res
    return canvas

# --- CARGA DEL SISTEMA ---
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: No se encuentra el modelo en {MODEL_PATH}")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH, 'r') as f:
    clases = f.read().splitlines()

print(f"✅ Sistema Iker Celaya cargado (OCR + QR).")

while True:
    nombre = input("\n📝 Nombre de la imagen de prueba o 'salir': ")
    if nombre.lower() == 'salir': break
    
    img = cv2.imread(os.path.join(BASE_DIR, nombre))
    if img is None:
        print("❌ Archivo no encontrado.")
        continue

    print("🔍 Buscando códigos QR...")
    detectar_qr(img) 

    # --- 2. PIPELINE DE OCR  ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if h > 7:
            rects.append((x, y, w, h))
    
    rects = sorted(rects, key=lambda b: b[0])
    
    resultado = ""
    for (x, y, w, h) in rects:
        roi = thresh[y:y+h, x:x+w]
        roi_ia = normalizar_para_ia(roi)
        input_data = roi_ia.reshape(1, 64, 64, 1).astype('float32') / 255.0
        pred = model.predict(input_data, verbose=0)
        resultado += clases[np.argmax(pred)]
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(img, clases[np.argmax(pred)], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    print(f"🔍 TEXTO DETECTADO: {resultado}")
    cv2.imshow("OCR + QR Iker Celaya", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()