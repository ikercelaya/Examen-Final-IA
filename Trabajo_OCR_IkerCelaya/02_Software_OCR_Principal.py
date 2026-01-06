import cv2
import numpy as np
import tensorflow as tf
import os
import time

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelos', 'ocr_manual_64.h5')
CLASSES_PATH = os.path.join(BASE_DIR, 'modelos', 'clases.txt')

# MODO DEPURACIÓN: Muestra qué ve la IA
DEBUG_MODE = True 

class OCRAjustado:
    def __init__(self):
        print("Cargando IA...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, 'r') as f:
            self.clases = f.read().splitlines()
        print(f"✅ IA lista. Clases: {len(self.clases)}")

    def preprocesar_suave(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur suave
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # --- CAMBIO CLAVE 1: Umbral menos agresivo ---
        # Block Size 11 (antes 19), C 2 (antes 5). 
        # Esto detecta trazos más finos y no hace "pegotes".
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # --- CAMBIO CLAVE 2: Limpieza mínima ---
        # Kernel muy pequeño (2x2) solo para quitar ruido diminuto.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # Solo 'opening' para limpiar. HE QUITADO EL 'DILATE' que deformaba las letras.
        processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        if DEBUG_MODE:
            cv2.imshow("Limpieza Suave (Input IA)", processed)
            cv2.waitKey(1)

        return processed

    def preparar_roi_centrada(self, roi_img):
        # Función para centrar la letra en 64x64 con margen
        h, w = roi_img.shape
        # Dimensiones objetivo (dejando margen)
        target_dim = 50
        
        # Calcular factor de escala manteniendo relación de aspecto
        scale = target_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
            
        resized = cv2.resize(roi_img, (new_w, new_h))
        
        # Crear lienzo negro 64x64
        final_img = np.zeros((64, 64), dtype=np.uint8)
        
        # Calcular centro
        x_off = (64 - new_w) // 2
        y_off = (64 - new_h) // 2
        
        # Pegar
        final_img[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
        return final_img

    def procesar(self, nombre_archivo):
        img_path = os.path.join(BASE_DIR, nombre_archivo)
        if not os.path.exists(img_path): return

        start_time = time.time()
        original_img = cv2.imread(img_path)
        
        # 1. Preprocesamiento Suave
        processed_img = self.preprocesar_suave(original_img)

        # 2. Segmentación
        cnts, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            # Filtros para asegurar que es una letra
            aspect = w / float(h)
            if w > 8 and h > 12 and aspect < 3:
                boxes.append((x, y, w, h))
        
        boxes = sorted(boxes, key=lambda b: b[0])

        # 3. Clasificación
        texto_final = ""
        debug_img = original_img.copy()
        
        for i, (x, y, w, h) in enumerate(boxes):
            roi = processed_img[y:y+h, x:x+w]
            roi_ready = self.preparar_roi_centrada(roi)
            
            if DEBUG_MODE:
                # Muestra cada letra individualmente como la ve la IA
                cv2.imshow(f"Letra {i}", roi_ready)

            roi_final = roi_ready.reshape(1, 64, 64, 1).astype('float32') / 255.0
            pred = self.model.predict(roi_final, verbose=0)
            
            letra_detectada = self.clases[np.argmax(pred)]
            texto_final += letra_detectada
            
            # Dibujar resultado sobre la imagen
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_img, letra_detectada, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if DEBUG_MODE:
            cv2.imshow("Resultado Visual", debug_img)
            print("Pulsa cualquier tecla en las ventanas para continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        total_time = time.time() - start_time

        print("\n" + "═"*45)
        print(f"📂 RESULTADO: {nombre_archivo}")
        # AHORA SÍ DEBERÍA SALIR ALGO PARECIDO A "OBAC"
        print(f"🔍 TEXTO DETECTADO: {texto_final}") 
        print(f"⏱️  TIEMPO: {total_time:.4f}s")
        print("═"*45 + "\n")

if __name__ == "__main__":
    app = OCRAjustado()
    print("--- Introduce nombres de imágenes ---")
    while True:
        f = input("Imagen (ej: prueba.png) o 'salir': ")
        if f.lower() == 'salir': break
        app.procesar(f)