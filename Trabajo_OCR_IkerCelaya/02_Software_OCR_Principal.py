import cv2
import numpy as np
import tensorflow as tf
import pytesseract # Para tipografía de imprenta obligatoria

# Cargar modelo entrenado y etiquetas
model = tf.keras.models.load_model('modelo_ocr_manual.h5')
CLASS_NAMES = ["0", "1", "A", "B", ...] # Deben coincidir con tu dataset

class SoftwareOCR:
    def __init__(self, image_path):
        self.original = cv2.imread(image_path)
        self.processed = None
        self.text_result = ""

    def preprocesar(self):
        # Paso 1: Escala de grises
        gris = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        # Paso 2: Filtro para limpiar ruido (Gaussian Blur)
        desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)
        # Paso 3: Binarización (Otsu) para separar trazo de fondo
        _, self.processed = cv2.threshold(desenfoque, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return self.processed

    def segmentar_y_predecir(self):
        # Encontrar contornos de cada letra
        cnts, _ = cv2.findContours(self.processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Ordenar contornos de izquierda a derecha
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][0]))

        for x, y, w, h in boundingBoxes:
            if w > 5 and h > 10: # Filtrar ruido pequeño
                # Extraer el carácter
                roi = self.processed[y:y+h, x:x+w]
                # Redimensionar al tamaño del modelo (32x32) con padding
                roi = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
                roi = cv2.resize(roi, (32, 32))
                roi = roi.reshape(1, 32, 32, 1) / 255.0
                
                # Predicción con la CNN
                pred = model.predict(roi)
                char = CLASS_NAMES[np.argmax(pred)]
                self.text_result += char
        
        return self.text_result

    def ocr_imprenta(self):
        # Uso de Tesseract para la funcionalidad obligatoria de imprenta
        return pytesseract.image_to_string(self.original, lang='spa')

# --- Ejecución ---
app = SoftwareOCR('prueba_mano.jpg')
app.preprocesar()
resultado = app.segmentar_y_predecir()

with open("resultado.txt", "w") as f:
    f.write(f"Texto Manual Detectado: {resultado}")

print(f"Resultado final: {resultado}")