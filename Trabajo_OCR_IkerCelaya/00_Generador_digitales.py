import os
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
IMG_SIZE = 64

# Definición de caracteres por categorías
categorias = {
    "mayusculas": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "minusculas": "abcdefghijklmnopqrstuvwxyz",
    "numeros": "0123456789"
}

# Fuentes estándar de Windows (puedes añadir más si quieres)
fuentes_sistema = [
    {"nombre": "arial", "path": "arial.ttf"},
    {"nombre": "times", "path": "times.ttf"},
    {"nombre": "courier", "path": "cour.ttf"},
    {"nombre": "calibri", "path": "calibri.ttf"}
]

def generar_dataset_digital():
    print("🚀 Iniciando generación de caracteres digitales...")
    
    for cat_nombre, caracteres in categorias.items():
        for char in caracteres:
            # Crear la ruta: dataset/categoria/caracter/
            # Usamos str(char) para asegurar que los números se traten como carpetas
            ruta_carpeta = os.path.join(DATASET_PATH, cat_nombre, str(char))
            os.makedirs(ruta_carpeta, exist_ok=True)

            for f in fuentes_sistema:
                try:
                    # Probamos un tamaño que llene bien el espacio de 64x64
                    font = ImageFont.truetype(f["path"], 50)
                    
                    # Crear lienzo blanco
                    img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
                    draw = ImageDraw.Draw(img)
                    
                    # Centrar el texto
                    bbox = draw.textbbox((0, 0), char, font=font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    pos = ((IMG_SIZE - w) // 2, (IMG_SIZE - h) // 2 - bbox[1])
                    
                    # Dibujar en negro
                    draw.text(pos, char, fill=0, font=font)
                    
                    # Nombre solicitado: caracter_digital_fuente.png
                    nombre_archivo = f"{char}_digital_{f['nombre']}.png"
                    img.save(os.path.join(ruta_carpeta, nombre_archivo))
                    
                except Exception as e:
                    # Si no encuentra una fuente, simplemente pasa a la siguiente
                    continue

    print(f"✅ ¡Hecho! Los caracteres digitales se han guardado en {DATASET_PATH}")

if __name__ == "__main__":
    generar_dataset_digital()