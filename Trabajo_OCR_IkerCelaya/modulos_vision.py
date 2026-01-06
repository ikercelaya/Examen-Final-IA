from pyzbar import pyzbar

def detectar_qr(imagen):
    qrs = pyzbar.decode(imagen)
    for qr in qrs:
        print(f"Contenido QR: {qr.data.decode('utf-8')}")