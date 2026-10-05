"""
Predicciones desde la consola, sin abrir Streamlit.

Uso:
    python src/predecir.py "You have won a free prize, call now"
    python src/predecir.py          (sin argumentos corre unos ejemplos de prueba)
"""

import os
import sys
import numpy as np
import tensorflow as tf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from limpieza import limpiar_texto

RUTA_MODELO = os.path.join(BASE, "modelo", "modelo_spam.keras")

EJEMPLOS = [
    "Congratulations! You have won a FREE iPhone 15. Click here to claim your prize now",
    "URGENT: Your account has been suspended. Verify your details at www.secure-login.com",
    "WINNER!! You have been selected to receive a 900 prize reward. Call 09061701461 now",
    "Free entry in a weekly competition to win tickets, text WIN to 80086",
    "Hey are we still meeting tomorrow at the library?",
    "Mom says dinner is ready, come home soon",
    "Can you send me the notes from yesterday class please",
    "I will call you back in 10 minutes, im in a meeting right now",
    "Happy birthday bro, hope you have a great day",
    "Did you finish the homework for monday?"
]


def main():
    if not os.path.exists(RUTA_MODELO):
        print("No encuentro el modelo. Primero corra src/entrenamiento.py")
        sys.exit(1)

    modelo = tf.keras.models.load_model(RUTA_MODELO)

    mensajes = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else EJEMPLOS

    limpios = np.array([limpiar_texto(m) for m in mensajes], dtype=object)
    probs = modelo.predict(limpios, verbose=0).flatten()

    print()
    for m, p in zip(mensajes, probs):
        etiqueta = "SPAM" if p >= 0.5 else "ham "
        print("[%s]  %6.2f%%  |  %s" % (etiqueta, p * 100, m[:70]))
    print()


if __name__ == "__main__":
    main()
