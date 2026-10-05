"""
App de Streamlit para probar el detector de spam.
Carga la LSTM ya entrenada, uno escribe un mensaje y la app dice si es spam
o no, con la probabilidad.

Para ejecutarla:  streamlit run app.py
"""

import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf

from limpieza import limpiar_texto

BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE, "modelo", "modelo_spam.keras")

st.set_page_config(page_title="Detector de spam - LSTM", page_icon="📩")


@st.cache_resource
def cargar_modelo():
    # el cache es para que no cargue el modelo cada vez que uno toca algo
    return tf.keras.models.load_model(RUTA_MODELO)


def predecir(modelo, texto):
    limpio = limpiar_texto(texto)
    if limpio == "":
        return None, limpio
    entrada = np.array([limpio], dtype=object)
    prob = float(modelo.predict(entrada, verbose=0).flatten()[0])
    return prob, limpio


st.title("Detector de mensajes spam 📩")
st.write("Proyecto de Deep Learning - red LSTM entrenada con el dataset SMS Spam Collection.")
st.write("Escriba un mensaje y el modelo le dice si parece spam o si es un mensaje normal.")

try:
    modelo = cargar_modelo()
except Exception:
    st.error("No pude cargar el modelo. Revise que exista el archivo modelo/modelo_spam.keras")
    st.stop()

with st.sidebar:
    st.header("Opciones")
    umbral = st.slider("Umbral para marcar como spam", 0.10, 0.90, 0.50, 0.05,
                       help="Si lo baja, el modelo marca mas mensajes como spam (detecta mas "
                            "pero se equivoca mas). Si lo sube, es mas exigente.")
    st.markdown("---")
    st.caption("El modelo fue entrenado con mensajes en ingles, entonces con mensajes "
               "en español no va a funcionar bien.")

# unos ejemplos listos para que no toque escribir desde cero
ejemplos = {
    "-- escribir yo mismo --": "",
    "Ejemplo de spam (premio)": "Congratulations! You have won a FREE iPhone. Click here to claim your prize now",
    "Ejemplo de spam (cuenta)": "URGENT: Your account has been suspended. Verify your details at www.secure-login.com",
    "Ejemplo normal (amigo)": "Hey are we still meeting tomorrow at the library?",
    "Ejemplo normal (casa)": "Mom says dinner is ready, come home soon",
}

escogido = st.selectbox("Puede usar uno de estos ejemplos:", list(ejemplos.keys()))

mensaje = st.text_area("Mensaje:", value=ejemplos[escogido], height=130,
                       placeholder="Escriba aqui el mensaje que quiere analizar...")

if st.button("Analizar mensaje", type="primary"):
    if mensaje.strip() == "":
        st.warning("Primero escriba un mensaje.")
    else:
        prob, limpio = predecir(modelo, mensaje)

        if prob is None:
            st.warning("Despues de limpiar el mensaje no quedo ninguna palabra que el modelo "
                       "pueda analizar. Intente con un texto mas largo.")
        else:
            st.markdown("---")
            es_spam = prob >= umbral

            if es_spam:
                st.error("### Es SPAM")
            else:
                st.success("### Es un mensaje normal (ham)")

            col1, col2 = st.columns(2)
            col1.metric("Probabilidad de spam", "%.2f %%" % (prob * 100))
            col2.metric("Probabilidad de ser normal", "%.2f %%" % ((1 - prob) * 100))

            st.progress(prob)

            # si queda muy cerca del umbral aviso, porque ahi el modelo esta dudando
            if abs(prob - umbral) < 0.15:
                st.info("El resultado quedo cerca del umbral, o sea que el modelo no esta "
                        "muy seguro con este mensaje.")

            with st.expander("Ver como quedo el mensaje despues de la limpieza"):
                st.code(limpio if limpio else "(vacio)")
                st.caption("Los links, correos y numeros se reemplazan por palabras clave antes "
                           "de pasarlos por la red.")
