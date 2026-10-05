"""
Entrenamiento de la red LSTM para detectar mensajes spam.
Es el mismo codigo del notebook de Colab pero pasado a script, para poder
volver a entrenar rapido sin abrir el notebook.

Para correrlo:  python src/entrenamiento.py
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from limpieza import limpiar_texto

SEMILLA = 42
np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

CARPETA_MODELO = os.path.join(BASE, "modelo")
CARPETA_IMGS = os.path.join(BASE, "imagenes")
RUTA_DATOS = os.path.join(BASE, "datos", "sms.tsv")
URL_DATOS = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

os.makedirs(CARPETA_MODELO, exist_ok=True)
os.makedirs(CARPETA_IMGS, exist_ok=True)

MAX_PALABRAS = 8000    # tamaño del vocabulario
LARGO_MAX = 60         # a cuantas palabras recorto o relleno cada mensaje
EPOCAS = 15
BATCH = 32


# ---------------------------------------------------------------
# 1. Cargar los datos
# ---------------------------------------------------------------
def cargar_datos():
    ruta = RUTA_DATOS if os.path.exists(RUTA_DATOS) else URL_DATOS
    df = pd.read_csv(ruta, sep="\t", header=None, names=["etiqueta", "mensaje"])
    print("Mensajes cargados:", len(df))
    print(df["etiqueta"].value_counts())
    return df


# ---------------------------------------------------------------
# 2. Analisis exploratorio
# ---------------------------------------------------------------
def explorar(df):
    conteo = df["etiqueta"].value_counts()

    fig, (g1, g2) = plt.subplots(1, 2, figsize=(12, 4.5))
    g1.bar(["ham (normal)", "spam"], [conteo["ham"], conteo["spam"]],
           color=["#4c72b0", "#c44e52"])
    g1.set_ylabel("Cantidad de mensajes")
    g1.set_title("Cantidad de mensajes por clase")
    for i, v in enumerate([conteo["ham"], conteo["spam"]]):
        g1.text(i, v + 50, str(v), ha="center")

    porcentajes = [conteo["ham"] / len(df) * 100, conteo["spam"] / len(df) * 100]
    g2.pie(porcentajes, labels=["ham (normal)", "spam"], autopct="%1.1f%%",
           colors=["#4c72b0", "#c44e52"], startangle=90)
    g2.set_title("Proporcion de cada clase")

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_IMGS, "01_distribucion_clases.png"), dpi=110)
    plt.close()

    # longitud de los mensajes
    df["largo"] = df["mensaje"].apply(lambda t: len(str(t).split()))
    print("\nLargo promedio ham: %.1f palabras" % df[df.etiqueta == "ham"]["largo"].mean())
    print("Largo promedio spam: %.1f palabras" % df[df.etiqueta == "spam"]["largo"].mean())

    plt.figure(figsize=(10, 4.5))
    plt.hist(df[df.etiqueta == "ham"]["largo"], bins=40, alpha=0.65, label="ham", color="#4c72b0")
    plt.hist(df[df.etiqueta == "spam"]["largo"], bins=40, alpha=0.65, label="spam", color="#c44e52")
    plt.xlabel("Cantidad de palabras del mensaje")
    plt.ylabel("Frecuencia")
    plt.title("Largo de los mensajes segun la clase")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_IMGS, "02_largo_mensajes.png"), dpi=110)
    plt.close()

    # palabras mas usadas en cada clase
    from collections import Counter
    vacias = {"the", "to", "you", "a", "i", "and", "in", "is", "it", "my", "of",
              "for", "me", "that", "have", "on", "are", "your", "but", "not",
              "so", "do", "be", "at", "with", "will", "can", "if", "we", "this",
              "am", "its", "just", "now", "no", "or", "u", "numero", "s", "t"}

    def top(clase, n=12):
        palabras = []
        for m in df[df.etiqueta == clase]["mensaje"]:
            palabras += [p for p in limpiar_texto(m).split() if p not in vacias and len(p) > 2]
        return Counter(palabras).most_common(n)

    top_ham, top_spam = top("ham"), top("spam")

    fig, (g1, g2) = plt.subplots(1, 2, figsize=(13, 4.8))
    g1.barh([p for p, _ in top_ham][::-1], [c for _, c in top_ham][::-1], color="#4c72b0")
    g1.set_title("Palabras mas comunes en mensajes normales")
    g2.barh([p for p, _ in top_spam][::-1], [c for _, c in top_spam][::-1], color="#c44e52")
    g2.set_title("Palabras mas comunes en spam")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_IMGS, "03_palabras_frecuentes.png"), dpi=110)
    plt.close()

    print("\nTop palabras spam:", [p for p, _ in top_spam])


# ---------------------------------------------------------------
# 3. Limpieza y division
# ---------------------------------------------------------------
def preparar(df):
    df["mensaje_limpio"] = df["mensaje"].apply(limpiar_texto)
    df["y"] = (df["etiqueta"] == "spam").astype(int)

    # reviso si quedo algun mensaje vacio despues de limpiar
    vacios = (df["mensaje_limpio"].str.len() == 0).sum()
    print("\nMensajes que quedaron vacios despues de limpiar:", vacios)
    df = df[df["mensaje_limpio"].str.len() > 0].copy()

    # tambien quito duplicados, encontre varios mensajes repetidos
    antes = len(df)
    df = df.drop_duplicates(subset="mensaje_limpio").copy()
    print("Duplicados eliminados:", antes - len(df))

    # uso tolist() en vez de .values porque en las versiones nuevas de pandas
    # las columnas de texto no salen como arreglo de numpy y sklearn se queja
    # uso dtype=object porque Keras no recibe los arreglos de texto de numpy
    # normales (los de tipo <U...), toca pasarlos como objeto
    x = np.array(df["mensaje_limpio"].tolist(), dtype=object)
    y = np.array(df["y"].tolist())

    # primero separo el 15% de prueba, y de lo que queda saco otro 15% de validacion
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x, y, test_size=0.15, random_state=SEMILLA, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=0.1765, random_state=SEMILLA, stratify=y_tmp)

    print("\nEntrenamiento:", len(x_train), " (spam: %d)" % y_train.sum())
    print("Validacion:   ", len(x_val), " (spam: %d)" % y_val.sum())
    print("Prueba:       ", len(x_test), " (spam: %d)" % y_test.sum())

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), df


# ---------------------------------------------------------------
# 4. El modelo
# ---------------------------------------------------------------
def construir_modelo(x_train):
    # esta capa convierte el texto en numeros. La dejo dentro del modelo para
    # que al guardarlo quede guardado tambien el vocabulario, asi la app solo
    # tiene que pasarle el texto y ya
    vectorizador = layers.TextVectorization(
        max_tokens=MAX_PALABRAS,
        output_sequence_length=LARGO_MAX,
        standardize=None,   # la limpieza ya la hice yo antes a mano
        name="vectorizador"
    )
    vectorizador.adapt(x_train)
    print("Tamaño real del vocabulario:", len(vectorizador.get_vocabulary()))

    modelo = keras.Sequential([
        layers.Input(shape=(), dtype=tf.string),
        vectorizador,
        layers.Embedding(MAX_PALABRAS, 64, mask_zero=True),
        layers.Bidirectional(layers.LSTM(48, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.4),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ], name="lstm_spam")

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )
    return modelo


# ---------------------------------------------------------------
# 5. Graficas
# ---------------------------------------------------------------
def graficar_historial(hist):
    fig, (g1, g2) = plt.subplots(1, 2, figsize=(12, 4.5))

    g1.plot(hist.history["accuracy"], label="entrenamiento")
    g1.plot(hist.history["val_accuracy"], label="validacion")
    g1.set_title("Exactitud por epoca")
    g1.set_xlabel("Epoca"); g1.set_ylabel("Accuracy")
    g1.legend(); g1.grid(alpha=0.3)

    g2.plot(hist.history["loss"], label="entrenamiento")
    g2.plot(hist.history["val_loss"], label="validacion")
    g2.set_title("Perdida por epoca")
    g2.set_xlabel("Epoca"); g2.set_ylabel("Loss")
    g2.legend(); g2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_IMGS, "04_curvas_entrenamiento.png"), dpi=110)
    plt.close()


def evaluar(modelo, x_test, y_test):
    probabilidades = modelo.predict(x_test, verbose=0).flatten()
    y_pred = (probabilidades >= 0.5).astype(int)

    exactitud = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, probabilidades)

    print("\n--- Resultados sobre el conjunto de prueba ---")
    print("Accuracy:  %.4f" % exactitud)
    print("Precision: %.4f  (de los que dijo spam, cuantos si eran)" % prec)
    print("Recall:    %.4f  (de todos los spam, cuantos detecto)" % rec)
    print("F1-score:  %.4f" % f1)
    print("ROC-AUC:   %.4f" % auc)

    reporte = classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=4)
    print("\n", reporte)

    with open(os.path.join(BASE, "resultados_metricas.txt"), "w", encoding="utf-8") as f:
        f.write("Accuracy:  %.4f\n" % exactitud)
        f.write("Precision: %.4f\n" % prec)
        f.write("Recall:    %.4f\n" % rec)
        f.write("F1-score:  %.4f\n" % f1)
        f.write("ROC-AUC:   %.4f\n\n" % auc)
        f.write(reporte)

    mc = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(mc, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel("Prediccion"); plt.ylabel("Valor real")
    plt.title("Matriz de confusion")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_IMGS, "05_matriz_confusion.png"), dpi=110)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, probabilidades)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#c44e52", label="LSTM (AUC = %.4f)" % auc)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="azar")
    plt.xlabel("Falsos positivos"); plt.ylabel("Verdaderos positivos")
    plt.title("Curva ROC")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_IMGS, "06_curva_roc.png"), dpi=110)
    plt.close()

    return probabilidades, y_pred, mc


# ---------------------------------------------------------------
def main():
    df = cargar_datos()
    explorar(df)
    (x_train, y_train), (x_val, y_val), (x_test, y_test), df = preparar(df)

    modelo = construir_modelo(x_train)
    modelo.summary()

    # el dataset tiene como 6 veces mas mensajes normales que spam, entonces le
    # doy mas peso a la clase spam para que no se acostumbre a decir "ham" siempre
    n = len(y_train)
    pesos = {0: n / (2 * (n - y_train.sum())), 1: n / (2 * y_train.sum())}
    print("\nPesos por clase:", pesos)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
    ]

    historial = modelo.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCAS,
        batch_size=BATCH,
        class_weight=pesos,
        callbacks=callbacks,
        verbose=2
    )

    graficar_historial(historial)
    probabilidades, y_pred, mc = evaluar(modelo, x_test, y_test)

    # 12 predicciones de ejemplo. Saco 6 de cada clase a proposito, porque si las
    # cojo al azar casi todas salen ham (el dataset tiene mucho mas ham que spam)
    # y no se alcanza a ver como se comporta con los dos casos
    np.random.seed(5)
    idx_ham = np.random.choice(np.where(y_test == 0)[0], 6, replace=False)
    idx_spam = np.random.choice(np.where(y_test == 1)[0], 6, replace=False)
    indices = np.concatenate([idx_ham, idx_spam])

    print("\n--- Predicciones de prueba ---")
    for i, idx in enumerate(indices):
        p = probabilidades[idx]
        acerto = "ok" if (p >= 0.5) == bool(y_test[idx]) else "FALLO"
        print("\n%d) %s" % (i + 1, x_test[idx][:75]))
        print("   Real: %-5s | Predicho: %-5s | Prob. de spam: %6.2f%%  %s" %
              ("spam" if y_test[idx] else "ham",
               "spam" if p >= 0.5 else "ham", p * 100, acerto))

    # grafica con esas mismas 12 predicciones
    etiquetas = [(x_test[i][:40] + "...") for i in indices]
    valores = [probabilidades[i] * 100 for i in indices]
    colores = ["#c44e52" if y_test[i] else "#4c72b0" for i in indices]

    plt.figure(figsize=(10, 6.5))
    barras = plt.barh(range(len(indices)), valores, color=colores)
    # le pongo el numero encima porque los ham dan casi 0% y la barra no se ve
    for b, v in zip(barras, valores):
        plt.text(max(v, 0) + 1.5, b.get_y() + b.get_height() / 2,
                 "%.2f%%" % v, va="center", fontsize=8)
    plt.yticks(range(len(indices)), etiquetas, fontsize=8)
    plt.axvline(50, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Probabilidad de que sea spam (%)")
    plt.title("12 predicciones de prueba (rojo = spam real, azul = ham real)")
    plt.xlim(0, 112)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_IMGS, "07_predicciones.png"), dpi=110)
    plt.close()

    # ademas pruebo con mensajes que escribi yo, para ver si el modelo sirve con
    # textos que no salieron del dataset
    mios = [
        "Congratulations! You have won a FREE iPhone 15. Click here to claim your prize now",
        "URGENT: Your account has been suspended. Verify your details at www.secure-login.com",
        "WINNER!! You have been selected to receive a 900 prize reward. Call 09061701461 now",
        "Hey are we still meeting tomorrow at the library?",
        "Mom says dinner is ready, come home soon",
        "Can you send me the notes from yesterday class please"
    ]
    print("\n--- Prueba con mensajes escritos por mi ---")
    limpios = np.array([limpiar_texto(m) for m in mios], dtype=object)
    probs_mios = modelo.predict(limpios, verbose=0).flatten()
    for m, p in zip(mios, probs_mios):
        print("%-6s (%5.2f%%) -> %s" % ("SPAM" if p >= 0.5 else "ham", p * 100, m[:62]))

    ruta = os.path.join(CARPETA_MODELO, "modelo_spam.keras")
    modelo.save(ruta)
    with open(os.path.join(CARPETA_MODELO, "historial.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in historial.history.items()}, f, indent=2)

    print("\nModelo guardado en:", ruta)


if __name__ == "__main__":
    main()
