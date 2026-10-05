# Detector de mensajes spam con una red LSTM

Proyecto de Deep Learning para la materia de Inteligencia Artificial.
Es una red recurrente (LSTM) que lee un mensaje de texto y decide si es spam o si es un
mensaje normal. Tambien tiene una aplicacion en Streamlit donde uno escribe un mensaje y
ve el resultado al instante con la probabilidad.

**Estudiante:** David Tellez
**Programa:** Ingenieria de Software - Noveno semestre

---

## El dataset

Use el **SMS Spam Collection**, un dataset publico de la UCI con mensajes de texto reales
recolectados en Reino Unido. Cada mensaje viene etiquetado como `ham` (normal) o `spam`.

- 5.572 mensajes en total
- Despues de quitar duplicados y mensajes vacios quedan 5.095
- Esta desbalanceado: como el 87% son normales y solo el 13% son spam

Escogi un problema de texto porque las redes recurrentes estan hechas justamente para datos
secuenciales. Un mensaje es una secuencia de palabras donde el orden importa, no es lo mismo
"free call me" que "call me free".

El dataset lo deje dentro del repo en `datos/sms.tsv`, pero el codigo tambien lo puede bajar
solo desde GitHub si el archivo no esta, asi funciona igual en Colab.

## Resultados

| Metrica | Valor |
|---|---|
| Accuracy | 98.04% |
| Precision (spam) | 0.9213 |
| Recall (spam) | 0.9111 |
| F1-score (spam) | 0.9162 |
| ROC-AUC | 0.9747 |

El detalle completo esta en `resultados_metricas.txt`.

Para este problema el accuracy solo no dice mucho, porque como el 87% de los mensajes son
normales, un modelo que dijera siempre "ham" ya sacaria 87% sin haber aprendido nada. Por eso
lo importante son la precision y el recall de la clase spam.

---

## Como esta organizado el repositorio

```
detector-spam-lstm/
├── app.py                      <- la aplicacion de Streamlit
├── limpieza.py                 <- la funcion de limpieza del texto
├── requirements.txt
├── README.md
├── resultados_metricas.txt
├── datos/
│   └── sms.tsv                 <- el dataset
├── notebooks/
│   └── Proyecto_LSTM_Spam.ipynb    <- el notebook de Colab con todo el proceso
├── src/
│   ├── entrenamiento.py        <- el mismo codigo del notebook pero como script
│   └── predecir.py             <- predicciones desde la consola
├── modelo/
│   ├── modelo_spam.keras       <- el modelo ya entrenado
│   └── historial.json
└── imagenes/                   <- las graficas que genera el entrenamiento
    ├── 01_distribucion_clases.png
    ├── 02_largo_mensajes.png
    ├── 03_palabras_frecuentes.png
    ├── 04_curvas_entrenamiento.png
    ├── 05_matriz_confusion.png
    ├── 06_curva_roc.png
    └── 07_predicciones.png
```

La funcion de limpieza la deje en un archivo aparte (`limpieza.py`) a proposito, porque la
necesito en dos lados: cuando entreno y cuando hago predicciones en la app. Si la copiaba en
los dos archivos, el dia que cambiara algo en uno se me olvidaba en el otro y el modelo
empezaba a predecir mal sin que me diera cuenta.

---

## Como ejecutarlo

### 1. Clonar e instalar

```bash
git clone https://github.com/DavidCaicedo0777/detector-spam-lstm.git
cd detector-spam-lstm
pip install -r requirements.txt
```

Recomiendo un entorno virtual:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux o Mac
pip install -r requirements.txt
```

### 2. Correr la aplicacion

El modelo ya viene entrenado en el repo, asi que no hay que entrenar nada:

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`. Uno escribe un mensaje (o escoge uno de los ejemplos que
deje en el desplegable), le da a "Analizar mensaje" y la app muestra si es spam con su
probabilidad.

En la barra lateral hay un control para mover el **umbral**. Por defecto esta en 0.50, pero si
uno lo sube el modelo se vuelve mas exigente para marcar algo como spam y bajan los falsos
positivos; si lo baja detecta mas spam pero tambien se come mas mensajes buenos.

> **Importante:** el modelo fue entrenado con mensajes en ingles, entonces con mensajes en
> español no va a funcionar bien.

### 3. Volver a entrenar (opcional)

```bash
python src/entrenamiento.py
```

Entrena la red, genera las graficas en `imagenes/` y guarda el modelo. En un computador sin
GPU se demora como 2 minutos, porque el dataset es pequeño y el EarlyStopping corta solo
despues de unas pocas epocas.

### 4. Predecir desde la consola

```bash
python src/predecir.py "You have won a free prize, call now"
python src/predecir.py                  # sin argumentos corre 10 ejemplos de prueba
```

### 5. Abrir el notebook

`notebooks/Proyecto_LSTM_Spam.ipynb` se abre directo en Google Colab (Archivo > Abrir cuaderno
> GitHub y se pega la URL del repo) o localmente con Jupyter. Ya trae todas las salidas y
graficas guardadas, entonces se puede revisar sin ejecutarlo.

---

## Como funciona

### Limpieza del texto

Mirando los mensajes se nota que el spam esta lleno de numeros de telefono, precios en libras
y links. En vez de borrar todo eso, lo **reemplazo por una palabra clave**: la direccion exacta
del link no dice nada, pero el hecho de que el mensaje traiga un link si es una señal fuerte.

| Lo que encuentra | Lo reemplaza por |
|---|---|
| `http://...` o `www...` | `link` |
| correos electronicos | `correo` |
| numeros de 5 cifras o mas | `numerolargo` |
| numeros cortos | `numero` |
| `£` y `$` | `libras` / `dolares` |

Despues pasa todo a minuscula y quita los signos de puntuacion.

### La arquitectura

```
Texto crudo
   |
TextVectorization  (vocabulario de 8000 palabras, mensajes de 60 palabras)
   |
Embedding (64 dimensiones)
   |
Bidirectional LSTM (48)  -> Dropout(0.3)
   |
Bidirectional LSTM (32)  -> Dropout(0.4)
   |
Dense(32) -> Dropout(0.3) -> Dense(1, sigmoid)
```

| | |
|---|---|
| Optimizador | Adam (learning rate 0.001) |
| Funcion de perdida | binary_crossentropy |
| Batch size | 32 |
| Epocas | hasta 15, con EarlyStopping |
| Division | 70% entrenamiento / 15% validacion / 15% prueba |
| Desbalance | se maneja con `class_weight` |

Use LSTM y no una RNN simple porque la RNN normal se le olvida lo que leyo al principio cuando
el texto se alarga. Y las puse bidireccionales porque en un mensaje el contexto importa en los
dos sentidos.

La capa `TextVectorization` la meti **dentro del modelo** en lugar de usarla por fuera. Asi
cuando guardo el modelo el vocabulario queda guardado ahi mismo, y la app solo tiene que
pasarle el texto. Si la usara por fuera tocaria guardar el vocabulario en otro archivo y
acordarse de cargarlo igualito, y si se desincroniza el modelo predice cualquier cosa.

---

## Cosas que aprendi / problemas que tuve

- Lo que mas mejoro el modelo no fue la arquitectura sino la **limpieza**. Cuando empece a
  cambiar los numeros y los links por palabras clave en vez de borrarlos, el resultado subio,
  porque esas marcas son justo las señales mas fuertes del spam.
- Al principio no usaba `class_weight` y el recall de spam quedaba mas bajito, porque a la red
  le salia barato decir "ham" siempre (como el 87% de los mensajes lo son).
- Me demore un rato con un error raro de dtype al pasarle el texto a Keras. Resulta que los
  arreglos de texto normales de numpy no los recibe, toca convertirlos con `dtype=object`.
- Probando con mensajes que escribi yo note que el modelo detecta muy bien el spam de premios
  y sorteos, pero el **phishing** (del tipo "su cuenta fue suspendida, verifique sus datos") a
  veces se le escapa. Es logico, el dataset casi no trae ese tipo de mensajes porque es de la
  epoca en que el spam por SMS era mas que todo publicidad y concursos.
