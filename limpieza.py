"""
Aca dejo la funcion de limpieza aparte porque la necesito en dos lados:
cuando entreno y cuando hago predicciones en la app. Si la copio y pego en los
dos archivos despues se me olvida cambiarla en uno y el modelo empieza a fallar.
"""

import re


def limpiar_texto(texto):
    texto = str(texto).lower()

    # los links los cambio por una palabra, porque la direccion exacta no importa
    # pero el hecho de que el mensaje traiga un link si es una señal de spam
    texto = re.sub(r"http\S+|www\.\S+", " link ", texto)

    # lo mismo con los correos
    texto = re.sub(r"\S+@\S+", " correo ", texto)

    # los numeros largos casi siempre son telefonos o codigos de premios
    texto = re.sub(r"\b\d{5,}\b", " numerolargo ", texto)
    texto = re.sub(r"\b\d+\b", " numero ", texto)

    # el simbolo de libra aparece harto porque los mensajes son de Reino Unido
    texto = texto.replace("£", " libras ")
    texto = texto.replace("$", " dolares ")

    # quito todo lo que no sea letra o espacio
    texto = re.sub(r"[^a-z\s]", " ", texto)

    # dejo un solo espacio entre palabras
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto
