# -*- coding: utf-8 -*-
"""
Práctica 2.3 - Teachable Machine
Completa la clasificación de tres imágenes:
1. Una imagen clara que pertenezca a una categoría.
2. Una imagen ambigua.
3. Una imagen que no pertenezca a ninguna categoría.

Antes de ejecutar:
- Exporta el modelo de Teachable Machine en formato TensorFlow (Keras).
- Coloca keras_model.h5 y labels.txt en la misma carpeta que este archivo.
- Guarda tus tres imágenes en esa carpeta con estos nombres:
    imagen_clara.jpg
    imagen_ambigua.jpg
    imagen_fuera_categoria.jpg
"""

import os

# Debe establecerse antes de importar TensorFlow/Keras.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import tf_keras as keras
from PIL import Image, ImageOps


# ---------------- CONFIGURACIÓN ----------------

RUTA_MODELO = "keras_model.h5"
RUTA_ETIQUETAS = "labels.txt"

IMAGENES = {
    "Imagen clara": "imagen_clara.jpg",
    "Imagen ambigua": "imagen_ambigua.jpg",
    "Imagen fuera de categoría": "imagen_fuera_categoria.jpg",
}


# ---------------- CARGA DEL MODELO ----------------

def cargar_modelo():
    """Carga el modelo exportado desde Teachable Machine y sus etiquetas."""
    if not os.path.exists(RUTA_MODELO):
        raise FileNotFoundError(
            f"No se encuentra {RUTA_MODELO}. "
            "Copia el modelo exportado a esta carpeta."
        )

    if not os.path.exists(RUTA_ETIQUETAS):
        raise FileNotFoundError(
            f"No se encuentra {RUTA_ETIQUETAS}. "
            "Copia el archivo de etiquetas a esta carpeta."
        )

    modelo = keras.models.load_model(RUTA_MODELO, compile=False)

    with open(RUTA_ETIQUETAS, "r", encoding="utf-8") as archivo:
        etiquetas = [linea.strip() for linea in archivo.readlines()]

    print("Modelo cargado correctamente.")
    print("Versión de TensorFlow:", tf.__version__)
    print("Versión de TF-Keras:", keras.__version__)
    return modelo, etiquetas


# ---------------- CLASIFICACIÓN DE IMÁGENES ----------------

def clasificar_imagen(modelo, etiquetas, ruta_imagen):
    """
    Prepara una imagen con el mismo formato esperado por el modelo
    y devuelve la categoría predicha y su nivel de confianza.
    """
    if not os.path.exists(ruta_imagen):
        print(f"No se encuentra la imagen: {ruta_imagen}")
        return None

    # Abrir la imagen y convertirla a RGB.
    imagen = Image.open(ruta_imagen).convert("RGB")

    # Redimensionar y recortar al tamaño de entrada del modelo.
    imagen = ImageOps.fit(
        imagen,
        (224, 224),
        method=Image.Resampling.LANCZOS
    )

    # Convertir a un array y normalizar los píxeles entre -1 y 1.
    array_imagen = np.asarray(imagen, dtype=np.float32)
    imagen_normalizada = (array_imagen / 127.5) - 1

    # El modelo espera una tanda de imágenes: (1, 224, 224, 3).
    datos = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    datos[0] = imagen_normalizada

    # Obtener predicción y elegir la categoría con mayor puntuación.
    prediccion = modelo.predict(datos, verbose=0)
    indice = int(np.argmax(prediccion[0]))
    confianza = float(prediccion[0][indice])

    # Las etiquetas exportadas suelen tener el formato "0 nombre".
    nombre_clase = etiquetas[indice]
    partes = nombre_clase.split(" ", 1)
    if len(partes) == 2 and partes[0].isdigit():
        nombre_clase = partes[1]

    resultado = {
        "clase": nombre_clase.strip(),
        "confianza": confianza,
    }

    print(f"Imagen: {ruta_imagen}")
    print(f"Clase predicha: {resultado['clase']}")
    print(f"Confianza: {resultado['confianza']:.2%}")
    print("-" * 45)

    return resultado


# ---------------- EJECUCIÓN DE LOS TRES EJERCICIOS ----------------

def main():
    modelo, etiquetas = cargar_modelo()

    resultados = {}

    for tipo, ruta in IMAGENES.items():
        print(f"\n--- {tipo} ---")
        resultados[tipo] = clasificar_imagen(modelo, etiquetas, ruta)

    print("\nRESUMEN DE RESULTADOS")
    print("=" * 60)

    for tipo, resultado in resultados.items():
        if resultado is None:
            print(f"{tipo}: no se pudo analizar (falta el archivo).")
        else:
            print(
                f"{tipo}: {resultado['clase']} "
                f"({resultado['confianza']:.2%} de confianza)"
            )


if __name__ == "__main__":
    main()


# ---------------- RESPUESTA A LA PREGUNTA TEÓRICA ----------------
#
# ¿Cómo se usa el modelo exportado?
#
# Primero se exporta desde Teachable Machine en formato TensorFlow/Keras.
# Después se cargan el archivo keras_model.h5 y las etiquetas labels.txt.
# Cada imagen se convierte a RGB, se adapta a 224 x 224 píxeles y se
# normalizan sus valores para que estén entre -1 y 1. A continuación,
# se introduce en el modelo mediante model.predict(). La función devuelve
# las puntuaciones de las categorías. Con argmax se selecciona la categoría
# que obtiene la puntuación más alta.
#
# ¿Cómo se comporta?
#
# En una imagen clara, que se parece a los ejemplos usados para entrenar
# una categoría, normalmente se espera que el modelo prediga esa categoría
# con una confianza alta.
#
# En una imagen ambigua, el modelo puede dudar y repartir la confianza
# entre varias categorías, aunque siempre selecciona la que obtiene la
# puntuación más alta.
#
# En una imagen que no pertenece a ninguna categoría, el modelo también
# selecciona obligatoriamente una de las categorías conocidas. Por tanto,
# una predicción no significa que la imagen pertenezca realmente a ella.
# Hay que observar la confianza y comprobar visualmente el resultado.
#
# IMPORTANTE: las predicciones y porcentajes concretos deben copiarse de
# la ejecución con vuestro propio modelo y vuestras tres imágenes. No se
# pueden conocer sin esos archivos.
