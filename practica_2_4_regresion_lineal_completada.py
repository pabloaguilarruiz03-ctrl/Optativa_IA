# -*- coding: utf-8 -*-
"""
PRÁCTICA 2.4 - REGRESIÓN LINEAL

Antes de ejecutar:
    pip install numpy pandas matplotlib scikit-learn

El programa descarga el conjunto de datos de Zenodo cuando se ejecuta.
Se necesita conexión a Internet para esa parte.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ============================================================
# PARTE 1. REGRESIÓN LINEAL CON DATOS SIMULADOS
# ============================================================

print("\n" + "=" * 65)
print("PARTE 1. REGRESIÓN LINEAL CON DATOS SIMULADOS")
print("=" * 65)

np.random.seed(238)
X = np.linspace(0, 10, 50)
y = 2 * X + 1 + np.random.randn(50) * 2

plt.figure(figsize=(8, 5))
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Datos simulados")
plt.grid(True, alpha=0.3)
plt.show()

print("Tipo de X:", type(X))
print("Tipo de y:", type(y))
print("Primeros 10 valores de X:", X[:10])
print("Primeros 10 valores de y:", y[:10])


def entrenar_regresion_simple(X_datos, y_datos, learning_rate, epochs):
    """Entrena y = a*x + b mediante descenso de gradiente."""
    a, b = 0.0, 0.0
    historial = []

    for _ in range(epochs):
        y_pred = a * X_datos + b
        da = -2 * np.mean(X_datos * (y_datos - y_pred))
        db = -2 * np.mean(y_datos - y_pred)
        a -= learning_rate * da
        b -= learning_rate * db

        mse = np.mean((a * X_datos + b - y_datos) ** 2)
        historial.append(mse)

        if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(mse):
            break

    return a, b, historial


# Se prueban los valores de learning rate y epochs propuestos en la práctica.
print("\nComparación de hiperparámetros (datos simulados):")
for lr in [0.1, 0.5, 0.0001, 0.001]:
    for num_epochs in [10, 50, 100]:
        a_prueba, b_prueba, historial = entrenar_regresion_simple(
            X, y, lr, num_epochs
        )
        if historial and np.isfinite(historial[-1]):
            print(
                f"learning_rate={lr:<8} epochs={num_epochs:<3} "
                f"a={a_prueba:.4f} b={b_prueba:.4f} "
                f"MSE={historial[-1]:.4f}"
            )
        else:
            print(
                f"learning_rate={lr:<8} epochs={num_epochs:<3} "
                "No converge (los valores se disparan)."
            )

# Configuración elegida para mostrar el ajuste.
learning_rate = 0.001
epochs = 100
a, b, historial = entrenar_regresion_simple(X, y, learning_rate, epochs)

print(f"\nRecta aprendida con datos simulados: y = {a:.2f} * x + {b:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Datos reales")
plt.plot(X, a * X + b, color="red", label=f"Ajuste: y={a:.2f}x+{b:.2f}")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Regresión lineal con datos simulados")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

mse_simulado = np.mean((a * X + b - y) ** 2)
mae_simulado = np.mean(np.abs(a * X + b - y))
print(f"Error cuadrático medio: {mse_simulado:.2f}")
print(f"Error absoluto medio: {mae_simulado:.2f}")


# ============================================================
# PARTE 2. DATASET REAL DE VIVIENDAS EN ESPAÑA
# ============================================================

print("\n" + "=" * 65)
print("PARTE 2. REGRESIÓN LINEAL CON DATOS REALES")
print("=" * 65)

url = "https://zenodo.org/records/14028180/files/properties_Spain.csv"
print("Descargando el conjunto de datos...")
df = pd.read_csv(url)

print("\nPrimeros registros:")
print(df.head())

# Limpieza de datos siguiendo las indicaciones de la práctica.
df = df.dropna(subset=["Sale_Price", "Surface"])
df = df[df["Contrat"] == "venta"]
df = df[(df["Sale_Price"] > 10000) & (df["Sale_Price"] < 2000000)]
df = df[(df["Surface"] > 20) & (df["Surface"] < 500)]

print("\nNúmero de viviendas después de la limpieza:", len(df))
print(df[["Sale_Price", "Surface", "Bedrooms", "Bathrooms"]].head())
print("\nEstadísticas:")
print(df[["Sale_Price", "Surface"]].describe())

plt.figure(figsize=(8, 5))
plt.scatter(df["Surface"], df["Sale_Price"], alpha=0.4, s=15)
plt.xlabel("Superficie (m²)")
plt.ylabel("Precio de venta (€)")
plt.title("Precio por superficie — datos reales")
plt.grid(True, alpha=0.3)
plt.show()

# Mostrar tipos y dimensiones, como pide el guion.
print("\nInformación de Surface:")
print("Tipo de la columna:", type(df["Surface"]))
print("Tipo del array:", type(df["Surface"].values))
print("Tipo del array bidimensional:", type(df["Surface"].values.reshape(-1, 1)))
print("Primeros valores:", df["Surface"].values[:5])
print("Primeros valores en 2D:", df["Surface"].values.reshape(-1, 1)[:5])
print("Dimensiones del array:", df["Surface"].values.ndim)
print("Forma del array:", df["Surface"].values.shape)
print("Tamaño del array:", df["Surface"].values.size)
print("Dimensiones del array 2D:", df["Surface"].values.reshape(-1, 1).ndim)
print("Forma del array 2D:", df["Surface"].values.reshape(-1, 1).shape)
print("Tamaño del array 2D:", df["Surface"].values.reshape(-1, 1).size)

# X contiene la superficie (entrada) e y contiene el precio (salida).
X = df["Surface"].values.reshape(-1, 1)
y = df["Sale_Price"].values

# 80 % para entrenamiento y 20 % para prueba.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nEntrenamiento: {len(X_train)} viviendas")
print(f"Prueba: {len(X_test)} viviendas")


# ============================================================
# PARTE 3. ENTRENAMIENTO Y COMPARACIÓN DE HIPERPARÁMETROS
# ============================================================

def entrenar_con_escalado(X_entrenamiento, y_entrenamiento, learning_rate, epochs):
    """
    Descenso de gradiente con datos estandarizados.
    El escalado evita que los valores grandes de superficie y precio
    hagan que el algoritmo diverja con facilidad.
    Devuelve la pendiente y la ordenada en unidades originales.
    """
    x_media = np.mean(X_entrenamiento)
    x_std = np.std(X_entrenamiento)
    y_media = np.mean(y_entrenamiento)
    y_std = np.std(y_entrenamiento)

    if x_std == 0:
        x_std = 1.0
    if y_std == 0:
        y_std = 1.0

    x_norm = (X_entrenamiento.flatten() - x_media) / x_std
    y_norm = (y_entrenamiento - y_media) / y_std

    a_norm, b_norm = 0.0, 0.0
    historial = []

    for epoch in range(epochs):
        pred = a_norm * x_norm + b_norm
        error = pred - y_norm

        da = (2 / len(x_norm)) * np.sum(error * x_norm)
        db = (2 / len(x_norm)) * np.sum(error)

        a_norm -= learning_rate * da
        b_norm -= learning_rate * db

        mse = np.mean((a_norm * x_norm + b_norm - y_norm) ** 2)
        historial.append(mse)

        if not np.isfinite(mse):
            break

    # Convertir los parámetros a la escala original (euros y m²).
    a_original = (a_norm * y_std) / x_std
    b_original = y_media + (b_norm * y_std) - a_original * x_media

    return a_original, b_original, historial


# Se prueban los learning rate y epochs indicados en el documento.
configuraciones = [
    (0.1, 5), (0.1, 20), (0.1, 100),
    (0.000001, 5), (0.000001, 20), (0.000001, 100),
    (0.00000001, 5), (0.00000001, 20), (0.00000001, 100),
    (0.01, 100), (0.05, 100)
]

resultados = []
print("\nComparación de hiperparámetros:")
for lr, num_epochs in configuraciones:
    a_test, b_test, historial = entrenar_con_escalado(
        X_train, y_train, lr, num_epochs
    )
    pred_test = a_test * X_test.flatten() + b_test
    mse_test = np.mean((pred_test - y_test) ** 2)

    resultados.append({
        "learning_rate": lr,
        "epochs": num_epochs,
        "a": a_test,
        "b": b_test,
        "MSE_prueba": mse_test,
    })

    print(
        f"LR={lr:g} | épocas={num_epochs:3d} | "
        f"a={a_test:.2f} | b={b_test:.2f} | "
        f"MSE prueba={mse_test:,.2f}"
    )

# Elegir la configuración con menor error de prueba entre las probadas.
mejor = min(resultados, key=lambda r: r["MSE_prueba"])
learning_rate = mejor["learning_rate"]
epochs = mejor["epochs"]
a = mejor["a"]
b = mejor["b"]

print("\nMejor configuración entre las probadas:")
print(f"Learning rate: {learning_rate}")
print(f"Épocas: {epochs}")
print(f"Pendiente a: {a:.2f}")
print(f"Ordenada b: {b:.2f}")
print(f"Recta aprendida: precio = {a:.2f} × superficie + {b:.2f}")
print(f"MSE de prueba: {mejor['MSE_prueba']:,.2f} €²")

# Representación de entrenamiento, prueba y recta aprendida.
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.3, s=15, label="Datos de entrenamiento")
plt.scatter(X_test, y_test, alpha=0.3, color="orange", s=15, label="Datos de prueba")

x_line = np.linspace(X.min(), X.max(), 100)
y_line = a * x_line + b
plt.plot(
    x_line, y_line, color="red", linewidth=2,
    label=f"Recta: y = {a:.2f}x + {b:.2f}"
)
plt.xlabel("Superficie (m²)")
plt.ylabel("Precio (€)")
plt.title(f"Regresión lineal — LR={learning_rate}, epochs={epochs}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# PARTE 4. EVALUACIÓN FINAL CON DATOS DE PRUEBA
# ============================================================

y_test_pred = a * X_test.flatten() + b
mse_test = np.mean((y_test_pred - y_test) ** 2)
mae_test = np.mean(np.abs(y_test_pred - y_test))

print("\nEvaluación final:")
print(f"Error cuadrático medio en PRUEBA: {mse_test:,.0f} €²")
print(f"Error absoluto medio en PRUEBA: {mae_test:,.0f} €")


# ============================================================
# RESPUESTAS A LAS PREGUNTAS DE LA PRÁCTICA
# ============================================================

print("\n" + "=" * 65)
print("RESPUESTAS TEÓRICAS")
print("=" * 65)

print("""
1. ¿Cómo funciona el algoritmo usando diferentes hiperparámetros?
El learning rate indica cuánto se modifican los parámetros en cada paso.
Si es demasiado pequeño, el aprendizaje es muy lento. Si es demasiado
grande, puede sobrepasar el mínimo del error y no converger. Las epochs
indican cuántas veces se repite el entrenamiento. Con pocas épocas puede
quedarse corto; al aumentar las épocas, el error suele disminuir hasta
estabilizarse.

2. ¿Con qué hiperparámetros se obtiene el mejor modelo?
Depende de los datos. En esta práctica se comparan varias combinaciones
y se selecciona la que obtiene el menor error cuadrático medio (MSE)
en el conjunto de prueba. El programa muestra los valores concretos
obtenidos al ejecutarlo.

3. ¿Hay algún número de épocas a partir del cual el modelo ya no aprende?
Sí. Llega un momento en el que el error apenas disminuye y los parámetros
cambian muy poco. A partir de ahí, aumentar las épocas aporta una mejora
mínima. El número exacto depende del learning rate y de los datos.

4. ¿Por qué predecir el precio solo con la superficie no es suficiente?
Porque el precio también depende de otros factores, como la ciudad,
la ubicación, la planta, el año de construcción, el número de habitaciones
y los baños. Para tener en cuenta varias características se puede utilizar
una regresión lineal múltiple.
""")

print("Práctica finalizada.")
