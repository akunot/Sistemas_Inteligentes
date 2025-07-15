import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Activación tanh
def tanh(x, deriv=False):
    if deriv:
        return 1.0 - np.tanh(x) ** 2
    return np.tanh(x)

# Cargar datos
data = pd.read_csv("datosMLP.csv", sep=";", decimal=",", encoding="utf-8")

# Extraer entradas y salidas
X = data[['X']].values
y = data[['Yreal']].values

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar pesos con distribución normal pequeña
np.random.seed(0)
syn0 = np.random.randn(1, 20) * 0.1
syn1 = np.random.randn(20, 1) * 0.1

eta = 0.01  # tasa de aprendizaje
errors = []

# Entrenamiento
for iter in range(5000):
    # Propagación hacia adelante
    neta1 = np.dot(X_train, syn0)       # (n,1) x (1,5) → (n,5)
    l1 = tanh(neta1)                    # capa oculta
    neta2 = np.dot(l1, syn1)            # (n,5) x (5,1) → (n,1)
    l2 = tanh(neta2)                    # salida final

    # Cálculo del error
    l2_error = y_train - l2
    errors.append(np.mean(np.abs(l2_error)))

    # Mostrar error cada 1000 iteraciones
    if iter % 1000 == 0:
        print(f"Iter {iter} - Error MAE: {errors[-1]}")

    # Retropropagación
    l2_delta = l2_error * tanh(neta2, deriv=True) * eta
    l1_error = l2_delta.dot(syn1.T)                 # (n,1) x (1,5) → (n,5)
    l1_delta = l1_error * tanh(neta1, deriv=True) * eta

    # Actualizar pesos
    syn1 += l1.T.dot(l2_delta)
    syn0 += X_train.T.dot(l1_delta)

ultimos = errors[-3:]
print(ultimos)

# Evaluación en el conjunto de prueba
neta1_test = np.dot(X_test, syn0)
l1_test = tanh(neta1_test)
neta2_test = np.dot(l1_test, syn1)
l2_test = tanh(neta2_test)

test_error = np.mean(np.abs(y_test - l2_test))

print("\n*********************************")
print("Salida de la red en datos de prueba:\n", l2_test)
print("Y esperado:\n", y_test)
print("Error de prueba:", test_error)
print("Pesos finales:")
print("syn0:\n", syn0)
print("syn1:\n", syn1)

# 🔍 Comparación visual entre valores reales y predichos
sorted_idx = X_test[:, 0].argsort()
plt.plot(X_test[sorted_idx], y_test[sorted_idx], label="Real", marker='o')
plt.plot(X_test[sorted_idx], l2_test[sorted_idx], label="Predicho", marker='x')
plt.legend()
plt.title("Comparación de salida real vs predicha")
plt.grid(True)
plt.show()