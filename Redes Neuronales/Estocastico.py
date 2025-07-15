import numpy as np

# Función de activación escalón
def step(x):
    return 1 if x >= 0 else 0

# Datos de entrada: cada fila = [bias=1, x1, x2]
X = np.array([
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 1],
    [1, 0, 0]
])

# Salidas esperadas
y = np.array([1, 1, 0, 1])

# Pesos iniciales (aleatorios o fijos)
w = np.random.rand(3)

# Tasa de aprendizaje
eta = 0.1

# Número de épocas
epocas = 50

for epoch in range(epocas):
    errores = 0
    print(f"\nÉpoca {epoch + 1}")
    for i in range(len(X)):
        entrada = X[i]
        salida_real = y[i]
        neta = np.dot(entrada, w)
        salida_predicha = step(neta)
        error = salida_real - salida_predicha

        # Actualización de pesos
        w += eta * error * entrada
        errores += abs(error)

        print(f"Entrada: {entrada[1:]}, Salida esperada: {salida_real}, Salida red: {salida_predicha}, Error: {error}")
        print(f"  Pesos actualizados: {w}")

    if errores == 0:
        print("\n✔️ ¡Aprendizaje completo en la época", epoch + 1, "!")
        break

print("\nPesos finales encontrados:")
print(f"Bias (w1): {w[0]}")
print(f"W2 (x1):  {w[1]}")
print(f"W3 (x2):  {w[2]}")

# Verificación final
print("\n🔍 Verificación con pesos finales:")
for i in range(len(X)):
    neta = np.dot(X[i], w)
    salida = step(neta)
    print(f"Entrada: {X[i][1:]}, Salida predicha: {salida}")
