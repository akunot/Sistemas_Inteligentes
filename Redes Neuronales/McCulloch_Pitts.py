# Definicion de una funcion de activacion simple
def step_function(x, umbral):
    return 1 if x >= umbral else 0

# Implementacion de una neurona de McCulloch-Pitts
def mcculloch_pitts_neuron(inputs, weights, b, umbral):
    # Calculo de la suma ponderada de las entradas
    weighted_sum = sum(i * w for i, w in zip(inputs, weights)) + b
    # Aplicacion de la funcion de activacion
    output = step_function(weighted_sum, umbral)
    return output

# Entradas para la neurona
inputs = [0, 1]
# Pesos para las entradas
weights = [0.5, 0.5]
# Termino aditivo
b = 0.5
# Umbral para la funcion de activacion
umbral = 0.6

# Calcular la salida de la neurona de McCoulloch-Pitts
output = mcculloch_pitts_neuron(inputs, weights, b, umbral)
print(f"Entradas: {inputs}")
print(f"Pesos: {weights}")
print(f"Umbral: {umbral}")
print(f"Salida de la neurona: {output}")