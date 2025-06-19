# Definicion de una funcion de activacion simple
def step_function(x, umbral):
    return 1 if x >= umbral else 0

# Implementacion de una neurona de McCulloch-Pitts
def mcculloch_pitts_neuron(inputs, weights, b, umbral):
    weighted_sum = sum(i * w for i, w in zip(inputs, weights)) + b
    output = step_function(weighted_sum, umbral)
    return output

# Entradas para la neurona
inputs_list = [[1, 0], [0, 1], [1, 1], [0, 0]]
# Pesos para las entradas
weights_list = [[1, 1], [0, 0]]
# Termino aditivo
b = 0
# Umbral para la funcion de activacion
umbral = 2

for inputs in inputs_list:
    for weights in weights_list:
        output = mcculloch_pitts_neuron(inputs, weights, b, umbral)
        print(f"Entradas: {inputs}, Pesos: {weights}, Salida: {output}")
