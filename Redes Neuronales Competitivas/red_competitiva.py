from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

# Definir los datos
data = np.array([[1.2, 0.7, 0.3], [0.7, 0.6, 0.8], [1.0, 0.9, 0.5], [0.7, 0.8, 1.0]])

# Definir el som
som = MiniSom(x=3, y=3, input_len=3, sigma=1.0, learning_rate=0.5) # x y son los tamaños de la red

# Inicializar los pesos
som.random_weights_init(data)

# Entrenar el som
som.train(data, 100)

# Obtener la asignacion de clases por cada uno de los datos
class_assignment = np.array([som.winner(x) for x in data])

# Mostrar la asignacion de clases
print(class_assignment)

# Visualiza los cluster
plt.figure(figsize=(8, 8))

# Marcadores para cada especie
markers= ['o', 's', 'D', '^']
colors = ['r', 'g', 'b', 'm']

for i, x in enumerate(data):
    w= som.winner(x) # Obtiene el nodo ganador para el punto de los datos
    # Graficar el punto de los datos en la posicion del nodo ganador
    plt.plot(w[0] + 0.5, # Suma 0.5 para centrar el marcador 
             w[1] + 0.5, 
             markers[i],
             markeredgecolor= colors[i],
             markerfacecolor= 'None',
             markersize=10,
             markeredgewidth=2)
    
plt.title('Clusters de flores Iris usando SOM')
plt.xlabel('Dimension X')
plt.ylabel('Dimension Y')
plt.show()