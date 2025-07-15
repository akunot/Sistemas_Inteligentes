from minisom import MiniSom
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from collections import Counter

# Cargar el dataset Iris
#iris = load_iris()
#data = iris.data # Caracteristicas (longitud y ancho)

# Cargamos los datos del csv
df = pd.read_csv('sobar-72.csv')
data = df.drop(columns=['ca_cervix'])  # Solo las variables de entrada

# Prepocesamiento de los datos
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Definir el som
som = MiniSom(x=2, y=1, input_len=19, sigma=0.3, learning_rate=0.5) # x y son los tamaños de la red

# Inicializar los pesos
som.random_weights_init(data)

# Entrenar el som
som.train(data, 280)

# Obtener la asignacion de clases por cada uno de los datos
class_assignment = np.array([som.winner(x) for x in data])

# Imprimir asignacion de clases para cada instancia
for i, instance in enumerate(data):
    cluster = class_assignment[i]
    print(f'Instancia {i}: Clase {cluster}')

cluster_ids = [tuple(x) for x in class_assignment]  # Convertir a tuplas para conteo
print("Distribución de datos por nodo:")
print(Counter(cluster_ids))

# Comparar con las clases verdaderas
print("\nDistribución real (ca_cervix):")
print(Counter(df['ca_cervix']))

labels = df['ca_cervix'].values  # 0 o 1
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(data):
    w = som.winner(x)
    label = int(labels[i])  # 0 o 1
    plt.plot(w[0] + 0.5, 
             w[1] + 0.5, 
             markers[label],
             markeredgecolor=colors[label],
             markerfacecolor='None',
             markersize=10,
             markeredgewidth=2)
    
plt.title('Clusters de flores Iris usando SOM')
plt.xlabel('Dimension X')
plt.ylabel('Dimension Y')
plt.show()

