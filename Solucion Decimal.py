## CREADO NDD Sept 2019
import random
import numpy as np
import math as m
import matplotlib.pyplot as plt

mejores_fitness = []


"""   Comentarios son Una Linea: #
O triple comilla doble: Un bloque"""

"""Si se desea una población inicial no aleatoria
cromosoma1 = [1, 0, 0, 0, 1]
cromosoma2 = [0, 1, 0, 0, 0]
cromosoma3 = [1, 1, 0, 0, 1]
cromosoma4 = [1, 1, 1, 0, 1]
poblInicial = np.array([cromosoma1, cromosoma2, cromosoma3, cromosoma4]
"""

# MEJORA: Tamaño de la Población como parametro 
#random.seed(1)
#print("\n","aletorio:", random.randrange(2)) #Entero 0 o 1

##### FUNCIONES PARA OPERADORES


def evalua(n, poblIt):
  fitness = np.zeros(n)
  total = 0
  xi = np.empty(n)
  cromosoma = np.empty(n, dtype=int)

  for i in range(n):
    global Xmin
    global Xmax
    global lind
    cromosoma[i] = int("".join(str(bit) for bit in poblIt[i]))
    binary_str = "".join(str(bit) for bit in poblIt[i])
    bin2dec = int(binary_str, 2)
    xi[i] = Xmin+bin2dec*(Xmax-Xmin)/(2**lind-1)

     # Penalización por muerte si está en (3, 4). Se usa en caso de que el valor de xi sea inválido
    if 3 < xi[i] < 4:
      fitness[i] = -np.inf  # Valor inválido, lo eliminaremos luego
    else:
      fitness[i] = (32 * xi[i] - 2 * xi[i] ** 2)
      total += fitness[i]

  return fitness, total, xi, cromosoma

def imprime(total,fitness,poblIt,xi,cromosoma):
    global mejores_fitness
    acumula = 0
    print("\nTabla Iteración:")
    print("{:^10} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15}".format(
        "Individuo", "Población", "Cromosoma", "Xi", "Fitness", "Probabilidad", "Acumulado"))

    for i in range(len(poblIt)):
        if fitness[i] == -np.inf or total == 0:
            probab = 0
        else:
            probab = fitness[i] / total
        acumula += probab
        print("{:^10} {:^15} {:^15} {:^15.3f} {:^15} {:^15.3f} {:^15.3f}".format(
            i + 1, str(poblIt[i]), str(cromosoma[i]), xi[i],
            "-" if fitness[i] == -np.inf else f"{fitness[i]:.3f}",
            probab, acumula))

    print("Suma Z:      ", total)
    best = float(np.max(fitness[fitness != -np.inf])) if np.any(fitness != -np.inf) else 0
    mejores_fitness.append(best)
def seleccion(acumulado):
  escoje = np.random.rand()
  print("escoje:      ", escoje)
  
  for i in range(len(acumulado)):
    if acumulado[i] > escoje:
      return poblIt[i]
  return poblIt[-1]
    
    
def cruce(a1,p1,p2):
    if a1<Pcruce:
      print("Mas grande", Pcruce, "que ", a1, "-> Si Cruzan")

      cp = random.randint(1, len(p1)-1)
      temp1 = p1[:cp]
      temp2 = p1[cp:]
      print("Padre1 corte en", cp, ":", temp1, temp2)
      
      temp3 = p2[:cp]
      temp4 = p2[cp:]
      print("Padre2 corte en", cp, ":", temp3, temp4)
# Convert to list and then back to ndarray with dtype=int:
      hijo1 = np.array(temp1.tolist() + temp4.tolist(), dtype=int)
      hijo2 = np.array(temp3.tolist() + temp2.tolist(), dtype=int)
    else:
      print("Menor", Pcruce, "que ", a1, "-> NO Cruzan")
      hijo1=p1
      hijo2=p2

    hijo1=mutar(hijo1, Pmuta)
    hijo2=mutar(hijo2, Pmuta)

    return hijo1, hijo2

def mutar(individuo, Pmuta):
    for i in range(len(individuo)):
        if random.random() < Pmuta:
            individuo[i] = 1 - individuo[i]  # Flip bit
    print("Mutación: ", individuo)
    return individuo
    
      
    
#### Parametros #####
num_decimales=1
Xmin=0
Xmax=12
lind = int(round(m.log2((Xmax - Xmin) * 10**num_decimales)))
x= 5  #numero de individuos en la poblacion
Pcruce=0.98  #Probabilidad de Cruce
Pmuta=0.15   #Probabilidad de Mutación
elitismo=1   #Hace referencia a la cantidad de individuos elitistas

acumulado= np.empty((x))
fitness= np.empty((lind))
acumulado= np.empty((lind))
xi= np.empty((lind))
cromosoma= np.empty((lind), dtype=int)
suma=0
total=0

#Individuos, soluciones o cromosomas 
poblInicial = np.random.randint(0, 2, (x, lind)) # aleatorios (n por x) enteros entre [0 y2)
#random.random((4,5)) # 4 individuos 5 genes

# Ingresar los datos del Problema de la Mochila - Peso y Utilidad de los Elementos
#pesos = [7, 6, 8, 2]
#utilidad = [4, 5, 6, 3]
#pesos = [5, 7, 10, 30, 25]
#utilidad = [10, 20, 15, 30,15]
### CAPACIDAD 60

print("Poblacion inicial Aleatoria:","\n", poblInicial)  
poblIt=poblInicial

######  FIN DE LOS DATOS INICIALES

##Llama función evalua, para calcular el fitness de cada individuo
fitness,total,xi,cromosoma =evalua(x,poblIt)
#####print("\n","Funcion Fitness por individuos",  fitness)
#####print("\n","Suma fitness: ",  total)

##### imprime la tabla de la iteracion
imprime(total,fitness,poblIt,xi,cromosoma)

##### ***************************************
# Inicia Iteraciones

# Crear vector de 5x2 vacio  a = numpy.zeros(shape=(5,2))
for iter in range(3):
  print("\n Iteración ", iter + 1)

  # ELITISMO: guardar el mejor individuo
  elite_idx = np.argmax(fitness)
  elite_individuo = np.copy(poblIt[elite_idx])
  for i in range(0, x - 1, 2):

    for i in range(0, x - 1, 2):
        # Calculate cumulative probabilities from fitness values
        probabilities = []
        cum = 0
        for fit in fitness:
            p = 0 if fit == -np.inf or total == 0 else fit / total
            cum += p
            probabilities.append(cum)
        acumulado = np.array(probabilities)

        papa1 = seleccion(acumulado)
        print("padre 1:", papa1)
        papa2 = seleccion(acumulado)
        print("padre 2:", papa2)

        hijoA, hijoB = cruce(np.random.rand(), papa1, papa2)

        print("hijo1: ", hijoA)
        poblIt[i] = hijoA
        print("hijo2: ", hijoB)
        poblIt[i+1] = hijoB
  replace_idx = random.randint(0, x - 1)
  # Insertar al elite al azar reemplazando a alguien (excepto si ya está)
  replace_idx = random.randint(0, x - 1)
  print("Elitismo aplicado: Reemplazando índice", replace_idx, "con elite")
  poblIt[replace_idx] = elite_individuo

  fitness, total, xi, cromosoma = evalua(x, poblIt)

  # Eliminar individuos no factibles (fitness -inf) al final:
  for i in range(len(fitness)):
      if fitness[i] == -np.inf:
          print("Eliminando individuo no factible:", poblIt[i], "Xi:", xi[i])
          poblIt[i] = np.random.randint(0, 2, lind)  # reemplazar aleatorio
          fitness[i] = 0

  imprime(total, fitness, poblIt, xi, cromosoma)


max_index = np.argmax(fitness)
print("Hijo con mayor fitness:", poblIt[max_index], "con fitness:", fitness[max_index])

def plot_fitness():
  global mejores_fitness
  plt.plot(mejores_fitness)
  plt.xlabel('Iteración')
  plt.ylabel('Fitness')
  plt.title('Mejores Fitness')
  plt.show()
    
plot_fitness()