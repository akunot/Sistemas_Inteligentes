## CREADO NDD Sept 2019
import random
import numpy as np
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


def evalua(n, x, poblIt, utilidad, pesos):
    fitness = np.zeros(n)  # Inicializado correctamente
    total = 0
    capacidad_max = 15
    
    for i in range(n):
        peso = np.sum(poblIt[i] * pesos)
        if peso > capacidad_max:
            fitness[i] = 0  # Penalización
        else:
            fitness[i] = np.sum(poblIt[i] * utilidad)
        total += fitness[i]
    
    return fitness, total

def imprime(n,total,fitness,poblIt):
    global mejores_fitness
    #Tabla de evaluación de la Población
    acumula=0
    print ("\n",'Tabla Iteración:',"\n")
    for i in range(0, n):
      probab=fitness[i]/total
      acumula+=probab
      print(f"{i+1:<10} {str(poblIt[i]):<40} {fitness[i]:<10.2f} {probab:<15.3f} {acumula:<10.3f}")
      acumulado[i]=acumula
    print("Suma Z:      ", total)

    # Agregar el mayor fitness de la iteración a la lista de mejores_fitness
    best = float(max(fitness))
    mejores_fitness.append(best)
    return acumulado

def seleccion(acumulado):  #Metodo de seleccion por ruleta
    escoje=np.random.rand()
    print("escoje:      ", escoje)
    
    for i in range(0,n):
      if acumulado[i]>escoje:
         padre=poblIt[i]
         break
    return (padre)
    
    
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

    capacidad_max = 15
    if np.sum(hijo1 * pesos) > capacidad_max:
      print("Hijo1 descartado por exceder capacidad máxima.")
      hijo1 = 0
    if np.sum(hijo2 * pesos) > capacidad_max:
      print("Hijo2 descartado por exceder capacidad máxima.")
      hijo2 = 0

    return hijo1, hijo2

def mutar(individuo, Pmuta):
    for i in range(len(individuo)):
        if random.random() < Pmuta:
            individuo[i] = 1 - individuo[i]  # Flip bit
    print("Mutación: ", individuo)
    return individuo
    
      
    
#### Parametros #####
x=4  #numero de variables de decision - Elementos diferentes: x
n=4  #numero de individuos en la poblacion - cromosomas: n
Pcruce=0.98  #Probabilidad de Cruce
Pmuta=0.1   #Probabilidad de Mutación


fitness= np.empty((n))
acumulado= np.empty((n))
suma=0
total=0

#Individuos, soluciones o cromosomas 
poblInicial = np.random.randint(0, 2, (n, x)) # aleatorios (n por x) enteros entre [0 y2)
#random.random((4,5)) # 4 individuos 5 genes

# Ingresar los datos del Problema de la Mochila - Peso y Utilidad de los Elementos
pesos = [7, 6, 8, 2]
utilidad = [4, 5, 6, 3]
#pesos = [5, 7, 10, 30, 25]
#utilidad = [10, 20, 15, 30,15]
### CAPACIDAD 60

print("Poblacion inicial Aleatoria:","\n", poblInicial)
print("\n","Utilidad:", utilidad) 
print("\n","Pesos", pesos )   
poblIt=poblInicial

######  FIN DE LOS DATOS INICIALES



##Llama función evalua, para calcular el fitness de cada individuo
fitness,total=evalua(n,x,poblIt,utilidad, pesos)
#####print("\n","Funcion Fitness por individuos",  fitness)
#####print("\n","Suma fitness: ",  total)

##### imprime la tabla de la iteracion
imprime(n,total,fitness,poblIt)

##### ***************************************
# Inicia Iteraciones

# Crear vector de 5x2 vacio  a = numpy.zeros(shape=(5,2))
for iter in range(20): #Hace referencia a las generaciones de cromosomas
  print("\n Iteración ", iter+1)
  
  # Iterar de 2 en 2 hasta completar todos los hijos (n es par)
  for i in range(0, n, 2):
    papa1 = seleccion(acumulado)  # Padre 1
    print("padre 1:", papa1)
    papa2 = seleccion(acumulado)  # Padre 2
    print("padre 2:", papa2)
    
    hijoA, hijoB = cruce(np.random.rand(), papa1, papa2)
    if np.array_equal(hijoA, 0) or np.array_equal(hijoB, 0):
      print("Uno de los hijos fue descartado, realizando nuevo cruce")
      hijoA, hijoB = cruce(np.random.rand(), papa1, papa2)
    print("hijo1: ", hijoA)
    poblIt[i] = hijoA
    print("hijo2: ", hijoB)
    poblIt[i+1] = hijoB
  
  print("\n Poblacion Iteración ", iter+1, "\n", poblIt)
  fitness, total = evalua(n, x, poblIt, utilidad, pesos)
  imprime(n, total, fitness, poblIt)

expected = 15
# Filtrar individuos válidos (fitness no nulo) ya que fitness 0 indica violación de restricciones
valid = fitness > 0
if np.any(valid):
  # Calcular la diferencia absoluta entre el fitness válido y el resultado esperado
  valid_indices = np.where(valid)[0]
  diferencias = np.abs(fitness[valid_indices] - expected)
  min_index_valid = np.argmin(diferencias)
  min_index = valid_indices[min_index_valid]
  print("Hijo con fitness más cercano a", expected, ":",
      poblIt[min_index], "con fitness:", fitness[min_index])
else:
  print("Ningún hijo cumple las restricciones de capacidad.")

def plot_fitness():
  global mejores_fitness
  plt.plot(mejores_fitness)
  plt.xlabel('Iteración')
  plt.ylabel('Fitness')
  plt.title('Mejores Fitness')
  plt.show()
    
plot_fitness()