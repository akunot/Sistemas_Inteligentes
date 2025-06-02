## CREADO NDD Sept 2019
import random
import numpy as np
import math as m


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
    capacidad_max = 60

    def bin_to_int(bits):
      # Convierte una lista de bits a entero: [1,0] -> 1*2^1 + 0*2^0
      valor = 0
      for idx, bit in enumerate(bits[::-1]):
        valor += bit * (2 ** idx)
      return valor

    for i in range(n):
      if poblIt[i] == 0:
          fitness[i] = 0
          continue
      total_peso = 0
      total_util = 0
      # Cada individuo es una lista de sublistas (cada variable)
      for j in range(len(poblIt[i])):
        if isinstance(poblIt[i][j], list):
          valor = bin_to_int(poblIt[i][j])
        else:
          valor = poblIt[i][j]
        total_peso += valor * pesos[j]
        total_util += valor * utilidad[j]
      if total_peso > capacidad_max:
        fitness[i] = 0  # Penalización al exceder capacidad
      else:
        fitness[i] = total_util
      total += fitness[i]

    return fitness, total

def imprime(n,total,fitness,poblIt):
    # Tabla de evaluación de la Población con títulos y conversión de bits a decimal
    def bin_to_int(bits):
      valor = 0
      for idx, bit in enumerate(bits[::-1]):
        valor += bit * (2 ** idx)
      return valor

    print("\nTabla Iteración:\n")
    print(f"{'Individuo':<10} {'Genotipo':<40} {'Fenotipo':<25} {'Fitness':<10} {'Probabilidad':<15} {'Acumulado':<10}")
    acumula = 0
    for i in range(n):
      probab = fitness[i] / total if total != 0 else 0
      acumula += probab
      # Convertir cada sublista (variable) de bits a su valor decimal correspondiente.
      if poblIt[i] == 0:
          decimal_values = "Invalid"
      else:
          decimal_values = [bin_to_int(bits) for bits in poblIt[i]]
      print(f"{i+1:<10} {str(poblIt[i]):<40} {str(decimal_values):<25} {fitness[i]:<10} {probab:<15.3f} {acumula:<10.3f}")
      acumulado[i] = acumula
    print("Suma Z:      ", total)
    return acumulado

def seleccion(acumulado):
    escoje = np.random.rand()
    print("escoje:      ", escoje)
    
    padre = None
    for i in range(0, n):
      if acumulado[i] > escoje and poblIt[i] != 0:
          padre = poblIt[i]
          break
    if padre is None:
      valid_individuals = [ind for ind in poblIt if ind != 0]
      if valid_individuals:
          padre = random.choice(valid_individuals)
      else:
          padre = [[0]]  # fallback individual to avoid errors
    # Unificar listas de listas en una sola lista
    flattened = [bit for sublist in padre for bit in sublist]
    return flattened

def bin_to_int(bits):
    raise NotImplementedError
    
    
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
# Convert merged lists to ndarray with dtype=int:
      hijo1 = np.array(temp1 + temp4, dtype=int)
      hijo2 = np.array(temp3 + temp2, dtype=int)
    else:
      print("Menor", Pcruce, "que ", a1, "-> NO Cruzan")
      hijo1=p1
      hijo2=p2

    hijo1=mutar(hijo1, Pmuta)
    hijo2=mutar(hijo2, Pmuta)

    # Reconvertir hijo1 y hijo2 de su forma "aplanada" a la estructura original de sublistas,
    # utilizando los tamaños especificados en diccionario_cromosomas.
    if not np.array_equal(hijo1, 0):
      unflattened1 = []
      idx = 0
      for var in sorted(diccionario_cromosomas.keys(), key=lambda k: int(k[1:])):
        num_bits = diccionario_cromosomas[var]
        unflattened1.append(hijo1[idx: idx + num_bits])
        idx += num_bits
      hijo1 = unflattened1

    if not np.array_equal(hijo2, 0):
      unflattened2 = []
      idx = 0
      for var in sorted(diccionario_cromosomas.keys(), key=lambda k: int(k[1:])):
        num_bits = diccionario_cromosomas[var]
        unflattened2.append(hijo2[idx: idx + num_bits])
        idx += num_bits
      hijo2 = unflattened2

    hijo1 = [[int(bit) for bit in sublist] if isinstance(sublist, (list, np.ndarray)) else int(sublist) for sublist in hijo1]
    hijo2 = [[int(bit) for bit in sublist] if isinstance(sublist, (list, np.ndarray)) else int(sublist) for sublist in hijo2]
    
    print("hijo1: ", hijo1)
    print("hijo2: ", hijo2)

    # Convertir cada sublista (variable) de bits a su valor decimal correspondiente.
    def bin_to_int(bits):
      valor = 0
      for idx, bit in enumerate(bits[::-1]):
        valor += bit * (2 ** idx)
      return valor

    capacidad_max = 60
    hijo1_int = [bin_to_int(bits) for bits in hijo1]
    if np.sum(np.multiply(hijo1_int, pesos)) > capacidad_max:
      print("Hijo1 descartado por exceder capacidad máxima.")
      hijo1 = 0
    hijo2_int = [bin_to_int(bits) for bits in hijo2]
    if np.sum(np.multiply(hijo2_int, pesos)) > capacidad_max:
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
n= int(m.log(x,2) * x)  #numero de individuos en la poblacion - cromosomas: n

diccionario_cromosomas={} #Diccionario para almacenar la cantidad de objetos para cada variable
for i in range(x):    
  cantidad = int(input(f"Ingrese la cantidad de objetos para x{i+1}: "))
  if cantidad == 1:
     diccionario_cromosomas[f'x{i+1}'] = cantidad
  else:
     diccionario_cromosomas[f'x{i+1}'] = int(m.log(cantidad+1,2))
print(diccionario_cromosomas)

Pcruce=0.98  #Probabilidad de Cruce
Pmuta=0.1   #Probabilidad de Mutación


fitness= np.empty((n))
acumulado= np.empty((n))
suma=0
total=0

# Generar población inicial como lista de listas basado en la cantidad de elementos de cada variable en el diccionario
poblInicial = []
# Asegurar orden en las variables: x1, x2, …, xN
variables = sorted(diccionario_cromosomas.keys(), key=lambda k: int(k[1:]))

for i in range(n):  # Para cada individuo en la población
  individuo = []
  for var in variables:
    num_bits = diccionario_cromosomas[var]
    # Crear una lista de bits aleatorios de tamaño num_bits
    sublista = [random.randint(0, 1) for _ in range(num_bits)]
    individuo.append(sublista)
  poblInicial.append(individuo)

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
for iter in range(5): #Hace referencia a las generaciones de cromosomas
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

expected = 60
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

    


