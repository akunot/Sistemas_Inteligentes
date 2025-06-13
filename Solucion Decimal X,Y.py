import random
import numpy as np
import matplotlib.pyplot as plt

mejores_fitness = []

# Conversión de binario a real
def bin2real(bin_array, Xmin, Xmax, bits):
    bin_str = ''.join(str(int(b)) for b in bin_array)
    bin_dec = int(bin_str, 2)
    return Xmin + bin_dec * (Xmax - Xmin) / (2**bits - 1)

# Evaluación de individuos
def evalua(n, poblIt):
    fitness = np.zeros(n)
    xi = np.zeros(n)
    yi = np.zeros(n)
    for i in range(n):
        crom = poblIt[i]
        crom_x = crom[:Lx]
        crom_y = crom[Lx:]
        x_real = bin2real(crom_x, Xmin_x, Xmax_x, Lx)
        y_real = bin2real(crom_y, Xmin_y, Xmax_y, Ly)
        xi[i] = x_real
        yi[i] = y_real
        f = np.sin(2 * x_real) * 4 * np.cos(y_real) + 5 #Función de evaluación
        fitness[i] = f
        # fitness[i] = 1 / (1 + f)  # Minimizar f, usar en caso de minimizar
    total = np.sum(fitness)
    return fitness, total, xi, yi

# Imprimir tabla por iteración
def imprime(n, total, fitness, poblIt, xi, yi, cromosoma):
    global mejores_fitness
    print("\nTabla Iteración:")
    print("{:^10} {:^20} {:^15} {:^10} {:^10} {:^12} {:^12}".format(
        "Individuo", "Cromosoma (XY)", "Población", "x", "y", "Fitness", "Probacum"))
    
    acumula = 0
    for i in range(n):
        probab = 0 if total == 0 or fitness[i] == -np.inf else fitness[i] / total
        acumula += probab
        fitness_str = "-" if fitness[i] == -np.inf else f"{fitness[i]:.3f}"
        print("{:^10} {:^20} {:^15} {:^10.3f} {:^10.3f} {:^12} {:^12.3f}".format(
            i + 1,
            cromosoma[i],
            ''.join(str(bit) for bit in poblIt[i]),
            xi[i], yi[i],
            fitness_str, acumula))
        acumulado[i] = acumula

    print("Suma Z:      ", total)
    best = np.max(fitness[fitness != -np.inf]) if np.any(fitness != -np.inf) else 0
    mejores_fitness.append(best)
    return acumulado

# Selección por ruleta
def seleccion(acumulado):
    escoje = np.random.rand()
    for i in range(len(acumulado)):
        if acumulado[i] > escoje:
            return poblIt[i]
    return poblIt[-1]

# Cruce y mutación
def cruce(a1, p1, p2):
    if a1 < Pcruce:
        cp = random.randint(1, len(p1) - 1)
        hijo1 = np.array(list(p1[:cp]) + list(p2[cp:]), dtype=int)
        hijo2 = np.array(list(p2[:cp]) + list(p1[cp:]), dtype=int)
    else:
        hijo1 = p1.copy()
        hijo2 = p2.copy()
    hijo1 = mutar(hijo1, Pmuta)
    hijo2 = mutar(hijo2, Pmuta)
    return hijo1, hijo2

# Mutación binaria
def mutar(individuo, Pmuta):
    for i in range(len(individuo)):
        if random.random() < Pmuta:
            individuo[i] = 1 - individuo[i]
    return individuo

# Parámetros de codificación y GA
num_decimales = 1 #Numero de decimales
Xmin_x, Xmax_x = 0, 2 #Rango de X
Xmin_y, Xmax_y = 2, 5 #Rango de Y
Lx = int(np.ceil(np.log2(1 + (Xmax_x - Xmin_x) * 10**num_decimales))) #Longitud de X
Ly = int(np.ceil(np.log2(1 + (Xmax_y - Xmin_y) * 10**num_decimales))) #Longitud de Y
lind = Lx + Ly #Longitud total

x = 6  # población
Pcruce = 0.8 #Probabilidad de Cruce
Pmuta = 0.2 #Probabilidad de Mutación
elitismo = 1 #Hace referencia a la cantidad de individuos elitistas

# Inicialización
poblInicial = np.random.randint(0, 2, (x, lind))
poblIt = poblInicial.copy()
acumulado = np.zeros(x)

# Evaluación inicial
fitness, total, xi, yi = evalua(x, poblIt)
cromosoma = [''.join(str(bit) for bit in ind) for ind in poblIt]
imprime(x, total, fitness, poblIt, xi, yi, cromosoma)

# Iteraciones
for iter in range(1000): #Hace referencia a las generaciones de cromosomas
    print("\n Iteración ", iter + 1)
    elite_idx = np.argmax(fitness)
    elite_ind = poblIt[elite_idx].copy()
    nueva_pobl = []

    for _ in range(x // 2):
        p1 = seleccion(acumulado)
        p2 = seleccion(acumulado)
        hijo1, hijo2 = cruce(np.random.rand(), p1, p2)
        nueva_pobl.extend([hijo1, hijo2])

    poblIt = np.array(nueva_pobl[:x])
    
    if elitismo > 0:
        replace_idx = random.choice([i for i in range(x) if i != elite_idx])
        poblIt[replace_idx] = elite_ind
        print(f"Elitismo aplicado en índice {replace_idx}")

    # Evaluación y tabla
    fitness, total, xi, yi = evalua(x, poblIt)
    cromosoma = [''.join(str(bit) for bit in ind) for ind in poblIt]
    acumulado = np.zeros(x)
    imprime(x, total, fitness, poblIt, xi, yi, cromosoma)

# Mejor individuo
max_index = np.argmax(fitness)
print("\nMejor individuo final:", poblIt[max_index])
print("Fitness final:", fitness[max_index])

# Graficar progreso
def graficar_superficie_fitness():
    x_vals = np.linspace(Xmin_x, Xmax_x, 100) 
    y_vals = np.linspace(Xmin_y, Xmax_y, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.sin(2 * X) * 4 * np.cos(Y) + 5

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_title('Superficie del Fitness: f(x, y) = sin(2x) * 4cos(y) + 5')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Fitness')
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label='Fitness')
    plt.show()

graficar_superficie_fitness()
