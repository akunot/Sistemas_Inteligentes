# coding: utf-8
import numpy as np

# Entrads en los patrones (dataset)
X = np.array([  [0,1,1],
                 [1,0,1],
                 [0,0,1],
                 [2,3,5],
                 [5,1,0],
                 [0,3,1] ])


# salidas en los patrones (dataset)           
y = np.array([[2,4,1,14,16,4]]).T

# semilla para random
np.random.seed(0)

# Inicialización de Pesos sinapticos

syn0 = np.array([[0.401],
                 [0.302],
                 [0.817] ])

print ('\n','Pesos Syn0','\n',syn0)

#Incrementar el numero de iteraciones, para entrenar ...352 iter
for iter in range(352):

    # forward propagation
	neta=np.dot(X,syn0)
	print('La neta es: ','\n',neta)

    # Funcion Lineal:
	l1 = (neta)
	print('La salida es: ','\n',l1)
    # Cálculo del error
	l1_error = y - l1
	print('El error es: ','\n',l1_error)
        
    # Constante de aprendizaje eta  (entre 0 y 1) -- Que pasa si lo cambio?
	eta=0.01
	# Cálculo del delta =(Error*eta*derivada de función)  ->  la Derivada, que para la función lineal es 1.
	derivada=1
	l1_delta = l1_error * eta * derivada
	deltaW=np.dot(X.T,l1_delta)
	print('delta completa','\n',deltaW)
	print('Delta por pasos: ','\n',np.dot(X.T,l1_delta))
	
	# Entrenamiento: Actualizacion de pesos
	syn0 += np.dot(X.T,l1_delta)
	print('Los nuevos pesos son: ','\n',syn0)

print ('Salida despues del entrenamiento:')

print ('Patrones Entrada','\n', X.T,'\n','Patrones Salida','\n',y,'\n','Salida Red','\n', l1,'\n','Error','\n',l1_error)
print ('\n','Pesos Syn0','\n',syn0)