"""
Parametros: pc, pm y n

1. Inicializar la poblacion
2. Evaluar Fitness de cada individuo
3. SELECCION padres , aquel que tenga mejor valor de Fitness
4. CROSSOVER
5. MUTACION
6. Obtenemos un descendiente
7. Repetimos hasta tener NUEVA GENERACION
8. Regresamos al paso (2)

MAX ONE
Encontrar una cadena binaria con la mayor cantidad 1's de longitud N
c = [1 0 0 0 1 0 0 1 ]
o = [1 1 1 1 1 1 1 1] -> MINIMO GLOBAL

"""

import random
import time
import matplotlib.pyplot as plt

def crear_individuo(longitud):
    individuo = []
    for i in range(longitud):
        gen = random.randint(0,1)
        individuo.append(gen)
    return individuo

def crear_poblacion(tam_poblacion, longitud):
    poblacion = []
    for i in range(tam_poblacion):
        poblacion.append(crear_individuo(longitud))

    return poblacion

def fitness(individuo):
    """
    ejemplo de fitness:
    Sea g = [0 1 0 0 1 0 1 0 1] -> fitness = 4
    """
    return sum(individuo)

def seleccion(poblacion):
    """
    Primer operador estocastico
    Seleccion sencilla
    1. Ordenar por valor fitness
    2. Tomamos la mitad superior
    3. Elegimos al azar de ese grupo a dos parents
    """
    poblacion_ordenada = sorted(poblacion, key=fitness, reverse=True)
    mitad = len(poblacion_ordenada) // 2
    top = poblacion_ordenada[:mitad]
    parent1 = random.choice(top)
    parent2 = random.choice(top)
    # vale la pena validar que parent1 != parent2
    return parent1, parent2

def cruza(parent1, parent2):
    # Usaremos la cruza de un punto
    punto = random.randint(1, len(parent1)-1)
    offspring = parent1[:punto] + parent2[punto:]
    return offspring, punto

def mutacion(individuo, probabilidad):
    pos_mutacion = []
    """
    Cada gen puede cambiar : 0->1, 1->0
    """
    for i in range(len(individuo)):
        if random.random() < probabilidad:
            individuo[i] = 1 - individuo[i]
            pos_mutacion.append(i)

    return individuo, pos_mutacion

def mostrar_poblacion(poblacion):
    for i in range (len(poblacion)):
        print(f"Individuo{i+1}: {poblacion[i]} , fitness: {fitness(poblacion[i])}")

def graficar_fitness(hm, hp, hmo):
    generaciones = list(range(len(hm)))
    plt.figure(figsize = (9,5))
    plt.plot(generaciones, hm, marker="o", label = "mejor fitness de la generacion")
    plt.plot(generaciones, hp, marker = 's', label = "fitness promedio(presion selectiva)")
    plt.plot(generaciones, hmo, linestyle = "--", label = "mejor fitness observado")

    plt.xlabel("Generaciones")
    plt.ylabel("Fitness")
    plt.title("EVOLUCION DEL FITNESS - ONEMAX")

    plt.grid()
    plt.legend()
    plt.show()

#parametros generales
longitud = 6

# tamaño de la poblacion
tam_poblacion = 6

# probabilidad de crossover (cruza) entre dos parents
pc = 0.6

# probalidad de mutacion por gen
pm = 0.1

#tiempos delay
global to
global tg

to = 1
tg = 3

def one_max_agen(tam_poblacion, longitud, pc, pm):

    historial_mejores = []
    historial_mejores_observado = []
    historial_promedio = []
    mejor_observado = 0


    # paso 1 inicializacion
    poblacion = crear_poblacion(tam_poblacion, longitud)

    # paso 2 iteraciones generacionales
    generacion = 0
    #Ciclo evolutivo
    while True:

        print("###" * 30)
        print(f"GENERACION: {generacion}")
        print("###" * 30)

        print("\nPOBLACION ACTUAL:")
        mostrar_poblacion(poblacion)
        time.sleep(to)

        ### pedagogicamente mostrar el mejor individuo

        mejor = max(poblacion, key=fitness)
        mejor_fitness = fitness(mejor)

        print("\n El mejor individuo de la población es:")
        print("-----------------------------------------")
        print("El individuo: ", mejor)
        print("Fitness: ", mejor_fitness)

        # CALCULAR EL PRESION SELECTIVA
        suma_fitness = 0
        for ind in poblacion:
            suma_fitness += fitness(ind)

        psel = suma_fitness / len(poblacion)
        print("Fitness promedio (presion selectiva):", round(psel, 2))

        #guardar datos para la grafica fitness
        # metodo de burbuja
        if mejor_fitness > mejor_observado:
            mejor_observado = mejor_fitness

        historial_mejores.append(mejor_fitness)
        historial_promedio.append(psel)
        historial_mejores_observado.append(mejor_observado)

        time.sleep(to)


        # el paro
        if mejor_fitness == longitud:
            print("\n ENCONTRAMOS LA MEJOR SOLUCION!!")
            print(mejor)
            break


        sigPoblacion = []
        print("###" * 30)
        print(f"CREANDO SIG GENERACION")
        print("###" * 30)

        for n_o in range(tam_poblacion):

            print("\n")
            print(f"CREANDO HIJO {n_o+1}")
            print("---" * 30)

            time.sleep(to)

            ## Seleccion
            parent1, parent2 = seleccion(poblacion)

            print("\nSELECION:")
            print(f"Parent1: {parent1}, Fitness: {fitness(parent1)}")
            print(f"Parent2: {parent2}, Fitness: {fitness(parent2)}")
            print("---" * 30)

            time.sleep(to)

            ## Crossover / Cruza
            print("\nCRUZA:")
            print("la probabilidad de cruza es de: ", pc)

            alea = random.random()

            if alea < pc:
                hijo, punto = cruza(parent1, parent2)
                print("SI OCURRIÓ CRUZA")
                print("Punto de CRUZA: ", punto)

                print("PARENT 1 :", parent1[:punto], "|", parent1[punto:])
                print("PARENT2 :", parent2[:punto], "|", parent2[punto:])
                print("Se cruza:")
                print("Primera Parte del padre 1:",  parent1[:punto])
                print("Segunda Parte del padre 2:",  parent2[punto:])
            else:
                hijo = parent1.copy()
                punto = None

                print("NO OCURRIÓ CRUZA")
                print("el hijo es una copia del padre 1")


            # Mutacion
            hijo_antes = hijo.copy()

            hijo, posiciones_mutadas = mutacion(hijo, pm)
            print("\nMUTACION:")
            print("Probabilidad de mutacion: ", pc)
            print("Hijo antes: ", hijo_antes)

            if len(posiciones_mutadas) == 0:
                print("NO OCURRIÓ MUTACION")
            else:
                print("posiciones mutadas: ", posiciones_mutadas)
                print("Las posicines mutadas empiezan el indice 0")
                print("Despues de mutación:", hijo)

            time.sleep(to)

            sigPoblacion.append(hijo)
            print("Hijo agregado a la nueva generación..")
            time.sleep(to)

        print("###" * 30)
        print(f"NUEVA GENERACION COMPLETA")
        print("###" * 30)

        mostrar_poblacion(sigPoblacion)

        # REEMPLAZO
        generacion += 1
        poblacion = sigPoblacion

        print(f"\nSIGUIENTE GENERACION en: {tg}")
        time.sleep(tg)


    graficar_fitness(historial_mejores, historial_promedio, historial_mejores_observado)


one_max_agen(tam_poblacion, longitud, pc, pm)




