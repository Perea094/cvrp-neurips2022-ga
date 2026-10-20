"""
Algoritmo genetico para el VRPTW (instancias EURO Meets NeurIPS 2022).
"""

import random
import time
import matplotlib.pyplot as plt
import tools  # read_vrplib y validate_static_solution (quickstart de ORTEC)


# datos

ARCHIVO = "data/instances/test/ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35.txt"


def cargar_datos(archivo):
    """
    Lee la instancia con read_vrplib y guarda los datos en listas de Python
    """
    instancia = tools.read_vrplib(archivo)

    datos = {
        "instancia": instancia,  # para validar al final
        "tiempos": instancia["duration_matrix"].tolist(), # t_ij en segundos
        "demandas": instancia["demands"].tolist(), # q_i
        "ventanas": instancia["time_windows"].tolist(), # [e_i, l_i]
        "servicio": instancia["service_times"].tolist(), # s_i
        "capacidad": instancia["capacity"],  # Q
        "vehiculos": instancia["vehicles"],   # K del archivo
        "n": len(instancia["demands"]) - 1,  # numero de clientes
    }
    return datos


# decodificador: permutacion -> rutas

def decodificar(individuo, datos):

    t = datos["tiempos"]
    q = datos["demandas"]
    ventanas = datos["ventanas"]
    s = datos["servicio"]
    Q = datos["capacidad"]
    cierre_deposito = ventanas[0][1]

    rutas = []
    ruta = []
    carga = 0
    hora = 0  # hora en que el vehiculo termina en el ultimo cliente
    anterior = 0  # ultimo nodo visitado (empieza en el deposito)

    for c in individuo:
        inicio = max(hora + t[anterior][c], ventanas[c][0])   # espera si llega antes

        cabe = carga + q[c] <= Q
        a_tiempo = inicio <= ventanas[c][1]
        regresa = inicio + s[c] + t[c][0] <= cierre_deposito

        if len(ruta) > 0 and cabe and a_tiempo and regresa:
            # se agrega a la ruta actual
            ruta.append(c)
        else:
            # se cierra la ruta actual y se abre una nueva desde el deposito
            if len(ruta) > 0:
                rutas.append(ruta)
            ruta = [c]
            carga = 0
            inicio = max(t[0][c], ventanas[c][0])

        carga += q[c]
        hora = inicio + s[c]
        anterior = c

    rutas.append(ruta)
    return rutas


def costo_rutas(rutas, datos):
    """
    Z = tiempo total de manejo: deposito -> clientes -> deposito, en cada ruta.
    """
    t = datos["tiempos"]
    Z = 0
    for ruta in rutas:
        Z += t[0][ruta[0]]  # salida del deposito
        for i in range(len(ruta) - 1):
            Z += t[ruta[i]][ruta[i + 1]]  # entre clientes
        Z += t[ruta[-1]][0] # regreso al deposito
    return Z


# individuo población y fitness

def crear_individuo(n):
    individuo = list(range(1, n + 1))
    random.shuffle(individuo)
    return individuo


def crear_poblacion(tam_poblacion, n):
    poblacion = []
    for i in range(tam_poblacion):
        poblacion.append(crear_individuo(n))
    return poblacion


def fitness(individuo, datos):
    """
    Aptitud = Z de las rutas decodificadas.
    Como el decodificador respeta capacidad y ventanas, toda solucion es
    factible y no hace falta penalizar
    """
    rutas = decodificar(individuo, datos)
    return costo_rutas(rutas, datos)


# seleccion por torneo

def seleccion_torneo(poblacion, fitnesses, k):
    """
    Se eligen k individuos al azar y gana el de MENOR fitness (minimizacion).
    k grande -> mas presion selectiva (explotacion).
    k chico  -> menos presion selectiva (exploracion).
    """
    participantes = random.sample(range(len(poblacion)), k)
    ganador = participantes[0]
    for i in participantes:
        if fitnesses[i] < fitnesses[ganador]:
            ganador = i
    return poblacion[ganador]


# cruza OX

def cruza_ox(padre1, padre2):
    n = len(padre1)
    a = random.randint(0, n - 2)
    b = random.randint(a + 1, n - 1)

    hijo = [None] * n
    hijo[a:b + 1] = padre1[a:b + 1]
    en_hijo = set(padre1[a:b + 1])

    pos = 0
    for cliente in padre2:
        if cliente not in en_hijo:
            while hijo[pos] is not None:
                pos += 1
            hijo[pos] = cliente
    return hijo


# mutacion swap

def mutacion_swap(individuo, pm):
    n = len(individuo)
    for i in range(n):
        if random.random() < pm:
            j = random.randint(0, n - 1)
            individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo


# GA

def algoritmo_genetico(datos, tam_poblacion, generaciones, pc, pm, k,
                       elitismo=2, semilla=None, imprimir=True):
    random.seed(semilla)
    n = datos["n"]

    historial_mejores = []  # mejor Z de cada generacion
    historial_promedio = [] # Z promedio de cada generacion
    historial_mejores_observado = []  # mejor Z visto hasta esa generacion
    historial_rutas = [] # num. de rutas del mejor observado

    mejor_observado = None
    mejor_observado_fit = float("inf")

    # inicializacion
    poblacion = crear_poblacion(tam_poblacion, n)

    for gen in range(generaciones):

        # evaluar fitness
        fitnesses = [fitness(ind, datos) for ind in poblacion]

        mejor_idx = fitnesses.index(min(fitnesses))
        if fitnesses[mejor_idx] < mejor_observado_fit:
            mejor_observado_fit = fitnesses[mejor_idx]
            mejor_observado = poblacion[mejor_idx].copy()

        historial_mejores.append(fitnesses[mejor_idx])
        historial_promedio.append(sum(fitnesses) / len(fitnesses))
        historial_mejores_observado.append(mejor_observado_fit)
        historial_rutas.append(len(decodificar(mejor_observado, datos)))

        if imprimir and gen % 50 == 0:
            print(f"Gen {gen:4d} | mejor Z = {fitnesses[mejor_idx]:,} | "
                  f"promedio = {historial_promedio[-1]:,.0f} | "
                  f"rutas = {historial_rutas[-1]}")

        # elitismo, los mejores pasan directo a la siguiente generacion
        orden = sorted(range(tam_poblacion), key=lambda i: fitnesses[i])
        sig_poblacion = [poblacion[i].copy() for i in orden[:elitismo]]

        # crear hijos hasta completar la poblacion
        while len(sig_poblacion) < tam_poblacion:
            padre1 = seleccion_torneo(poblacion, fitnesses, k)
            padre2 = seleccion_torneo(poblacion, fitnesses, k)

            if random.random() < pc:
                hijo = cruza_ox(padre1, padre2)
            else:
                hijo = padre1.copy()

            hijo = mutacion_swap(hijo, pm)
            sig_poblacion.append(hijo)

        # reemplazo
        poblacion = sig_poblacion

    rutas = decodificar(mejor_observado, datos)
    historial = {
        "mejores": historial_mejores,
        "promedio": historial_promedio,
        "mejor_observado": historial_mejores_observado,
        "rutas": historial_rutas,
    }
    return mejor_observado, rutas, mejor_observado_fit, historial


# graficas

def graficar_convergencia(historial, titulo="Convergencia del AG - VRPTW", archivo=None):
    generaciones = list(range(len(historial["mejores"])))
    plt.figure(figsize=(9, 5))
    plt.plot(generaciones, historial["mejores"], label="mejor Z de la generación")
    plt.plot(generaciones, historial["promedio"], label="Z promedio de la generación")
    plt.plot(generaciones, historial["mejor_observado"], linestyle="--", label="mejor Z observado")
    plt.xlabel("Generación")
    plt.ylabel("Tiempo total de manejo Z (s)")
    plt.title(titulo)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if archivo:
        plt.savefig(archivo, dpi=200)
    plt.show()


def graficar_rutas(rutas, datos, titulo="Mejor solución", archivo=None):
    coords = datos["instancia"]["coords"]
    plt.figure(figsize=(8, 7))
    for ruta in rutas:
        recorrido = [0] + ruta + [0]
        xs = [coords[i][0] for i in recorrido]
        ys = [coords[i][1] for i in recorrido]
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=0.8)
    plt.scatter(coords[0][0], coords[0][1], c="red", marker="s", s=100, label="Depósito", zorder=5)
    plt.title(f"{titulo} ({len(rutas)} rutas)")
    plt.legend()
    plt.tight_layout()
    if archivo:
        plt.savefig(archivo, dpi=200)
    plt.show()


# ejecucion de prueba

if __name__ == "__main__":

    # hiperparametros 
    TAM_POBLACION = 100
    GENERACIONES = 500
    PC = 0.9
    PM = 0.01
    K_TORNEO = 3
    ELITISMO = 2
    SEMILLA = 1

    datos = cargar_datos(ARCHIVO)
    print("Instancia:", datos["instancia"]["name"], "| clientes:", datos["n"])

    inicio = time.time()
    mejor, rutas, Z, historial = algoritmo_genetico(
        datos, TAM_POBLACION, GENERACIONES, PC, PM, K_TORNEO, ELITISMO, SEMILLA)
    duracion = time.time() - inicio

    # verificacion independiente con el validador de ORTEC
    import numpy as np
    Z_validado = tools.validate_static_solution(datos["instancia"], [np.array(r) for r in rutas])

    print("\nMejor Z:", Z, "| validado:", Z_validado)
    print("Rutas:", len(rutas), "| vehiculos en el archivo:", datos["vehiculos"])
    print(f"Tiempo: {duracion:.1f} s")

    graficar_convergencia(historial)
    graficar_rutas(rutas, datos)
