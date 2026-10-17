"""
Algoritmo Genético - Problema de la Mochila con análisis de esquemas
====================================================================

Este programa conserva la lógica del ejemplo de mochila usado en clase
y agrega una capa de análisis mediante esquemas.

OBJETIVO DEL PROBLEMA
---------------------
Seleccionar objetos para maximizar el valor total sin superar la
capacidad de la mochila.

Cada individuo es una cadena binaria:

    [1, 0, 1, 0, 1, 0]

donde:
    1 = el objeto se incluye
    0 = el objeto no se incluye

OBJETIVO DEL ANÁLISIS DE ESQUEMAS
---------------------------------
Los esquemas NO modifican el algoritmo genético.

Se utilizan para observar familias de soluciones y responder:

    1. ¿Cuántos individuos pertenecen al esquema H?
          M(H,t)

    2. ¿Cuál es el fitness promedio de esos individuos?
          F_prom(H,t)

    3. ¿El patrón aumenta, disminuye o desaparece durante
       las generaciones?

GRÁFICAS
--------
Al terminar se muestran dos figuras:

1. Fitness general de la población:
   - mejor fitness de la generación
   - fitness promedio de la población
   - mejor fitness observado

2. Fitness promedio de los esquemas:
   - F_prom(H_A,t)
   - F_prom(H_B,t)
   - F_prom(H_C,t)

Librerías:
    random
    matplotlib
"""

import random
import matplotlib.pyplot as plt


# ============================================================
# DATOS DEL PROBLEMA DE LA MOCHILA
# ============================================================

valores = [10, 8, 15, 4, 12, 6]
pesos   = [4,  3,  6, 2,  5, 3]

capacidad = 12


# ============================================================
# ESQUEMAS A ANALIZAR
# ============================================================

# None representa el comodín *
#
# H_A = [1,1,*,*,1,*]
# Incluye obligatoriamente los objetos 1, 2 y 5.
#
# H_B = [1,*,*,*,1,*]
# Incluye obligatoriamente los objetos 1 y 5.
#
# H_C = [*,*,1,*,*,1]
# Incluye obligatoriamente los objetos 3 y 6.

ESQUEMAS = {
    "H_A": [1, 1, None, None, 1, None],
    "H_B": [1, None, None, None, 1, None],
    "H_C": [None, None, 1, None, None, 1]
}


# ============================================================
# PARÁMETROS DEL ALGORITMO GENÉTICO
# ============================================================

longitud = len(valores)

tamano_poblacion = 8

probabilidad_mutacion = 0.10

numero_generaciones = 20

# Descomenta esta línea si quieres repetir siempre
# exactamente la misma ejecución.
# random.seed(2)


# ============================================================
# 1. CREAR INDIVIDUO
# ============================================================

def crear_individuo(longitud):

    individuo = []

    for i in range(longitud):

        gen = random.randint(0, 1)

        individuo.append(gen)

    return individuo


# ============================================================
# 2. CREAR POBLACIÓN
# ============================================================

def crear_poblacion(tamano, longitud):

    poblacion = []

    for i in range(tamano):

        poblacion.append(
            crear_individuo(longitud)
        )

    return poblacion


# ============================================================
# 3. CALCULAR PESO
# ============================================================

def calcular_peso(individuo):

    peso_total = 0

    for i in range(len(individuo)):

        if individuo[i] == 1:

            peso_total += pesos[i]

    return peso_total


# ============================================================
# 4. CALCULAR VALOR
# ============================================================

def calcular_valor(individuo):

    valor_total = 0

    for i in range(len(individuo)):

        if individuo[i] == 1:

            valor_total += valores[i]

    return valor_total


# ============================================================
# 5. FUNCIÓN DE APTITUD
# ============================================================

def fitness(individuo):
    """
    Si la solución excede la capacidad:
        fitness = 0

    En otro caso:
        fitness = valor total
    """

    peso_total = calcular_peso(individuo)

    valor_total = calcular_valor(individuo)

    if peso_total > capacidad:

        return 0

    return valor_total


# ============================================================
# 6. SELECCIÓN
# ============================================================

def seleccionar_padres(poblacion):
    """
    Misma selección sencilla usada en la presentación:

    1. Ordenar por fitness.
    2. Tomar la mitad superior.
    3. Elegir dos padres al azar de esa mitad.
    """

    poblacion_ordenada = sorted(
        poblacion,
        key=fitness,
        reverse=True
    )

    mitad = len(poblacion_ordenada) // 2

    mejores = poblacion_ordenada[:mitad]

    padre1 = random.choice(mejores)

    padre2 = random.choice(mejores)

    return padre1, padre2


# ============================================================
# 7. CRUZA
# ============================================================

def cruza(padre1, padre2):
    """
    Cruza de un punto.
    """

    punto = random.randint(
        1,
        len(padre1) - 1
    )

    hijo = (
        padre1[:punto]
        +
        padre2[punto:]
    )

    return hijo


# ============================================================
# 8. MUTACIÓN
# ============================================================

def mutacion(individuo, probabilidad):
    """
    Mutación bit-flip:

        0 -> 1
        1 -> 0
    """

    for i in range(len(individuo)):

        if random.random() < probabilidad:

            individuo[i] = 1 - individuo[i]

    return individuo


# ============================================================
# 9. ¿UN INDIVIDUO PERTENECE A UN ESQUEMA?
# ============================================================

def pertenece(individuo, esquema):
    """
    Comprueba si un individuo coincide con todas las
    posiciones fijas de un esquema.

    None representa * y, por tanto, no impone restricción.

    Ejemplo:

        individuo = [1, 1, 0, 0, 1, 0]
        esquema    = [1, 1, *, *, 1, *]

    Resultado:
        True
    """

    for gen, patron in zip(individuo, esquema):

        if patron is not None and gen != patron:

            return False

    return True


# ============================================================
# 10. ORDEN DEL ESQUEMA O(H)
# ============================================================

def orden(esquema):
    """
    O(H) = número de posiciones fijas del esquema.
    """

    return sum(
        x is not None
        for x in esquema
    )


# ============================================================
# 11. LONGITUD DEFINITORIA delta(H)
# ============================================================

def longitud_definida(esquema):
    """
    delta(H) = distancia entre la primera y la última
    posición fija.
    """

    posiciones = []

    for i, x in enumerate(esquema):

        if x is not None:

            posiciones.append(i)

    if len(posiciones) <= 1:

        return 0

    return (
        max(posiciones)
        -
        min(posiciones)
    )


# ============================================================
# 12. ESTADÍSTICAS DE UN ESQUEMA
# ============================================================

def estadisticas_esquema(poblacion, esquema):
    """
    Calcula:

        M(H,t)
            número de individuos de la población
            que pertenecen al esquema.

        F_prom(H,t)
            fitness promedio de los individuos
            pertenecientes al esquema.

    Si ningún individuo pertenece al esquema:
        M = 0
        F_prom = 0
    """

    miembros = []

    for individuo in poblacion:

        if pertenece(individuo, esquema):

            miembros.append(individuo)

    M = len(miembros)

    if M == 0:

        return 0, 0

    fitness_promedio = (
        sum(
            fitness(ind)
            for ind in miembros
        )
        / M
    )

    return M, fitness_promedio


# ============================================================
# 13. MOSTRAR POBLACIÓN
# ============================================================

def mostrar_poblacion(poblacion):

    for i, individuo in enumerate(poblacion):

        print(
            f"Individuo {i + 1}: "
            f"{individuo}  "
            f"Peso={calcular_peso(individuo):2d}  "
            f"Valor={calcular_valor(individuo):2d}  "
            f"Fitness={fitness(individuo):2d}"
        )


# ============================================================
# 14. MOSTRAR INFORMACIÓN DE LOS ESQUEMAS
# ============================================================

def mostrar_esquemas():

    print("\n" + "=" * 72)
    print("ESQUEMAS ANALIZADOS")
    print("=" * 72)

    for nombre, esquema in ESQUEMAS.items():

        esquema_texto = [
            "*" if x is None else str(x)
            for x in esquema
        ]

        print(
            nombre,
            "=",
            "".join(esquema_texto),
            "| O(H)=",
            orden(esquema),
            "| delta(H)=",
            longitud_definida(esquema)
        )


# ============================================================
# 15. GRÁFICA DE FITNESS GENERAL
# ============================================================

def graficar_fitness_general(
    historial_mejor,
    historial_promedio,
    historial_mejor_observado
):

    generaciones = list(
        range(len(historial_mejor))
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        generaciones,
        historial_mejor,
        marker="o",
        label="Mejor fitness de la generación"
    )

    plt.plot(
        generaciones,
        historial_promedio,
        marker="s",
        label="Fitness promedio de la población"
    )

    plt.plot(
        generaciones,
        historial_mejor_observado,
        linestyle="--",
        label="Mejor fitness observado"
    )

    plt.xlabel("Generación")

    plt.ylabel("Fitness")

    plt.title(
        "Evolución general del fitness - Mochila"
    )

    plt.xticks(generaciones)

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 16. GRÁFICA DE FITNESS DE LOS ESQUEMAS
# ============================================================

def graficar_fitness_esquemas(
    historial_fitness_esquemas
):

    nombres = list(
        historial_fitness_esquemas.keys()
    )

    numero_datos = len(
        historial_fitness_esquemas[nombres[0]]
    )

    generaciones = list(
        range(numero_datos)
    )

    plt.figure(figsize=(9, 5))

    for nombre in nombres:

        plt.plot(
            generaciones,
            historial_fitness_esquemas[nombre],
            marker="o",
            label=f"Fitness promedio {nombre}"
        )

    plt.xlabel("Generación")

    plt.ylabel(
        "Fitness promedio del esquema"
    )

    plt.title(
        "Evolución del fitness de los esquemas"
    )

    plt.xticks(generaciones)

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 17. GRÁFICA DE FRECUENCIA DE LOS ESQUEMAS
# ============================================================

def graficar_frecuencia_esquemas(
    historial_M_esquemas
):
    """
    Esta tercera gráfica es opcional, pero es muy útil para
    relacionar directamente el código con M(H,t) del teorema
    de esquemas.
    """

    nombres = list(
        historial_M_esquemas.keys()
    )

    numero_datos = len(
        historial_M_esquemas[nombres[0]]
    )

    generaciones = list(
        range(numero_datos)
    )

    plt.figure(figsize=(9, 5))

    for nombre in nombres:

        plt.plot(
            generaciones,
            historial_M_esquemas[nombre],
            marker="o",
            label=f"M({nombre},t)"
        )

    plt.xlabel("Generación")

    plt.ylabel(
        "Número de individuos M(H,t)"
    )

    plt.title(
        "Frecuencia de los esquemas en la población"
    )

    plt.xticks(generaciones)

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# POBLACIÓN INICIAL
# ============================================================

poblacion = crear_poblacion(
    tamano_poblacion,
    longitud
)


# ============================================================
# HISTORIALES GENERALES
# ============================================================

historial_mejor = []

historial_promedio = []

historial_mejor_observado = []

mejor_observado = 0

mejor_individuo_observado = None


# ============================================================
# HISTORIALES DE LOS ESQUEMAS
# ============================================================

historial_M_esquemas = {
    nombre: []
    for nombre in ESQUEMAS
}

historial_fitness_esquemas = {
    nombre: []
    for nombre in ESQUEMAS
}


# ============================================================
# INFORMACIÓN INICIAL
# ============================================================

mostrar_esquemas()


# ============================================================
# CICLO EVOLUTIVO
# ============================================================

# Usamos numero_generaciones + 1 para registrar también
# la generación 0 antes de producir descendientes.

for generacion in range(
    numero_generaciones + 1
):

    print("\n")
    print("=" * 72)
    print(
        f"GENERACIÓN {generacion}"
    )
    print("=" * 72)

    mostrar_poblacion(
        poblacion
    )


    # --------------------------------------------------------
    # FITNESS GENERAL DE LA POBLACIÓN
    # --------------------------------------------------------

    mejor = max(
        poblacion,
        key=fitness
    )

    mejor_fitness = fitness(
        mejor
    )

    fitness_promedio_poblacion = (
        sum(
            fitness(ind)
            for ind in poblacion
        )
        /
        len(poblacion)
    )


    # Mejor observado durante toda la ejecución
    if mejor_individuo_observado is None:

        mejor_observado = mejor_fitness

        mejor_individuo_observado = (
            mejor.copy()
        )

    elif mejor_fitness > mejor_observado:

        mejor_observado = mejor_fitness

        mejor_individuo_observado = (
            mejor.copy()
        )


    historial_mejor.append(
        mejor_fitness
    )

    historial_promedio.append(
        fitness_promedio_poblacion
    )

    historial_mejor_observado.append(
        mejor_observado
    )


    # --------------------------------------------------------
    # FITNESS Y FRECUENCIA DE LOS ESQUEMAS
    # --------------------------------------------------------

    print("\nANÁLISIS DE ESQUEMAS")
    print("-" * 72)

    for nombre, esquema in ESQUEMAS.items():

        M, fitness_prom_esquema = (
            estadisticas_esquema(
                poblacion,
                esquema
            )
        )

        historial_M_esquemas[
            nombre
        ].append(M)

        historial_fitness_esquemas[
            nombre
        ].append(
            fitness_prom_esquema
        )

        print(
            f"{nombre}: "
            f"M(H,t)={M}   "
            f"F_prom(H,t)="
            f"{fitness_prom_esquema:.2f}"
        )


    # --------------------------------------------------------
    # RESUMEN DE LA GENERACIÓN
    # --------------------------------------------------------

    print("\nRESUMEN")

    print(
        "Mejor individuo:",
        mejor
    )

    print(
        "Peso:",
        calcular_peso(mejor)
    )

    print(
        "Valor:",
        calcular_valor(mejor)
    )

    print(
        "Mejor fitness de la generación:",
        mejor_fitness
    )

    print(
        "Fitness promedio de la población:",
        round(
            fitness_promedio_poblacion,
            2
        )
    )

    print(
        "Mejor fitness observado:",
        mejor_observado
    )


    # --------------------------------------------------------
    # NO GENERAR DESCENDIENTES DESPUÉS DE LA ÚLTIMA MEDICIÓN
    # --------------------------------------------------------

    if generacion == numero_generaciones:

        break


    # --------------------------------------------------------
    # CREAR NUEVA GENERACIÓN
    # --------------------------------------------------------

    nueva_poblacion = []

    for i in range(
        tamano_poblacion
    ):

        padre1, padre2 = (
            seleccionar_padres(
                poblacion
            )
        )

        hijo = cruza(
            padre1,
            padre2
        )

        hijo = mutacion(
            hijo,
            probabilidad_mutacion
        )

        nueva_poblacion.append(
            hijo
        )


    # Reemplazo generacional:
    # los hijos sustituyen a los padres.
    poblacion = nueva_poblacion


# ============================================================
# RESULTADO FINAL
# ============================================================

print("\n")
print("=" * 72)
print("MEJOR SOLUCIÓN OBSERVADA")
print("=" * 72)

print(
    "Individuo:",
    mejor_individuo_observado
)

print(
    "Peso:",
    calcular_peso(
        mejor_individuo_observado
    )
)

print(
    "Valor:",
    calcular_valor(
        mejor_individuo_observado
    )
)

print(
    "Fitness:",
    mejor_observado
)


# ============================================================
# TABLA FINAL DE LOS ESQUEMAS
# ============================================================

print("\n")
print("=" * 72)
print("RESUMEN FINAL DE ESQUEMAS")
print("=" * 72)

for nombre in ESQUEMAS:

    print(
        nombre,
        "| M final =",
        historial_M_esquemas[nombre][-1],
        "| F_prom final =",
        round(
            historial_fitness_esquemas[
                nombre
            ][-1],
            2
        )
    )


# ============================================================
# GRÁFICAS
# ============================================================

graficar_fitness_general(
    historial_mejor,
    historial_promedio,
    historial_mejor_observado
)

graficar_fitness_esquemas(
    historial_fitness_esquemas
)

# Esta gráfica adicional muestra directamente M(H,t),
# que es la cantidad central del teorema de esquemas.
graficar_frecuencia_esquemas(
    historial_M_esquemas
)
