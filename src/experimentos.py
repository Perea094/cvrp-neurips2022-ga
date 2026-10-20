"""
AG con 3 config de hiperparametros y 10 semillas cada 1 
"""

import os
import csv
import time
import statistics
import numpy as np
import matplotlib.pyplot as plt
import tools
from ga_vrptw import cargar_datos, algoritmo_genetico, graficar_rutas, ARCHIVO

# hiperparametros: tamaño poblacion, pm y k torneo
CONFIGS = {
    "Config 1": {"tam_poblacion": 50,  "pm": 0.02,  "k": 2},   # mas exploracion
    "Config 2": {"tam_poblacion": 100, "pm": 0.01,  "k": 3},   # intermedia
    "Config 3": {"tam_poblacion": 150, "pm": 0.005, "k": 5},   # mas explotacion
}

# condiciones iguales para todas las configuraciones
GENERACIONES = 500
PC = 0.9
ELITISMO = 2
SEMILLAS = list(range(1, 11))    #10 ej dif

CARPETA = "resultados"


def main():
    os.makedirs(CARPETA, exist_ok=True)
    datos = cargar_datos(ARCHIVO)

    filas = []   #fila por ejecución
    historiales = {}           
    mejor_global = None   # (Z, rutas, config, semilla)

    for nombre, cfg in CONFIGS.items():
        historiales[nombre] = []
        for semilla in SEMILLAS:
            inicio = time.time()
            mejor, rutas, Z, historial = algoritmo_genetico(
                datos, cfg["tam_poblacion"], GENERACIONES, PC, cfg["pm"], cfg["k"],
                ELITISMO, semilla, imprimir=False)
            duracion = time.time() - inicio

            # factibilidad
            try:
                tools.validate_static_solution(datos["instancia"], [np.array(r) for r in rutas])
                factible = 1
            except AssertionError:
                factible = 0

            filas.append([nombre, semilla, Z, len(rutas), round(duracion, 2), factible])
            historiales[nombre].append(historial["mejor_observado"])
            print(f"{nombre} | semilla {semilla:2d} | Z = {Z:,} | rutas = {len(rutas)} | {duracion:.1f} s")

            if mejor_global is None or Z < mejor_global[0]:
                mejor_global = (Z, rutas, nombre, semilla)

    # tabla por ejecucion
    with open(os.path.join(CARPETA, "resultados.csv"), "w", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["config", "semilla", "Z", "rutas", "tiempo_s", "factible"])
        escritor.writerows(filas)

    # resumen por configuracion
    print("\nRESUMEN")
    print("Config   | Mejor     | Promedio  | Desv.est. | Tiempo prom (s) | Factibilidad (%)")
    resumen = []
    for nombre in CONFIGS:
        Zs = [f[2] for f in filas if f[0] == nombre]
        tiempos = [f[4] for f in filas if f[0] == nombre]
        fact = [f[5] for f in filas if f[0] == nombre]
        r = [nombre, min(Zs), round(statistics.mean(Zs)), round(statistics.stdev(Zs)),
             round(statistics.mean(tiempos), 1), 100 * sum(fact) / len(fact)]
        resumen.append(r)
        print(f"{r[0]} | {r[1]:,} | {r[2]:,} | {r[3]:,} | {r[4]} | {r[5]:.0f}")

    with open(os.path.join(CARPETA, "resumen.csv"), "w", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["config", "mejor", "promedio", "desv_est", "tiempo_prom_s", "factibilidad_pct"])
        escritor.writerows(resumen)

    # grafic convergencia
    plt.figure(figsize=(9, 5))
    for nombre, curvas in historiales.items():
        promedio = np.mean(np.array(curvas), axis=0)
        plt.plot(promedio, label=nombre)
    plt.xlabel("Generación")
    plt.ylabel("Mejor Z observado (s), promedio de 10 ejecuciones")
    plt.title("Convergencia por configuración")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA, "convergencia.png"), dpi=200)
    plt.close()

    # mejor z al final
    plt.figure(figsize=(7, 5))
    plt.boxplot([[f[2] for f in filas if f[0] == nombre] for nombre in CONFIGS])
    plt.xticks(range(1, len(CONFIGS) + 1), list(CONFIGS.keys()))
    plt.ylabel("Mejor Z final (s)")
    plt.title("Distribución del mejor Z por configuración (10 ejecuciones)")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA, "boxplot.png"), dpi=200)
    plt.close()

    # mejor solucion encontrada
    Z, rutas, nombre, semilla = mejor_global
    print(f"\nMejor solución: Z = {Z:,} ({nombre}, semilla {semilla}), {len(rutas)} rutas "
          f"(vehículos en el archivo: {datos['vehiculos']})")
    with open(os.path.join(CARPETA, "mejor_solucion.txt"), "w") as f:
        for i, ruta in enumerate(rutas, start=1):
            f.write(f"Route #{i}: {' '.join(map(str, ruta))}\n")
        f.write(f"Cost {Z}\n")
    plt.switch_backend("Agg")
    graficar_rutas(rutas, datos, titulo=f"Mejor solución, Z = {Z:,}",
                   archivo=os.path.join(CARPETA, "mejor_solucion.png"))


if __name__ == "__main__":
    main()
