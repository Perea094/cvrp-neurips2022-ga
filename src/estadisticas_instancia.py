import math
import os
import numpy as np
import matplotlib.pyplot as plt
import tools  

ARCHIVO = "data/instances/test/ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35.txt"

instancia = tools.read_vrplib(ARCHIVO)

demandas = instancia["demands"]
ventanas = instancia["time_windows"]
servicio = instancia["service_times"]
tiempos = instancia["duration_matrix"]
Q = instancia["capacity"]

n = len(demandas) - 1           #sin deposito
demanda_total = demandas.sum()
k_min = math.ceil(demanda_total / Q)  
amplitud = ventanas[1:, 1] - ventanas[1:, 0]

print("Archivo:", instancia["name"])
print("Clientes n:", n)
print("Capacidad Q:", Q)
print("VEHICLES en el archivo:", instancia["vehicles"])
print("Demanda total:", demanda_total)
print("Vehiculos minimos:", k_min)
print("Demanda por cliente (min, max, promedio):", demandas[1:].min(), demandas[1:].max(), round(demandas[1:].mean(), 2))
print("Horizonte del deposito:", ventanas[0])
print("Amplitud de ventanas (promedio, min, max) en s:", round(amplitud.mean(), 1), amplitud.min(), amplitud.max())
print("Tiempo de servicio (promedio, min, max) en s:", round(servicio[1:].mean(), 1), servicio[1:].min(), servicio[1:].max())

# Asimetria
fuera_diagonal = ~np.eye(len(tiempos), dtype=bool)
asimetricos = (tiempos != tiempos.T)[fuera_diagonal].mean() * 100
print("Tiempo de viaje promedio entre nodos (s):", round(tiempos[fuera_diagonal].mean(), 1))
print("Pares asimetricos (%):", round(asimetricos, 1))

# Mapa donde color = hora de apertura de la ventana
coords = instancia["coords"]
plt.figure(figsize=(7, 6))
puntos = plt.scatter(coords[1:, 0], coords[1:, 1], c=ventanas[1:, 0] / 3600, cmap="viridis", s=18)
plt.scatter(coords[0, 0], coords[0, 1], c="red", marker="s", s=90, label="Depósito")
plt.colorbar(puntos, label="Apertura de la ventana (horas desde el inicio)")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Clientes de la instancia " + instancia["name"].split("-")[3])
plt.tight_layout()
os.makedirs("figuras", exist_ok=True)
plt.savefig("figuras/mapa_clientes.png", dpi=200)
print("Mapa guardado en figuras/mapa_clientes.png")
