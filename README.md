# Algoritmo Genético para VRPTW (EURO Meets NeurIPS 2022)

Este repositorio presenta un Algoritmo Genético (GA) diseñado para resolver el Problema de Enrutamiento de Vehículos con Ventanas de Tiempo (VRPTW). El código está adaptado para procesar y evaluar las instancias utilizadas en la competencia de enrutamiento de vehículos **EURO Meets NeurIPS 2022** (instancias de ORTEC).

## Características Principales

*   **Indiviudos en cromosomas (permutación):** Representación de los individuos mediante vector, preservando el orden y evitando la repetitividad de los nodos.
*   **Decodificación(Factibilidad):** Decodificador que separa el arreglo de cada indiviudo en nuevas rutas respetando en todo momento la capacidad máxima ($Q$) y los límites superiores de las ventanas de tiempo. No aplica penalizaciones.
*   **Operadores Genéticos:**
    * Selección eltista para conservación de mejores resultados
    *   Selección por Torneo para diversificación
    *   Cruza de Orden (OX) especializada en problemas de permutación para mantener la validez de los cromosomas.
    *   Mutación Swap para promover la exploración y mantener la no repetitividad.
*   **Validación Integrada:** Utiliza el módulo de evaluación oficial de la competencia (`tools.validate_static_solution`) para comprobar matemáticamente la validez de los resultados.
*   **Visualización:** Generación de gráficas de convergencia (historial de *fitness*) y ploteo de las rutas finales sobre las coordenadas geográficas de los clientes usando `matplotlib`. Generación de archovos de texto para la exportación de la mejor ruta con los nodos de cada una.

## Estructura del Proyecto
*   `data`: Contiene todas las instancias obtenidas del EURO Meets NeurIPS 2022.
*   `doc`: Algoritmos en formato `.py` tomados como base del proyecto (clase y talleres).
*   `figuras`: Gráfica del mapeo de los clientes.
*   `resultados`: Resultados `.txt` de la mejor solución, gráficos `.png` de las configuraciones y `.csv` con los datos finales.
*   `src`: Herramientas `tools.py`, algoritmo genético `ga_vrptw.py`, definición de configuraciones y métod de experimentación `experimentos.py`, lectura de las instancias `exploracion-de-datos.ipynb` y definición de constantes y valores `estadisticas_instacia.py`.

Intalacion de dependencias con:
pip install requirements.txt
