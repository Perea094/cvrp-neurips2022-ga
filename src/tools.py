# Retrieved from https://github.com/ortec/euro-neurips-vrp-2022-quickstart



import json  # Para serialización y deserialización en formato JSON
import os  # Para interactuar con el sistema de archivos (rutas, borrado de archivos/carpetas)
import numpy as np  # Para cálculos numéricos y manipulación de matrices/arreglos

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyJSONEncoder(json.JSONEncoder):
    """
    Codificador personalizado que extiende json.JSONEncoder para convertir
    tipos de datos nativos de NumPy a tipos estándar de Python serializables en JSON.
    """
    def default(self, obj):
        # Convierte enteros de NumPy (int32, int64, etc.) a int de Python
        if isinstance(obj, np.integer):
            return int(obj)
        # Convierte flotantes de NumPy (float32, float64, etc.) a float de Python
        if isinstance(obj, np.floating):
            return float(obj)
        # Convierte ndarrays de NumPy a listas recursivas de Python
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Para cualquier otro tipo, delega en la implementación por defecto de JSONEncoder
        return json.JSONEncoder.default(self, obj)


def json_dumps_np(data):
    """
    Serializa un objeto de Python a una cadena de texto en formato JSON,
    soportando estructuras con tipos de datos de NumPy gracias a NumpyJSONEncoder.
    """
    return json.dumps(data, cls=NumpyJSONEncoder)


def json_loads_np(json_string):
    """
    Deserializa una cadena de texto JSON a estructuras de Python,
    convirtiendo recursivamente todas las listas a arrays de NumPy.
    """
    return lists_to_np(json.loads(json_string))


def lists_to_np(obj):
    """
    Convierte recursivamente listas dentro de diccionarios o estructuras anidadas
    a arrays de NumPy (np.array).
    """
    # Si es un diccionario, procesa recursivamente sus valores
    if isinstance(obj, dict):
        return {k: lists_to_np(v) for k, v in obj.items()}
    # Si es una lista, la transforma directamente en un arreglo de NumPy
    if isinstance(obj, list):
        return np.array(obj)
    # Si es un valor escalar u otro tipo, lo retorna intacto
    return obj


def cleanup_tmp_dir(tmp_dir):
    """
    Elimina de manera segura un directorio temporal y los archivos de problemas
    VRPTW contenidos en él, evitando el uso directo de shutil.rmtree.
    """
    # Si el directorio no existe, termina la función inmediatamente
    if not os.path.isdir(tmp_dir):
        return
    # Itera sobre los archivos dentro del directorio temporal
    # Don't use shutil.rmtree for safety :)
    for filename in os.listdir(tmp_dir):
        filepath = os.path.join(tmp_dir, filename)
        # Solo borra archivos regulares que contengan 'problem.vrptw' en su nombre
        if 'problem.vrptw' in filename and os.path.isfile(filepath):
            os.remove(filepath)
    # Asegura que el directorio esté completamente vacío antes de eliminarlo
    assert len(os.listdir(tmp_dir)) == 0, "Unexpected files in tmp_dir"    
    # Elimina el directorio vacío
    os.rmdir(tmp_dir)


def to_giant_tour(routes, with_depot=True):
    """
    Convierte una lista de rutas independientes en un único 'giant tour',
    separando o delimitando las rutas opcionalmente con el depósito (nodo 0).
    """
    # Converts a list of routes to a giant tour with depots
    # Si la lista de rutas está vacía, devuelve una lista vacía
    if len(routes) == 0:
        return []
    # Inicia el recorrido con los nodos de la primera ruta
    tour = list(routes[0])
    # Recorre las rutas subsecuentes
    for route in routes[1:]:
        # Si se requiere incluir el depósito, se inserta el nodo 0 entre rutas
        if with_depot:
            tour.append(0)
        # Concatena los nodos de la siguiente ruta
        tour.extend(list(route))
    return tour


def compute_solution_driving_time(instance, solution):
    """
    Calcula el tiempo total de conducción de una solución completa,
    sumando el tiempo de conducción de cada una de las rutas individuales.
    """
    return sum([
        compute_route_driving_time(route, instance['duration_matrix']) 
        for route in solution
    ])


def validate_static_solution(instance, solution, allow_skipped_customers=False):
    """
    Valida una solución estática para un problema VRPTW:
    - Verifica visitas de clientes.
    - Valida capacidad máxima de carga en cada ruta.
    - Valida ventanas de tiempo de atención y viaje.
    Retorna el tiempo total de conducción si la solución es válida.
    """
    # Si no se permiten clientes omitidos, valida que todos hayan sido visitados
    if not allow_skipped_customers:
        validate_all_customers_visited(solution, len(instance['coords']) - 1)

    # Itera por cada ruta de la solución para validar restricciones físicas y temporales
    for route in solution:
        # Comprueba que la demanda acumulada no exceda la capacidad del vehículo
        validate_route_capacity(route, instance['demands'], instance['capacity'])
        # Comprueba que los arribos cumplan los horarios de apertura/cierre de ventanas
        validate_route_time_windows(route, instance['duration_matrix'], instance['time_windows'], instance['service_times'])

    # Si pasa todas las aserciones, calcula y retorna el tiempo total de manejo
    return compute_solution_driving_time(instance, solution)


def validate_dynamic_epoch_solution(epoch_instance, epoch_solution):
    """
    Valida una solución para una época en un entorno dinámico de VRPTW:
    - Remapea índices de solicitudes dinámicas a índices locales continuos (0, 1, ..., n).
    - Asegura que todas las solicitudes marcadas como 'must_dispatch' se hayan atendido.
    - Valida la solución como un subproblema estático.
    Retorna el tiempo total de conducción (sin tiempos de espera).
    """
    
    # Renumber requests (and depot) to 0,1...n
    request_idx = epoch_instance['request_idx']
    # El primer índice siempre debe corresponder al depósito (índice 0)
    assert request_idx[0] == 0
    # Asegura que los índices de solicitudes estén estrictamente ordenados de forma ascendente
    assert (request_idx[1:] > request_idx[:-1]).all()
    
    # Look up positions of request idx
    # Traduce los identificadores globales de las solicitudes a índices locales contiguos usando búsqueda binaria
    solution = [np.searchsorted(request_idx, route) for route in epoch_solution]
    
    # Check that all 'must_dispatch' requests are dispatched
    # if 'must_dispatch' in instance:
    # Crea una copia de la máscara booleana de solicitudes obligatorias a despachar
    must_dispatch = epoch_instance['must_dispatch'].copy()
    # Marca como False las solicitudes que han sido incluidas en alguna ruta
    for route in solution:
        must_dispatch[route] = False
    # Verifica que no quede ninguna solicitud obligatoria sin despachar
    assert not must_dispatch.any(), f"Some requests must be dispatched but were not: {request_idx[must_dispatch]}"

    # Filtra los metadatos específicos del entorno dinámico para construir una instancia estática equivalente
    static_instance = {
        k: v for k, v in epoch_instance.items() 
        if k not in ('request_idx', 'customer_idx', 'must_dispatch')
    }

    # Valida la instancia reducida permitiendo clientes omitidos (aquellos que pueden esperar a futuras épocas)
    return validate_static_solution(static_instance, solution, allow_skipped_customers=True)


def compute_route_driving_time(route, duration_matrix):
    """
    Calcula el tiempo de conducción de una sola ruta excluyendo tiempos de espera:
    Depósito -> Primer cliente -> ... -> Último cliente -> Depósito.
    """
    # From depot to first stop + first to last stop + last stop to depot
    return duration_matrix[0, route[0]] + duration_matrix[route[:-1], route[1:]].sum() + duration_matrix[route[-1], 0]


def validate_all_customers_visited(solution, num_customers):
    """
    Verifica que cada cliente (desde 1 hasta num_customers) sea visitado exactamente una vez.
    Lanza una aserción si falta algún cliente o si el total no coincide.
    """
    # Aplana todas las rutas en un único arreglo 1D con todos los nodos visitados
    flat_solution = np.array([stop for route in solution for stop in route])
    # Comprueba que el total de paradas sea igual al número de clientes
    assert len(flat_solution) == num_customers, "Not all customers are visited"
    # Arreglo de seguimiento booleano para marcar nodos visitados (índice 0 reservado para depósito)
    visited = np.zeros(num_customers + 1)  # Add padding for depot
    # Marca con True los nodos que aparecen en las rutas
    visited[flat_solution] = True
    # Valida que todos los clientes (índices 1 en adelante) hayan sido visitados
    assert visited[1:].all(), "Not all customers are visited"


def validate_route_capacity(route, demands, capacity):
    """
    Verifica que la suma total de las demandas de los clientes en una ruta
    no exceda la capacidad máxima del vehículo asignado.
    """
    assert sum(demands[route]) <= capacity, f"Capacity validated for route, {sum(demands[route])} > {capacity}"


def validate_route_time_windows(route, dist, timew, service_t, release_t=None):
    """
    Simula la ejecución temporal de una ruta para verificar el cumplimiento
    de ventanas de tiempo (earliest/latest arrival) y tiempos de liberación:
    - Ajusta el tiempo de salida inicial según el release_time si existe.
    - Añade tiempos de viaje y servicio.
    - Considera esperas si el vehículo llega antes de que abra la ventana.
    - Asegura que el vehículo regrese al depósito antes de su cierre.
    """
    depot = 0  # For readability, define variable: El depósito se identifica con el nodo 0
    earliest_start_depot, latest_arrival_depot = timew[depot]  # Ventana de operación del depósito
    
    # Si hay tiempos de liberación de pedidos, el vehículo no puede salir antes de que estén listos
    if release_t is not None:
        earliest_start_depot = max(earliest_start_depot, release_t[route].max())
    
    # Tiempo actual al iniciar el servicio/salida en el depósito
    current_time = earliest_start_depot + service_t[depot]

    prev_stop = depot
    # Itera sobre cada parada/cliente de la ruta
    for stop in route:
        earliest_arrival, latest_arrival = timew[stop]
        # Calcula el momento de llegada sumando el tiempo de viaje desde la parada anterior
        arrival_time = current_time + dist[prev_stop, stop]
        
        # Wait if we arrive before earliest_arrival
        # Si se llega antes de la apertura de la ventana, el vehículo espera hasta la hora de apertura
        current_time = max(arrival_time, earliest_arrival)
        # Comprueba que la llegada/atención no sobrepase el límite superior de la ventana
        assert current_time <= latest_arrival, f"Time window violated for stop {stop}: {current_time} not in ({earliest_arrival}, {latest_arrival})"
        # Añade la duración del servicio en la parada actual
        current_time += service_t[stop]
        prev_stop = stop
        
    # Agrega el tiempo de viaje desde el último cliente de regreso al depósito
    current_time += dist[prev_stop, depot]
    # Comprueba que el vehículo regrese al depósito antes del cierre de este
    assert current_time <= latest_arrival_depot, f"Time window violated for depot: {current_time} not in ({earliest_start_depot}, {latest_arrival_depot})"


def readlines(filename):
    """
    Lee todas las líneas de un archivo de texto con manejo de codificación:
    intenta primero con la codificación por defecto (UTF-8) y si falla,
    hace una lectura binaria ignorando caracteres no decodificables.
    """
    try:
        # Intento de lectura estándar en modo texto
        with open(filename, 'r') as f:
            return f.readlines()
    except:
        # Fallback en caso de error de decodificación o formato de bytes inesperado
        with open(filename, 'rb') as f:
            return [line.decode('utf-8', errors='ignore').strip() for line in f.readlines()]


def read_vrptw_solution(filename, return_extra=False):
    """
    Lee y parsea un archivo de solución VRPTW en formato VRPLib (una ruta por línea).
    Opcionalmente devuelve metadatos adicionales del archivo si existen pares clave-valor.
    """
    solution = []  # Lista que contendrá las rutas (cada ruta como array de numpy)
    extra = {}  # Diccionario para almacenar metadatos adicionales presentes en el archivo
    
    # Procesa cada línea del archivo
    for line in readlines(filename):
        # Las líneas de ruta comienzan con el prefijo 'Route'
        if line.startswith('Route'):
            # Extrae la secuencia de nodos posterior al caracter ':' y la convierte a array de enteros
            solution.append(np.array([int(node) for node in line.split(":")[-1].strip().split(" ")]))
        else:
            # Si la línea tiene formato clave-valor (separada por espacio), la almacena en 'extra'
            if len(line.strip().split(" ")) == 2:
                key, val = line.strip().split(" ")
                extra[key] = val
    
    # Retorna las rutas junto con la información extra si se solicita
    if return_extra:
        return solution, extra
    return solution   


def read_vrplib(filename, rounded=True):
    """
    Lee y parsea una instancia de problema en formato estándar VRPLib / Solomon / TSPLIB.
    Extrae coordenadas, demandas, capacidad, ventanas de tiempo, tiempos de servicio y distancias.
    Retorna un diccionario estructurado con los arreglos correspondientes en NumPy.
    """
    # Inicialización de listas y variables para almacenar las distintas secciones del archivo
    loc = []  # Coordenadas de los clientes
    demand = []  # Demandas de clientes y depósito
    mode = ''  # Estado actual de lectura para saber qué sección se está parseando
    capacity = None  # Capacidad máxima del vehículo
    edge_weight_type = None  # Tipo de pesos en las aristas (e.g., EXPLICIT, EUC_2D)
    edge_weight_format = None  # Formato de la matriz de distancias (e.g., FULL_MATRIX)
    duration_matrix = []  # Matriz de tiempos o distancias entre nodos
    service_t = []  # Tiempos de servicio por nodo
    timewi = []  # Ventanas de tiempo [lower_bound, upper_bound] por nodo
    
    with open(filename, 'r') as f:
        # Itera línea por línea sobre el archivo
        for line in f:
            line = line.strip(' \t\n')
            # Ignora líneas vacías
            if line == "":
                continue
            # Parseo de metadatos de cabecera
            elif line.startswith('CAPACITY'):
                capacity = int(line.split(" : ")[1])
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                edge_weight_type = line.split(" : ")[1]
            elif line.startswith('EDGE_WEIGHT_FORMAT'):
                edge_weight_format = line.split(" : ")[1]
            # Cambio de secciones del archivo según encabezados estándar de VRPLib
            elif line == 'NODE_COORD_SECTION':
                mode = 'coord'
            elif line == 'DEMAND_SECTION':
                mode = 'demand'
            elif line == 'DEPOT_SECTION':
                mode = 'depot'
            elif line == "EDGE_WEIGHT_SECTION":
                mode = 'edge_weights'
                # Verifica que los pesos sean explícitos y en formato matriz completa
                assert edge_weight_type == "EXPLICIT"
                assert edge_weight_format == "FULL_MATRIX"
            elif line == "TIME_WINDOW_SECTION":
                mode = "time_windows"
            elif line == "SERVICE_TIME_SECTION":
                mode = "service_t"
            elif line == "EOF":
                # Fin del archivo alcanzado
                break
            # Parseo de la sección de coordenadas geográficas o euclidianas
            elif mode == 'coord':
                node, x, y = line.split()  # Split by whitespace or \t, skip duplicate whitespace
                node = int(node)
                # Convierte coordenadas a enteros o flotantes según el argumento 'rounded'
                x, y = (int(x), int(y)) if rounded else (float(x), float(y))
                
                # El nodo 1 corresponde al depósito por convención
                if node == 1:
                    depot = (x, y)
                else:
                    # Verifica la correlación ordenada de los índices de clientes
                    assert node == len(loc) + 2 # 1 is depot, 2 is 0th location
                    loc.append((x, y))
            # Parseo de la sección de demandas de cada nodo
            elif mode == 'demand':
                node, d = [int(v) for v in line.split()]
                if node == 1:
                    # El depósito siempre debe tener demanda 0
                    assert d == 0
                demand.append(d)
            # Parseo de la sección de matriz de duraciones/distancias
            elif mode == 'edge_weights':
                duration_matrix.append(list(map(int if rounded else float, line.split())))
            # Parseo de los tiempos de servicio
            elif mode == 'service_t':
                node, t = line.split()
                node = int(node)
                t = int(t) if rounded else float(t)
                if node == 1:
                    # El depósito no tiene tiempo de servicio
                    assert t == 0
                assert node == len(service_t) + 1
                service_t.append(t)
            # Parseo de las ventanas de tiempo [l = inicio, u = fin]
            elif mode == 'time_windows':
                node, l, u = line.split()
                node = int(node)
                l, u = (int(l), int(u)) if rounded else (float(l), float(u))
                assert node == len(timewi) + 1
                timewi.append([l, u])
    
    # Construcción y retorno del diccionario con la instancia procesada en arrays NumPy
    return {
        'is_depot': np.array([1] + [0] * len(loc), dtype=bool),  # Máscara booleana: True solo para el depósito (nodo 0)
        'coords': np.array([depot] + loc),  # Arreglo con coordenadas de depósito seguido por clientes
        'demands': np.array(demand),  # Demandas de cada parada
        'capacity': capacity,  # Capacidad máxima del vehículo
        'time_windows': np.array(timewi),  # Arreglo bidimensional de ventanas temporales
        'service_times': np.array(service_t),  # Duraciones del servicio
        'duration_matrix': np.array(duration_matrix) if len(duration_matrix) > 0 else None  # Matriz de costos/tiempos
    }

def write_vrplib(filename, instance, name="problem", euclidean=False, is_vrptw=True):
    """
    Escribe una instancia de problema en disco siguiendo el estándar de formato VRPLib.
    Soporta configuraciones para problemas CVRP y VRPTW, distancias euclidianas o explícitas,
    así como secciones de tiempos de servicio, ventanas de tiempo y tiempos de liberación.
    """
    # LKH/VRP does not take floats (HGS seems to do)
    
    # Extracción de los componentes de la instancia
    coords = instance['coords']
    demands = instance['demands']
    is_depot = instance['is_depot']
    duration_matrix = instance['duration_matrix']
    capacity = instance['capacity']
    
    # Validación de consistencia: diagonal de la matriz de tiempos debe ser 0 (distancia nodo a sí mismo)
    assert (np.diag(duration_matrix) == 0).all()
    # Validación de consistencia: todos los clientes (nodos que no son depósito) deben tener demanda > 0
    assert (demands[~is_depot] > 0).all()
        
    # Escritura del archivo de texto
    with open(filename, 'w') as f:
        # Escritura de los metadatos de cabecera
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in [
                ("NAME", name),
                ("COMMENT", "ORTEC"),  # For HGS we need an extra row...
                ("TYPE", "CVRP"),
                ("DIMENSION", len(coords)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D" if euclidean else "EXPLICIT"),
            ] + ([] if euclidean else [
                ("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")
            ]) + [("CAPACITY", capacity)]
        ]))
        f.write("\n")
        
        # Si las distancias no son euclidianas, escribe la matriz de pesos completa de las aristas
        if not euclidean:
            f.write("EDGE_WEIGHT_SECTION\n")
            for row in duration_matrix:
                f.write("\t".join(map(str, row)))
                f.write("\n")
        
        # Sección de coordenadas de los nodos (índice 1-based para compatibilidad con VRPLib)
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate(coords)
        ]))
        f.write("\n")
        
        # Sección de demandas por nodo (índice 1-based)
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate(demands)
        ]))
        f.write("\n")
        
        # Sección indicadora del depósito terminada en -1 según especificación VRPLib
        f.write("DEPOT_SECTION\n")
        for i in np.flatnonzero(is_depot):
            f.write(f"{i+1}\n")
        f.write("-1\n")
        
        # Escritura de secciones adicionales específicas de VRPTW (ventanas de tiempo y servicio)
        if is_vrptw:
            
            service_t = instance['service_times']
            timewi = instance['time_windows']
            
            # Following LKH convention: sección de tiempos de servicio por nodo
            f.write("SERVICE_TIME_SECTION\n")
            f.write("\n".join([
                "{}\t{}".format(i + 1, s)
                for i, s in enumerate(service_t)
            ]))
            f.write("\n")
            
            # Sección de ventanas de tiempo [límite inferior, límite superior] por nodo
            f.write("TIME_WINDOW_SECTION\n")
            f.write("\n".join([
                "{}\t{}\t{}".format(i + 1, l, u)
                for i, (l, u) in enumerate(timewi)
            ]))
            f.write("\n")

            # Si la instancia incluye tiempos de liberación (release dates), agrega su respectiva sección
            if 'release_times' in instance:
                release_times = instance['release_times']

                f.write("RELEASE_TIME_SECTION\n")
                f.write("\n".join([
                    "{}\t{}".format(i + 1, s)
                    for i, s in enumerate(release_times)
                ]))
                f.write("\n")
            
        # Marca de fin de archivo estándar (End Of File)
        f.write("EOF\n")