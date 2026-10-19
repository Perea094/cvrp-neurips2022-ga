# Retrieved from https://github.com/ortec/euro-neurips-vrp-2022-quickstart



import json
import os
import numpy as np

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyJSONEncoder(json.JSONEncoder):
    """
    Qué hace:
        Codificador personalizado para JSON que permite serializar tipos de datos
        propios de NumPy (enteros, flotantes y ndarrays) convirtiéndolos a tipos
        nativos de Python (int, float, list).

    Consideraciones:
        - Solo intercepta instancias de np.integer, np.floating y np.ndarray;
          cualquier otro tipo no estándar se delega al encoder por defecto.
        - Para ndarrays grandes, la conversión a listas anidadas de Python puede
          incrementar significativamente el uso de memoria.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def json_dumps_np(data):
    """
    Qué hace:
        Serializa un objeto o estructura de datos de Python que contiene tipos
        de NumPy a una cadena en formato JSON.

    Consideraciones:
        - Utiliza internamente la clase NumpyJSONEncoder para evitar errores de
          tipo "TypeError: Object of type ndarray is not JSON serializable".
    """
    return json.dumps(data, cls=NumpyJSONEncoder)


def json_loads_np(json_string):
    """
    Qué hace:
        Deserializa una cadena JSON y convierte de manera recursiva todas las
        listas contenidas en arrays de NumPy.

    Consideraciones:
        - Depende de la función lists_to_np, por lo que toda lista (incluidas listas
          heterogéneas o anidadas) se transformará en un np.ndarray.
    """
    return lists_to_np(json.loads(json_string))


def lists_to_np(obj):
    """
    Qué hace:
        Recorre recursivamente una estructura de datos (diccionarios y listas)
        convirtiendo cada lista encontrada en un array de NumPy.

    Consideraciones:
        - Si un diccionario contiene listas internas a distintos niveles de profundidad,
          todas serán transformadas.
        - Las listas con elementos de distintos tipos generarán arrays de tipo object
          o strings según la inferencia de tipos de NumPy.
    """
    if isinstance(obj, dict):
        return {k: lists_to_np(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return np.array(obj)
    return obj


def cleanup_tmp_dir(tmp_dir):
    """
    Qué hace:
        Elimina los archivos temporales de instancias VRPTW que contengan la cadena
        'problem.vrptw' dentro del directorio temporal indicado y luego borra el directorio.

    Consideraciones:
        - Si la ruta no es un directorio válido, finaliza silenciosamente sin error.
        - Por seguridad, no utiliza shutil.rmtree; sin embargo, lanza un AssertionError
          si quedan archivos no reconocidos en la carpeta antes de llamar a os.rmdir.
        - Falla si alguno de los archivos o el directorio no tienen permisos de escritura/borrado.
    """
    if not os.path.isdir(tmp_dir):
        return
    # Don't use shutil.rmtree for safety :)
    for filename in os.listdir(tmp_dir):
        filepath = os.path.join(tmp_dir, filename)
        if 'problem.vrptw' in filename and os.path.isfile(filepath):
            os.remove(filepath)
    assert len(os.listdir(tmp_dir)) == 0, "Unexpected files in tmp_dir"    
    os.rmdir(tmp_dir)


def to_giant_tour(routes, with_depot=True):
    """
    Qué hace:
        Concatena una lista de rutas independientes en una única secuencia o
        "recorrido gigante" (giant tour), insertando opcionalmente el nodo del depósito (0)
        entre cada ruta.

    Consideraciones:
        - Asume de manera estricta que el depósito corresponde al identificador 0.
        - No antepone el depósito al inicio de la primera ruta ni lo coloca al final de la última;
          únicamente lo inserta como separador intermedio si with_depot=True.
        - Si la lista de rutas está vacía, devuelve una lista vacía.
    """
    if len(routes) == 0:
        return []
    tour = list(routes[0])
    for route in routes[1:]:
        if with_depot:
            tour.append(0)
        tour.extend(list(route))
    return tour


def compute_solution_driving_time(instance, solution):
    """
    Qué hace:
        Calcula el tiempo total de conducción de una solución completa sumando
        los tiempos de manejo individuales de cada ruta.

    Consideraciones:
        - La instancia debe contener la clave 'duration_matrix'.
        - Excluye tiempos de espera generados por ventanas de tiempo y tiempos de servicio;
          únicamente mide desplazamiento efectivo.
    """
    return sum([
        compute_route_driving_time(route, instance['duration_matrix']) 
        for route in solution
    ])


def validate_static_solution(instance, solution, allow_skipped_customers=False):
    """
    Qué hace:
        Valida que una solución estática para un problema VRPTW sea factible, comprobando
        la cobertura de clientes (opcional), el límite de capacidad de los vehículos y
        las ventanas de tiempo. Retorna el tiempo total de conducción si la solución es válida.

    Consideraciones:
        - Lanza AssertionError si se incumple alguna restricción física o temporal.
        - Si allow_skipped_customers es False, exige que todos los clientes de la instancia
          hayan sido visitados exactamente una vez.
        - Asume que la instancia cuenta con las claves: 'coords', 'demands', 'capacity',
          'duration_matrix', 'time_windows' y 'service_times'.
    """
    if not allow_skipped_customers:
        validate_all_customers_visited(solution, len(instance['coords']) - 1)

    for route in solution:
        validate_route_capacity(route, instance['demands'], instance['capacity'])
        validate_route_time_windows(route, instance['duration_matrix'], instance['time_windows'], instance['service_times'])

    return compute_solution_driving_time(instance, solution)


def validate_dynamic_epoch_solution(epoch_instance, epoch_solution):
    """
    Qué hace:
        Valida una solución parcial generada para un lote o época dinámica en un problema
        DVRPTW. Comprueba que las solicitudes obligatorias ('must_dispatch') hayan sido asignadas,
        remapea los índices de los clientes a coordenadas locales y valida la factibilidad estática.

    Consideraciones:
        - Exige mediante aserciones que 'request_idx' comience con 0 (depósito) y esté estrictamente ordenado.
        - Utiliza np.searchsorted para convertir los IDs globales de solicitudes en índices locales (0 a n).
        - Permite clientes sin visitar (allow_skipped_customers=True) siempre y cuando no estén
          marcados como 'must_dispatch'.
    """
    # Renumber requests (and depot) to 0,1...n
    request_idx = epoch_instance['request_idx']
    assert request_idx[0] == 0
    assert (request_idx[1:] > request_idx[:-1]).all()
    # Look up positions of request idx
    solution = [np.searchsorted(request_idx, route) for route in epoch_solution]
    
    # Check that all 'must_dispatch' requests are dispatched
    # if 'must_dispatch' in instance:
    must_dispatch = epoch_instance['must_dispatch'].copy()
    for route in solution:
        must_dispatch[route] = False
    assert not must_dispatch.any(), f"Some requests must be dispatched but were not: {request_idx[must_dispatch]}"

    static_instance = {
        k: v for k, v in epoch_instance.items() 
        if k not in ('request_idx', 'customer_idx', 'must_dispatch')
    }

    return validate_static_solution(static_instance, solution, allow_skipped_customers=True)


def compute_route_driving_time(route, duration_matrix):
    """
    Qué hace:
        Calcula el tiempo total de conducción de una única ruta, sumando el viaje del
        depósito (0) al primer cliente, el trayecto entre clientes intermedios y el
        retorno del último cliente al depósito (0).

    Consideraciones:
        - Excluye tiempos de espera y tiempos de servicio.
        - Asume de forma fija que el depósito corresponde al índice 0.
        - Requiere que duration_matrix sea un array bidimensional indexable por NumPy.
        - Falla si la ruta está vacía (indexación route[0] y route[-1]).
    """
    return duration_matrix[0, route[0]] + duration_matrix[route[:-1], route[1:]].sum() + duration_matrix[route[-1], 0]


def validate_all_customers_visited(solution, num_customers):
    """
    Qué hace:
        Comprueba que todos los clientes (del 1 al num_customers) aparezcan visitados
        exactamente una sola vez en el conjunto de rutas de la solución.

    Consideraciones:
        - Lanza AssertionError si algún cliente no es visitado o si existen clientes duplicados.
        - Las rutas no deben incluir explícitamente el depósito (nodo 0); si el nodo 0 aparece
          dentro de una ruta, desajustará el conteo total len(flat_solution) == num_customers.
    """
    flat_solution = np.array([stop for route in solution for stop in route])
    assert len(flat_solution) == num_customers, "Not all customers are visited"
    visited = np.zeros(num_customers + 1)  # Add padding for depot
    visited[flat_solution] = True
    assert visited[1:].all(), "Not all customers are visited"


def validate_route_capacity(route, demands, capacity):
    """
    Qué hace:
        Verifica que la demanda total acumulada de los clientes asignados a una ruta
        no exceda la capacidad máxima del vehículo.

    Consideraciones:
        - Lanza AssertionError si sum(demands[route]) > capacity.
        - demands debe ser una estructura indexable por listas/arrays (como un array de NumPy).
    """
    assert sum(demands[route]) <= capacity, f"Capacity validated for route, {sum(demands[route])} > {capacity}"


def validate_route_time_windows(route, dist, timew, service_t, release_t=None):
    """
    Qué hace:
        Simula cronológicamente el recorrido de un vehículo a lo largo de una ruta para validar
        que se respeten los horarios de atención (ventanas de tiempo), sumando tiempos de viaje,
        esperas por llegada temprana, tiempos de servicio y horas de disponibilidad (release_times).

    Consideraciones:
        - Asume que si el vehículo llega antes del inicio de una ventana de tiempo, espera hasta
          earliest_arrival sin penalización ni error.
        - Lanza AssertionError si el vehículo llega después de latest_arrival en cualquier parada
          o al regresar al depósito.
        - Si release_t está presente, la hora de salida del depósito se posterga hasta el máximo
          tiempo de disponibilidad de los pedidos de esa ruta.
    """
    depot = 0  # For readability, define variable
    earliest_start_depot, latest_arrival_depot = timew[depot]
    if release_t is not None:
        earliest_start_depot = max(earliest_start_depot, release_t[route].max())
    current_time = earliest_start_depot + service_t[depot]

    prev_stop = depot
    for stop in route:
        earliest_arrival, latest_arrival = timew[stop]
        arrival_time = current_time + dist[prev_stop, stop]
        # Wait if we arrive before earliest_arrival
        current_time = max(arrival_time, earliest_arrival)
        assert current_time <= latest_arrival, f"Time window violated for stop {stop}: {current_time} not in ({earliest_arrival}, {latest_arrival})"
        current_time += service_t[stop]
        prev_stop = stop
    current_time += dist[prev_stop, depot]
    assert current_time <= latest_arrival_depot, f"Time window violated for depot: {current_time} not in ({earliest_start_depot}, {latest_arrival_depot})"


def readlines(filename):
    """
    Qué hace:
        Lee y devuelve todas las líneas de un archivo de texto, implementando un mecanismo
        de respaldo en modo binario con decodificación UTF-8 tolerante a fallos si la lectura estándar falla.

    Consideraciones:
        - Si la lectura en modo texto falla por codificación, el bloque except decodifica ignorando errores
          y aplica .strip() a cada línea (eliminando posibles espacios o saltos al inicio y final).
        - Usa una cláusula except genérica sin especificar la excepción.
    """
    try:
        with open(filename, 'r') as f:
            return f.readlines()
    except:
        with open(filename, 'rb') as f:
            return [line.decode('utf-8', errors='ignore').strip() for line in f.readlines()]


def read_vrptw_solution(filename, return_extra=False):
    """
    Qué hace:
        Lee e interpreta un archivo de solución en formato estándar de VRPLib (una ruta por línea
        con prefijo 'Route'), extrayendo las secuencias de clientes por ruta y metadatos adicionales.

    Consideraciones:
        - Identifica las rutas buscando líneas que comiencen con 'Route'. Los números de nodos
          deben estar separados por espacios tras los dos puntos ':'.
        - Las líneas que contienen exactamente dos tokens separados por espacio se almacenan
          como pares clave-valor en el diccionario 'extra'.
        - Devuelve únicamente la lista de rutas, a menos que return_extra=True.
    """
    solution = []
    extra = {}
    
    for line in readlines(filename):
        if line.startswith('Route'):
            solution.append(np.array([int(node) for node in line.split(":")[-1].strip().split(" ")]))
        else:
            if len(line.strip().split(" ")) == 2:
                key, val = line.strip().split(" ")
                extra[key] = val
    
    if return_extra:
        return solution, extra
    return solution   


def read_vrplib(filename, rounded=True):
    """
    Qué hace:
        Parsea un archivo de instancia de problema en formato VRPLib / Solomon, extrayendo
        coordenadas de clientes, demandas, capacidad del vehículo, tiempos de servicio,
        ventanas de tiempo y matrices de distancia o duración explícitas.

    Consideraciones:
        - El parámetro rounded determina si los valores numéricos se convierten a enteros (int)
          o se conservan con punto flotante (float).
        - Asume que los identificadores de nodos en el archivo inician en 1 (donde el nodo 1 es el depósito)
          y los adapta a indexación basada en 0 para Python.
        - Requiere que las matrices de distancia explícitas usen EDGE_WEIGHT_FORMAT = FULL_MATRIX.
    """
    loc = []
    demand = []
    mode = ''
    capacity = None
    edge_weight_type = None
    edge_weight_format = None
    duration_matrix = []
    service_t = []
    timewi = []

    # New implemented variables
    name = ''
    vehicles = None

    with open(filename, 'r') as f:
        
        for line in f:
            line = line.strip(' \t\n')
            if line == "":
                continue
            elif line.startswith('CAPACITY'):
                capacity = int(line.split(" : ")[1])
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                edge_weight_type = line.split(" : ")[1]

            # New implemented variables
            elif line.startswith('NAME'):
                name = line.split(" : ")[1]
            elif line.startswith('VEHICLES'):
                vehicles = int(line.split(" : ")[1])

            elif line.startswith('EDGE_WEIGHT_FORMAT'):
                edge_weight_format = line.split(" : ")[1]
            elif line == 'NODE_COORD_SECTION':
                mode = 'coord'
            elif line == 'DEMAND_SECTION':
                mode = 'demand'
            elif line == 'DEPOT_SECTION':
                mode = 'depot'
            elif line == "EDGE_WEIGHT_SECTION":
                mode = 'edge_weights'
                assert edge_weight_type == "EXPLICIT"
                assert edge_weight_format == "FULL_MATRIX"
            elif line == "TIME_WINDOW_SECTION":
                mode = "time_windows"
            elif line == "SERVICE_TIME_SECTION":
                mode = "service_t"
            elif line == "EOF":
                break
            elif mode == 'coord':
                node, x, y = line.split()  # Split by whitespace or \t, skip duplicate whitespace
                node = int(node)
                x, y = (int(x), int(y)) if rounded else (float(x), float(y))
                
                if node == 1:
                    depot = (x, y)
                else:
                    assert node == len(loc) + 2 # 1 is depot, 2 is 0th location
                    loc.append((x, y))
            elif mode == 'demand':
                node, d = [int(v) for v in line.split()]
                if node == 1:
                    assert d == 0
                demand.append(d)
            elif mode == 'edge_weights':
                duration_matrix.append(list(map(int if rounded else float, line.split())))
            elif mode == 'service_t':
                node, t = line.split()
                node = int(node)
                t = int(t) if rounded else float(t)
                if node == 1:
                    assert t == 0
                assert node == len(service_t) + 1
                service_t.append(t)
            elif mode == 'time_windows':
                node, l, u = line.split()
                node = int(node)
                l, u = (int(l), int(u)) if rounded else (float(l), float(u))
                assert node == len(timewi) + 1
                timewi.append([l, u])
    
    return {
        'is_depot': np.array([1] + [0] * len(loc), dtype=bool),
        'coords': np.array([depot] + loc),
        'demands': np.array(demand),
        'capacity': capacity,
        'time_windows': np.array(timewi),
        'service_times': np.array(service_t),
        'duration_matrix': np.array(duration_matrix) if len(duration_matrix) > 0 else None,
        'name': name, # New implemented variable
        'vehicles': vehicles
    }

def write_vrplib(filename, instance, name="problem", euclidean=False, is_vrptw=True):
    """
    Qué hace:
        Escribe un diccionario con los datos de una instancia de ruteo de vehículos
        en un archivo estructurado bajo la especificación de formato VRPLib.

    Consideraciones:
        - Convierte internamente los índices basados en 0 de Python a identificadores basados
          en 1 para cumplir con el estándar de solvers externos como LKH o HGS.
        - Lanza aserciones si la diagonal de la matriz de duraciones no es cero o si algún cliente
          (nodo que no es depósito) tiene demanda menor o igual a cero.
        - Si euclidean=False, escribe obligatoriamente la matriz completa de distancias bajo EDGE_WEIGHT_SECTION.
        - Si is_vrptw=True, añade las secciones SERVICE_TIME_SECTION, TIME_WINDOW_SECTION y
          RELEASE_TIME_SECTION (esta última solo si la clave 'release_times' existe en la instancia).
    """
    # LKH/VRP does not take floats (HGS seems to do)
    
    coords = instance['coords']
    demands = instance['demands']
    is_depot = instance['is_depot']
    duration_matrix = instance['duration_matrix']
    capacity = instance['capacity']
    assert (np.diag(duration_matrix) == 0).all()
    assert (demands[~is_depot] > 0).all()
        
    with open(filename, 'w') as f:
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
        
        if not euclidean:
            f.write("EDGE_WEIGHT_SECTION\n")
            for row in duration_matrix:
                f.write("\t".join(map(str, row)))
                f.write("\n")
        
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate(coords)
        ]))
        f.write("\n")
        
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate(demands)
        ]))
        f.write("\n")
        
        f.write("DEPOT_SECTION\n")
        for i in np.flatnonzero(is_depot):
            f.write(f"{i+1}\n")
        f.write("-1\n")
        
        if is_vrptw:
            
            service_t = instance['service_times']
            timewi = instance['time_windows']
            
            # Following LKH convention
            f.write("SERVICE_TIME_SECTION\n")
            f.write("\n".join([
                "{}\t{}".format(i + 1, s)
                for i, s in enumerate(service_t)
            ]))
            f.write("\n")
            
            f.write("TIME_WINDOW_SECTION\n")
            f.write("\n".join([
                "{}\t{}\t{}".format(i + 1, l, u)
                for i, (l, u) in enumerate(timewi)
            ]))
            f.write("\n")

            if 'release_times' in instance:
                release_times = instance['release_times']

                f.write("RELEASE_TIME_SECTION\n")
                f.write("\n".join([
                    "{}\t{}".format(i + 1, s)
                    for i, s in enumerate(release_times)
                ]))
                f.write("\n")
            
        f.write("EOF\n")