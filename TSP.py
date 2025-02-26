import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Plots the graph showing only the best path
def plot_graph(best_route, distance_matrix, num_cities):
    graph = nx.Graph()
    positions = nx.circular_layout(range(num_cities))

    for i in range(num_cities):
        graph.add_node(i, pos=positions[i])

    for i in range(len(best_route) - 1):
        current_city = best_route[i]
        next_city = best_route[i + 1]
        graph.add_edge(current_city, next_city, weight=distance_matrix[current_city][next_city])

    graph.add_edge(best_route[-1], best_route[0], weight=distance_matrix[best_route[-1]][best_route[0]])

    plt.figure(figsize=(8, 6))
    labels = {edge: f"{graph[edge[0]][edge[1]]['weight']}" for edge in graph.edges()}
    nx.draw(graph, positions, with_labels=True, node_color='lightblue', node_size=700, edge_color='red', font_size=10)
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=labels, font_size=8)
    plt.show()

# Calculates the distance between two cities using the distance matrix
def calculate_distance(city1, city2, distance_matrix):
    return distance_matrix[city1][city2]

# Calculates the total distance of the ant's route
def calculate_total_distance(route, distance_matrix):
    distance = 0  
    for i in range(len(route) - 1):  
        distance += calculate_distance(route[i], route[i + 1], distance_matrix)  
    
    distance += calculate_distance(route[-1], route[0], distance_matrix)  
    return distance  


# Selects the next city based on the probability function
def select_city(visited_cities, cities, pheromones, visibility, alpha, beta):

    available_cities = [i for i in range(len(cities)) if i not in visited_cities] 
    
    if not available_cities:
        return None

    probabilities = []

    for i in available_cities:
        total = sum(
            (pheromones[i][j] ** alpha) * ((visibility[i][j]) ** beta)
            for j in available_cities
        )

        Pki = (pheromones[visited_cities[-1]][i] ** alpha) * ((visibility[visited_cities[-1]][i]) ** beta)
        probabilities.append(Pki if total > 0 else 0)

    probabilities = np.array(probabilities)
    
    
    sum_probabilities = probabilities.sum()
    if sum_probabilities == 0:
        probabilities.fill(1 / len(probabilities))

    else:
        probabilities /= sum_probabilities

    return np.random.choice(available_cities, p=probabilities)


# Updates the pheromones based on the update function
def update_pheromones(pheromones, routes, distances, evaporation_rate, pheromone_intensity):
    num_cities = len(pheromones)
    
    for i in range(num_cities):
        for j in range(num_cities):
            pheromones[i][j] *= (1 - evaporation_rate)
            pheromones[j][i] *= (1 - evaporation_rate)
    
    for k in range(len(routes)):
        route = routes[k] 
        distance = distances[k]
        deposit = pheromone_intensity / distance  
        
        for i in range(len(route) - 1):
            current_city = route[i]
            next_city = route[i + 1]
            pheromones[current_city][next_city] += deposit
            pheromones[next_city][current_city] += deposit
        
        pheromones[route[-1]][route[0]] += deposit  
        pheromones[route[0]][route[-1]] += deposit  

# The ant system algorithm
def ant_system(alpha, beta, evaporation_rate):
    distance_matrix = [
        [0, 29, 82, 46, 68, 52, 72, 42, 51, 55, 29, 74, 23, 72, 46],
        [29, 0, 55, 46, 42, 43, 43, 23, 23, 31, 41, 51, 11, 52, 21],
        [82, 55, 0, 68, 46, 55, 23, 43, 41, 29, 79, 21, 64, 31, 51],
        [46, 46, 68, 0, 82, 15, 72, 31, 62, 42, 21, 51, 51, 43, 64],
        [68, 42, 46, 82, 0, 74, 23, 52, 21, 46, 82, 58, 46, 65, 23],
        [52, 43, 55, 15, 74, 0, 61, 23, 55, 31, 33, 37, 51, 29, 59],
        [72, 43, 23, 72, 23, 61, 0, 42, 23, 31, 77, 37, 51, 46, 33],
        [42, 23, 43, 31, 52, 23, 42, 0, 33, 15, 37, 33, 33, 31, 37],
        [51, 23, 41, 62, 21, 55, 23, 33, 0, 29, 62, 46, 29, 51, 11],
        [55, 31, 29, 42, 46, 31, 31, 15, 29, 0, 51, 21, 41, 23, 37],
        [29, 41, 79, 21, 82, 33, 77, 37, 62, 51, 0, 65, 42, 59, 61],
        [74, 51, 21, 51, 58, 37, 37, 33, 46, 21, 65, 0, 61, 11, 55],
        [23, 11, 64, 51, 46, 51, 51, 33, 29, 41, 42, 61, 0, 62, 23],
        [72, 52, 31, 43, 65, 29, 46, 31, 51, 23, 59, 11, 62, 0, 59],
        [46, 21, 51, 64, 23, 59, 33, 37, 11, 37, 61, 55, 23, 59, 0],
    ]

   
    num_cities = 15
    num_ants = 15
    num_iterations = 30



    pheromone_intensity = 100

    cities = list(range(num_cities))

    pheromones = [[10**-16 for _ in range(num_cities)] for _ in range(num_cities)]

    visibility = [[1 / calculate_distance(i, j, distance_matrix) if i != j else 0 for j in range(num_cities)] for i in range(num_cities)]

    best_route = None
    best_distance = float('inf')


    for t in range(num_iterations):
        routes = []
        distances = []
        
        for k in range(num_ants):
            visited_cities = [k]
            while len(visited_cities) < num_cities:
                next_city = select_city(visited_cities, cities, pheromones, visibility, alpha, beta)
                if next_city is None:
                    break
                visited_cities.append(next_city)
            
            distance_k = calculate_total_distance(visited_cities, distance_matrix)
            routes.append(visited_cities)
            distances.append(distance_k)

            if distance_k < best_distance:
                best_distance = distance_k
                best_route = visited_cities

            update_pheromones(pheromones, [visited_cities], [distance_k], evaporation_rate, pheromone_intensity)

    # Final result
    print("Best route:", " -> ".join(map(str, map(int, best_route))))
    print("Best distance:", best_distance)
    print("\nFinal pheromones (edges with pheromones):")
    for i in range(len(pheromones)):
        for j in range(i + 1, len(pheromones[i])):
            if pheromones[i][j] > (10**-16):
                print(f"City {i} -> City {j}: {pheromones[i][j]:.6f}")

    plot_graph(best_route, distance_matrix, num_cities)


def main():

    alpha = 1 # Pheromone influence
    beta = 5 # Visibility influence
    evaporation_rate = 0.5

    alpha1 = 2
    beta1 = 6
    evaporation_rate1 = 0.01

    alpha2 = 5
    beta2 = 5
    evaporation_rate2 = 0.1

    ant_system(alpha, beta, evaporation_rate)
    print("\n")
    ant_system(alpha1, beta1, evaporation_rate1)
    print("\n")
    ant_system(alpha2, beta2, evaporation_rate2)


main()