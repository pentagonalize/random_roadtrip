import networkx as nx
import osmnx as ox
import pandas as pd
import numpy as np
import random
from collections import deque
import math

ox.config(use_cache=True, log_console=True)

def create_adjacency_matrix(edges):
    # Find the maximum node value
    max_node = max(max(edge) for edge in edges)

    # Create an empty adjacency matrix
    matrix = [[0] * (max_node + 1) for _ in range(max_node + 1)]

    # Create a dictionary to map node values to row indices
    node_to_row = {node: row for row, node in enumerate(range(max_node + 1))}

    # Fill the adjacency matrix based on the edges
    for edge in edges:
        source, target = edge
        source_row = node_to_row[source]
        target_row = node_to_row[target]
        matrix[source_row][target] = 1
        matrix[target_row][source] = 1  # If the graph is undirected, uncomment this line

    return matrix, node_to_row

def euclidean_distance(node1, node2):
        # Assuming nodes have (x, y) coordinates as node attributes
        x1, y1 = node1['x'], node1['y']
        x2, y2 = node2['x'], node2['y']
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

class Route:
    def __init__(self, filepath):
        self.G = ox.load_graphml(filepath)
        self.gdf_edges = ox.graph_to_gdfs(self.G, nodes=False)
        self.nodes = list(self.G)
        self.orig = None
        self.dest = None
        self.path = None
        self.pathLength = 0

    def setOrigByCoordinates(self, x, y):
        self.orig = ox.nearest_nodes(self.G, x, y)

    def setDestByCoordinates(self, x, y):
        self.dest = ox.nearest_nodes(self.G, x, y)

    def setPathLength(self):
        edge_lengths = ox.utils_graph.get_route_edge_attributes(self.G, self.path, 'length')
        self.pathLength = sum(edge_lengths)

    def random_walk(self, steps):
        position = self.orig
        path = [position]
        visited = set()

        for _ in range(steps):
            neighbors = list(self.G.neighbors(position))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            position = next_node
            path.append(position)

        self.path = path
        self.setPathLength()

    def self_avoiding_random_walk_with_backtracking(self, steps):
        position = self.orig
        path = [position]
        maxpath = []
        visited = set()

        for i in range(steps):
            visited.add(position)
            neighbors = [n for n in list(self.G.neighbors(position)) if n not in visited]
            if not neighbors:
                if len(path) > len(maxpath):
                    maxpath = path
                path.pop()
                position = path[-1]
            else:
                next_node = random.choice(neighbors)
                position = next_node
                path.append(position)

        if len(path) < len(maxpath):
            self.path = maxpath
        else:
            self.path = path
        self.setPathLength()

    def self_avoiding_random_walk(self, steps):
        position = self.orig
        path = [position]
        visited = set()
        # goes until dead end

        for i in range(steps):
            print(i)
            neighbors = [n for n in list(self.G.neighbors(position)) if n not in visited]
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            visited.add(next_node)
            position = next_node
            path.append(position)

        self.path = path
        self.setPathLength()

    def loop_erasing_random_walk(self, steps):
        iters = 0
        node = self.orig
        path = [node]
        while iters < steps:
            neighbors = [n for n in self.G.neighbors(node)]
            if neighbors == []:
                i = random.choice(range(len(path)))
                node = path[i]
                path = path[:i+1]
            else:
                node = random.choice(neighbors)
                if node in path:
                    i = random.choice(range(int(len(path)/2), len(path)))
                    node = path[i]
                    path = path[:i+1]
                else:
                    path.append(node)
            iters += 1
            print(path)

        self.path = path
        self.setPathLength()

    def random_path_DFS(self):
        visited = set()
        stack = [(self.orig, [self.orig])]

        while stack:
            node, path = stack.pop()

            if node == self.dest:
                self.path = path
                self.setPathLength()
                return

            if node not in visited:
                visited.add(node)

                neighbors = [n for n in self.G.neighbors(node)]
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))

        print("not connected")
        self.path = None

    def random_path_BFS(self):
        visited = set()
        queue = deque([(self.orig, [self.orig])])

        while queue:
            node, path = queue.popleft()

            if node == self.dest:
                self.path = path
                self.setPathLength()
                return

            if node not in visited:
                visited.add(node)
                neighbors = [n for n in self.G.neighbors(node)]
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))

        print("not connected")
        self.path = None

    def random_path_BFS_fuzzy(self, noise):

        # This simply just goes down dead ends?
        visited = set()
        queue = deque([(self.orig, [self.orig])])

        while queue:
            node, path = queue.popleft()

            if node == self.dest:
                self.path = path
                self.setPathLength()
                return

            if node not in visited:
                visited.add(node)
                neighbors = [n for n in self.G.neighbors(node)]
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        # randomly go down depth
                        nneighbor = neighbor
                        additional_path = [nneighbor]
                        while(random.random() < noise):
                            nneighbors = [n for n in self.G.neighbors(nneighbor) if n not in visited]
                            if nneighbors == []:
                                break
                            nneighbor = random.choice(nneighbors)
                            visited.add(nneighbor)
                            additional_path.append(nneighbor)
                            if nneighbor == self.dest:
                                self.path = (path + additional_path)
                        # visited.add(neighbor)
                        queue.append((nneighbor, path + additional_path))

        print("not connected")
        self.path = None

    def random_path_IDDFS(self):
        max_depth = 0
        while True:
            visited = set()
            stack = [(self.orig, [self.orig], 0)]

            while stack:
                node, path, depth = stack.pop()

                if node == self.dest:
                    self.path = path
                    self.setPathLength()
                    return

                if depth < max_depth and node not in visited:
                    visited.add(node)
                    neighbors = [n for n in self.G.neighbors(node)]
                    random.shuffle(neighbors)
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            stack.append((neighbor, path + [neighbor], depth + 1))

            max_depth += 1

            if max_depth > len(self.G.nodes):
                break

        print("not connected")
        self.path = None

    def loop_erasing_random_walk_with_dest(self):
        visited = set()

        # this seems to not work..
        node = self.orig
        path = [node]
        while node != self.dest:
            if len(visited) == len(self.nodes):
                print("not connected")
                self.path = None
            neighbors = [n for n in self.G.neighbors(node)]
            if neighbors == []:
                i = random.choice(range(len(path)))
                node = path[i]
                path = path[:i+1]
            else:
                node = random.choice(neighbors)
                if node in path:
                    i = random.choice(range(len(path)))
                    node = path[i]
                    path = path[:i+1]
                else:
                    path.append(node)

        self.path = path
        self.setPathLength()

    def astar_path(self, h):
        def custom_comparison(node):  # Custom comparison function for the priority queue
            return f_score[node]

        open_list = [self.orig]  # List of nodes to be visited
        came_from = {}  # Keep track of the path from each node to the previous node
        g_score = {node: float('inf') for node in self.G.nodes()}  # Cost from start to node
        g_score[self.orig] = 0

        f_score = {node: float('inf') for node in self.G.nodes()}  # Estimated total cost from start to goal through node
        f_score[self.orig] = h(self.G.nodes[self.orig], self.G.nodes[self.dest])

        while open_list:
            current_node = min(open_list, key=custom_comparison)  # Node with the lowest f(n) value

            if current_node == self.dest:
                # Reconstruct path from the goal node to the start node
                path = [self.dest]
                while current_node in came_from:
                    current_node = came_from[current_node]
                    path.insert(0, current_node)
                self.path = path
                self.setPathLength()
                return

            open_list.remove(current_node)

            for neighbor in self.G.neighbors(current_node):
                tentative_g_score = g_score[current_node] + self.G[current_node][neighbor].get('weight', 1)

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + h(self.G.nodes[neighbor], self.G.nodes[self.dest])

                    if neighbor not in open_list:
                        open_list.append(neighbor)

        self.path = None  # No path found

    def plot(self):
        fig, ax = ox.plot_graph_route(self.G, self.path, route_linewidth=6, node_size=0, bgcolor='k')

def path_to_coordinates(graph, path):
    coordinates = []
    for node in path:
        coordinates.append((graph._node[node]['y'], graph._node[node]['x']))
    return coordinates

def coordinates_to_gpx(coordinates, filename):
    header = '<?xml version="1.0" encoding="UTF-8"?><gpx xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://www.topografix.com/GPX/1/1" xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd http://www.garmin.com/xmlschemas/GpxExtensions/v3 http://www.garmin.com/xmlschemas/GpxExtensionsv3.xsd http://www.garmin.com/xmlschemas/TrackPointExtension/v1 http://www.garmin.com/xmlschemas/TrackPointExtensionv1.xsd http://www.topografix.com/GPX/gpx_style/0/2 http://www.topografix.com/GPX/gpx_style/0/2/gpx_style.xsd" xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1" xmlns:gpxx="http://www.garmin.com/xmlschemas/GpxExtensions/v3" xmlns:gpx_style="http://www.topografix.com/GPX/gpx_style/0/2" version="1.1" creator="https://gpx.studio">'
    footer = '</gpx>'
    gpx_content = header

    gpx_content += "\n<trk>\n <trkseg>"

    for coordinate in coordinates:
        lat, lon = coordinate
        gpx_content += f'<trkpt lat="{lat}" lon="{lon}"></trkpt>\n'

    gpx_content += "</trkseg>\n</trk>" + footer

    with open(filename, 'w') as f:
        f.write(gpx_content)

    print(f"GPX file '{filename}' has been created.")

def remove_loops(path):
    result = path
    start = 0
    end = 0
    length = len(result)
    while start <= length:
        while end < length:
            if result[start] == result[end]:
                result = result[0:start] + result[end:]
                length -= (end-start)
                end -= (end-start)
            end += 1
        start += 1
        end = start
    return result

def generate_graph(num_nodes, num_edges, weight_range):
    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    for node in range(num_nodes):
        G.add_node(node)

    # Generate random edges
    edges = set()  # Use a set to avoid self-loops (edges between the same node)
    while len(edges) < num_edges:
        edge = (random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1))
        if edge[0] != edge[1]:  # Avoid self-loops
            edges.add(edge)
    for edge in edges:
        weight = random.randint(weight_range[0], weight_range[1])
        G.add_edge(edge[0], edge[1], weight=weight)
    # Add edges to the graph
    for edge in edges:
        G.add_edge(*edge)

    return G


# saves output to here
filepath = './data/portland.graphml'

portland = Route(filepath)
portland.setOrigByCoordinates(-122.69580860261648, 45.52233007283795) # pearl district
portland.setDestByCoordinates(-122.661531, 45.562707) # our hotel location
portland.self_avoiding_random_walk_with_backtracking(1000)
print(portland.path)
portland.plot()

