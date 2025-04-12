import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import networkx as nx
import random
import logging

logging.basicConfig(level=logging.DEBUG)
random.seed(4)

# UPDATE SEARCH ALGORITHM GRAPHICS
def update_search_algorithm_graphics(traversed_list:list[list[int]], visited_list:list[list[int]], labels_list:list[str], path:list[int], visited:list[int], label:str) -> None:
    """ updates graphical information for search algorithms

        * traversed_list:list[list[int]] - list of lists of traversed nodes
        * visited_list:list[list[int]] - list of lists of visited nodes
        * labels_list:list[str] - list of strings of labels
        * path:list[int] - traversed nodes list to be added to traversed_list
        * visited:list[int] - visited nodes list to be added to visited_list
        * label:str - label to be added to labels

    """
    logging.info("Calling Update Search Algorithm Graphics")

    traversed_list.append(path.copy())
    visited_list.append(visited.copy())
    labels_list.append(label)

# BREADTH FIRST SEARCH ALGORITHM
def find_breadthfirstsearch_path(graph:nx.Graph, starting_node:int, destination_node:int):
    """ finds path in the graph between the starting node and the destination node using breadth first search
        returns title, traversed_list, visited_list, labels_list used for graphics

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the starting node
        * destination_node:int - index of the destination node

    """
    logging.info("Calling Iterative Breadth First Search")

    # final path
    path = []
    path.append(starting_node)

    # list of paths
    queue = []
    queue.append(path.copy())

    # list of visited nodes
    visited = []
    visited.append(starting_node)

    # list of all traversed paths, visited paths, and labels for graphics
    traversed_list = []
    visited_list = []
    labels_list = []
    label = "Finding Path From Node " + str(starting_node) + " to Node " + str(destination_node)
    update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

    # while queue not empty, check each path in queue
    while len(queue) > 0:
        # set path to path popped from queue
        path = queue.pop(0)

        # get node at end of path
        current_node = path[-1]

        # graphics
        label = "Checking Neighbors of Node " + str(current_node)
        update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

        # check each neighbor
        neighbors = sorted(graph.neighbors(current_node))
        neighbor_nodes = []
        for node in neighbors:
            if node not in visited:
                neighbor_nodes.append(node)

        # graphics
        if len(neighbor_nodes) == 0:
            label = "No Neighbors Found"
            update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

        else:
            for neighbor_node in neighbor_nodes:
                # for each not visited neighbor, mark as visited, add to path, and push path to queue
                visited.append(neighbor_node)
                new_path = path.copy()
                new_path.append(neighbor_node)
                queue.append(new_path.copy())

                # graphics
                label = "Neighbor Node " + str(neighbor_node) + " Visited"
                update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=new_path, visited=visited, label=label)

                # if node at end of path is destination, return path found
                if neighbor_node == destination_node:
                    logging.debug(f"Valid Path From Node {starting_node} to Node {destination_node}")
                    logging.debug(f"Path: {new_path}")

                    # graphics
                    label = "Path Found at " + str(new_path)
                    update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=new_path, visited=visited, label=label)

                    return "Breadth First Search:", traversed_list, visited_list, labels_list
                    
                # graphics
                label = "Checking Neighbors of Node " + str(current_node)
                update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

                # graphics
                if neighbor_node == neighbor_nodes[len(neighbor_nodes) - 1]:
                    label = "No Neighbors Found"
                    update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

    # if destination not found, return path not found 
    logging.debug(f"No Valid Path From Node {starting_node} to Node {destination_node}")

    # graphics
    label = "No Path Found"
    update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

    return "Breadth First Search:", traversed_list, visited_list, labels_list

# DEPTH FIRST SEARCH ALGORITHM
def find_depthfirstsearch_path(graph:nx.Graph, starting_node:int, destination_node:int):
    """ finds path in the graph between the starting node and the destination node using depth first search
        returns title, traversed_list, visited_list, labels_list used for graphics

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the starting node
        * destination_node:int - index of the destination node

    """
    logging.info("Calling Depth First Search")

    # final path
    path = []
    path.append(starting_node)

    # list of paths
    stack = []
    stack.append(path.copy())

    # list of visited nodes
    visited = []
    visited.append(starting_node)

    # list of all traversed paths, visited paths, and labels for graphics
    traversed_list = []
    visited_list = []
    label = "Finding Path From Node " + str(starting_node) + " to Node " + str(destination_node)
    labels_list = []
    update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

    # while stack not empty, check each path in queue
    while len(stack) > 0:
        # set path to path from stack
        path = stack[-1]

        # get node at end of path
        current_node = path[-1]

        # graphics
        label = "Checking Neighbors of Node " + str(current_node)
        update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

        # check if unvisited neighbor exists
        neighbors = sorted(graph.neighbors(current_node))
        neighbor_node = None
        for node in neighbors:
            if node not in visited:
                neighbor_node = node
                break

        # for the first not visited neighbor, mark as visited, add to path, and push path to stack
        if neighbor_node:
            visited.append(neighbor_node)
            new_path = path.copy()
            new_path.append(neighbor_node)
            stack.append(new_path.copy())

            # graphics
            label = "Neighbor Node " + str(neighbor_node) + " Visited"
            update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=new_path, visited=visited, label=label)

            # if node at end of path is destination, return path found
            if neighbor_node == destination_node:
                logging.debug(f"Valid Path From Node {starting_node} to Node {destination_node}")
                logging.debug(f"Path: {new_path}")

                # graphics
                label = "Path Found at " + str(new_path)
                update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=new_path, visited=visited, label=label)

                return "Depth First Search:", traversed_list, visited_list, labels_list
            
        else:
            # graphics
            path = stack.pop(-1)
            label = "No Neighbors Found"
            update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

    # if destination not found, return path not found 
    logging.debug(f"No Valid Path From Node {starting_node} to Node {destination_node}")

    # graphics
    label = "No Path Found"
    update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

    return "Depth First Search:", traversed_list, visited_list, labels_list

# UPDATE SSSP ALGORITHM GRAPHICS
def update_sssp_algorithm_graphics(sssp_list:list[list[int]], neighbors_list:list[list[int]], distances_list:list[dict], edges_list:list[list[tuple]], visited_edges_list:list[list[tuple]], labels_list:list[str], sssp:list[int], neighbors:list[int], distances:dict, edges:list[tuple], visited_edges:list[tuple], label:str):
    """ updates graphical information for sssp algorithms

        * sssp_list:list[list[int]] - list of lists of sssp nodes
        * neighbors_list:list[list[int]] - list of list of neighbor nodes
        * distances_list:list[dict] - list of dict of distances from starting node
        * edges_list:list[list[tuple]] - list of list of edges
        * visited_edges_list:list[list[tuple]] - list of list of visited edges
        * labels_list:list[str] - list of strings of labels
        * sssp:list[int] - sssp nodes list to be added to sssp_list
        * neighbors:list[int] - neighbor nodes list to be added to neighbors_list
        * distances:dict - distances dict to be formatted and added to distances_list
        * edges:list[tuple] - edge tuples list to be added to edges_list
        * visited_edges:list[tuple] - edge tuples list to be added to visited_edges_list
        * label:str - label to be added to labels

    """
    logging.info("Calling Update SSSP Algorithm Graphics")

    distances_formatted = distances.copy()
    for i in range(len(distances_formatted)):
        distances_formatted[i] = str(i) + ": " + str(distances[i])
    
    sssp_list.append(sssp.copy())
    neighbors_list.append(neighbors.copy())
    distances_list.append(distances_formatted.copy())
    edges_list.append(edges.copy())
    visited_edges_list.append(visited_edges.copy())
    labels_list.append(label)

# GET PATH NEIGHBORS
def get_path_neighbors(graph:nx.Graph, path:list[int]):
    """ updates graphical information for search algorithms

        * graph:nx.Graph - graph to find path
        * path:list[int] - path nodes list

    """
    logging.info("Calling Update Search Algorithm Graphics")

    neighbors = []

    for node in path:
        for neighbor in graph.neighbors(node):
            if neighbor not in neighbors and neighbor not in path:
                neighbors.append(neighbor)
    
    return neighbors

# DIJKSTRA'S ALGORITHM
def find_dijkstra_path(graph:nx.Graph, starting_node:int):
    """ finds shortest path in the graph between all nodes using dikstra's algorithm
        returns title, sssp_list, neighbors_list, distances_list, edges_list, visited_edges_list, labels_list used for graphics

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the first graph node

    """
    logging.info("Calling Dijkstra's Algorithm")

    # final sssp
    sssp = []
    edges = []
    visited_edges = []
    neighbors = []

    # list of distances from starting node to all nodes
    distances = {i:np.inf for i in range(0, graph.number_of_nodes())}
    distances[starting_node] = 0

    # list of all sssp nodes, sssp edges, neighbor nodes, and labels for graphics
    sssp_list = []
    neighbors_list = []
    distances_list = []
    edges_list = []
    visited_edges_list = []
    label = "Finding SSSP From Node " + str(starting_node)
    labels_list = []
    update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, visited_edges=visited_edges, label=label)

    # local memory used to help graphics
    minimum_distance_neighbor_node = starting_node
    minimum_distance_sssp_node = starting_node
    minimum_distance = 0

    # get neighbors of sssp
    neighbors = get_path_neighbors(graph=graph, path=sssp)

    # graphics
    label = "Checking Neighbors of SSSP Nodes " + str(sssp)
    update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, visited_edges=visited_edges, label=label)

    # while sssp doesn't include all nodes
    while len(sssp) < graph.number_of_nodes():        
        # get minimum distance node
        sssp.append(minimum_distance_neighbor_node)

        # graphics
        label = "Node " + str(minimum_distance_neighbor_node) + " With Distance " + str(minimum_distance) + " Added to SSSP"
        update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, visited_edges=visited_edges, label=label)

        # get neighbors of sssp
        neighbors = get_path_neighbors(graph=graph, path=sssp)

        # get next minimum distance edge
        minimum_distance = np.inf

        # for all neighbor nodes of sssp
        for sssp_node in sssp:
            for neighbor_node in graph.neighbors(sssp_node):
                neighbor_distance = distances[sssp_node] + graph.get_edge_data(sssp_node, neighbor_node)['weight']

                # graphics
                edge = (sssp_node, neighbor_node)
                if edge not in visited_edges:
                    visited_edges.append(edge)

                # if neighbor node has shorter distance
                if neighbor_node not in sssp and neighbor_distance <= distances[neighbor_node]:
                    distances[neighbor_node] = neighbor_distance

                    # local memory used to help graphics
                    if neighbor_distance <= minimum_distance:
                        minimum_distance = neighbor_distance
                        minimum_distance_sssp_node = sssp_node
                        minimum_distance_neighbor_node = neighbor_node

        # graphics
        label = "Checking Neighbors of SSSP Nodes " + str(sssp)
        update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, visited_edges=visited_edges, label=label)
        
        edges.append((minimum_distance_sssp_node, minimum_distance_neighbor_node))
        
    logging.debug(f"Valid SSSP From Node {starting_node}")
    logging.debug(f"SSSP: {sssp}")
    logging.debug(f"SSSP Edges: {edges}")

    # graphics
    label = "SSSP Found With Edges " + str(edges)
    update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, visited_edges=visited_edges, label=label)

    return "Dijkstra's Algorithm:", sssp_list, neighbors_list, distances_list, edges_list, visited_edges_list, labels_list

# BELLMAN-FORD ALGORITHM
def find_bellmanford_path(graph:nx.Graph, starting_node:int):
    """ finds shortest path in the graph between all nodes using bellman-ford algorithm
        returns title, sssp_list, neighbors_list, distances_list, edges_list, labels_list used for graphics

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the first graph node

    """
    logging.info("Calling Bellman-Ford Algorithm")

    # final sssp
    sssp = []

    # list of distances from starting node to all nodes
    edges = [(0, 0) for i in range(0, graph.number_of_nodes())]
    visited_edges = []
    neighbors = []
    distances = {i:np.inf for i in range(0, graph.number_of_nodes())}
    distances[starting_node] = 0

    # list of all sssp nodes, sssp edges, neighbor nodes, and labels for graphics
    sssp_list = []
    neighbors_list = []
    distances_list = []
    edges_list = []
    visited_edges_list = []
    label = "Finding SSSP From Node " + str(starting_node)
    labels_list = []
    update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, visited_edges=visited_edges, label=label)

    # iterate for the number of nodes in graph - 1, if sssp changes in iteration number of nodes in graph, graph contains negative cycle
    for iteration in range(graph.number_of_nodes()):   
        # graphics
        new_sssp_found = False
        sssp = []
        edges = [(0, 0) for i in range(0, graph.number_of_nodes())]
        visited_edges = []
        label = "Iteration " + str(iteration)
        update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, visited_edges=visited_edges, label=label)

        # check each edge in graph
        for edge in graph.edges():
            # edge nodes
            edge_node_one = edge[0]
            edge_node_two = edge[1]

            # edge distance
            edge_distance = graph.get_edge_data(edge_node_one, edge_node_two)['weight']
            edge = (edge_node_one, edge_node_two)

            # graphics
            if edge not in visited_edges:
                visited_edges.append(edge)
            
            # graphics
            label = "Checking Edge " + str(edge)
            update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, visited_edges=visited_edges, label=label)

            # if new edge shortens distance to node
            new_edge_added = False
            node_u_dist = distances[edge_node_two] + edge_distance
            node_v_dist = distances[edge_node_one] + edge_distance

            if node_u_dist <= distances[edge_node_one]:
                new_edge_added = True
                edges[edge_node_one] = (edge_node_one, edge_node_two)

                # graphics
                if edge_node_one not in sssp:
                    sssp.append(edge_node_one) 

                if edge_node_two not in sssp:
                    sssp.append(edge_node_two)

                # update distances to be the smallest calculated distance
                if node_u_dist < distances[edge_node_one]:
                    distances[edge_node_one] = node_u_dist
                    new_sssp_found = True

            if node_v_dist <= distances[edge_node_two]:
                new_edge_added = True
                edges[edge_node_two] = (edge_node_one, edge_node_two)

                # graphics
                if edge_node_one not in sssp:
                    sssp.append(edge_node_one) 

                if edge_node_two not in sssp:
                    sssp.append(edge_node_two)

                # update distances to be the smallest calculated distance
                if node_v_dist < distances[edge_node_two]:
                    distances[edge_node_two] = node_v_dist
                    new_sssp_found = True
            
            if new_edge_added:
                # graphics
                label = "Adding Edge " + str(edge)
                update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, visited_edges=visited_edges, label=label)
        
        # if sssp same as last iteration
        if not new_sssp_found:
            # graphics
            label = "SSSP Repeated From Last Iteration"
            update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, visited_edges=visited_edges, label=label)
            break

    logging.debug(f"Valid SSSP From Node {starting_node}")
    logging.debug(f"SSSP: {sssp}")
    logging.debug(f"SSSP Edges: {edges}")

    # graphics
    label = "SSSP Found With Edges " + str(edges)
    update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, visited_edges=visited_edges, label=label)

    return "Bellman-Ford Algorithm:", sssp_list, neighbors_list, distances_list, edges_list, visited_edges_list, labels_list

# UPDATE MST ALGORITHM GRAPHICS
def update_mst_algorithm_graphics(mst_list:list[list[int]], edges_list:list[list[tuple]], visited_edges_list:list[list[tuple]], labels_list:list[str], mst:list[int], edges:list[tuple], visited_edges:list[tuple], label:str):
    """ updates graphical information for mst algorithms

        * mst_list:list[list[int]] - list of lists of mst nodes
        * edges_list:list[list[tuple]] - list of list of edges
        * visited_edges_list:list[list[tuple]] - list of list of visited edges
        * labels_list:list[str] - list of strings of labels
        * mst:list[int] - mst nodes list to be added to mst_list
        * edges:list[tuple] - edge tuples list to be added to edges_list
        * visited_edges:list[tuple] - edge tuples list to be added to edges_list
        * label:str - label to be added to labels

    """
    logging.info("Calling Update MST Algorithm Graphics")

    mst_list.append(mst.copy())
    edges_list.append(edges.copy())
    visited_edges_list.append(visited_edges.copy())
    labels_list.append(label)
    
# FIND ALGORITHM
def find(node:int, representatives:list[int]):
    """ finds representative node of node

        * node:int - node to find representative node
        * representatives:list[int] - list of representatives of each node

    """
    logging.info("Calling Find Algorithm")

    if node == representatives[node]:
        return node
    
    parent = find(node=representatives[node], representatives=representatives)
    representatives[node] = parent
    return parent

# UNION ALGORITHM
def union(node_one:int, node_two:int, representatives:list[int]):
    """ gets union of two node sets

        * node_one:int - node_one to find representative node
        * node_two:int - node_two to find representative node and join under node_one's set
        * representatives:list[int] - list of representatives of each node

    """
    logging.info("Calling Union Algorithm")

    node_one_representative = find(node=node_one, representatives=representatives)
    node_two_representative = find(node=node_two, representatives=representatives)
    representatives[node_two_representative] = node_one_representative

# KRUSKAL'S ALGORITHM
def find_kruskal_path(graph:nx.Graph):
    """ finds minimum spanning tree in the graph between all nodes using kruskal's algorithm
        returns title, mst_list, edges_list, visited_edges_list, labels_list

        * graph:nx.Graph - graph to find path

    """
    logging.info("Calling Kruskal's Algorithm")

    # final mst
    mst = []
    edges = []
    visited_edges = []
    representatives = [i for i in range(0, graph.number_of_nodes())]

    # list of ordered edge weights
    edges_unsorted = graph.edges.data()
    edges_sorted = sorted(edges_unsorted, key=lambda tup:tup[2]['weight'])

    # list of all mst nodes, mst edges, mst visited edges, and labels for graphics
    mst_list = []
    edges_list = []
    visited_edges_list = []
    label = "Finding MST"
    labels_list = []
    update_mst_algorithm_graphics(mst_list=mst_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, mst=mst, edges=edges, visited_edges=visited_edges, label=label)
    
    # iterate through each edge from smallest to largest weight
    for edge in edges_sorted:
        # if mst is not complete
        if len(mst) < graph.number_of_nodes():
            node_one = edge[0]
            node_two = edge[1]
            edge = (node_one, node_two)
            weight = graph.get_edge_data(node_one, node_two)['weight']
            visited_edges.append(edge)

            # graphics
            label = "Checking Edge " + str(edge) + " With Weight " + str(weight)
            update_mst_algorithm_graphics(mst_list=mst_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, mst=mst, edges=edges, visited_edges=visited_edges, label=label)

            # if cycle created
            if find(node=node_one, representatives=representatives) == find(node=node_two, representatives=representatives):
                # graphics
                label = "Cycle Created, Dropping Edge"
                update_mst_algorithm_graphics(mst_list=mst_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, mst=mst, edges=edges, visited_edges=visited_edges, label=label)

                continue
            
            # union set of node one and node two
            union(node_one=node_one, node_two=node_two, representatives=representatives)

            # add nodes to mst list
            if node_one not in mst:
                mst.append(node_one)

            if node_two not in mst:
                mst.append(node_two)

            # add edge to mst list
            edges.append(edge)

            # graphics
            label = "Edge Added to MST"
            update_mst_algorithm_graphics(mst_list=mst_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, mst=mst, edges=edges, visited_edges=visited_edges, label=label)

        else:
            # graphics
            label = "MST Contains All Nodes" + str(mst)
            update_mst_algorithm_graphics(mst_list=mst_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, mst=mst, edges=edges, visited_edges=visited_edges, label=label)

            break

    # graphics
    label = "MST Found With Edges " + str(edges)
    update_mst_algorithm_graphics(mst_list=mst_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, mst=mst, edges=edges, visited_edges=visited_edges, label=label)

    logging.debug(f"Valid MST")
    logging.debug(f"MST: {mst}")
    logging.debug(f"MST Edges: {edges}")

    return "Kruskal's Algorithm:", mst_list, edges_list, visited_edges_list, labels_list

# PRIM'S ALGORITHM
def find_prim_path(graph:nx.Graph):
    """ finds minimum spanning tree in the graph between all nodes using prim's algorithm
        returns title, mst_list, edges_list, visited_edges_list, labels_list

        * graph:nx.Graph - graph to find path

    """
    logging.info("Calling Prim's Algorithm")

    # final mst
    mst = []
    edges = []
    visited_edges = []
    representatives = [i for i in range(0, graph.number_of_nodes())]

    # list of all mst nodes, mst edges, mst visited edges, and labels for graphics
    mst_list = []
    edges_list = []
    visited_edges_list = []
    label = "Finding MST"
    labels_list = []
    update_mst_algorithm_graphics(mst_list=mst_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, mst=mst, edges=edges, visited_edges=visited_edges, label=label)

    # choose random starting node
    starting_node = random.randrange(0, nx.number_of_nodes(graph))
    mst.append(starting_node)

    # graphics
    label = "Starting From Random Node " + str(starting_node)
    update_mst_algorithm_graphics(mst_list=mst_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, mst=mst, edges=edges, visited_edges=visited_edges, label=label)

    # local memory used to help graphics
    minimum_weight_neighbor_node = starting_node
    minimum_weight_mst_node = None
    minimum_weight = 0

    # while mst doesn't include all nodes
    while len(mst) < graph.number_of_nodes():
        visited_edges = []        

        # get next minimum distance edge
        minimum_weight = np.inf

        # graphics
        label = "Checking Neighbors of MST Nodes " + str(mst)
        update_mst_algorithm_graphics(mst_list=mst_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, mst=mst, edges=edges, visited_edges=visited_edges, label=label)

        # for all neighbor edges of mst
        for mst_node in mst:
            for neighbor_node in graph.neighbors(mst_node):
                if neighbor_node not in mst:
                    edge = (mst_node, neighbor_node)

                    if edge not in visited_edges and edge not in edges:
                        # graphics
                        visited_edges.append(edge)
                        label = "Checking Neighbor Edge " + str(edge) + " of MST"
                        update_mst_algorithm_graphics(mst_list=mst_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, mst=mst, edges=edges, visited_edges=visited_edges, label=label)

                        # if cycle created
                        if find(node=mst_node, representatives=representatives) == find(node=neighbor_node, representatives=representatives):
                            # graphics
                            label = "Cycle Created, Dropping Edge"
                            update_mst_algorithm_graphics(mst_list=mst_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, mst=mst, edges=edges, visited_edges=visited_edges, label=label)

                            continue

                        # get weight of edge
                        edge_weight = graph.get_edge_data(mst_node, neighbor_node)['weight']
                        
                        # if neighbor edge has smaller weight 
                        if neighbor_node not in mst and edge_weight < minimum_weight:
                            minimum_weight = edge_weight
                            minimum_weight_mst_node = mst_node
                            minimum_weight_neighbor_node = neighbor_node

        # add smallest weight edge to mst
        edges.append((minimum_weight_mst_node, minimum_weight_neighbor_node))
        mst.append(minimum_weight_neighbor_node)

        # union set of mst_node and and added neighbor_node
        union(node_one=mst_node, node_two=neighbor_node, representatives=representatives)

        # graphics
        label = "Adding Smallest Weight Edge " + str(edge) + " to MST"
        update_mst_algorithm_graphics(mst_list=mst_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, mst=mst, edges=edges, visited_edges=visited_edges, label=label)

    # graphics
    label = "MST Found With Edges " + str(edges)
    update_mst_algorithm_graphics(mst_list=mst_list, edges_list=edges_list, visited_edges_list=visited_edges_list, labels_list=labels_list, mst=mst, edges=edges, visited_edges=visited_edges, label=label)

    logging.debug(f"Valid MST")
    logging.debug(f"MST: {mst}")
    logging.debug(f"MST Edges: {edges}")

    return "Prim's Algorithm:", mst_list, edges_list, visited_edges_list, labels_list

# GENERATE UNWEIGHTED UNDIRECTED GRAPH
def generate_unweighted_undirected_graph(nodes:list[int], edges:list[tuple]):
    """ generates unweighted graph with given nodes and edges

        * nodes:list[int] - list of nodes to add to graph
        * edges:list[tuple] - list of edges to add to graph

    """
    logging.info("Calling Generate Graph")

    graph = nx.Graph()
    logging.debug(f"Number of Nodes: {len(nodes)}")
    logging.debug(f"Number of Edges: {len(edges)}")

    # add nodes [1, number_nodes] to graph
    graph.add_nodes_from(nodes_for_adding=nodes)
    logging.debug(f"Number of Nodes Added: {graph.number_of_nodes()}")
    logging.debug(f"Nodes in Graph:", graph.nodes())

    # add edges [1, number_edges] to graph
    graph.add_edges_from(ebunch_to_add=edges)
    logging.debug(f"Number of Edges Added: {graph.number_of_edges()}")
    logging.debug(f"Edges in Graph: {graph.edges()}")

    return graph

# GENERATE WEIGHTED UNDIRECTED GRAPH
def generate_weighted_undirected_graph(nodes:list[int], edges:list[tuple]):
    """ generates weighted graph with given nodes and edges

        * nodes:list[int] - list of nodes to add to graph
        * edges:list[tuple] - list of edges to add to graph

    """
    logging.info("Calling Generate Graph")

    graph = nx.Graph()
    logging.debug(f"Number of Nodes: {len(nodes)}")
    logging.debug(f"Number of Edges: {len(edges)}")

    # add nodes [1, number_nodes] to graph
    graph.add_nodes_from(nodes_for_adding=nodes)
    logging.debug(f"Number of Nodes Added: {graph.number_of_nodes()}")
    logging.debug(f"Nodes in Graph: {graph.nodes()}")

    # add edges [1, number_edges] to graph
    graph.add_weighted_edges_from(ebunch_to_add=edges)
    logging.debug(f"Number of Edges Added: {graph.number_of_edges()}")
    logging.debug(f"Edges in Graph: {graph.edges()}")

    return graph

# GENERATE GRAPH SEARCH ANIMATION
def generate_graph_search_animation(function:callable) -> None:
    """ generates graph, finds path from starting_node to destination_node, and animates process

        * function:callable - search algorithm to animate

    """
    logging.info("Calling Generate Graph Search Animation")

    # generate nodes, random edges, and graph
    num_nodes = 8
    starting_node = 0
    destination_node = num_nodes - 1

    nodes = [i for i in range(0, num_nodes)]
    logging.debug(f"{len(nodes)} Nodes Generated")

    edges = [(0, 3), (0, 1), (1, 2), (1, 4), (4, 5), (4, 6), (6, 7)]
    logging.debug(f"{len(edges)} Edges Generated")

    graph = generate_unweighted_undirected_graph(nodes=nodes, edges=edges)

    # run search algorithm
    title, traversed_list, visited_list, labels_list = function(graph=graph, starting_node=starting_node, destination_node=destination_node)

    pos = nx.spring_layout(G=graph)
    fig, ax = plt.subplots()

    # font 
    font = {'fontname':"Trebuchet MS"}

    # update function used to iterate through animation
    def update(frame:int) -> None:
        ax.clear()

        path_nodes = traversed_list[frame]
        path_edges = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)] if len(path_nodes) > 1 else []
        visited_nodes = visited_list[frame]
        label = labels_list[frame]

        # background frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=graph.edges(), edge_color="black", width=1)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=graph.nodes(), node_color="white", edgecolors="black", node_size=400, linewidths=1)

        # visited nodes frame
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=visited_nodes, node_color="lightgray", edgecolors="black", node_size=450, linewidths=1)

        # animation nodes frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=path_edges, edge_color="yellow", width=2)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=path_nodes, node_color="yellow", edgecolors="black", node_size=450, linewidths=2)

        # labels frame
        nx.draw_networkx_labels(G=graph, pos=pos, ax=ax, font_color="black", font_size=8)
        ax.set_title(title, **font)
        ax.set_xlabel(label, **font)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(traversed_list), interval=1200, repeat=True, repeat_delay=1200)
    plt.show()

# GENERATE GRAPH SSSP ANIMATION
def generate_graph_sssp_animation(function:callable) -> None:
    """ generates graph, finds sssp, and animates process

        * function:callable - search algorithm to animate

    """
    logging.info("Calling Generate Graph SSSP Animation")

    # generate nodes, random edges, and graph
    num_nodes = 6
    starting_node = 0
    destination_node = num_nodes - 1

    nodes = [i for i in range(0, num_nodes)]
    logging.debug(f"{len(nodes)} Nodes Generated")

    edges = [(0, 1, 8), (0, 2, 8), (1, 2, 7), (1, 3, 4), (1, 4, 2), (2, 3, 7), (2, 5, 9), (3, 4, 14), (3, 5, 10), (4, 5, 2)]
    logging.debug(f"{len(edges)} Edges Generated")

    graph = generate_weighted_undirected_graph(nodes=nodes, edges=edges)

    # run sssp algorithm
    title, sssp_list, neighbors_list, distances_list, edges_list, visited_edges_list, labels_list = function(graph=graph, starting_node=starting_node)

    pos = nx.spring_layout(G=graph)
    fig, ax = plt.subplots()

    # font
    font = {'fontname':"Trebuchet MS"}

    # update function used to iterate through animation
    def update(frame:int) -> None:
        ax.clear()

        sssp_nodes = sssp_list[frame]
        sssp_edges = edges_list[frame]
        visited_edges = visited_edges_list[frame]
        neighbor_nodes = neighbors_list[frame]
        distances = distances_list[frame]
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        label = labels_list[frame]

        # background frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=graph.edges(), edge_color="black", width=1)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=graph.nodes(), node_color="white", edgecolors="black", node_size=400, linewidths=1)

        # visited edges frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=visited_edges, edge_color="lightgray", width=1)
        nx.draw_networkx_edge_labels(G=graph, pos=pos, ax=ax, edge_labels=edge_labels)

        # neighbor nodes frame
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=neighbor_nodes, node_color="lightgray", edgecolors="black", node_size=450, linewidths=1)

        # animation nodes frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=sssp_edges, edge_color="yellow", width=2)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=sssp_nodes, node_color="yellow", edgecolors="black", node_size=450, linewidths=2)
        nx.draw_networkx_edge_labels(G=graph, pos=pos, ax=ax, edge_labels=edge_labels)

        # labels frame
        nx.draw_networkx_labels(G=graph, pos=pos, ax=ax, labels=distances, font_color="black", font_size=8)
        ax.set_title(title, **font)
        ax.set_xlabel(label, **font)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(sssp_list), interval=1200, repeat=True, repeat_delay=1200)
    plt.show()

# GENERATE GRAPH MST ANIMATION
def generate_graph_mst_animation(function:callable) -> None:
    """ generates graph, finds mst, and animates process

        * function:callable - search algorithm to animate

    """
    logging.info("Calling Generate Graph MST Animation")

    # generate nodes, random edges, and graph
    num_nodes = 6
    starting_node = 0
    destination_node = num_nodes - 1

    nodes = [i for i in range(0, num_nodes)]
    logging.debug(f"{len(nodes)} Nodes Generated")

    edges = [(0, 1, 8), (0, 2, 8), (1, 2, 7), (1, 3, 4), (1, 4, 2), (2, 3, 7), (2, 5, 9), (3, 4, 14), (3, 5, 10), (4, 5, 2)]
    logging.debug(f"{len(edges)} Edges Generated")

    graph = generate_weighted_undirected_graph(nodes=nodes, edges=edges)

    # run mst algorithm
    title, mst_list, edges_list, visited_edges_list, labels_list = function(graph=graph)

    pos = nx.spring_layout(G=graph)
    fig, ax = plt.subplots()

    # font
    font = {'fontname':"Trebuchet MS"}

    # update function used to iterate through animation
    def update(frame:int) -> None:
        ax.clear()

        mst_nodes = mst_list[frame]
        mst_edges = edges_list[frame]
        visited_edges = visited_edges_list[frame]
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        label = labels_list[frame]

        # background frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=graph.edges(), edge_color="black", width=1)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=graph.nodes(), node_color="white", edgecolors="black", node_size=400, linewidths=1)

        # visited edges frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=visited_edges, edge_color="lightgray", width=1)
        nx.draw_networkx_edge_labels(G=graph, pos=pos, ax=ax, edge_labels=edge_labels)

        # animation edges frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=mst_edges, edge_color="yellow", width=2)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=mst_nodes, node_color="yellow", edgecolors="black", node_size=450, linewidths=2)
        nx.draw_networkx_edge_labels(G=graph, pos=pos, ax=ax, edge_labels=edge_labels)

        # labels frame
        nx.draw_networkx_labels(G=graph, pos=pos, ax=ax, font_color="black", font_size=8)
        ax.set_title(title, **font)
        ax.set_xlabel(label, **font)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(mst_list), interval=1200, repeat=True, repeat_delay=1200)
    plt.show()

# call animation functions

generate_graph_search_animation(function=find_breadthfirstsearch_path)
generate_graph_search_animation(function=find_depthfirstsearch_path)

generate_graph_sssp_animation(function=find_dijkstra_path)
generate_graph_sssp_animation(function=find_bellmanford_path)

generate_graph_mst_animation(function=find_kruskal_path)
generate_graph_mst_animation(function=find_prim_path)