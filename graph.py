from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from task import read_tasks_csv, Task
from datetime import datetime, timedelta
from random import choice, choices
import random
import numpy as np


NUM_ANTS: int = 100
W_MAX: int = 1000
ALPHA = 8
BETA = 4
RHO = 0.3
T_MAX: int = 50

HOURS_DAY: int = 10
START_DAY_HOUR: int = 8


def tasks_to_bipartite(tasks: list[Task]) -> (nx.Graph, int, dict):
    """
    creates a bipartite graph from a list of tasks
    cannot use nx.algorithms.bipartite because the graph can be disconnected
    """
    G = nx.Graph()
    # dict to log to which taske each node belongs to
    node_to_task = {}

    first, last = get_first_and_last_day(tasks)

    # find the starting index for u_i in U
    idx_u: int = HOURS_DAY * (last - first + 1)

    # v_j in V: j in {0, ..., idx_u - 1}
    # add node for each work hour of the day, i.e. 08:00 to 18:00
    # add 10 nodes for each day (08:00 - 09:00, ...,  17:00 - 18:00)
    G.add_nodes_from([j for j in range(idx_u)])

    # u_i in U: i in {idx_u, ..., idx_u + #number of task segments - 1}
    # add a node for each 1h segment of a task
    # t13 := 3rd segment of task 1
    G.add_nodes_from(
        [i for i in range(
            idx_u, idx_u + sum(t['duration'] for t in tasks)
        )]
    )

    j: int = 0
    for t in tasks:
        # add edge iff task can be done in this hour segment,
        # i.e. is between the start time and the end time of the task
        # e.g. start: 08:00, end: 10:00 -> edge only to idx_v and
        # idx_v + 1 (NOT idx_v + 2)

        day_diff: int = t['start'].day - first
        for _ in range(t['duration']):
            # log in dict, which task each node belongs to
            node_to_task[j + idx_u] = t
            add_edges_from_node(
                G, j + idx_u, t['start'], t['end'], t['prio'], day_diff
            )
            j += 1

    return (G, idx_u, node_to_task)


def add_edges_from_node(G: nx.Graph, node: int, start: datetime, end: datetime,
                        weight: int, day_diff: int):
    """
    adds edges to the given graph from one node to all possible time nodes
    with the given weight.
    e. g. node = 0, start = 08:00, end = 10:00
    [{0, idx(8)}, {0, idx(9)}]
    """
    while (start < end):

        # adds an edge from the current node the time nodes
        # that are between start and end
        G.add_edge(
            node,
            day_diff * HOURS_DAY + start.hour - START_DAY_HOUR,
            weight=weight
        )
        start += timedelta(hours=1)


def get_first_and_last_day(tasks: list[Task]):
    """
    find the first and last day for all tasks to know
    how many nodes must be added to the graph
    """
    first = tasks[0]['start'].day
    last = tasks[0]['end'].day
    for task in tasks:
        if task['start'].day < first:
            first = task['start'].day
        if task['end'].day > last:
            last = task['end'].day
    return (first, last)


def choose_by_pheromones(avail_u: list[int], pheromones: np.array,
                         u_prev: int) -> int | None:
    """
    Chooses an edge based on the pheromones.
    See Eq 3.1 in paper
    """
    probs = np.zeros(len(avail_u))
    densities = np.zeros(len(avail_u))

    for i, u in enumerate(avail_u):
        densities[i] = pheromones[u_prev, u] ** ALPHA

    sum_dens = sum(densities)

    if sum_dens == 0:
        return choice(avail_u)

    for i in range(len(avail_u)):
        probs[i] = densities[i] / sum_dens

    return choices(avail_u, weights=probs, k=1)[0]


def choose_by_pheromones_and_ETA(u: int, N_u: list[int], G: nx.Graph,
                                 pheromones: np.array) -> int:
    """
    Chooses an edge based on the pheromones and ETA = 1 / weight_ij
    See Eq 2.1 in paper
    """
    probs = np.zeros(len(N_u))
    densities = np.zeros(len(N_u))

    # v in Neighbours(u)
    i: int = 0
    for v in N_u:
        densities[i] = (pheromones[u, v] ** ALPHA) * (1 / G[u][v]['weight']) ** BETA
        i += 1

    sum_dens = sum(densities)

    if sum_dens == 0:
        return choice(N_u)

    for i in range(len(N_u)):
        probs[i] = densities[i] / sum_dens

    return choices(N_u, weights=probs, k=1)[0]


def ant_matching(G: nx.Graph, t_max: int, idx_v: int) -> (
        list[(int, int)], int):
    """
    uses ACO to find a matching on the given bipartite graph G
    """
    best_matching_list = []
    weight_min = 1000000  # should be infinity

    # dimension: #nodes * #edges
    pheromones: np.array = np.zeros((len(G), len(G)))

    for _ in range(t_max):
        best_matching_list, weight_min = generate_solution(
            G, weight_min, best_matching_list, pheromones, idx_v
        )
        update_pheromone(best_matching_list, pheromones, weight_min, G)

    return best_matching_list, weight_min


def generate_solution(G: nx.Graph, weight_min: int, best_matching_list: list,
                      pheromones: np.array, idx_v: int):
    for _ in range(NUM_ANTS):
        avail_u: list[int] = list(G.nodes())[:idx_v]
        visited_v: set[int] = set()

        # random task and segment as first
        u_prev: int = choice(avail_u)

        matching_list = []
        weight = 0
        while (len(avail_u) > 0):
            u = choose_by_pheromones(avail_u, pheromones, u_prev)
            avail_u.remove(u)
            to_visit: list[int] = list(set(list(G[u])).difference(visited_v))
            if len(to_visit) > 0:
                v = choose_by_pheromones_and_ETA(
                    u, to_visit, G, pheromones
                )
                matching_list.append((u, v))
                weight += G[u][v]['weight']
                visited_v.add(v)
            else:
                matching_list.append((u, None))
                weight += W_MAX

            u_prev = u

        if weight <= weight_min:
            best_matching_list = matching_list
            weight_min = weight

    return best_matching_list, weight_min


def update_pheromone(matching_list: list, pheromones: np.array, weight: int,
                     G: nx.Graph):
    delta_tau = np.zeros((NUM_ANTS, len(G), len(G)))  # very inefficient
    for k in range(NUM_ANTS):
        last: int | None = None
        for (x, y) in matching_list:
            if last is not None:
                delta_tau[k, last, x] = 1 / weight
            last = x
            if y is None:
                delta_tau[k, x, x] = 1 / weight
            else:
                delta_tau[k, x, y] = 1 / weight

    for (u, v) in G.edges():
        s = sum(delta_tau[k, u, v] for k in range(NUM_ANTS))
        pheromones[u, v] = (1 - RHO) * pheromones[u, v] + s

    for u_i in G.nodes():
        for u_j in G.nodes():
            if u_i != u_j:

                s = sum(delta_tau[k, u_i, u_j] for k in range(NUM_ANTS))
                pheromones[u_i, u_j] = (1 - RHO) * pheromones[u_i, u_j] + s


def draw_bipartite(G: nx.Graph, idx_v: int):
    plt.subplot()
    top = list(G.nodes())[:idx_v]  # 1st partition (U)
    nx.draw(G, pos=nx.bipartite_layout(G, top), with_labels=True)
    plt.show()


def print_timetable(matching: list, idx_v: int, first: int, last: int,
                    node_to_task: dict):
    """
    map the matching back to the tasks and print a timetable
    """

    print("Timetable: ")

    hour_to_node = {}
    # this works, because every 'a' is unique
    for a, b in matching:
        if(a > b):
            hour_to_node.setdefault(b, a)
        else:
            hour_to_node.setdefault(a, b)

    for i in range(first, last+1):
        for j in range(HOURS_DAY):
            string_task = " "
            if (j+8 < 10):
                string_task += " "
            key = ((i-first)*HOURS_DAY + j)

            # check if there is a task done in this hour:
            if (key in hour_to_node):
                key2 = hour_to_node[key]

                t = node_to_task[key2]
                string_task += "    do task "
                string_task += str(t['id'])
            else:
                string_task += "    nothing to do"
            print("day: ", i, " hour: ", j+8, string_task)

        print()


def draw_with_weights(G: nx.Graph):
    plt.subplot()
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


if __name__ == "__main__":
    random.seed(0)
    tasks = read_tasks_csv(Path("tasks_large.csv"))
    G, idx_v, node_to_task = tasks_to_bipartite(tasks)
    matching, weight = ant_matching(G, T_MAX, idx_v)
    for (u, v) in reversed(deepcopy(matching)):
        if u is None or v is None:
            matching.remove((u, v))
            weight -= W_MAX
    min_matching = nx.algorithms.min_weight_matching(G)
    s = 0
    for (u, v) in min_matching:
        s += G[u][v]['weight']

    (first, last) = get_first_and_last_day(tasks)

    print("ANT_MATCHING: ", matching, weight)
    print_timetable(matching, idx_v, first, last, node_to_task)

    print("MIN_MATCHING: ", min_matching, s)
    print_timetable(min_matching, idx_v, first, last, node_to_task)

    G: nx.Graph
    is_valid_matching: bool = nx.is_matching(G, matching)
    assert (is_valid_matching)

    # Ant matching result
    H: nx.Graph = deepcopy(G.copy())
    H.clear_edges()
    H.add_edges_from(matching)

    # Max matching result
    K: nx.Graph = deepcopy(G.copy())
    K.clear_edges()
    K.add_edges_from(min_matching)

    # draw_bipartite(G, idx_v)
    # draw_bipartite(K, idx_v)
    # draw_bipartite(H, idx_v)
