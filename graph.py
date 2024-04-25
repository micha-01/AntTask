from copy import deepcopy
import networkx as nx
from networkx.algorithms import min_weight_matching
from pathlib import Path
from task import read_tasks_csv, Task
from datetime import timedelta
from random import choice, choices
import numpy as np

tasks = read_tasks_csv(Path("tasks.csv"))

NUM_ANTS: int = 10
W_MAX: int = 100
ALPHA = 3
BETA = 3
RHO = 0.5


def tasks_to_bipartite(tasks: list[Task]) -> (nx.Graph, int):
    """
    creates a bipartite graph from a list of tasks
    cannot use nx.algorithms.bipartite because the graph can be disconnected
    """
    G = nx.Graph()

    # find the starting index for v_j in V
    idx_v: int = sum(map(lambda task: task['duration'], tasks))

    # u_i in U: i in {0, ..., idx_v - 1}
    # add a node for each 1h segment of a task
    # t13 := 3rd segment of task 1
    G.add_nodes_from([i for i in range(idx_v)])

    # v_j in V: j in {idx_v, ..., idx_v + 8}
    # add node for each work hour of the day, i.e. 08:00 to 18:00
    G.add_nodes_from([i for i in range(idx_v, idx_v + 10)])

    i: int = 0
    for t in tasks:
        # add edge iff task can be done in this hour segment,
        # i.e. is between the start time and the end time of the task
        # e.g. start: 08:00, end: 10:00 -> edge only to 8 and 9 (NOT 10)
        start = t['start']
        end = t['end']
        j: int = 0
        while (j < t['duration']):
            G.add_edge(i + j, (end.day - start.day) * 10 + start.hour + idx_v, weight=t['prio'])
            start += timedelta(hours=1)
            if start >= end:
                start = t['start']
                j += 1
        i += j
    return (G, idx_v)


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


def ant_matching(G: nx.Graph, t_max: int, idx_v: int) -> (list[(int, int)], int):
    """
    uses ACO to find a matching on the given bipartite graph G
    """
    best_matching_list = []
    weight_min = 1000000  # should be infinity

    # dimension: #nodes * #edges
    pheromones: np.array = np.zeros((len(G), len(G)))

    for t in range(t_max):
        best_matching_list, weight_min = generate_solution(
            G, t, weight_min, best_matching_list, pheromones, idx_v
        )
        update_pheromone(best_matching_list, pheromones, weight_min, G)

    return best_matching_list, weight_min


def generate_solution(G: nx.Graph, t: int, weight_min: int,
                      best_matching_list: list, pheromones: np.array,
                      idx_v: int):
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


if __name__ == "__main__":
    G, idx_v = tasks_to_bipartite(tasks)
    matching, weight = ant_matching(G, 100, idx_v)
    for (u, v) in reversed(deepcopy(matching)):
        if u is None or v is None:
            matching.remove((u, v))
            weight -= W_MAX
    print(matching, weight)

    min_matching = list(min_weight_matching(G))
    s = 0
    for (u, v) in min_matching:
        s += G[u][v]['weight']
    print(min_matching, s)
