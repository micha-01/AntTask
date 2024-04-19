import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from task import read_tasks_csv, Task
from datetime import timedelta
from copy import deepcopy

tasks = read_tasks_csv(Path("tasks.csv"))



def tasks_to_bipartite(tasks: list[Task],
                       weight: int | None = None) -> nx.Graph:
    """
    creates a bipartite graph from a list of tasks
    cannot use nx.algorithms.bipartite because the graph can be disconnected
    """
    G = nx.Graph()

    for i, t in enumerate(tasks):
        # add a node for each 1h segment of a task
        # t13 := 3rd segment of task 1
        G.add_nodes_from([f"t{i}_{x}" for x in range(t['duration'])])

        # add node for each work hour of the day, i.e. 08:00 to 18:00
        G.add_nodes_from([i for i in range(8, 19)])

        # add edge iff task can be done in this hour segment,
        # i.e. is between the start time and the end time of the task
        # e.g. start: 08:00, end: 10:00 -> edge only to 8 and 9 (NOT 10)
        start = t['start']
        end = t['end']
        j: int = 0
        while (j < t['duration']):
            G.add_edge(f"t{i}_{j}", start.hour, weight=t['prio'])
            start += timedelta(hours=1)
            if start >= end:
                start = t['start']
                j += 1
    return G


def transform_to_TSP(G: nx.Graph) -> nx.Graph:
    """
    transforms the graph G to H (G' in the paper)
    """
    H = deepcopy(G.copy())

    # add virtual edges for all task nodes: t0_0 -> vt0_0, ...
    U = list(filter(lambda x: type(x) is str, G.nodes))
    for u in U:
        virt: str = f"v{u}"
        print(u, virt)
        H.add_node(virt)
        for v in G[u]:
            H.add_edge(virt, v, weight=0)
        H.add_edge(u, virt, weight=W_MAX)
        for uu in U:
            if uu != u:
                H.add_edge(virt, uu, weight=0)
    return H
if __name__ == "__main__":
    G = tasks_to_bipartite(tasks)
    ax = plt.subplot()
    top = list(filter(lambda x: type(x) is str, G.nodes))
    nx.draw(G, pos=nx.bipartite_layout(G, top), with_labels=True)
    plt.show()
