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


    """
    """
if __name__ == "__main__":
    G = tasks_to_bipartite(tasks)

    # draw bipartite graph
    ax = plt.subplot()
    top = list(filter(lambda x: type(x) is str, G.nodes))  # 1st partition (U)
    nx.draw(G, pos=nx.bipartite_layout(G, top), with_labels=True)
    plt.show()

    # Also draw weights
    ax = plt.subplot()
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
