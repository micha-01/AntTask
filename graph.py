import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from task import read_tasks_csv, Task
from datetime import timedelta
from copy import deepcopy

tasks = read_tasks_csv(Path("tasks.csv"))



def tasks_to_bipartite(tasks: list[Task]) -> nx.Graph:
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
            G.add_edge(i + j, start.hour - idx_v, weight=t['prio'])
            start += timedelta(hours=1)
            if start >= end:
                start = t['start']
                j += 1
        i += j
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
