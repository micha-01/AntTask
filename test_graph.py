from datetime import datetime
import numpy as np
import unittest
import random
import graph
import networkx as nx
from pathlib import Path

from task import read_tasks_csv


class GraphTest(unittest.TestCase):
    seed = 0
    tasks = graph.read_tasks_csv(Path("tasks.csv"))
    G, idx_v = graph.tasks_to_bipartite(tasks)
    pheromones: np.array = np.zeros((len(G), len(G)))

    def setUp(self) -> None:
        random.seed(self.seed)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_tasks_to_bipartite_very_small(self):
        tasks = read_tasks_csv(Path("tasks_test_very_small.csv"))

        G, H, idx_v = self.init_G_H(tasks)

        # add nodes to TEST graph H for task a, b & 10 nodes for the time
        H.add_nodes_from([i for i in range(idx_v + 10)])

        # add edges to TEST graph H
        one = tasks[0]
        two = tasks[1]
        j: int = 0
        for _ in range(0, one['duration']):
            self.add_edges_from_node(
                H, j, one['start'].hour, one['end'].hour,
                idx_v, one['prio'], 0
            )
            j += 1

        self.assertEqual(j, one['duration'])

        for _ in range(one['duration'], one['duration'] + two['duration']):
            self.add_edges_from_node(
                H, j, two['start'].hour, two['end'].hour,
                idx_v, two['prio'], 0
            )
            j += 1

        self.assertEqual(j, one['duration'] + two['duration'])

        self.check_G_H(G, H)

    def test_tasks_to_bipartite_small(self):
        tasks = read_tasks_csv(Path("tasks_test_small.csv"))

        G, H, idx_v = self.init_G_H(tasks)

        # add nodes to TEST graph H for task a, b & 10 nodes for the time
        H.add_nodes_from([i for i in range(idx_v + 10)])

        # add edges to TEST graph H
        one = tasks[0]
        two = tasks[1]
        three = tasks[2]

        j: int = 0
        for _ in range(0, one['duration']):
            self.add_edges_from_node(
                H, j, one['start'].hour, one['end'].hour,
                idx_v, one['prio'], 0
            )
            j += 1

        self.assertEqual(j, one['duration'])

        for _ in range(one['duration'], one['duration'] + two['duration']):
            self.add_edges_from_node(
                H, j, two['start'].hour, two['end'].hour,
                idx_v, two['prio'], 0
            )
            j += 1

        start_three = one['duration'] + two['duration']
        self.assertEqual(j, one['duration'] + two['duration'])

        for _ in range(start_three, start_three + three['duration']):
            self.add_edges_from_node(
                H, j, three['start'].hour, three['end'].hour,
                idx_v, three['prio'], 0
            )
            j += 1

        self.assertEqual(j,
                         one['duration'] + two['duration'] + three['duration'])

        self.check_G_H(G, H)

    def test_tasks_to_bipartite_small_two_days(self):
        tasks = read_tasks_csv(Path("tasks_test_small_two_days.csv"))

        G, H, idx_v = self.init_G_H(tasks)

        # add nodes to TEST graph H for task a, b & 10 nodes for the time
        H.add_nodes_from([i for i in range(idx_v + 10 * 2)])

        # add edges to TEST graph H
        one = tasks[0]
        two = tasks[1]
        three = tasks[2]

        j: int = 0
        for _ in range(0, one['duration']):
            self.add_edges_from_node(
                H, j, one['start'].hour, one['end'].hour,
                idx_v, one['prio'], 0
            )
            j += 1

        self.assertEqual(j, one['duration'])

        for _ in range(one['duration'], one['duration'] + two['duration']):
            self.add_edges_from_node(
                H, j, two['start'].hour, two['end'].hour,
                idx_v, two['prio'], 0
            )
            j += 1

        start_three = one['duration'] + two['duration']
        self.assertEqual(j, one['duration'] + two['duration'])

        for _ in range(start_three, start_three + three['duration']):
            self.add_edges_from_node(
                H, j, three['start'].hour, three['end'].hour,
                idx_v, three['prio'], 1
            )
            j += 1

        self.assertEqual(j,
                         one['duration'] + two['duration'] + three['duration'])

        self.check_G_H(G, H)

    def test_tasks_to_bipartite_medium(self):
        tasks = read_tasks_csv(Path("tasks_test_medium.csv"))
        self.tasks_to_bipartite_general(tasks, 2)

    ################### HELPER METHODS ###################
    def check_G_H(self, G, H):
        """
        test if the given graph G and the TEST graph H have
        the same dimensions, nodes, edges and weights
        """
        # same dimensions
        self.assertEqual(len(G), len(H))
        self.assertEqual(G.size(), H.size())

        # check nodes, edges and weights
        self.assertEqual(G.nodes, H.nodes)
        self.assertEqual(G.edges, H.edges)
        for (u, v, c) in H.edges.data('weight'):
            self.assertEqual(G[u][v]['weight'], c)

    def init_G_H(self, tasks):
        """
        initializes graph G and TEST graph H
        """
        G: nx.Graph
        G, g_idx_v = graph.tasks_to_bipartite(tasks)
        H: nx.Graph = nx.Graph()
        idx_v: int = sum(t['duration'] for t in tasks)
        self.assertEqual(g_idx_v, idx_v)
        return (G, H, idx_v)

    def tasks_to_bipartite_general(self, tasks, number_days: int):
        """
        general test to assert the correct execution of tasks_to_bipartite
        """
        G, H, idx_v = self.init_G_H(tasks)

        # add nodes for task a to c & 10 nodes for the time (10 per day)
        H.add_nodes_from([i for i in range(idx_v + 10 * number_days)])

        j = 0
        first_day: datetime = tasks[0]['start']
        last_task: int = 0
        snd_day_idx: int = 0

        # split tasks in two lists one for each day
        for i, t in enumerate(tasks):
            if t['start'].day != first_day.day:
                snd_day_idx = i
                break

        tasks_day_one = tasks[:snd_day_idx]
        tasks_day_two = tasks[snd_day_idx:]

        for task in tasks_day_one:
            for _ in range(last_task, last_task + task['duration']):
                self.add_edges_from_node(
                    H, j, task['start'].hour, task['end'].hour,
                    idx_v, task['prio'], 0
                )
                j += 1
                last_task += task['duration']

        for task in tasks_day_two:
            for _ in range(last_task, last_task + task['duration']):
                self.add_edges_from_node(
                    H, j, task['start'].hour, task['end'].hour,
                    idx_v, task['prio'], 1
                )
                j += 1
                last_task += task['duration']

        self.check_G_H(G, H)

    def add_edges_from_node(self, G: nx.Graph, node: int, start: int, end: int,
                            idx_v: int, weight: int, day_diff: int = 0):
        while (start < end):
            G.add_edge(node, day_diff * 10 + start - 8 + idx_v, weight=weight)
            start += 1


if __name__ == "__main__":
    unittest.main()
