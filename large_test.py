from datetime import datetime, timedelta
import numpy as np
import unittest
import random
from task import read_tasks_csv
import networkx as nx
from pathlib import Path
import csv
from graph import tasks_to_bipartite, ant_matching, T_MAX, W_MAX, get_first_and_last_day, print_timetable
from copy import deepcopy

#from task import read_tasks_csv

def create_tasks(days: int, nr_of_tasks: int, multiple_days=False) -> str:  #returns the name of the file
    tasks = []
    #put it in an csv and then read it
    with open('t0.csv', 'w', newline='') as csvfile:
        taskwriter = csv.writer(csvfile, delimiter=',',
                            quotechar=',', quoting=csv.QUOTE_MINIMAL)
        if(multiple_days):
            #for tasks spanning multiple days
            None
        else:
            date = datetime(2024, 4, 15, 8, 0)
            #print(date)
            #date += timedelta(days=1)
            #print(date)
            #date += timedelta(hours=1)
            #print(date)
            
            counter = 0 #priority
            for i in range(days):
                for j in range(nr_of_tasks):
                    row = []
                    k = random.randint(0,6)
                    l = random.randint(k,7)
                    #create random start k in[0,6] and random end l in [k+1,7]  (task is at least 1 hour)
                    #create line for task
                    #{'id': 0, 'name': 'a', 'start': datetime.datetime(2024, 4, 15, 8, 0), 'end': datetime.datetime(2024, 4, 15, 17, 0), 'prio': 1, 'duration': 6}
                    taskwriter.writerow([str(counter),str(counter),str(date+timedelta(hours=k)),str(date+timedelta(hours=l)),str(counter+1),str(l-k)])
                    counter += 1
                date += timedelta(days=1)
            None

    
    print("Tasks created!")
    return "t0.csv"






if __name__ == "__main__":
    random.seed(0)
    
    name = create_tasks(10,10)
    #read from the created csv
    tasks = read_tasks_csv(Path(name))
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

    (first,last) = get_first_and_last_day(tasks)

    print("ANT_MATCHING: ", matching, weight)
    print_timetable(matching,idx_v,first,last,node_to_task)

    print("MIN_MATCHING: ", min_matching, s)
    print_timetable(min_matching,idx_v,first,last,node_to_task)

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

    #print(tasks)
    #do tasks


    #record time it takes

    