'''
Collect training data without using grammar rules
This is only for comparison purpose
This would unavoidably generate a lot of invalid designs
'''

import torch
import torch.nn.functional as F
import numpy as np
import os

n_nodes_max = 25
n_types = 42
iterations = 500000


def random_graph():
    n_nodes = np.random.randint(2, n_nodes_max)
    attr = np.random.randint(n_types, size=n_nodes)

    connection = [0]
    child_list = np.zeros(n_nodes)
    for i in range(n_nodes-1):
        while True:
            proposed_parent = np.random.randint(i+1)
            budget = 3
            if proposed_parent == 0:
                budget += 1
            if child_list[proposed_parent] < budget:
                break
        child_list[proposed_parent] += 1
        connection.append(proposed_parent)

    conn_one_hot = F.one_hot(torch.tensor(connection), num_classes=n_nodes).type(torch.float)
    conn_one_hot[0,0] = 0
    return attr, (conn_one_hot + conn_one_hot.T).numpy()


adj_data = []
features_data = []

for i in range(iterations):
    features_np, adj_matrix_np  = random_graph()
    adj_data.append(adj_matrix_np)
    features_data.append(features_np)
    if i % 10000 == 0:
        print(f"iteration {i}")
save_dir = "../data/random_data"
if not os.path.exists(save_dir):
    # Create a new directory because it does not exist
    os.makedirs(save_dir)
np.save(os.path.join(save_dir, "adj.npy"), adj_data)
np.save(os.path.join(save_dir, "feat.npy"), features_data)