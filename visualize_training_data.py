'''
This file is mostly for testing purpose
In the current configuration it is used for testing contact points
'''

import numpy as np
import torch
import argparse
from fast_jtnn import *
from robot_utils import *
import numpy as np
import robot_utils.tasks as tasks
import pyrobotdesign as rd
from design_search import make_graph, build_normalized_robot
import random



task_name = "FlatTerrainTask"

adj_data_name = "data/new_train_data_loc_prune/adj1000.npy"
feat_data_name = "data/new_train_data_loc_prune/feat1000.npy"
loc_data = "data/new_train_data_loc_prune/loc1000.npy"

print(f"loading pruned data from {adj_data_name}")
attr = np.load(feat_data_name, allow_pickle=True)
conn = np.load(adj_data_name, allow_pickle=True)
loc_data = np.load(loc_data, allow_pickle=True)
n_sample = 1

for i in range(n_sample):
    ind = 78 # np.random.randint(attr.shape[0])
    adj_matrix_np, features_np = conn[ind], attr[ind]
    robot = graph_to_robot(adj_matrix_np, features_np)
    task_class = getattr(tasks, task_name)
    task = task_class(episode_len=128)
    robot_init_pos, has_self_collision = presimulate(robot)
    if has_self_collision:
        print("Warning: robot self-collides in initial configuration")
    # print(robot_init_pos)
    loc, valid = presimulate_contact(robot, robot_init_pos)
    print(loc)
    print(valid)
    print(loc_data[ind])
    main_sim = rd.BulletSimulation(task.time_step)
    task.add_terrain(main_sim)
    # Rotate 180 degrees around the y axis, so the base points to the right
    main_sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    robot_idx = main_sim.find_robot_index(robot)
    input_sequence = None
    camera_params, record_step_indices = view_trajectory(
        main_sim, robot_idx, input_sequence, task)
