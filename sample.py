'''
Validation File
All sorts of visualization and examination for the trained VAE model and the optimized designs

example run:
For testing distribution of the generative model:
- python sample.py --sample --calc_stats --model MODEL

For testing reconstruction of the generative model:
- python sample.py --reconstruct --visualize_robot --model MODEL
add --perturb to see samples around the data point

For decoding a latent vector (place the latent vector in the corresponding part first):
- python sample.py --decode --visualize_robot --model MODEL

For visualizing results from BO:
- python sample.py --model sum_ls28_pred20/model.iter-400000 --visualize_from_file  --task FrozenLakeTask
'''


import torch
import argparse
from fast_jtnn import *
from robot_utils import *
import numpy as np
import robot_utils.tasks as tasks
import pyrobotdesign as rd
from design_search import make_graph, build_normalized_robot
import random
import matplotlib.pyplot as plt

def sample_graph(model):
    root, pred_nodes = model.sample_prior()
    n_nodes = len(pred_nodes)
    adj_matrix_np = np.zeros([n_nodes, n_nodes])
    features_np = np.zeros(n_nodes)
    idx_offset = root.idx
    for i in range(n_nodes):
        node = pred_nodes[i]
        features_np[i] = node.wid
        for nei in node.neighbors:
            true_idx = nei.idx - idx_offset
            adj_matrix_np[true_idx, i] = 1
            adj_matrix_np[i, true_idx] = 1
    return adj_matrix_np, features_np,

def decode_graph(model, tree_vec):
    root, pred_nodes = model.decode(tree_vec, prob_decode=False)
    n_nodes = len(pred_nodes)
    adj_matrix_np = np.zeros([n_nodes, n_nodes])
    features_np = np.zeros(n_nodes)
    idx_offset = root.idx
    for i in range(n_nodes):
        node = pred_nodes[i]
        features_np[i] = node.wid
        for nei in node.neighbors:
            true_idx = nei.idx - idx_offset
            adj_matrix_np[true_idx, i] = 1
            adj_matrix_np[i, true_idx] = 1
    return adj_matrix_np, features_np


def get_robot_image(robot, task):
  sim = rd.BulletSimulation(task.time_step)
  task.add_terrain(sim)
  viewer = rd.GLFWViewer()
  if robot is not None:
    robot_init_pos, _ = presimulate(robot)
    # Rotate 180 degrees around the y axis, so the base points to the right
    sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    robot_idx = sim.find_robot_index(robot)

    # Get robot position and bounds
    base_tf = np.zeros((4, 4), order='f')
    lower = np.zeros(3)
    upper = np.zeros(3)
    sim.get_link_transform(robot_idx, 0, base_tf)
    sim.get_robot_world_aabb(robot_idx, lower, upper)
    viewer.camera_params.position = base_tf[:3,3]
    viewer.camera_params.yaw = - np.pi / 3
    viewer.camera_params.pitch = -np.pi / 4.5
    viewer.camera_params.distance = np.linalg.norm(upper - lower) * 1.5
  else:
    viewer.camera_params.position = [1.0, 0.0, 0.0]
    viewer.camera_params.yaw = -np.pi / 3
    viewer.camera_params.pitch = -np.pi / 6
    viewer.camera_params.distance = 5.0

  viewer.update(task.time_step)
  return viewer.render_array(sim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsample', type=int, default=1)
    parser.add_argument('--model', required=True)

    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--latent_size', type=int, default=28)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--decode', default=False, action="store_true")
    parser.add_argument('--sample', default=False, action="store_true")

    parser.add_argument("--grammar_file", type=str, default="data/designs/grammar_apr30.dot",
                        help="Grammar file (.dot)")

    # for visualization of the actual robot
    parser.add_argument("--visualize_robot", default=False, action="store_true")

    # for visualization of the robot graph
    parser.add_argument("--visualize_graph", default=False, action="store_true")

    # Mostly for testing purpose, examine the behavior of the generator as time goes on
    parser.add_argument("--calc_stats", default=False, action="store_true")

    # FlatTerrainTask
    parser.add_argument("--task", type=str, default="FlatTerrainTask", help="Task (Python class name)")
    parser.add_argument("-l", "--episode_len", type=int, default=128,
                        help="Length of episode")
    parser.add_argument("-j", "--jobs", type=int, default=4,
                        help="Number of jobs/threads")
    parser.add_argument("-o", "--optim", default=False, action="store_true",
                        help="Optimize a trajectory")
    parser.add_argument("-s", "--opt_seed", type=int, default=None,
                        help="Trajectory optimization seed")
    parser.add_argument("--reconstruct", default=False, action="store_true")
    parser.add_argument("--use_grammar", default=False, action="store_true")

    parser.add_argument("--no_noise", default=False, action="store_true")
    parser.add_argument("--visualize_from_file", default=False, action="store_true")
    parser.add_argument("--perturb", default=False, action="store_true")
    parser.add_argument('--encode', type=str, default="sum")
    parser.add_argument("--pred", default=True, action="store_false")
    args = parser.parse_args()

    model = JTNNVAE(args.hidden_size, args.latent_size, args.depthT, args.encode, args.pred)
    model.load_state_dict(torch.load(args.model))
    model = model.cuda()

    # A set of rules for visualization purpose
    rule_list = {"RidgedTerrainTask":
                     "0, 7, 1, 13, 1, 2, 16, 12, 13, 6, 4, 19, 4, 17, 5, 3, 2, 16, 4, 5, 18, 9, 8, 9, 9, 8",
                 "FlatTerrainTask":
                     "0, 12, 7, 1, 12, 3, 10, 1, 3, 1, 12, 12, 1, 3, 10, 2, 16, 8, 1, 3, "
                     "12, 4, 1, 3, 2, 12, 18, 9, 18, 8, 5, 5, 1, 12, 6, 3",
                 "GapTerrainTask":
                     "0, 1, 1, 7, 1, 6, 10, 3, 2, 4, 10, 10, 3, 16, 4, 16, "
                     "18, 2, 5, 16, 8, 4, 8, 8, 18, 4, 5, 15, 9, 8, 8",
                 "FrozenLakeTask":
                     "0, 1, 1, 1, 6, 7, 10, 11, 13, 2, 4, 3, 4, 16, 8, 14, 4, 8, 3, 15, 15, 5, 3, 9, 8"}

    if args.reconstruct:
        print("using grammar file...")
        rule_graphs = rd.load_graphs(args.grammar_file)
        rules = [rd.create_rule_from_graph(g) for g in rule_graphs]
        all_labels = set()
        for rule in rules:
            for node in rule.lhs.nodes:
                all_labels.add(node.attrs.require_label)
        all_labels = sorted(list(all_labels))
        preprocessor = Preprocessor(all_labels=all_labels)

        rule_sequence = [int(s) for s in rule_list[args.task].split(", ")]
        graph = make_graph(rules, rule_sequence)
        cur_conn, cur_attr = preprocessor.preprocess(graph)

        if args.use_grammar:
            adj_matrix_np, features_np = cur_conn, cur_attr
        else:
            # we convert the robot to latent, and reconstruct the graph
            batch = tensorize(np.expand_dims(cur_attr, 0), np.expand_dims(cur_conn, 0))
            tree_mean, tree_var = model.encode_latent(batch)
            vec = tree_mean
            print(f"The latent vector is: {vec.cpu().detach().numpy()}")

            if args.perturb:
                uniform_rec = gaus_2_uni(vec)
                noise = np.random.rand(args.latent_size) * 0.1
                new_vec = np.clip(uniform_rec + noise, 0.001, 0.999)
                vec = uniform_2_gaussian(new_vec)
                print(f"perturbed latent vector is : {vec}")
            else:
                vec = vec.cpu().detach().numpy()
            adj_matrix_np, features_np = decode_graph(model, torch.tensor(vec.astype(np.float32), device="cuda"))
    elif args.decode:
        vec_str = "-0.42105492  2.07314603 -0.81671534  2.05360516  0.25225775  1.89400168\
                 -1.56602309 -0.28385988 -0.81614669 -1.19536242 -2.68024078  0.54721525\
                 -2.95561138  2.1500398  -1.55778335  0.83260687  1.46759942  1.93973844\
                 -1.17244679 -2.02176059 -2.45239294 -2.30022974  2.50066512  1.8945962\
                  2.887226    1.62168349  0.60159975 -1.47454025"
        vec = [float(r) for r in vec_str.split()]
        vec = torch.tensor(vec).float().reshape(1, -1).cuda()
        adj_matrix_np, features_np  = decode_graph(model, vec)
        print(adj_matrix_np.tolist())
        print(features_np.tolist())
    elif args.sample:
        if args.calc_stats:
            from collections import defaultdict
            d = defaultdict(lambda: 0)
            num_collision = 0
            for i in range(1000):
                adj_matrix_np, features_np,  = sample_graph(model)

                draw_graph(features_np, adj_matrix_np, show=True)

                robot = graph_to_robot(adj_matrix_np, features_np)
                robot_init_pos, has_self_collision = presimulate(robot)

                g_length = adj_matrix_np.shape[0]
                d[g_length] += 1
                if has_self_collision:
                    num_collision += 1
            print(f"num of collision: {num_collision}")
            for key in sorted(d):
                print("%s: %s" % (key, d[key]))
        else:
            for i in range(args.nsample):
                adj_matrix_np, features_np,  = sample_graph(model)
    else:
        print("please specify which operation to take (unless visualizing list)")

    if args.visualize_graph:
        draw_graph(features_np, adj_matrix_np, show=True)

    if args.visualize_robot:
        if args.opt_seed is not None:
            opt_seed = args.opt_seed
        else:
            opt_seed = random.getrandbits(32)
            print("Using optimization seed:", opt_seed)

        task_class = getattr(tasks, args.task)
        if args.no_noise:
            task = task_class(force_std=0.0, torque_std=0.0, episode_len=args.episode_len)
        else:
            task = task_class(episode_len=args.episode_len)

        robot = graph_to_robot(adj_matrix_np, features_np)

        if args.optim:
            input_sequence, result = simulate(robot, task, opt_seed, args.jobs, 1)
            print("Result:", result)
        else:
            input_sequence = None

        robot_init_pos, has_self_collision = presimulate(robot)
        if has_self_collision:
            print("Warning: robot self-collides in initial configuration")
        main_sim = rd.BulletSimulation(task.time_step)
        task.add_terrain(main_sim)
        # Rotate 180 degrees around the y axis, so the base points to the right
        main_sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
        robot_idx = main_sim.find_robot_index(robot)

        camera_params, record_step_indices = view_trajectory(
            main_sim, robot_idx, input_sequence, task)

    # visualize the BO samples and the corresponding performance of the robots
    if args.visualize_from_file:

        import pickle
        import os

        terrain_name = args.task  # "FlatTerrainTask"
        exp_type = "compare"
        category = "GLSO"


        folder_path = os.path.join("new_log", terrain_name, exp_type, category)
        target_lists = []
        next_point_lists = []
        print(folder_path)
        list_of_files = {}
        length = 500
        for (dirpath, dirnames, filenames) in os.walk(folder_path):
            for filename in filenames:
                current_file = os.sep.join([dirpath, filename])
                if filename == "point":
                    with open(current_file, 'rb') as f:
                        points = pickle.load(f)[:length]
                        pad_length = length - len(points)
                        points = np.pad(points, ((0, 0), (0, pad_length)), 'constant')
                        next_point_lists.append(points)
                elif filename == "point.npy":
                    points = np.load(current_file)[:length]
                    pad_length = length - points.shape[0]
                    points = np.pad(points, ((0, 0), (0, pad_length)), 'constant')
                    next_point_lists.append(points)
                elif filename == "target":
                    with open(current_file, 'rb') as f:
                        points = pickle.load(f)[:length]
                        pad_length = length - len(points)
                        points = np.pad(points, (0, pad_length), 'constant')
                        target_lists.append(points)
                elif filename == "target.npy":
                    points = np.load(current_file)[:length]
                    pad_length = length - points.shape[0]
                    points = np.pad(points, (0, pad_length), 'constant')
                    target_lists.append(points)
                else:
                    print("unknown file spotted")
                    exit()

        from bayes_opt import BayesianOptimization

        pbounds = {}
        for i in range(args.latent_size):
            pbounds[str(i)] = (-3, 3)
        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
        )

        target_lists = np.array(target_lists)
        next_point_lists = np.array(next_point_lists)

        index = np.unravel_index(target_lists.argmax(), target_lists.shape)
        print(target_lists[index])
        point = next_point_lists[index]


        try:
            real_point = optimizer.space.params_to_array(point)
        except:
            real_point = point

        task_class = getattr(tasks, args.task)
        if args.no_noise:
            task = task_class(force_std=0.0, torque_std=0.0, episode_len=args.episode_len)
        else:
            task = task_class(episode_len=args.episode_len)
        real_point = torch.tensor([real_point], dtype=torch.float32).cuda()
        adj_matrix_np, features_np = decode_graph(model, real_point)
        robot = graph_to_robot(adj_matrix_np, features_np)

        # need to build robot here
        image = get_robot_image(robot, task)
        plt.axis('off')
        plt.imshow(image, origin='lower')
        plt.show()


