'''
This file optimize the target function.
Some example commands:
python run_bo.py --model sum_ls28_pred20 --model_iter 400000 --log_dir new_log --latent_size 28 --no_noise \
--alpha 1e-3 --acq ei --rd_explore
python run_bo.py --no_noise --rd_explore --domain_reduction --task FrozenLakeTask
'''

from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from fast_jtnn import *
from sample import decode_graph
from robot_utils import *
import random
import robot_utils.tasks as tasks
import torch
import numpy as np
from bayes_opt import UtilityFunction
import time
import pickle
import argparse
import os
import uuid



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="FlatTerrainTask")
    parser.add_argument('--bound', type=float, default=3)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--init', type=int, default=50)
    parser.add_argument('--n_iter', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=25)
    parser.add_argument('--latent_size', type=int, default=28)
    parser.add_argument('--model', type=str, default="sum_ls28_pred20")
    parser.add_argument('--model_iter', type=int, default=400000)
    parser.add_argument('--log_dir', type=str, default="new_log")
    parser.add_argument('--no_noise', default=False, action="store_true")
    parser.add_argument('--verbose', default=False, action="store_true")
    parser.add_argument('--jobs', type=int, default=16)
    parser.add_argument('--gaussian', default=False, action="store_true")
    parser.add_argument('--encode', type=str, default="sum")
    parser.add_argument('--cache', default=False, action="store_true")
    parser.add_argument('--acq', type=str, default="ei") # ucb
    parser.add_argument('--alpha', type=float, default=1e-3) # 1e-6
    parser.add_argument('--rd_explore', default=False, action="store_true")
    parser.add_argument('--domain_reduction', default=False, action="store_true")
    parser.add_argument('--pred', default=True, action="store_false")

    args = parser.parse_args()

    # initialized trained model
    hidden_size = 450
    depthT = 20
    latent_size = args.latent_size
    print(f"using latent size {latent_size}")
    cuda = args.cuda
    bound = args.bound
    n_iter = args.n_iter
    task_name = args.task
    init_points = args.init
    log_interval = args.log_interval
    model_dir = args.model
    model_file_name = "model.iter-" + str(args.model_iter)
    model_path = os.path.join(model_dir, model_file_name)
    model = JTNNVAE(hidden_size, latent_size, depthT, args.encode, args.pred)
    print(f"loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path))

    log_folder_name = args.model + "_" + task_name + "_iter_" + str(args.model_iter) + "_ls" + str(args.latent_size)
    if args.gaussian:
        log_folder_name += "_gaussian"
    if args.rd_explore:
        log_folder_name += "_rdexplore"
    if args.domain_reduction:
        log_folder_name += "_reduce"
    log_folder_name += f"_alpha={args.alpha}_acq={args.acq}"
    # add a random number so that we can run multiple in parallel
    log_folder_name += "_" + uuid.uuid4().hex[:5]

    log_file_path = os.path.join(args.log_dir, log_folder_name)
    if not os.path.exists(log_file_path):
        # Create a new directory because it does not exist
        os.makedirs(log_file_path)
        print(f"creating log filepath {log_file_path}...")

    if cuda:
        model = model.cuda()

    # other parameters
    jobs = args.jobs
    episodes = 1

    # tmp cache related
    if args.cache:
        result_cache = dict()
        result_cache_hit_count = 0

    pbounds = {}
    if args.gaussian:
        print("using gaussian transformation on latent space")
        for i in range(latent_size):
            pbounds[str(i)] = (0.001, 0.999)
    else:
        for i in range(latent_size):
            pbounds[str(i)] = (-bound, bound)

    # initialize task to optimize
    episode_len = 128
    task_class = getattr(tasks, task_name)

    if args.no_noise:
        task = task_class(force_std=0.0, torque_std=0.0, episode_len=episode_len)
    else:
        task = task_class()

    if args.domain_reduction:
        bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.5)
        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
            bounds_transformer=bounds_transformer,
        )
    else:
        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
        )

    utility = UtilityFunction(kind=args.acq,
                              kappa=2.576,
                              kappa_decay=1.0,
                              xi=1e-2)

    target_list = []
    next_point_list = []

    def black_box_function(**kwargs):
        """Function with unknown internals we wish to maximize.
        """
        latent_vec = optimizer.space.params_to_array(kwargs)
        # latent_vec = np.array(list(kwargs.values()))
        if args.gaussian:
            latent_vec = uniform_2_gaussian(latent_vec)
        vect = torch.tensor(latent_vec).float().reshape(1, -1)
        if cuda:
            vect = vect.cuda()
        adj, attr = decode_graph(model, vect)

        if args.cache:
            global result_cache_hit_count
            result_cache_key = tuple(attr)
            if result_cache_key in result_cache:
                result = result_cache[result_cache_key]
                result_cache_hit_count += 1
                print(f"hitting cache, total = {result_cache_hit_count}")
            else:
                robot = graph_to_robot(adj, attr)
                opt_seed = random.getrandbits(32)
                input_sequence, result = simulate(robot, task, opt_seed, jobs, episodes)
                result_cache[result_cache_key] = result
            return result

        # igonore the cache otherwise
        robot = graph_to_robot(adj, attr)
        opt_seed = random.getrandbits(32)
        input_sequence, result = simulate(robot, task, opt_seed, jobs, episodes)
        return result


    optimizer.set_gp_params(alpha=args.alpha)
    np.random.seed()

    def random_acq(i):
        print(f"\nrandom iteration {i}")
        next_point_vect = optimizer.space.random_sample()
        next_point = optimizer.space.array_to_params(next_point_vect)
        start = time.time()
        target = black_box_function(**next_point)
        end = time.time()
        print(f"black box time: {end - start}")
        optimizer.register(params=next_point, target=target)
        print(f"reward: {target}")
        if args.verbose:
            print(f"latent: {list(next_point.values())}")
        target_list.append(target)
        next_point_list.append(optimizer.space.params_to_array(next_point))

    def bo_acq(i):
        print(f"\noptimization iteration: {i}")
        start = time.time()
        utility.update_params()  # with decay=1 it seems that this does nothing
        next_point = optimizer.suggest(utility)
        end = time.time()
        print(f"suggest time: {end - start}")

        start = time.time()
        target = black_box_function(**next_point)
        end = time.time()
        print(f"black box time: {end - start}")

        optimizer.register(params=next_point, target=target)
        print(f"reward: {target}")
        if args.verbose:
            print(f"latent: {list(next_point.values())}")

        target_list.append(target)
        next_point_list.append(optimizer.space.params_to_array(next_point))

    # init loop
    for i in range(init_points):
        random_acq(i)

    # optimization loop
    for i in range(n_iter):
        if (i+1) % 10 == 0 and args.rd_explore:
            random_acq(i)
        else:
            bo_acq(i)

        if args.domain_reduction:
            optimizer.set_bounds(
                optimizer._bounds_transformer.transform(optimizer._space))

        if (i+1) % log_interval == 0:
            print(f"current max: {optimizer.max['target']}")
            cur_max_latent = optimizer.space.params_to_array(optimizer.max['params'])
            print(f"max latent: {cur_max_latent}")
            if args.gaussian:
                print(f"max latent transformed: {uniform_2_gaussian(cur_max_latent)}")
            print("saving files...")
            try:
                with open(os.path.join(log_file_path, "target"), 'wb') as f:
                    pickle.dump(target_list, f)
                with open(os.path.join(log_file_path, "point"), 'wb') as f:
                    pickle.dump(next_point_list, f)
            except:
                print("save trajectory error: most likely not enough storage")
                continue
