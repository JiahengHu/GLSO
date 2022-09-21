'''
This file is in charge of converting between robot tree and graphs of format adj + feat
'''
import pyrobotdesign as rd
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import tasks
import time
import argparse

def graph_to_robot(adj, attr):
    """Convert a graph to robot graph."""
    robot = rd.Robot()
    link_list = []
    adj = np.array(adj, dtype=int)
    attr = np.array(attr, dtype=int)
    for idx in range(len(adj)):
        robot_attr = get_node_attributes(attr, adj, idx)
        new_link = rd.Link(*robot_attr)
        link_list.append(new_link)
    robot.links = np.array(link_list)
    robot = normalized_robot(robot)
    return robot


def rot_2_quat(angle_axis):
    angle = angle_axis[3]
    axis = angle_axis[:3]
    r = R.from_rotvec(angle / 180 * np.pi * np.array(axis))
    quat_comp = r.as_quat()  # x, y, z, w
    quat_final = rd.Quaterniond(quat_comp[3], quat_comp[0], quat_comp[1], quat_comp[2])
    return quat_final


# take in a feature, tell us how we should initialize these values
# The idx specifies which node we want the attributes
# Essentially, invert the "get_robot_data"
def get_node_attributes(features, adj, idx):
    quaternion_identity = rd.Quaterniond(1, 0, 0, 0)
    default_attributes = [-1, rd.JointType.NONE, 1.0, quaternion_identity, np.array([0, 0, 1], dtype="float64"),
                          rd.LinkShape.NONE, 1.0, 0.05, 1.0, 0.9,
                          0.01, 0.5, 1.0, rd.JointControlMode.POSITION,
                          np.array([0.45, 0.5, 0.55], dtype="float32"), np.array([1.0, 0.5, 0.3], dtype="float32"), "", ""]
    field_names = ["parent", "joint_type", "joint_pos", "joint_rot", "joint_axis",
                   "shape", "length", "radius", "density", "friction",
                   "joint_kp", "joint_kd", "joint_torque", "joint_control_mode",
                   "color", "joint_color", "label", "joint_label"]

    feature = features[idx]
    total_edge = 14
    node_types = feature // total_edge
    edge_types = feature % total_edge

    # angle_list[5] is default
    angle_list = [[0, 1, 0, 90], [0, 1, 0, -90], [0, 0, 1, 120],
                  [0, 0, 1, 60], [0, 0, 1, -60], [0, 0, 0, 0]]
    # axis_list[2] is default and should not be used
    axis_list = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype="float64")

    # If not the root node, modify its edge parameters
    if idx != 0:
        parent_idx = adj[idx].tolist().index(1)
        default_attributes[field_names.index("parent")] = parent_idx
        if edge_types == 0:
            default_attributes[field_names.index("joint_type")] = rd.JointType.FIXED
            default_attributes[field_names.index("joint_pos")] = 0.5
            default_attributes[field_names.index("joint_rot")] = rot_2_quat(angle_list[0])
        elif edge_types == 1:
            default_attributes[field_names.index("joint_type")] = rd.JointType.FIXED
            default_attributes[field_names.index("joint_pos")] = 0.5
            default_attributes[field_names.index("joint_axis")] = axis_list[3]
            default_attributes[field_names.index("joint_rot")] = rot_2_quat(angle_list[1])
        elif edge_types == 10:
            default_attributes[field_names.index("joint_type")] = rd.JointType.FIXED
        elif edge_types == 2:
            default_attributes[field_names.index("joint_type")] = rd.JointType.HINGE
        elif edge_types == 3:
            default_attributes[field_names.index("joint_type")] = rd.JointType.HINGE
            default_attributes[field_names.index("joint_axis")] = axis_list[1]
            default_attributes[field_names.index("joint_color")] = np.array([0, 0.5, 0], dtype="float32")
        elif edge_types == 4:
            default_attributes[field_names.index("joint_type")] = rd.JointType.HINGE
            default_attributes[field_names.index("joint_axis")] = axis_list[0]
        elif edge_types == 5:
            default_attributes[field_names.index("joint_type")] = rd.JointType.HINGE
            default_attributes[field_names.index("joint_axis")] = axis_list[3]
            default_attributes[field_names.index("joint_rot")] = rot_2_quat(angle_list[4])
        elif edge_types == 6:
            default_attributes[field_names.index("joint_type")] = rd.JointType.HINGE
            default_attributes[field_names.index("joint_axis")] = axis_list[3]
            default_attributes[field_names.index("joint_rot")] = rot_2_quat(angle_list[3])
        elif edge_types == 7:
            default_attributes[field_names.index("joint_type")] = rd.JointType.HINGE
            default_attributes[field_names.index("joint_axis")] = axis_list[3]
            default_attributes[field_names.index("joint_rot")] = rot_2_quat(angle_list[2])
        elif edge_types == 8:
            default_attributes[field_names.index("joint_type")] = rd.JointType.HINGE
            default_attributes[field_names.index("joint_axis")] = axis_list[0]
            default_attributes[field_names.index("joint_rot")] = rot_2_quat(angle_list[0])
        elif edge_types == 9:
            default_attributes[field_names.index("joint_type")] = rd.JointType.HINGE
            default_attributes[field_names.index("joint_axis")] = axis_list[0]
            default_attributes[field_names.index("joint_rot")] = rot_2_quat(angle_list[1])
        elif edge_types == 11:
            default_attributes[field_names.index("joint_type")] = rd.JointType.HINGE
            default_attributes[field_names.index("joint_axis")] = axis_list[2]
            default_attributes[field_names.index("joint_rot")] = rot_2_quat(angle_list[4])
        elif edge_types == 12:
            default_attributes[field_names.index("joint_type")] = rd.JointType.HINGE
            default_attributes[field_names.index("joint_axis")] = axis_list[2]
            default_attributes[field_names.index("joint_rot")] = rot_2_quat(angle_list[3])
        elif edge_types == 13:
            default_attributes[field_names.index("joint_type")] = rd.JointType.HINGE
            default_attributes[field_names.index("joint_axis")] = axis_list[2]
            default_attributes[field_names.index("joint_rot")] = rot_2_quat(angle_list[2])
        else:
            exit(f"unrecognized edge type at graph_io.py, edge type = {edge_types}")
    else:
        # Meaning it is the root node
        default_attributes[field_names.index("joint_type")] = rd.JointType.FREE
        default_attributes[field_names.index("joint_kp")] = 0
        default_attributes[field_names.index("joint_kd")] = 0
        default_attributes[field_names.index("joint_color")] = np.zeros(3)

    if node_types == 0:
        default_attributes[field_names.index("shape")] = rd.LinkShape.CAPSULE
        default_attributes[field_names.index("length")] = 0.15
        default_attributes[field_names.index("radius")] = 0.025
    elif node_types == 1:
        default_attributes[field_names.index("shape")] = rd.LinkShape.CAPSULE
        default_attributes[field_names.index("length")] = 0.15
        default_attributes[field_names.index("radius")] = 0.045
        default_attributes[field_names.index("density")] = 3.0
    elif node_types == 2:
        default_attributes[field_names.index("shape")] = rd.LinkShape.CAPSULE
        default_attributes[field_names.index("length")] = 0.1
        default_attributes[field_names.index("radius")] = 0.025
    else:
        exit(f"unrecognized node type at graph_io.py, node type = {node_types}")
    default_attributes[field_names.index("joint_axis")] \
        = default_attributes[field_names.index("joint_axis")].reshape([-1, 1])
    default_attributes[field_names.index("color")] \
        = default_attributes[field_names.index("color")].reshape([-1, 1])
    default_attributes[field_names.index("joint_color")] \
        = default_attributes[field_names.index("joint_color")].reshape([-1, 1])
    return default_attributes


def normalized_robot(robot):
    """normalize the mass of the body links."""
    body_links = []
    total_body_length = 0.0
    for link in robot.links:
        if np.isclose(link.radius, 0.045):
            # Link is a body link
            body_links.append(link)
            total_body_length += link.length
            target_mass = link.length * link.density

    if body_links:
        body_density = target_mass / total_body_length
        for link in body_links:
            link.density = body_density

    return robot


def presimulate(robot):
    """Find an initial position that will place the robot on the ground behind the
    x=0 plane, and check if the robot collides in its initial configuration."""
    temp_sim = rd.BulletSimulation()
    temp_sim.add_robot(robot, np.zeros(3), rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    temp_sim.step()
    robot_idx = temp_sim.find_robot_index(robot)
    lower = np.zeros(3)
    upper = np.zeros(3)
    temp_sim.get_robot_world_aabb(robot_idx, lower, upper)
    return [-upper[0], -lower[1], 0.0], temp_sim.robot_has_collision(robot_idx)


def simulate(robot, task, opt_seed, thread_count, episode_count=1):
    """Run trajectory optimization for the robot on the given task, and return the
    resulting input sequence and result."""
    robot_init_pos, has_self_collision = presimulate(robot)

    if has_self_collision:
        return None, 3.8  # set it to be the worst performing designs

    def make_sim_fn():
        sim = rd.BulletSimulation(task.time_step)
        task.add_terrain(sim)
        # Rotate 180 degrees around the y axis, so the base points to the right
        sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
        return sim

    main_sim = make_sim_fn()
    robot_idx = main_sim.find_robot_index(robot)

    dof_count = main_sim.get_robot_dof_count(robot_idx)
    if episode_count >= 2:
        value_estimator = rd.FCValueEstimator(main_sim, robot_idx, 'cpu', 64, 3, 1)
    else:
        value_estimator = rd.NullValueEstimator()
    input_sampler = rd.DefaultInputSampler()
    objective_fn = task.get_objective_fn()

    replay_obs = np.zeros((value_estimator.get_observation_size(), 0))
    replay_returns = np.zeros(0)

    for episode_idx in range(episode_count):
        optimizer = rd.MPPIOptimizer(1.0, task.discount_factor, dof_count,
                                     task.interval, task.horizon, 512,
                                     thread_count, opt_seed + episode_idx,
                                     make_sim_fn, objective_fn, value_estimator,
                                     input_sampler)

        optimizer.update()
        optimizer.set_sample_count(64)

        main_sim.save_state()

        input_sequence = np.zeros((dof_count, task.episode_len))
        obs = np.zeros((value_estimator.get_observation_size(),
                        task.episode_len + 1), order='f')
        rewards = np.zeros(task.episode_len * task.interval)
        for j in range(task.episode_len):
            optimizer.update()
            input_sequence[:, j] = optimizer.input_sequence[:, 0]
            optimizer.advance(1)

            value_estimator.get_observation(main_sim, obs[:, j])
            for k in range(task.interval):
                main_sim.set_joint_targets(robot_idx,
                                           input_sequence[:, j].reshape(-1, 1))
                task.add_noise(main_sim, j * task.interval + k)
                main_sim.step()
                rewards[j * task.interval + k] = objective_fn(main_sim)
        value_estimator.get_observation(main_sim, obs[:, -1])

        main_sim.restore_state()

        # Only train the value estimator if there will be another episode
        if episode_idx < episode_count - 1:
            returns = np.zeros(task.episode_len + 1)
            # Bootstrap returns with value estimator
            value_estimator.estimate_value(obs[:, task.episode_len], returns[-1:])
            for j in reversed(range(task.episode_len)):
                interval_reward = np.sum(
                    rewards[j * task.interval:(j + 1) * task.interval])
                returns[j] = interval_reward + task.discount_factor * returns[j + 1]
            replay_obs = np.hstack((replay_obs, obs[:, :task.episode_len]))
            replay_returns = np.concatenate((replay_returns,
                                             returns[:task.episode_len]))
            value_estimator.train(replay_obs, replay_returns)

    # discard impossible value
    eventual_reward = np.mean(rewards)
    if eventual_reward >= 10:
        eventual_reward = 4.0
    return input_sequence, eventual_reward

def finalize_robot(robot):
    for link in robot.links:
        link.label = ""
        link.joint_label = ""
        if link.shape == rd.LinkShape.NONE:
            link.shape = rd.LinkShape.CAPSULE
            link.length = 0.1
            link.radius = 0.025
            link.color = [1.0, 0.0, 1.0]
        if link.joint_type == rd.JointType.NONE:
            link.joint_type = rd.JointType.FIXED
            link.joint_color = [1.0, 0.0, 1.0]


def run_trajectory(sim, robot_idx, input_sequence, task, step_callback):
    step_callback(0)

    for j in range(input_sequence.shape[1]):
        for k in range(task.interval):
            step_idx = j * task.interval + k
            sim.set_joint_targets(robot_idx, input_sequence[:, j].reshape(-1, 1))
            task.add_noise(sim, step_idx)
            sim.step()
            step_callback(step_idx + 1)

def view_trajectory(sim, robot_idx, input_sequence, task):
    record_step_indices = set()

    sim.save_state()

    viewer = rd.GLFWViewer()

    # Get robot bounds
    lower = np.zeros(3)
    upper = np.zeros(3)
    sim.get_robot_world_aabb(robot_idx, lower, upper)

    # Set initial camera parameters
    task_name = type(task).__name__
    if 'Ridged' in task_name or 'Gap' in task_name:
        viewer.camera_params.yaw = 0.0
    elif 'Wall' in task_name:
        viewer.camera_params.yaw = -np.pi / 2
    else:
        viewer.camera_params.yaw = -np.pi / 4
    viewer.camera_params.pitch = -np.pi / 6
    viewer.camera_params.distance = 1.5 * np.linalg.norm(upper - lower)

    tracker = CameraTracker(viewer, sim, robot_idx)

    j = 0
    k = 0
    sim_time = time.time()
    while not viewer.should_close():
        current_time = time.time()
        while sim_time < current_time:
            step_idx = j * task.interval + k
            if input_sequence is not None:
                sim.set_joint_targets(robot_idx, input_sequence[:, j].reshape(-1, 1))
            task.add_noise(sim, step_idx)
            sim.step()
            tracker.update(task.time_step)
            viewer.update(task.time_step)
            if viewer.camera_controller.should_record():
                record_step_indices.add(step_idx)
            sim_time += task.time_step
            k += 1
            if k >= task.interval:
                j += 1
                k = 0
            if input_sequence is not None and j >= input_sequence.shape[1]:
                j = 0
                k = 0
                sim.restore_state()
                tracker.reset()
        viewer.render(sim)

    sim.restore_state()

    return viewer.camera_params, record_step_indices


class CameraTracker(object):
    def __init__(self, viewer, sim, robot_idx):
        self.viewer = viewer
        self.sim = sim
        self.robot_idx = robot_idx

        self.reset()

    def update(self, time_step):
        lower = np.zeros(3)
        upper = np.zeros(3)
        self.sim.get_robot_world_aabb(self.robot_idx, lower, upper)

        # Update camera position to track the robot smoothly
        target_pos = 0.5 * (lower + upper)
        camera_pos = self.viewer.camera_params.position.copy()
        camera_pos += 5.0 * time_step * (target_pos - camera_pos)
        self.viewer.camera_params.position = camera_pos

    def reset(self):
        lower = np.zeros(3)
        upper = np.zeros(3)
        self.sim.get_robot_world_aabb(self.robot_idx, lower, upper)

        self.viewer.camera_params.position = 0.5 * (lower + upper)

if __name__ == '__main__':
    adj = [[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]

    attr = [14.0, 28.0, 12.0, 29.0, 6.0, 17.0, 17.0, 17.0, 28.0, 40.0, 29.0, 34.0]
    robot = graph_to_robot(adj, attr)

    parser = argparse.ArgumentParser(description="Robot design viewer.")
    parser.add_argument("task", type=str, help="Task (Python class name)")
    parser.add_argument("-o", "--optim", default=False, action="store_true",
                        help="Optimize a trajectory")
    parser.add_argument("-s", "--opt_seed", type=int, default=None,
                        help="Trajectory optimization seed")
    parser.add_argument("-e", "--episodes", type=int, default=1,
                        help="Number of optimization episodes")
    parser.add_argument("-j", "--jobs", type=int, required=True,
                        help="Number of jobs/threads")
    parser.add_argument("--input_sequence_file", type=str,
                        help="File to save input sequence to (.csv)")
    parser.add_argument("--save_video_file", type=str,
                        help="File to save video to (.mp4)")
    parser.add_argument("-l", "--episode_len", type=int, default=128,
                        help="Length of episode")
    args = parser.parse_args()
    task_class = getattr(tasks, args.task)
    task = task_class(episode_len=args.episode_len)
    if args.opt_seed is not None:
        opt_seed = args.opt_seed
    else:
        opt_seed = random.getrandbits(32)
        print("Using optimization seed:", opt_seed)

    #########################
    ### Visualization part###
    #########################
    if args.optim:
        input_sequence, result = simulate(robot, task, opt_seed, args.jobs,
                                          args.episodes)
        print("Result:", result)
    else:
        input_sequence = None

    if args.input_sequence_file and input_sequence is not None:
        import csv

        with open(args.input_sequence_file, 'w', newline='') as input_seq_file:
            writer = csv.writer(input_seq_file)
            for col in input_sequence.T:
                writer.writerow(col)
        print("Saved input sequence to file:", args.input_sequence_file)

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


    if args.save_video_file and input_sequence is not None:
        import cv2

        if record_step_indices:
            print("Saving video for {} steps".format(len(record_step_indices)))

        viewer = rd.GLFWViewer()

        # Copy camera parameters from the interactive viewer
        viewer.camera_params = camera_params

        tracker = CameraTracker(viewer, main_sim, robot_idx)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video_file, fourcc, 60.0,
                                 viewer.get_framebuffer_size())
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)


        def write_frame_callback(step_idx):
            tracker.update(task.time_step)

            # 240 steps/second / 4 = 60 fps
            if step_idx % 4 == 0:
                # Flip vertically, convert RGBA to BGR
                frame = viewer.render_array(main_sim)[::-1, :, 2::-1]
                writer.write(frame)


        run_trajectory(main_sim, robot_idx, input_sequence, task,
                       write_frame_callback)

        writer.release()

