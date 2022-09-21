'''
This file combines get_robot_data, prune_dataset and further_prune.
Ideally collect a dataset with suitable robots and no self-collision

Furthermore, it collects the contact point of the robot with the world

Current run: python3 collect_data.py -i500000
'''

import numpy as np
import quaternion
from design_search import make_initial_graph, build_normalized_robot, get_applicable_matches, has_nonterminals, presimulate
import argparse
import env
import pyrobotdesign as rd
import random
from scipy.spatial.transform import Rotation as R
import os

def np_quaternion(q):
    """Create a np.quaternion from a rd.Quaternion."""
    return np.quaternion(q.w, q.x, q.y, q.z)


def one_hot_encode(enum_member):
    """Encode an enum member as a one-hot vector."""
    vec = np.zeros(len(type(enum_member).__members__))
    vec[int(enum_member)] = 1
    return vec


def quaternion_coords(q):
    """Get the coefficients of a rd.Quaternion as an np.ndarray."""
    return np.array([q.w, q.x, q.y, q.z])


def check_angle_same(angle_axis, quaternion):
    angle = angle_axis[3]
    axis = angle_axis[:3]
    r = R.from_rotvec(angle / 180 * np.pi * np.array(axis))
    quat_comp = r.as_quat()  # x, y, z, w
    if np.isclose(quat_comp[0], quaternion[1]) and np.isclose(quat_comp[1], quaternion[2]) \
            and np.isclose(quat_comp[2], quaternion[3]) and np.isclose(quat_comp[3], quaternion[0]):
        return True
    return False


def get_link_edge_type(link):
    if link.parent != -1:
        assert (np.isclose(link.joint_kp, 0.01) and np.isclose(link.joint_kd, 0.5)
                and np.isclose(link.joint_torque, 1.0))

    linktype = None
    jointtype = None
    if link.shape == rd.LinkShape.CAPSULE:
        if np.isclose(link.radius, 0.025) and np.isclose(link.length, 0.15):
            linktype = 0
        elif np.isclose(link.radius, 0.045) and np.isclose(link.length, 0.15):
            linktype = 1
        elif np.isclose(link.radius, 0.025) and np.isclose(link.length, 0.1):
            linktype = 2

    if link.parent == -1:
        jointtype = 0
    else:
        angle_list = [[0, 1, 0, 90], [0, 1, 0, -90], [0, 0, 1, 120], [0, 0, 1, 60], [0, 0, 1, -60], [0, 0, 0, 0]]
        quat_angle = quaternion_coords(link.joint_rot)
        angle_idx = None
        for i in range(len(angle_list)):
            if check_angle_same(angle_list[i], quat_angle):
                angle_idx = i
                break
        if angle_idx == None:
            print("no matching angle!")
            import pdb
            pdb.Pdb().set_trace()

        axis_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]
        axis_idx = None
        for i in range(len(axis_list)):
            if (link.joint_axis == axis_list[i]).all():
                axis_idx = i
                break
        if axis_idx == None:
            print("no matching axis!")
            import pdb
            pdb.Pdb().set_trace()

        if link.joint_type == rd.JointType.FIXED:
            if link.joint_pos == 0.5:
                if axis_idx == 2 and angle_idx == 0:
                    jointtype = 0
                elif axis_idx == 3 and angle_idx == 1:  # mirrored
                    jointtype = 1
            elif link.joint_pos == 1.0 and axis_idx == 2:
                jointtype = 10
        elif link.joint_type == rd.JointType.HINGE:
            if link.joint_pos == 1.0:
                if axis_idx == 2 and angle_idx == 5:
                    jointtype = 2
                elif axis_idx == 1 and angle_idx == 5:
                    jointtype = 3
                elif axis_idx == 0 and angle_idx == 5:
                    jointtype = 4
                elif axis_idx == 3 and angle_idx == 4:
                    jointtype = 5
                elif axis_idx == 3 and angle_idx == 3:
                    jointtype = 6
                elif axis_idx == 3 and angle_idx == 2:
                    jointtype = 7
                elif axis_idx == 0 and angle_idx == 0:
                    jointtype = 8
                elif axis_idx == 0 and angle_idx == 1:
                    jointtype = 9
                elif axis_idx == 2 and angle_idx == 4:
                    jointtype = 11
                elif axis_idx == 2 and angle_idx == 3:
                    jointtype = 12
                elif axis_idx == 2 and angle_idx == 2:
                    jointtype = 13

    if linktype == None or jointtype == None:
        print("Link or joint mismatch!!!!")
        print(f"link type: {linktype}, joint type: {jointtype}")
        print(f"axis idx {axis_idx}, angle idx {angle_idx}")
        import pdb
        pdb.Pdb().set_trace()
    total_number_joint = 14
    final_idx = linktype * total_number_joint + jointtype
    return final_idx


def loc_preprocess(loc):
    '''
    preprocess the robot contact location data
    '''
    # First, filter out all non-ground collision
    new_loc = np.zeros([loc.shape[0], 8, 2])
    for i in range(loc.shape[0]):
        ind_loc = loc[i]
        # zero out the body collision entry
        body_collision = np.abs(ind_loc[:, 1]) > 1e-2
        ind_loc[body_collision] = 0

        # filter out all zero entries
        zero_entries = (ind_loc[:, 0] == 0) * (ind_loc[:, 1] == 0) * (ind_loc[:, 2] == 0)
        true_entries = ind_loc[~zero_entries]

        # default sequencing is sorted already
        new_loc[i, :true_entries.shape[0], :] = true_entries[:, (0,2)]
    return new_loc

def single_loc_preprocess(loc):
    # First, filter out all non-ground collision
    new_loc = np.zeros([8, 2])
    ind_loc = loc
    # zero out the body collision entry
    body_collision = np.abs(ind_loc[:, 1]) > 1e-2
    ind_loc[body_collision] = 0

    # filter out all zero entries
    zero_entries = (ind_loc[:, 0] == 0) * (ind_loc[:, 1] == 0) * (ind_loc[:, 2] == 0)
    true_entries = ind_loc[~zero_entries]

    num_contact = true_entries.shape[0]

    valid = True

    if num_contact < 4:
        valid = False

    center_entries = (np.abs(true_entries[:, 0]) > 1e-2) * (np.abs(true_entries[:, 2]) < 1e-2)
    if np.sum(center_entries) > 0:
        valid = False

    new_loc[:true_entries.shape[0], :] = true_entries[:, (0,2)]
    new_loc = new_loc * 10

    # cover the zeros with 2 instead, for discrimination purpose
    new_loc[true_entries.shape[0]:, 0] = 2.0

    return new_loc, valid



def presimulate_contact(robot, robot_init_pos):
    """Obtain the initial contact of the robot with the ground"""
    sim = rd.BulletSimulation(1.0 / 24)
    floor = rd.Prop(rd.PropShape.BOX, 0.0, 0.5, [40.0, 1.0, 10.0])

    sim.add_prop(floor, [0.0, -1.0, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
    sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))

    for i in range(10):
        sim.step()
    robot_idx = sim.find_robot_index(robot)
    loc = np.ones(24)
    sim.robot_world_contact(robot_idx, loc)

    reformated_loc = loc.reshape([-1,3])
    reformated_loc, valid = single_loc_preprocess(reformated_loc)

    return reformated_loc, valid

class Preprocessor:
    def __init__(self, all_labels=None):
        self.all_labels = all_labels

    def preprocess(self, robot_graph):
        robot = build_normalized_robot(robot_graph)

        # Generate adjacency matrix
        adj_matrix = np.zeros((len(robot.links), len(robot.links)))
        for i, link in enumerate(robot.links):
            if link.parent >= 0:
                adj_matrix[link.parent, i] += 1

        # Generate features for links
        link_features = []
        for i, link in enumerate(robot.links):
            link_features.append(get_link_edge_type(link))
        link_features = np.array(link_features)

        # make adj_matrix symmetric
        adj_matrix = adj_matrix + np.transpose(adj_matrix)

        return adj_matrix, link_features


class RandomSearch(object):
    def __init__(self, env):
        self.env = env

    def select_action(self, state):
        available_actions = list(self.env.get_available_actions(state))
        if available_actions:
            return random.choice(available_actions)
        else:
            return None

    def run_iteration(self):
        while True:
            states = [self.env.initial_state]
            actions = []
            action = self.select_action(states[-1])
            while action is not None:
                states.append(self.env.get_next_state(states[-1], action))
                actions.append(action)
                action = self.select_action(states[-1])
            result = self.env.get_result(states[-1])
            if result is not None:
                # Result is valid
                return states, result



class RobotDesignEnv(env.Env):
    """Robot design environment where states are (graph, rule sequence) pairs and
    actions are rule applications."""

    def __init__(self, rules, max_rule_seq_len):
        self.rules = rules
        self.max_rule_seq_len = max_rule_seq_len
        self.initial_graph = make_initial_graph()
        self.result_cache = dict()
        self.result_cache_hit_count = 0

    @property
    def initial_state(self):
        return (self.initial_graph, [])

    def get_available_actions(self, state):
        graph, rule_seq = state
        if len(rule_seq) >= self.max_rule_seq_len:
            # No more actions should be available
            return
        for rule in self.rules:
            if list(get_applicable_matches(rule, graph)):
                # Rule has at least one applicable match
                yield rule

    def get_next_state(self, state, rule):
        graph, rule_seq = state
        applicable_matches = list(get_applicable_matches(rule, graph))
        return (rd.apply_rule(rule, graph, applicable_matches[0]),
                rule_seq + [rule])

    def get_result(self, state):
        graph, rule_seq = state
        if has_nonterminals(graph):
            # Graph is incomplete
            return None

        # instead: filter out small robots
        robot = build_normalized_robot(graph)
        if len(robot.links) < args.prune_length:
            return None

        robot_init_pos, has_self_collision = presimulate(robot)
        if has_self_collision:
            return None
        # where to calculate and store the contact?
        loc, valid = presimulate_contact(robot, robot_init_pos)

        # We further filter out all designs that are not good based on heuristics
        if not valid:
            return None

        return loc

    def get_key(self, state):
        return hash(state[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Robot design graph conversion demo.")
    parser.add_argument("--grammar_file", type=str, default="../data/designs/grammar_apr30.dot", help="Grammar file (.dot)")
    parser.add_argument("-i", "--iterations", type=int, required=True,
                        help="Number of iterations")
    parser.add_argument("-d", "--depth", type=int, default=40,
                        help="Maximum tree depth")
    parser.add_argument("--prune_length", type=int, default=11,
                        help="Prune limit for number of modules")
    args = parser.parse_args()

    graphs = rd.load_graphs(args.grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]
    env = RobotDesignEnv(rules, args.depth)
    search_alg = RandomSearch(env)

    # state preprocessor
    # Find all possible link labels, so they can be one-hot encoded
    all_labels = set()
    for rule in rules:
        for node in rule.lhs.nodes:
            all_labels.add(node.attrs.require_label)
    all_labels = sorted(list(all_labels))
    global preprocessor
    preprocessor = Preprocessor(all_labels=all_labels)

    adj_data = []
    features_data = []
    contact_data = []
    save_iter = 10000
    log_iter = 1000
    for i in range(args.iterations):
        states, contact_loc = search_alg.run_iteration()

        graph, rule_seq = states[-1]
        adj_matrix_np, features_np = preprocessor.preprocess(graph)

        adj_data.append(adj_matrix_np)
        features_data.append(features_np)
        contact_data.append(contact_loc)

        if i % log_iter == 0:
            print(f"collecting data at iteration {i}")

    save_dir = "../data/new_train_data_loc_prune"
    if not os.path.exists(save_dir):
        # Create a new directory because it does not exist
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, f"adj.npy"), adj_data)
    np.save(os.path.join(save_dir, f"feat.npy"), features_data)
    np.save(os.path.join(save_dir, f"loc.npy"), contact_data)

