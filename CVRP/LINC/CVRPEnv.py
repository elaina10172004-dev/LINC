from dataclasses import dataclass
import math
import pickle

import torch

from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    customer_distance_matrix: torch.Tensor = None
    # shape: (batch, problem, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    ROLLOUT_IDX: torch.Tensor = None
    # shape: (batch, rollout)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, rollout)
    current_node: torch.Tensor = None
    # shape: (batch, rollout)
    ninf_mask: torch.Tensor = None
    # shape: (batch, rollout, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, rollout)
    candidate_features: torch.Tensor = None
    # shape: (batch, rollout, problem+1, feature_dim)


class CVRPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.rollout_size = None

        self.enable_candidate_features = bool(env_params.get('enable_candidate_features', True))
        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        self.batch_size = None
        self.BATCH_IDX = None
        self.ROLLOUT_IDX = None
        self.depot_node_xy = None
        self.depot_node_demand = None
        self.distance_matrix = None
        self.distance_matrix_norm = None
        self.dist_scale = None
        self.node_angles = None
        self.dist_to_depot_norm = None
        self.demand_rollout = None
        self.dist_to_depot_rollout = None
        self._cand_buffer = None

        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None

        self.at_the_depot = None
        self.load = None
        self.visited_ninf_flag = None
        self.ninf_mask = None
        self.finished = None

        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_index = 0

    def use_pkl_saved_problems(self, filename, num_problems, index_begin=0):
        self.FLAG__use_saved_problems = True

        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)

        depot_data = list(data[i][0] for i in range(index_begin, index_begin + num_problems))
        self.saved_depot_xy = torch.tensor(depot_data)[:, None, :]

        node_data = list(data[i][1] for i in range(index_begin, index_begin + num_problems))
        self.saved_node_xy = torch.tensor(node_data)

        demand_data = list(data[i][2] for i in range(index_begin, index_begin + num_problems))
        capacity_data = list(data[i][3] for i in range(index_begin, index_begin + num_problems))
        capacity_tensor = torch.tensor(capacity_data, dtype=torch.float)
        self.saved_node_demand = torch.tensor(demand_data, dtype=torch.float) / capacity_tensor[:, None]

        self.saved_index = 0

    def use_random_problems(self):
        self.enable_candidate_features = bool(self.env_params.get('enable_candidate_features', True))
        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None

    def _prepare_candidate_static_cache(self):
        self.distance_matrix = torch.cdist(self.depot_node_xy, self.depot_node_xy)
        self.dist_scale = self.distance_matrix.amax(dim=2).amax(dim=1).clamp_min(1e-6)
        self.distance_matrix_norm = self.distance_matrix / self.dist_scale[:, None, None]

        customer_xy = self.depot_node_xy[:, 1:, :]
        depot_xy = self.depot_node_xy[:, :1, :]
        rel = customer_xy - depot_xy
        customer_angles = torch.atan2(rel[..., 1], rel[..., 0])
        depot_angles = torch.zeros((self.batch_size, 1), dtype=customer_angles.dtype, device=customer_angles.device)
        self.node_angles = torch.cat((depot_angles, customer_angles), dim=1)
        self.dist_to_depot_norm = torch.cat(
            (
                torch.zeros((self.batch_size, 1), dtype=self.depot_node_xy.dtype, device=self.depot_node_xy.device),
                torch.norm(rel, dim=2) / self.dist_scale[:, None],
            ),
            dim=1,
        )
        # Pre-compute angle difference matrix to avoid per-step atan2/sin/cos
        angle_diff = self.node_angles[:, None, :] - self.node_angles[:, :, None]
        self.angle_diff_matrix = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)).abs() / math.pi

    def load_problems(self, batch_size, rollout_size, aug_factor=1):
        self.batch_size = batch_size
        self.rollout_size = rollout_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index + batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index + batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index + batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(size=(self.batch_size, 1), dtype=node_demand.dtype, device=node_demand.device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        if self.enable_candidate_features:
            self._prepare_candidate_static_cache()
        else:
            self.distance_matrix = torch.cdist(self.depot_node_xy, self.depot_node_xy)

        self.BATCH_IDX = torch.arange(self.batch_size, device=self.depot_node_xy.device)[:, None].expand(self.batch_size, self.rollout_size)
        self.ROLLOUT_IDX = torch.arange(self.rollout_size, device=self.depot_node_xy.device)[None, :].expand(self.batch_size, self.rollout_size)
        self.demand_rollout = self.depot_node_demand[:, None, :].expand(self.batch_size, self.rollout_size, -1)
        if self.enable_candidate_features:
            self.dist_to_depot_rollout = self.dist_to_depot_norm[:, None, :].expand(self.batch_size, self.rollout_size, -1)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.customer_distance_matrix = self.distance_matrix[:, 1:, 1:]

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.ROLLOUT_IDX = self.ROLLOUT_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.rollout_size, self.problem_size * 2 + 1),
            dtype=torch.long,
            device=self.depot_node_xy.device,
        )

        self.at_the_depot = torch.ones(size=(self.batch_size, self.rollout_size), dtype=torch.bool, device=self.depot_node_xy.device)
        self.load = torch.ones(size=(self.batch_size, self.rollout_size), dtype=self.depot_node_xy.dtype, device=self.depot_node_xy.device)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.rollout_size, self.problem_size + 1), dtype=self.depot_node_xy.dtype, device=self.depot_node_xy.device)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.rollout_size, self.problem_size + 1), dtype=self.depot_node_xy.dtype, device=self.depot_node_xy.device)
        self.finished = torch.zeros(size=(self.batch_size, self.rollout_size), dtype=torch.bool, device=self.depot_node_xy.device)

        reward = None
        done = False
        return self.reset_state, reward, done

    def _ensure_selected_capacity(self, required_count):
        current_capacity = self.selected_node_list.size(2)
        if current_capacity >= required_count:
            return
        new_capacity = max(required_count, current_capacity * 2, self.problem_size * 2 + 1)
        expanded = torch.zeros(
            (self.batch_size, self.rollout_size, new_capacity),
            dtype=self.selected_node_list.dtype,
            device=self.selected_node_list.device,
        )
        expanded[:, :, :current_capacity] = self.selected_node_list
        self.selected_node_list = expanded

    def _compute_candidate_features(self):
        if not self.enable_candidate_features or self.current_node is None:
            return None

        actual_rollout = self.current_node.size(1)
        batch_idx = self.BATCH_IDX[:, :actual_rollout]

        travel_dist_norm = self.distance_matrix_norm[batch_idx, self.current_node, :]
        demand = self.demand_rollout[:, :actual_rollout, :]
        load_after_ratio = self.load[:, :, None] - demand
        load_after_ratio[:, :, 0] = 1.0
        dist_to_depot_norm = self.dist_to_depot_rollout[:, :actual_rollout, :]
        angle_diff = self.angle_diff_matrix[batch_idx, self.current_node, :]

        feature_dtype = torch.float16 if self.depot_node_xy.is_cuda and torch.is_autocast_enabled("cuda") else torch.float32
        cand = torch.zeros(
            (self.batch_size, actual_rollout, self.problem_size + 1, 5),
            dtype=feature_dtype,
            device=self.depot_node_xy.device,
        )
        cand[..., 0] = travel_dist_norm.to(dtype=feature_dtype)
        cand[..., 1] = demand.to(dtype=feature_dtype)
        cand[..., 2] = load_after_ratio.to(dtype=feature_dtype)
        cand[..., 3] = dist_to_depot_norm.to(dtype=feature_dtype)
        cand[..., 4] = angle_diff.to(dtype=feature_dtype)
        return cand

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.candidate_features = self._compute_candidate_features()

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        device = self.depot_node_xy.device
        if selected.device != device:
            selected = selected.to(device)
        self._ensure_selected_capacity(self.selected_count + 1)
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list[:, :, self.selected_count - 1] = self.current_node

        self.at_the_depot = selected == 0

        demand_list = self.demand_rollout[:, :self.load.size(1), :]
        gathering_index = selected[:, :, None]
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1

        self.visited_ninf_flag[self.BATCH_IDX, self.ROLLOUT_IDX, selected] = float('-inf')
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0

        self.ninf_mask.copy_(self.visited_ninf_flag)
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        self.ninf_mask[demand_too_large] = float('-inf')

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished = self.finished + newly_finished
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        done = self.finished.all()
        if done:
            self.step_state.candidate_features = None
            reward = -self._get_travel_distance()
        else:
            self.step_state.candidate_features = self._compute_candidate_features()
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        node_seq = self.selected_node_list[:, :, :self.selected_count]
        from_nodes = node_seq[:, :, :-1]
        to_nodes = node_seq[:, :, 1:]
        batch_idx = self.BATCH_IDX[:, :, None].expand(-1, -1, from_nodes.size(2))
        segment_lengths = self.distance_matrix[batch_idx, from_nodes, to_nodes]
        closing_lengths = self.distance_matrix[self.BATCH_IDX, node_seq[:, :, -1], node_seq[:, :, 0]]
        return segment_lengths.sum(2) + closing_lengths
