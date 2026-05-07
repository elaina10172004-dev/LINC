from dataclasses import dataclass
import math
import pickle

import torch

from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold


def _accurate_cdist(x, y):
    if not x.is_cuda:
        return torch.cdist(x, y)

    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return torch.cdist(x, y)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


@dataclass
class Reset_State:
    problems: torch.Tensor


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    ROLLOUT_IDX: torch.Tensor
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    candidate_features: torch.Tensor = None


class TSPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.rollout_size = None
        self.FLAG__use_saved_problems = False
        self.saved_problems = None
        self.saved_index = None

        self.batch_size = None
        self.BATCH_IDX = None
        self.ROLLOUT_IDX = None
        self.problems = None
        self.distance_matrix = None
        self.dist_scale = None
        self.centroid = None
        self.dist_to_centroid_norm = None
        self.dist_to_centroid_rollout = None
        self.centroid_angles = None
        self.start_dist_norm = None

        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None

    def use_saved_problems(self, filename):
        self.FLAG__use_saved_problems = True
        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        self.saved_problems = torch.tensor(data)
        self.saved_index = 0

    def use_pkl_saved_problems(self, filename, num_problems, index_begin=0):
        self.FLAG__use_saved_problems = True
        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        partial_data = list(data[i] for i in range(index_begin, index_begin + num_problems))
        self.saved_problems = torch.tensor(partial_data)
        self.saved_index = 0

    def use_random_problems(self):
        self.FLAG__use_saved_problems = False
        self.saved_problems = None

    def _prepare_candidate_static_cache(self):
        self.distance_matrix = _accurate_cdist(self.problems, self.problems)
        self.dist_scale = self.distance_matrix.amax(dim=2).amax(dim=1).clamp_min(1e-6)
        self.centroid = self.problems.mean(dim=1, keepdim=True)
        rel = self.problems - self.centroid
        self.dist_to_centroid_norm = torch.norm(rel, dim=2) / self.dist_scale[:, None]
        self.centroid_angles = torch.atan2(rel[..., 1], rel[..., 0])

    def load_problems(self, batch_size, rollout_size, aug_factor=1):
        self.batch_size = batch_size
        self.rollout_size = rollout_size

        if not self.FLAG__use_saved_problems:
            self.problems = get_random_problems(batch_size, self.problem_size)
        else:
            self.problems = self.saved_problems[self.saved_index:self.saved_index + batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
            else:
                raise NotImplementedError

        self._prepare_candidate_static_cache()
        self.BATCH_IDX = torch.arange(self.batch_size, device=self.problems.device)[:, None].expand(self.batch_size, self.rollout_size)
        self.ROLLOUT_IDX = torch.arange(self.rollout_size, device=self.problems.device)[None, :].expand(self.batch_size, self.rollout_size)
        self.dist_to_centroid_rollout = self.dist_to_centroid_norm[:, None, :].expand(self.batch_size, self.rollout_size, -1)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.start_dist_norm = None
        self.selected_node_list = torch.empty(
            (self.batch_size, self.rollout_size, self.problem_size),
            dtype=torch.long,
            device=self.problems.device,
        )
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, ROLLOUT_IDX=self.ROLLOUT_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.rollout_size, self.problem_size), dtype=self.problems.dtype, device=self.problems.device)
        self.step_state.candidate_features = None
        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def _compute_candidate_features(self):
        if self.current_node is None:
            return None

        current_idx = self.current_node

        batch_idx = torch.arange(self.batch_size, device=self.problems.device)[:, None].expand(self.batch_size, self.current_node.size(1))
        travel_dist_norm = self.distance_matrix[batch_idx, current_idx, :] / self.dist_scale[:, None, None]
        if self.start_dist_norm is None or self.start_dist_norm.size(1) != self.current_node.size(1):
            first_idx = self.selected_node_list[:, :, 0]
            self.start_dist_norm = self.distance_matrix[batch_idx, first_idx, :] / self.dist_scale[:, None, None]

        current_angles = self.centroid_angles[batch_idx, current_idx]
        angle_diff = self.centroid_angles[:, None, :] - current_angles[:, :, None]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)).abs() / math.pi

        return torch.stack(
            (
                travel_dist_norm,
                self.start_dist_norm,
                self.dist_to_centroid_norm[:, None, :].expand(self.batch_size, self.current_node.size(1), -1),
                angle_diff,
            ),
            dim=-1,
        )

    def pre_step(self):
        reward = None
        done = False
        self.step_state.candidate_features = self._compute_candidate_features()
        return self.step_state, reward, done

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list[:, :, self.selected_count - 1] = self.current_node

        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask[self.BATCH_IDX, self.ROLLOUT_IDX, self.current_node] = float('-inf')

        done = (self.selected_count == self.problem_size)
        if done:
            self.step_state.candidate_features = None
            reward = -self._get_travel_distance()
        else:
            self.step_state.candidate_features = self._compute_candidate_features()
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        node_seq = self.selected_node_list
        from_nodes = node_seq[:, :, :-1]
        to_nodes = node_seq[:, :, 1:]
        batch_idx = self.BATCH_IDX[:, :, None].expand(-1, -1, from_nodes.size(2))
        segment_lengths = self.distance_matrix[batch_idx, from_nodes, to_nodes]
        closing_lengths = self.distance_matrix[self.BATCH_IDX, node_seq[:, :, -1], node_seq[:, :, 0]]
        return segment_lengths.sum(2) + closing_lengths
