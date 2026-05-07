
from dataclasses import dataclass
import os
import torch

from envelope_generator import EnvelopeCVRPTWConfig, EnvelopeCVRPTWGenerator
from CVRPTWProblemDef import augment_xy_data_by_8_fold, get_random_problems, get_random_problems_from_data


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
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_tw: torch.Tensor = None
    # shape: (batch, problem, 2)
    service_t: torch.Tensor = None
    # shape: (batch, 1)



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
    visited_mask: torch.Tensor = None
    # shape: (batch, rollout, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, rollout)
    time = None
    # shape: (batch, rollout)
    candidate_features: torch.Tensor = None
    # shape: (batch, rollout, problem+1, cand_dim)


class CVRPTWCandidateFeatureView:
    """References needed to derive LINC candidate features without materializing them."""

    linc_cvrptw_feature_view = True

    def __init__(self, env, current_node, time):
        self.batch_size = env.batch_size
        self.rollout_size = env.rollout_size
        self.problem_size = env.problem_size
        self.current_node = current_node
        self.time = time
        self.BATCH_IDX = env.BATCH_IDX
        self.cand_step_customer_dist = env.cand_step_customer_dist
        self.cand_step_travel_dist_norm = env.cand_step_travel_dist_norm
        self.cand_step_angle_diff_norm = env.cand_step_angle_diff_norm
        self.cand_customer_ready = env.cand_customer_ready
        self.cand_customer_due = env.cand_customer_due
        self.cand_service_time = env.cand_service_time
        self.cand_max_customer_due = env.cand_max_customer_due
        self.cand_dist_scale = env.cand_dist_scale
        self.cand_depot_due = env.cand_depot_due
        self.cand_has_depot_due = env.cand_has_depot_due


class CVRPTWEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        if 'problem_size' in env_params:
            self.problem_size = env_params['problem_size']
        elif 'problem_size_list' in env_params:
            self.problem_size = env_params['problem_size_list'][0]
        else:
            raise KeyError("env_params must include 'problem_size' or 'problem_size_list'")
        if isinstance(self.problem_size, (list, tuple)):
            self.problem_size = int(self.problem_size[0])
        self.distribution = env_params.get('distribution', {'data_type': 'uniform'})
        self.grid_size = float(env_params.get('grid_size', 1000))
        self.grid_size_tensor = None
        self.input_scale_mode = str(env_params.get('input_scale_mode', 'grid'))
        self.model_input_scale_tensor = None
        self.model_xy_scale_tensor = None
        self.rollout_size = None
        self.enforce_depot_return = bool(env_params.get('enforce_depot_return', True))
        self.enable_candidate_features = bool(env_params.get('enable_candidate_features', True))
        selected_default = os.environ.get("LINC_SELECTED_CVRPTW_FEATURES", "0") != "0"
        self.use_selected_candidate_features = bool(
            env_params.get('use_selected_candidate_features', selected_default)
        )
        fused_default = os.environ.get("LINC_FUSED_CVRPTW_FEATURES", "1") != "0"
        self.use_fused_candidate_features = bool(env_params.get('use_fused_candidate_features', fused_default))

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_node_tw = None
        self.saved_service_t = None
        self.saved_depot_tw = None
        self.saved_travel_time_scale = None
        self.saved_grid_size = None
        self.saved_index = None
        self._envelope_generator = None
        self._envelope_generator_signature = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.ROLLOUT_IDX = None
        # IDX.shape: (batch, rollout)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_tw = None
        # shape: (batch, problem+1)
        self.distance_matrix = None
        # shape: (batch, problem+1, problem+1)
        self.travel_time_matrix = None
        self.capacity = None
        self.service_t = None
        self.travel_time_scale = None
        self.customer_dist_matrix = None
        self.node_angle_from_depot = None
        self.cand_angle_from_depot = None
        self.cand_step_customer_dist = None
        self.cand_step_travel_dist_norm = None
        self.cand_step_angle_diff_norm = None
        self.cand_max_customer_due = None
        self.cand_customer_ready = None
        self.cand_customer_due = None
        self.cand_service_time = None
        self.cand_depot_due = None
        self.cand_has_depot_due = None
        self.cand_dist_scale = None


        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, rollout)
        self.selected_node_list = None
        # shape: (batch, rollout, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, rollout)
        self.load = None
        # shape: (batch, rollout)
        self.visited_ninf_flag = None
        # shape: (batch, rollout, problem+1)
        self.ninf_mask = None
        # shape: (batch, rollout, problem+1)
        self.finished = None
        # shape: (batch, rollout)
        self.time = None
        # shape: (batch, rollout)
        self.used_vehicles = None
        # shape: (batch, rollout)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy'].float()
        self.saved_node_xy = loaded_dict['node_xy'].float()
        self.saved_node_demand = loaded_dict['node_demand'].float()
        self.saved_grid_size = self._normalize_grid_size(
            loaded_dict['grid_size'],
            self.saved_node_xy.shape[0],
            device=device,
        )
        self.grid_size = float(self.saved_grid_size.float().mean().item())
        self.capacity = loaded_dict['capacity']
        self.saved_service_t = None
        self.saved_depot_tw = None
        self.saved_travel_time_scale = None
        if 'node_tw' in loaded_dict.keys():
            self.saved_node_tw = loaded_dict['node_tw']
            service_value = loaded_dict.get('service_t', loaded_dict.get('service_duration'))
            if service_value is None:
                raise KeyError("Saved dataset includes node_tw but is missing service_t/service_duration")
            self.saved_service_t = self._normalize_service_tensor(service_value, self.saved_node_xy.shape[0], device=device)
            self.service_t = self.saved_service_t
        else:
            self.saved_node_tw = None
        if 'depot_tw' in loaded_dict:
            self.saved_depot_tw = loaded_dict['depot_tw'].float()
        elif 'depot_horizon' in loaded_dict:
            depot_horizon = loaded_dict['depot_horizon'].float()
            if depot_horizon.ndim == 1:
                depot_horizon = depot_horizon.unsqueeze(0)
            self.saved_depot_tw = depot_horizon.unsqueeze(1)
        travel_scale_value = loaded_dict.get('travel_time_scale', None)
        if travel_scale_value is not None:
            self.saved_travel_time_scale = self._normalize_travel_time_scale(
                travel_scale_value,
                self.saved_node_xy.shape[0],
                device=device,
            )
        self.saved_index = 0

    def _slice_saved_batch(self, tensor, batch_size):
        if tensor is None:
            return None
        total = int(tensor.shape[0])
        if total <= 0:
            raise ValueError("Saved problem tensor is empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        start = int(self.saved_index or 0) % total
        end = start + int(batch_size)
        if end <= total:
            return tensor[start:end]

        first = tensor[start:total]
        second = tensor[0:end - total]
        return torch.cat((first, second), dim=0)

    def load_problems(self, batch_size, rollout_size, device, aug_factor=1, problem_size=None, distribution=None):
        self.batch_size = batch_size
        self.rollout_size = rollout_size
        if problem_size is not None:
            self.problem_size = int(problem_size)
        if distribution is not None:
            if isinstance(distribution, str):
                self.distribution['data_type'] = distribution
            elif isinstance(distribution, dict):
                self.distribution.update(distribution)
            else:
                raise TypeError(f"Unsupported distribution type: {type(distribution)}")

        if self.FLAG__use_saved_problems:
            depot_xy = self._slice_saved_batch(self.saved_depot_xy, batch_size).to(device)
            node_xy = self._slice_saved_batch(self.saved_node_xy, batch_size).to(device)
            node_demand = self._slice_saved_batch(self.saved_node_demand, batch_size).to(device)
            capacity = self._slice_saved_batch(self.capacity, batch_size).to(device)
            self.grid_size_tensor = self._slice_saved_batch(self.saved_grid_size, batch_size).to(device)
            self.grid_size = float(self.grid_size_tensor.float().mean().item())
            self.travel_time_scale = self._normalize_travel_time_scale(
                (
                    self._slice_saved_batch(self.saved_travel_time_scale, batch_size)
                    if self.saved_travel_time_scale is not None
                    else 1.0
                ),
                batch_size,
                device=device,
            )
            depot_tw = None if self.saved_depot_tw is None else self._slice_saved_batch(self.saved_depot_tw, batch_size).to(device)
            if depot_tw is not None and depot_tw.ndim == 2:
                depot_tw = depot_tw[:, None, :]

            if self.saved_node_tw is not None:
                node_tw = self._slice_saved_batch(self.saved_node_tw, batch_size).to(device)
                if self.saved_service_t is None:
                    raise RuntimeError("saved_node_tw is set but saved_service_t is missing")
                self.service_t = self._slice_saved_batch(self.saved_service_t, batch_size).to(device)
            else:
                depot_xy, node_xy, node_demand, _, node_tw, self.service_t = get_random_problems_from_data(
                    depot_xy.to(device), node_xy.to(device), node_demand.to(device))
                self.service_t = self._normalize_service_tensor(self.service_t, batch_size, device=device)
                depot_tw = None

            self.saved_index = (int(self.saved_index or 0) + int(batch_size)) % int(self.saved_node_xy.shape[0])
        else:
            data_type = self._normalize_data_type(self.distribution.get('data_type', 'uniform'))
            if data_type in ('procedural', 'envelope'):
                generator = self._get_envelope_generator()
                batch = generator.sample_batch(
                    batch_size=batch_size,
                    problem_size=self.problem_size,
                    return_metadata=False,
                )
                self.grid_size_tensor = self._normalize_grid_size(
                    batch.get('grid_size', self.grid_size),
                    batch_size,
                    device=device,
                )
                self.grid_size = float(self.grid_size_tensor.float().mean().item())
                depot_xy = batch['depot_xy'].to(device)
                node_xy = batch['node_xy'].to(device)
                node_demand = batch['node_demand'].to(device)
                node_tw = batch['node_tw'].to(device)
                capacity = batch['capacity'].to(device)
                depot_tw = batch.get('depot_tw', None)
                if depot_tw is not None:
                    depot_tw = depot_tw.to(device)
                self.service_t = self._normalize_service_tensor(
                    batch.get('service_t', batch.get('service_duration')),
                    batch_size,
                    device=device,
                )
                self.travel_time_scale = self._normalize_travel_time_scale(
                    batch.get('travel_time_scale', 1.0),
                    batch_size,
                    device=device,
                )
            else:
                depot_xy, node_xy, node_demand, capacity = get_random_problems(
                    batch_size=batch_size,
                    problem_size=self.problem_size,
                    distribution=self.distribution,
                )
                self.grid_size = float(self.distribution.get('grid_size', self.grid_size))

                depot_xy = depot_xy.to(device)
                node_xy = node_xy.to(device)
                node_demand = node_demand.to(device)
                capacity = capacity.to(device)
                self.grid_size_tensor = self._normalize_grid_size(
                    self.distribution.get('grid_size', self.grid_size),
                    batch_size,
                    device=device,
                )
                self.grid_size = float(self.grid_size_tensor.float().mean().item())

                depot_xy, node_xy, node_demand, _, node_tw, self.service_t = get_random_problems_from_data(
                    depot_xy, node_xy, node_demand
                )
                self.service_t = self._normalize_service_tensor(self.service_t, batch_size, device=device)
                self.travel_time_scale = self._normalize_travel_time_scale(1.0, batch_size, device=device)
                depot_tw = None

        assert node_xy.shape[1] == self.problem_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy, self.grid_size_tensor)
                node_xy = augment_xy_data_by_8_fold(node_xy, self.grid_size_tensor)
                node_demand = node_demand.repeat(8, 1)
                capacity = capacity.repeat(8)
                node_tw = node_tw.repeat(8, 1, 1)
                self.grid_size_tensor = self.grid_size_tensor.repeat(8)
                self.travel_time_scale = self.travel_time_scale.repeat(8)
                self.service_t = self._repeat_service_time(self.service_t, 8)
                if depot_tw is not None:
                    depot_tw = depot_tw.repeat(8, 1, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1), device=device)
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        if depot_tw is None:
            depot_tw = torch.tensor([0.0, float('inf')], dtype=torch.float32, device=device)[None, None].expand(self.batch_size, 1, 2)
        self.depot_node_tw = torch.cat((depot_tw, node_tw), dim=1)
        # shape: (batch, problem+1, 2)
        self.distance_matrix = _accurate_cdist(self.depot_node_xy, self.depot_node_xy)
        self.travel_time_matrix = self.distance_matrix * self.travel_time_scale[:, None, None]
        # shape: (batch, problem+1, problem+1)
        self._prepare_candidate_static_cache()

        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(self.batch_size, self.rollout_size)
        self.ROLLOUT_IDX = torch.arange(self.rollout_size, device=device)[None, :].expand(self.batch_size, self.rollout_size)

        # Scale demand to [0, 1] based on vehilce capacity
        self.depot_node_demand /= capacity[:, None]

        # Create neural network input. Scale data
        grid_xy_scale = self.grid_size_tensor[:, None, None]
        self.model_input_scale_tensor, self.model_xy_scale_tensor = self._get_model_input_scales(depot_tw, node_tw)
        model_xy_scale = self.model_xy_scale_tensor[:, None, None]
        model_time_scale = self.model_input_scale_tensor[:, None]
        model_tw_scale = self.model_input_scale_tensor[:, None, None]
        self.reset_state.depot_xy = depot_xy / model_xy_scale
        self.reset_state.node_xy = node_xy / model_xy_scale
        self.reset_state.node_demand = self.depot_node_demand[:, 1:]
        self.reset_state.node_tw = node_tw / model_tw_scale
        self.reset_state.service_t = self.service_t / model_time_scale

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.ROLLOUT_IDX = self.ROLLOUT_IDX

    def _prepare_candidate_static_cache(self):
        time_matrix = self.travel_time_matrix if self.travel_time_matrix is not None else self.distance_matrix
        self.customer_dist_matrix = time_matrix[:, 1:, 1:]
        self.cand_dist_scale = time_matrix.amax(dim=2).amax(dim=1).clamp_min(1.0)
        depot_xy = self.depot_node_xy[:, 0:1, :]
        depot_vec = self.depot_node_xy - depot_xy
        self.node_angle_from_depot = torch.atan2(depot_vec[..., 1], depot_vec[..., 0])
        self.cand_angle_from_depot = self.node_angle_from_depot[:, 1:]
        self.cand_step_customer_dist = time_matrix[:, :, 1:]
        self.cand_step_travel_dist_norm = self.cand_step_customer_dist / self.cand_dist_scale[:, None, None]
        angle_diff = torch.remainder(
            self.cand_angle_from_depot[:, None, :] - self.node_angle_from_depot[:, :, None] + torch.pi,
            2 * torch.pi,
        ) - torch.pi
        self.cand_step_angle_diff_norm = angle_diff.abs() / torch.pi
        self.cand_max_customer_due = self.depot_node_tw[:, 1:, 1].float().max(dim=1, keepdim=True).values
        self.cand_customer_ready = self.depot_node_tw[:, None, 1:, 0].float()
        self.cand_customer_due = self.depot_node_tw[:, None, 1:, 1].float()
        self.cand_service_time = self._get_candidate_service_time()[:, None, :]
        self.cand_depot_due = self.depot_node_tw[:, 0, 1].float()[:, None, None]
        self.cand_has_depot_due = torch.isfinite(self.cand_depot_due) & (self.cand_depot_due < 1e8) & (self.cand_depot_due > 0.0)

    def _get_candidate_service_time(self):
        if torch.is_tensor(self.service_t):
            return self.service_t.to(device=self.depot_node_xy.device, dtype=torch.float32)
        return torch.full(
            (self.batch_size, 1),
            float(self.service_t),
            dtype=torch.float32,
            device=self.depot_node_xy.device,
        )

    def _get_model_input_scales(self, depot_tw, node_tw):
        mode = self.input_scale_mode.lower()
        if mode in {"grid", "grid_size"}:
            input_scale = self.grid_size_tensor.float().clamp_min(1.0)
            xy_scale = input_scale
        elif mode in {"horizon", "time_horizon"}:
            depot_due = depot_tw[:, 0, 1].float()
            finite_customer_due = torch.where(
                torch.isfinite(node_tw[:, :, 1]),
                node_tw[:, :, 1].float(),
                torch.zeros_like(node_tw[:, :, 1].float()),
            )
            customer_due = finite_customer_due.amax(dim=1).clamp_min(1.0)
            input_scale = torch.where(
                torch.isfinite(depot_due) & (depot_due > 1e-6),
                depot_due,
                customer_due,
            ).clamp_min(1.0)
            xy_scale = input_scale / self.travel_time_scale.float().clamp_min(1e-6)
        else:
            raise ValueError(f"Unsupported input_scale_mode: {self.input_scale_mode}")
        return input_scale, xy_scale

    def _get_model_time(self):
        scale = self.model_input_scale_tensor
        if scale is None:
            scale = self.grid_size_tensor
        return self.time / scale[:, None]

    def _compose_candidate_features(
        self,
        travel_dist_norm,
        depot_angle_diff_norm,
        arrival,
        wait,
        tw_slack,
        finish,
    ):
        eps = 1e-6
        horizon_fallback = torch.maximum(
            self.cand_max_customer_due[:, None, :],
            self.time[:, :, None] + 2.0 * self.cand_dist_scale[:, None, None],
        )
        horizon = torch.where(self.cand_has_depot_due, self.cand_depot_due, horizon_fallback).clamp_min(1.0)

        wait_norm = wait / (horizon + eps)
        tw_slack_ratio = tw_slack / (horizon + eps)
        arrival_time_norm = arrival / (horizon + eps)
        departure_time_norm = finish / (horizon + eps)

        feature_dtype = travel_dist_norm.dtype
        if travel_dist_norm.is_cuda and torch.is_autocast_enabled("cuda"):
            feature_dtype = torch.get_autocast_dtype("cuda")
        cand_phi = torch.zeros(
            (self.batch_size, self.rollout_size, self.problem_size + 1, 6),
            dtype=feature_dtype,
            device=travel_dist_norm.device,
        )
        customer_phi = cand_phi[:, :, 1:, :]
        customer_phi[..., 0] = travel_dist_norm.to(dtype=feature_dtype)
        customer_phi[..., 1] = wait_norm.to(dtype=feature_dtype)
        customer_phi[..., 2] = tw_slack_ratio.to(dtype=feature_dtype)
        customer_phi[..., 3] = arrival_time_norm.to(dtype=feature_dtype)
        customer_phi[..., 4] = departure_time_norm.to(dtype=feature_dtype)
        customer_phi[..., 5] = depot_angle_diff_norm.to(dtype=feature_dtype)
        cand_phi.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        return cand_phi

    def _compose_selected_candidate_features(
        self,
        travel_dist_norm,
        depot_angle_diff_norm,
        arrival,
        wait,
        tw_slack,
        finish,
    ):
        eps = 1e-6
        horizon_fallback = torch.maximum(
            self.cand_max_customer_due[:, None, :],
            self.time[:, :, None] + 2.0 * self.cand_dist_scale[:, None, None],
        )
        horizon = torch.where(self.cand_has_depot_due, self.cand_depot_due, horizon_fallback).clamp_min(1.0)

        feature_dtype = travel_dist_norm.dtype
        if travel_dist_norm.is_cuda and torch.is_autocast_enabled("cuda"):
            feature_dtype = torch.get_autocast_dtype("cuda")
        selected_phi = torch.zeros(
            (self.batch_size, self.rollout_size, self.problem_size + 1, 6),
            dtype=feature_dtype,
            device=travel_dist_norm.device,
        )
        customer_phi = selected_phi[:, :, 1:, :]
        inv_horizon = torch.reciprocal(horizon + eps)
        customer_phi[..., 0] = travel_dist_norm.to(dtype=feature_dtype)
        customer_phi[..., 1] = (wait * inv_horizon).to(dtype=feature_dtype)
        customer_phi[..., 2] = (tw_slack * inv_horizon).to(dtype=feature_dtype)
        customer_phi[..., 3] = (arrival * inv_horizon).to(dtype=feature_dtype)
        customer_phi[..., 4] = (finish * inv_horizon).to(dtype=feature_dtype)
        customer_phi[..., 5] = depot_angle_diff_norm.to(dtype=feature_dtype)
        selected_phi.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        return selected_phi

    def _get_candidate_features(self):
        if self.use_fused_candidate_features and not self.use_selected_candidate_features:
            return CVRPTWCandidateFeatureView(self, self.current_node, self.time)

        if self.current_node is None:
            current_idx = torch.zeros((self.batch_size, self.rollout_size), dtype=torch.long, device=self.depot_node_xy.device)
        else:
            current_idx = self.current_node

        dist_to_customers = self.cand_step_customer_dist[self.BATCH_IDX, current_idx]
        travel_dist_norm = self.cand_step_travel_dist_norm[self.BATCH_IDX, current_idx]
        depot_angle_diff_norm = self.cand_step_angle_diff_norm[self.BATCH_IDX, current_idx]
        arrival = self.time[:, :, None] + dist_to_customers

        wait = torch.relu(self.cand_customer_ready - arrival)
        start = torch.maximum(arrival, self.cand_customer_ready)
        tw_slack = self.cand_customer_due - start

        finish = start + self.cand_service_time
        if self.use_selected_candidate_features:
            return self._compose_selected_candidate_features(
                travel_dist_norm=travel_dist_norm,
                depot_angle_diff_norm=depot_angle_diff_norm,
                arrival=arrival,
                wait=wait,
                tw_slack=tw_slack,
                finish=finish,
            )
        return self._compose_candidate_features(
            travel_dist_norm=travel_dist_norm,
            depot_angle_diff_norm=depot_angle_diff_norm,
            arrival=arrival,
            wait=wait,
            tw_slack=tw_slack,
            finish=finish,
        )


    def reset(self):
        device = self.depot_node_xy.device
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, rollout)
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.rollout_size, self.problem_size * 2 + 1),
            dtype=torch.long,
            device=device,
        )
        # shape: (batch, rollout, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.rollout_size), dtype=torch.bool, device=device)
        # shape: (batch, rollout)
        self.load = torch.ones(size=(self.batch_size, self.rollout_size), device=device)
        # shape: (batch, rollout)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.rollout_size, self.problem_size+1), device=device)
        # shape: (batch, rollout, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.rollout_size, self.problem_size+1), device=device)
        # shape: (batch, rollout, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.rollout_size), dtype=torch.bool, device=device)
        # shape: (batch, rollout)
        self.time = torch.zeros(size=(self.batch_size, self.rollout_size), device=device)
        # shape: (batch, rollout)
        self.used_vehicles = torch.zeros(size=(self.batch_size, self.rollout_size), device=device)
        # shape: (batch, rollout)

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

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.visited_mask = self.visited_ninf_flag == float('-inf')
        self.step_state.finished = self.finished
        self.step_state.time = self._get_model_time()
        self.step_state.candidate_features = (
            self._get_candidate_features() if self.enable_candidate_features else None
        )

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected, refresh_candidate_features=True):
        # selected.shape: (batch, rollout)
        device = self.depot_node_xy.device
        if selected.device != device:
            selected = selected.to(device)
        if self.selected_node_list.device != device:
            self.selected_node_list = self.selected_node_list.to(device)

        # Dynamic-1
        ####################################
        self._ensure_selected_capacity(self.selected_count + 1)
        self.selected_count += 1
        prev_node = self.current_node
        self.current_node = selected
        # shape: (batch, rollout)
        self.selected_node_list[:, :, self.selected_count - 1] = self.current_node
        # shape: (batch, rollout, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.rollout_size, -1)
        # shape: (batch, rollout, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, rollout, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, rollout)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill loaded at the depot

        self.used_vehicles += self.at_the_depot.int() - self.finished.int()  # Maybe only calculate this after solution is complete?

        # Dynamic-TW
        ####################################
        travel_time = self._get_travel_distance_between(prev_node, selected)
        tw_start_list = self.depot_node_tw[:, None, :, 0].expand(self.batch_size, self.rollout_size, -1)
        selected_tw_start = torch.gather(tw_start_list, 2, gathering_index).squeeze(dim=2)
        self.time = torch.maximum(self.time + travel_time, selected_tw_start) + self.service_t
        self.time[self.at_the_depot] = 0  # reset time at the depot

        # Compute mask
        ####################################

        self.visited_ninf_flag[self.BATCH_IDX, self.ROLLOUT_IDX, selected] = float('-inf')
        # shape: (batch, rollout, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, rollout, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, rollout, problem+1)

        # Mask nodes that can not be reached before the time window end
        travel_time_to_all = self.travel_time_matrix[self.BATCH_IDX, selected, :]
        possible_arrival_time_all = self.time[:, :, None] + travel_time_to_all
        tw_end_all = self.depot_node_tw[:, None, :, 1].expand(-1, self.rollout_size, -1)
        can_not_be_reached_in_time = possible_arrival_time_all > tw_end_all
        # We inherit the behavior below (depot not masked by customer TW reachability)
        # from PolyNet, whose codebase does not enforce depot-return feasibility.
        # For fair comparison we retain this bypass, but LINC additionally enforces
        # a depot-return mask (see "Mask nodes from which..." block below) so that
        # every LINC solution is guaranteed to return to depot before the due time.
        # Returning to the depot is always allowed, even if depot due is finite.
        can_not_be_reached_in_time[:, :, 0] = False
        # shape: (batch, rollout, problem+1)
        self.ninf_mask[can_not_be_reached_in_time] = float('-inf')

        # Mask nodes from which the vehicle cannot return to depot before depot due time.
        if self.enforce_depot_return and self.cand_has_depot_due.any():
            ready_all = self.depot_node_tw[:, None, :, 0]
            wait_j = torch.clamp(ready_all - possible_arrival_time_all, min=0.0)
            service_at_node = torch.zeros_like(possible_arrival_time_all)
            service_at_node[:, :, 1:] = self.service_t.view(-1, 1, 1) if torch.is_tensor(self.service_t) else self.service_t
            finish_j = possible_arrival_time_all + wait_j + service_at_node
            travel_j_to_depot = self.travel_time_matrix[:, :, 0]  # (batch, problem+1): travel from each node to depot
            travel_j_to_depot = travel_j_to_depot[:, None, :]  # (batch, 1, problem+1)
            return_time = finish_j + travel_j_to_depot
            depot_due = self.depot_node_tw[:, 0, 1][:, None, None]
            cannot_return = return_time > depot_due + 1e-6
            cannot_return[:, :, 0] = False  # depot is always reachable from itself
            self.ninf_mask[cannot_return] = float('-inf')

        no_action_available = ~torch.isfinite(self.ninf_mask).any(dim=2)
        self.ninf_mask[:, :, 0][no_action_available] = 0


        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, rollout)
        self.finished = self.finished + newly_finished

        # Guard: force-finish rollouts that exceed max steps to prevent infinite depot loops
        max_steps = self.problem_size * 2
        forced_finish = (self.selected_count >= max_steps) & ~self.finished
        self.finished = self.finished | forced_finish
        # shape: (batch, rollout)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0


        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.visited_mask = self.visited_ninf_flag == float('-inf')
        self.step_state.finished = self.finished
        self.step_state.time = self._get_model_time()
        # returning values
        done = self.finished.all()
        if self.enable_candidate_features and refresh_candidate_features and not done:
            if self.use_fused_candidate_features and not self.use_selected_candidate_features:
                self.step_state.candidate_features = CVRPTWCandidateFeatureView(self, selected, self.time)
            else:
                current_travel_dist_norm = self.cand_step_travel_dist_norm[self.BATCH_IDX, selected]
                current_angle_diff_norm = self.cand_step_angle_diff_norm[self.BATCH_IDX, selected]
                customer_arrival = possible_arrival_time_all[:, :, 1:]
                customer_wait = torch.relu(self.cand_customer_ready - customer_arrival)
                customer_start = torch.maximum(customer_arrival, self.cand_customer_ready)
                customer_tw_slack = self.cand_customer_due - customer_start
                customer_finish = customer_start + self.cand_service_time
                if self.use_selected_candidate_features:
                    self.step_state.candidate_features = self._compose_selected_candidate_features(
                        travel_dist_norm=current_travel_dist_norm,
                        depot_angle_diff_norm=current_angle_diff_norm,
                        arrival=customer_arrival,
                        wait=customer_wait,
                        tw_slack=customer_tw_slack,
                        finish=customer_finish,
                    )
                else:
                    self.step_state.candidate_features = self._compose_candidate_features(
                        travel_dist_norm=current_travel_dist_norm,
                        depot_angle_diff_norm=current_angle_diff_norm,
                        arrival=customer_arrival,
                        wait=customer_wait,
                        tw_slack=customer_tw_slack,
                        finish=customer_finish,
                    )
        else:
            self.step_state.candidate_features = None

        if done:
            reward = -self._get_total_travel_distance()  # note the minus sign! -self.used_vehicles
        else:
            reward = None

        return self.step_state, reward, done

    def step_with_mask(self, selected, active_mask):
        active_mask = active_mask.to(dtype=torch.bool)
        old_current_node = None if self.current_node is None else self.current_node.clone()
        old_selected_node_list = self.selected_node_list.clone()
        old_at_the_depot = self.at_the_depot.clone()
        old_load = self.load.clone()
        old_visited_ninf_flag = self.visited_ninf_flag.clone()
        old_ninf_mask = self.ninf_mask.clone()
        old_finished = self.finished.clone()
        old_time = self.time.clone()
        old_used_vehicles = self.used_vehicles.clone()

        step_state, _, _ = self.step(selected, refresh_candidate_features=False)

        inactive_mask = ~active_mask
        if inactive_mask.any():
            if old_current_node is None:
                self.current_node[inactive_mask] = 0
            else:
                self.current_node[inactive_mask] = old_current_node[inactive_mask]
            if self.selected_node_list.size(2) == old_selected_node_list.size(2):
                self.selected_node_list[inactive_mask] = old_selected_node_list[inactive_mask]
            else:
                old_capacity = old_selected_node_list.size(2)
                self.selected_node_list[inactive_mask] = 0
                self.selected_node_list[:, :, :old_capacity][inactive_mask] = old_selected_node_list[inactive_mask]
            self.at_the_depot[inactive_mask] = old_at_the_depot[inactive_mask]
            self.load[inactive_mask] = old_load[inactive_mask]
            self.visited_ninf_flag[inactive_mask] = old_visited_ninf_flag[inactive_mask]
            self.ninf_mask[inactive_mask] = old_ninf_mask[inactive_mask]
            self.finished[inactive_mask] = old_finished[inactive_mask]
            self.time[inactive_mask] = old_time[inactive_mask]
            self.used_vehicles[inactive_mask] = old_used_vehicles[inactive_mask]

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.visited_mask = self.visited_ninf_flag == float('-inf')
        self.step_state.finished = self.finished
        self.step_state.time = self._get_model_time()
        self.step_state.candidate_features = (
            self._get_candidate_features() if self.enable_candidate_features else None
        )

        return self.step_state, None, bool(self.finished.all())

    def _get_travel_distance_between(self, from_node, to_node):
        if from_node is None:
            return torch.zeros((self.batch_size, self.rollout_size), dtype=torch.float32, device=self.depot_node_xy.device)
        return self.travel_time_matrix[self.BATCH_IDX, from_node, to_node]

    def _normalize_service_tensor(self, service_value, batch_size, device):
        service_t = torch.as_tensor(service_value, dtype=torch.float32, device=device)
        if service_t.ndim == 0:
            service_t = service_t.repeat(batch_size).unsqueeze(1)
        elif service_t.ndim == 1:
            if service_t.numel() == 1:
                service_t = service_t.repeat(batch_size).unsqueeze(1)
            elif service_t.numel() == batch_size:
                service_t = service_t.unsqueeze(1)
            else:
                raise ValueError("service_t length must be 1 or batch_size")
        elif service_t.ndim == 2 and service_t.shape[1] == 1:
            if service_t.shape[0] == 1:
                service_t = service_t.repeat(batch_size, 1)
            elif service_t.shape[0] != batch_size:
                raise ValueError("service_t first dimension must be 1 or batch_size")
        else:
            raise ValueError("service_t must be scalar, (batch,), or (batch, 1)")
        return service_t

    def _repeat_service_time(self, service_t, repeat_factor):
        if torch.is_tensor(service_t):
            if service_t.ndim == 0:
                return service_t
            return service_t.repeat(repeat_factor, 1)
        return service_t

    def _normalize_travel_time_scale(self, scale_value, batch_size, device):
        scale = torch.as_tensor(scale_value, dtype=torch.float32, device=device)
        if scale.ndim == 0:
            scale = scale.repeat(batch_size)
        elif scale.ndim == 1:
            if scale.numel() == 1:
                scale = scale.repeat(batch_size)
            elif scale.numel() != batch_size:
                raise ValueError("travel_time_scale length must be 1 or batch_size")
        else:
            raise ValueError("travel_time_scale must be scalar or (batch,)")
        return scale

    def _normalize_grid_size(self, grid_value, batch_size, device):
        grid = torch.as_tensor(grid_value, dtype=torch.float32, device=device)
        if grid.ndim == 0:
            grid = grid.repeat(batch_size)
        elif grid.ndim == 1:
            if grid.numel() == 1:
                grid = grid.repeat(batch_size)
            elif grid.numel() != batch_size:
                raise ValueError("grid_size length must be 1 or batch_size")
        else:
            raise ValueError("grid_size must be scalar or (batch,)")
        return grid

    def _get_envelope_generator(self):
        config = EnvelopeCVRPTWConfig.from_mapping(self.distribution)
        signature = (
            tuple(sorted((key, repr(value)) for key, value in self.distribution.items())),
            repr(config),
        )
        if self._envelope_generator is None or self._envelope_generator_signature != signature:
            self._envelope_generator = EnvelopeCVRPTWGenerator(
                config=config,
                seed=self.distribution.get('seed'),
            )
            self._envelope_generator_signature = signature
        return self._envelope_generator

    def _normalize_data_type(self, data_type):
        normalized = str(data_type).strip().lower()
        if normalized in {'envelopecvrptw', 'ecvrptw'}:
            return 'envelope'
        return normalized

    def _get_total_travel_distance(self):
        selected = self.selected_node_list[:, :, :self.selected_count]
        if selected.size(2) == 0:
            return torch.zeros((self.batch_size, self.rollout_size), dtype=torch.float32, device=self.depot_node_xy.device)
        if selected.size(2) == 1:
            return torch.zeros((self.batch_size, self.rollout_size), dtype=torch.float32, device=self.depot_node_xy.device)

        batch_idx = self.BATCH_IDX[:, :, None].expand(-1, -1, selected.size(2) - 1)
        segment_lengths = self.distance_matrix[
            batch_idx,
            selected[:, :, :-1],
            selected[:, :, 1:],
        ]
        closing_lengths = self.distance_matrix[
            self.BATCH_IDX,
            selected[:, :, -1],
            selected[:, :, 0],
        ]
        return segment_lengths.sum(2) + closing_lengths

    def _get_travel_distance_last_step(self):
        if self.selected_count < 2:
            return torch.zeros((self.batch_size, self.rollout_size), dtype=torch.float32, device=self.depot_node_xy.device)
        recent = self.selected_node_list[:, :, self.selected_count - 2:self.selected_count]
        return self.distance_matrix[self.BATCH_IDX, recent[:, :, 0], recent[:, :, 1]]
