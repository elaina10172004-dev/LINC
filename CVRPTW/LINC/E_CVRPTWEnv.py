import torch

from CVRPTWEnv import CVRPTWEnv


class E_CVRPTWEnv(CVRPTWEnv):
    """CVRPTW env extensions used by the official SGBS rollout pattern.

    This mirrors the state-management API in the official SGBS repository:
    repeat a beam-search env into rollout branches, gather the best branches
    back, and optionally merge rollout branches with existing beam branches.
    """

    def modify_rollout_size(self, new_rollout_size):
        self.rollout_size = int(new_rollout_size)
        device = self.depot_node_xy.device
        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(
            self.batch_size, self.rollout_size
        )
        self.ROLLOUT_IDX = torch.arange(self.rollout_size, device=device)[None, :].expand(
            self.batch_size, self.rollout_size
        )
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.ROLLOUT_IDX = self.ROLLOUT_IDX

    def reset_by_repeating_bs_env(self, bs_env, repeat):
        repeat = int(repeat)
        self.selected_count = int(bs_env.selected_count)
        self.current_node = _repeat_rollout_tensor(bs_env.current_node, repeat)
        self.selected_node_list = _repeat_rollout_tensor(bs_env.selected_node_list, repeat)
        self.at_the_depot = _repeat_rollout_tensor(bs_env.at_the_depot, repeat)
        self.load = _repeat_rollout_tensor(bs_env.load, repeat)
        self.visited_ninf_flag = _repeat_rollout_tensor(bs_env.visited_ninf_flag, repeat)
        self.ninf_mask = _repeat_rollout_tensor(bs_env.ninf_mask, repeat)
        self.finished = _repeat_rollout_tensor(bs_env.finished, repeat)
        self.time = _repeat_rollout_tensor(bs_env.time, repeat)
        self.used_vehicles = _repeat_rollout_tensor(bs_env.used_vehicles, repeat)
        self.modify_rollout_size(bs_env.rollout_size * repeat)
        self._sync_step_state()

    def reset_by_gathering_rollout_env(self, rollout_env, gathering_index):
        self.selected_count = int(rollout_env.selected_count)
        self.current_node = _gather_rollout_tensor(rollout_env.current_node, gathering_index)
        self.selected_node_list = _gather_rollout_tensor(rollout_env.selected_node_list, gathering_index)
        self.at_the_depot = _gather_rollout_tensor(rollout_env.at_the_depot, gathering_index)
        self.load = _gather_rollout_tensor(rollout_env.load, gathering_index)
        self.visited_ninf_flag = _gather_rollout_tensor(rollout_env.visited_ninf_flag, gathering_index)
        self.ninf_mask = _gather_rollout_tensor(rollout_env.ninf_mask, gathering_index)
        self.finished = _gather_rollout_tensor(rollout_env.finished, gathering_index)
        self.time = _gather_rollout_tensor(rollout_env.time, gathering_index)
        self.used_vehicles = _gather_rollout_tensor(rollout_env.used_vehicles, gathering_index)
        self.modify_rollout_size(gathering_index.size(1))
        self._sync_step_state()

    def merge(self, other_env):
        self.current_node = _cat_rollout_tensor(self.current_node, other_env.current_node)
        self.selected_node_list = _cat_rollout_tensor(self.selected_node_list, other_env.selected_node_list)
        self.at_the_depot = _cat_rollout_tensor(self.at_the_depot, other_env.at_the_depot)
        self.load = _cat_rollout_tensor(self.load, other_env.load)
        self.visited_ninf_flag = _cat_rollout_tensor(self.visited_ninf_flag, other_env.visited_ninf_flag)
        self.ninf_mask = _cat_rollout_tensor(self.ninf_mask, other_env.ninf_mask)
        self.finished = _cat_rollout_tensor(self.finished, other_env.finished)
        self.time = _cat_rollout_tensor(self.time, other_env.time)
        self.used_vehicles = _cat_rollout_tensor(self.used_vehicles, other_env.used_vehicles)
        self.modify_rollout_size(self.rollout_size + other_env.rollout_size)
        self._sync_step_state()

    def copy_dynamic_from(self, source_env):
        self.selected_count = int(source_env.selected_count)
        self.current_node = _copy_or_clone_tensor(self.current_node, source_env.current_node)
        self.selected_node_list = _copy_or_clone_tensor(self.selected_node_list, source_env.selected_node_list)
        self.at_the_depot = _copy_or_clone_tensor(self.at_the_depot, source_env.at_the_depot)
        self.load = _copy_or_clone_tensor(self.load, source_env.load)
        self.visited_ninf_flag = _copy_or_clone_tensor(self.visited_ninf_flag, source_env.visited_ninf_flag)
        self.ninf_mask = _copy_or_clone_tensor(self.ninf_mask, source_env.ninf_mask)
        self.finished = _copy_or_clone_tensor(self.finished, source_env.finished)
        self.time = _copy_or_clone_tensor(self.time, source_env.time)
        self.used_vehicles = _copy_or_clone_tensor(self.used_vehicles, source_env.used_vehicles)
        if self.rollout_size != source_env.rollout_size:
            self.modify_rollout_size(source_env.rollout_size)
        self._sync_step_state()

    def _sync_step_state(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.visited_mask = self.visited_ninf_flag == float("-inf")
        self.step_state.finished = self.finished
        self.step_state.time = self._get_model_time()
        self.step_state.candidate_features = None


def _repeat_rollout_tensor(value, repeat):
    if value is None:
        return None
    return value.repeat_interleave(repeat, dim=1)


def _gather_rollout_tensor(value, gathering_index):
    if value is None:
        return None
    if value.ndim == 2:
        return value.gather(dim=1, index=gathering_index)
    expand_index = gathering_index
    for _ in range(2, value.ndim):
        expand_index = expand_index.unsqueeze(-1)
    expand_index = expand_index.expand(-1, -1, *value.shape[2:])
    return value.gather(dim=1, index=expand_index)


def _cat_rollout_tensor(left, right):
    if left is None:
        return right
    if right is None:
        return left
    return torch.cat((left, right), dim=1)


def _clone_tensor(value):
    if value is None:
        return None
    return value.clone()


def _copy_or_clone_tensor(target, source):
    if source is None:
        return None
    if (
        target is not None
        and target.shape == source.shape
        and target.dtype == source.dtype
        and target.device == source.device
    ):
        target.copy_(source)
        return target
    return source.clone()
