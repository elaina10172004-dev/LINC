
from dataclasses import dataclass
import torch
import pickle

from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)


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


class CVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.rollout_size = None

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

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
        self.distance_matrix = None
        self.demand_rollout = None

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

        # states to return
        ####################################
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

        depot_data = list(data[i][0] for i in range(index_begin, index_begin+num_problems))
        self.saved_depot_xy = torch.tensor(depot_data)[:, None, :]
        # shape: (batch, 1, 2)

        node_data = list(data[i][1] for i in range(index_begin, index_begin+num_problems))
        self.saved_node_xy = torch.tensor(node_data)
        # shape: (batch, problem, 2)

        demand_data = list(data[i][2] for i in range(index_begin, index_begin+num_problems))
        capacity_data = list(data[i][3] for i in range(index_begin, index_begin+num_problems))
        capacity_tensor = torch.tensor(capacity_data, dtype=torch.float)
        self.saved_node_demand = torch.tensor(demand_data, dtype=torch.float)/capacity_tensor[:, None]

        self.saved_index = 0

    def use_random_problems(self):
        self.FLAG__use_saved_problems = False

        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None

    def load_problems(self, batch_size, rollout_size, aug_factor=1):
        self.batch_size = batch_size
        self.rollout_size = rollout_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
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
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1), dtype=node_demand.dtype, device=node_demand.device)
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        self.distance_matrix = torch.cdist(self.depot_node_xy, self.depot_node_xy)


        self.BATCH_IDX = torch.arange(self.batch_size, device=self.depot_node_xy.device)[:, None].expand(self.batch_size, self.rollout_size)
        self.ROLLOUT_IDX = torch.arange(self.rollout_size, device=self.depot_node_xy.device)[None, :].expand(self.batch_size, self.rollout_size)
        self.demand_rollout = self.depot_node_demand[:, None, :].expand(self.batch_size, self.rollout_size, -1)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.ROLLOUT_IDX = self.ROLLOUT_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, rollout)
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.rollout_size, self.problem_size * 2 + 1),
            dtype=torch.long,
            device=self.depot_node_xy.device,
        )
        # shape: (batch, rollout, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.rollout_size), dtype=torch.bool, device=self.depot_node_xy.device)
        # shape: (batch, rollout)
        self.load = torch.ones(size=(self.batch_size, self.rollout_size), dtype=self.depot_node_xy.dtype, device=self.depot_node_xy.device)
        # shape: (batch, rollout)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.rollout_size, self.problem_size+1), dtype=self.depot_node_xy.dtype, device=self.depot_node_xy.device)
        # shape: (batch, rollout, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.rollout_size, self.problem_size+1), dtype=self.depot_node_xy.dtype, device=self.depot_node_xy.device)
        # shape: (batch, rollout, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.rollout_size), dtype=torch.bool, device=self.depot_node_xy.device)
        # shape: (batch, rollout)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, rollout)
        device = self.depot_node_xy.device
        if selected.device != device:
            selected = selected.to(device)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, rollout)
        self.selected_node_list[:, :, self.selected_count - 1] = self.current_node
        # shape: (batch, rollout, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.demand_rollout[:, :self.load.size(1), :]
        # shape: (batch, rollout, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, rollout, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, rollout)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.ROLLOUT_IDX, selected] = float('-inf')
        # shape: (batch, rollout, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask.copy_(self.visited_ninf_flag)
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, rollout, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, rollout, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, rollout)
        self.finished = self.finished + newly_finished
        # shape: (batch, rollout)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
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
