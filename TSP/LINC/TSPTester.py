
import time

import torch

import os
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from utils.utils import *
import itertools
from torch.optim import Adam as Optimizer

class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

        self.binary_string_pool = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=model_params['z_dim'])])
        self.raw_saved_problems = None
        self.metric = "euclidean"

    def run(self):
        start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        rows = []

        test_num_episode = self.tester_params['test_episodes']

        if self.tester_params['test_data_load']['enable']:
            self.env.use_pkl_saved_problems(self.tester_params['test_data_load']['filename'], test_num_episode)
            raw = self.env.saved_problems.float()
            coord_scale = raw.amax(dim=(1, 2)).clamp_min(1.0)
            if bool((coord_scale > 2.0).any().item()):
                self.raw_saved_problems = raw
                self.env.saved_problems = raw / coord_scale[:, None, None]
                self.metric = "tsplib_euc2d"

        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            if not self.tester_params['EAS_params']['enable']:
                score, aug_score, batch_rows = self._test_one_batch(batch_size)
            else:
                score, aug_score = self._search_one_batch(batch_size)
                batch_rows = []

            for row in batch_rows:
                row = dict(row)
                row["instance"] = int(episode + row["instance"])
                rows.append(row)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        else:
            peak_memory = None
        return {
            "episodes": int(test_num_episode),
            "batch_size": int(self.tester_params["test_batch_size"]),
            "score_mean": float(score_AM.avg),
            "aug_score_mean": float(aug_score_AM.avg),
            "mean_cost": float(aug_score_AM.avg),
            "elapsed_sec": float(time.perf_counter() - start),
            "peak_memory_mb": peak_memory,
            "z_samples": int(self.tester_params["test_z_sample_size"]),
            "aug_factor": int(self.tester_params["aug_factor"] if self.tester_params["augmentation_enable"] else 1),
            "eval_type": self.model_params["eval_type"],
            "metric": self.metric,
            "rows": rows,
        }

    def _test_one_batch(self, batch_size):
        z_sample_size = self.tester_params['test_z_sample_size']
        z_dim = self.model_params['z_dim']
        amp_inference = self.tester_params['amp_inference']
        device = "cuda" if self.tester_params['use_cuda'] else "cpu"
        greedy_action_selection = self.model_params['eval_type'] == 'argmax'

        if self.model_params['force_first_move']:
            starting_points = self.env_params['problem_size']
            rollout_size = starting_points * z_sample_size
        else:
            starting_points = 1
            rollout_size = z_sample_size

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            raw_batch = None
            if self.raw_saved_problems is not None:
                raw_batch = self.raw_saved_problems[self.env.saved_index:self.env.saved_index + batch_size]
            self.env.load_problems(batch_size, rollout_size, aug_factor)
            reset_state, _, _ = self.env.reset()

            # Sample z vectors
            z_idx = torch.multinomial((torch.ones(batch_size * aug_factor * starting_points, 2 ** z_dim) / 2 ** z_dim),
                                      z_sample_size, replacement=z_sample_size > 2**z_dim)
            z = self.binary_string_pool[z_idx].reshape(batch_size * aug_factor, starting_points, z_sample_size, z_dim)
            z = z.transpose(1, 2).reshape(batch_size * aug_factor, rollout_size, z_dim)

            self.model.pre_forward(reset_state, z)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            with torch.amp.autocast(device_type=device, enabled=amp_inference):
                while not done:
                    selected, _ = self.model(state, greedy_action_selection)
                    # shape: (batch, pomo)
                    state, reward, done = self.env.step(selected)

        # Return
        ###############################################
        if raw_batch is not None:
            tours = self.env.selected_node_list[:, :, :self.env.problem_size]
            raw_aug = raw_batch.repeat(int(aug_factor), 1, 1)
            cost = self._tsplib_euc2d_cost(raw_aug, tours).reshape(int(aug_factor), batch_size, rollout_size)
            no_aug_values = cost[0].min(dim=1).values.float()
            aug_values = cost.min(dim=2).values.min(dim=0).values.float()
            rows = [
                {
                    "instance": int(idx),
                    "cost": float(aug_values[idx].item()),
                    "no_aug_cost": float(no_aug_values[idx].item()),
                }
                for idx in range(batch_size)
            ]
            return no_aug_values.mean().item(), aug_values.mean().item(), rows

        aug_reward = reward.reshape(aug_factor, batch_size, rollout_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_values = -max_pomo_reward[0, :].float()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_values = -max_aug_pomo_reward.float()  # negative sign to make positive value

        rows = [
            {
                "instance": int(idx),
                "cost": float(aug_values[idx].item()),
                "no_aug_cost": float(no_aug_values[idx].item()),
            }
            for idx in range(batch_size)
        ]

        return no_aug_values.mean().item(), aug_values.mean().item(), rows

    @staticmethod
    def _tsplib_euc2d_cost(raw_coords, tours):
        batch_size, rollout_size, problem_size = tours.shape
        coords = raw_coords[:, None, :, :].expand(batch_size, rollout_size, problem_size, 2)
        ordered = coords.gather(dim=2, index=tours[:, :, :, None].expand(-1, -1, -1, 2))
        segment = ordered[:, :, 1:, :] - ordered[:, :, :-1, :]
        closing = ordered[:, :, :1, :] - ordered[:, :, -1:, :]
        edges = torch.cat((segment, closing), dim=2)
        return torch.floor(torch.linalg.vector_norm(edges, dim=-1) + 0.5).sum(dim=2)



    def _search_one_batch(self, batch_size):
        z_sample_size = self.tester_params['test_z_sample_size']
        z_dim = self.model_params['z_dim']
        amp_inference = self.tester_params['amp_inference']
        device = "cuda" if self.tester_params['use_cuda'] else "cpu"
        iterations = self.tester_params['EAS_params']['iterations']

        if self.model_params['force_first_move']:
            raise NotImplementedError
        else:
            starting_points = 1
            rollout_size = z_sample_size

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Prepare model
        ###############################################
        self.model.decoder.reset_EAS_layers(batch_size*aug_factor)  # initialize/reset EAS layers
        EAS_layer_parameters = self.model.decoder.get_EAS_parameters()

        # Only store gradients for new EAS layer weights
        self.model.requires_grad_(False)
        for t in EAS_layer_parameters:
            t.requires_grad_(True)

        optimizer = Optimizer(EAS_layer_parameters, lr=self.tester_params['EAS_params']['lr'])


        # Ready
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size, rollout_size, aug_factor)
        reset_state, _, _ = self.env.reset()

        # Sample z vectors
        def sample_z_vectors():
            z_idx = torch.multinomial((torch.ones(batch_size * aug_factor * starting_points, 2 ** z_dim) / 2 ** z_dim),
                                      z_sample_size, replacement=z_sample_size > 2**z_dim)
            z = self.binary_string_pool[z_idx].reshape(batch_size * aug_factor, starting_points, z_sample_size, z_dim)
            z = z.transpose(1, 2).reshape(batch_size * aug_factor, rollout_size, z_dim)
            return z

        z = sample_z_vectors()
        self.model.pre_forward(reset_state, z)

        incumbent_reward = torch.ones(batch_size).float() * float('-inf')
        incumbent_solution = None

        for iter in range(iterations):
            self.env.reset()

            if self.tester_params['EAS_params']['resample']:
                z = sample_z_vectors()
                self.model.decoder.set_z(z)

            prob_list = []

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            with torch.amp.autocast(device_type=device, enabled=amp_inference):
                while not done:

                    if incumbent_solution is not None:
                        incumbent_action = incumbent_solution[:, self.env.selected_count]
                    else:
                        incumbent_action = None

                    selected, prob = self.model(state, greedy_construction=False, EAS_incumbent_action=incumbent_action)
                    # shape: (batch, pomo)
                    state, reward, done = self.env.step(selected)
                    prob_list.append(prob)

            # Incumbent solution
            ###############################################
            max_reward, max_idx = reward.max(dim=1)  # get best results from rollouts + Incumbent
            # shape: (aug_batch,)
            incumbent_reward = max_reward

            gathering_index = max_idx[:, None, None].expand(-1, 1, self.env.selected_count)
            new_incumbent_solution = self.env.selected_node_list.gather(dim=1, index=gathering_index)
            new_incumbent_solution = new_incumbent_solution.squeeze(dim=1)
            # shape: (aug_batch, tour_len)

            solution_max_length = self.tester_params['solution_max_length']
            incumbent_solution = torch.zeros(size=(batch_size*aug_factor, solution_max_length), dtype=torch.long)
            incumbent_solution[:, :self.env.selected_count] = new_incumbent_solution

            # Loss: POMO RL
            ###############################################
            prob_list = torch.stack(prob_list, dim=2)
            pomo_prob_list = prob_list[:, :-1, :]
            # shape: (aug_batch, pomo, tour_len)
            pomo_reward = reward[:, :-1]
            # shape: (aug_batch, pomo)

            advantage = pomo_reward - pomo_reward.mean(dim=1, keepdim=True)
            # shape: (aug_batch, pomo)
            log_prob = pomo_prob_list.log().sum(dim=2)
            # size = (aug_batch, pomo)
            loss_RL = -advantage * log_prob  # Minus Sign: To increase REWARD

            # shape: (aug_batch, pomo)
            loss_RL = loss_RL.mean(dim=1)
            # shape: (aug_batch,)

            # Loss: IL
            ###############################################
            imitation_prob_list = prob_list[:, -1, :]
            # shape: (aug_batch, tour_len)
            log_prob = imitation_prob_list.log().sum(dim=1)
            # shape: (aug_batch,)
            loss_IL = -log_prob  # Minus Sign: to increase probability
            # shape: (aug_batch,)

            # Back Propagation
            ###############################################
            optimizer.zero_grad(set_to_none=True)

            loss = loss_RL + self.tester_params['EAS_params']['lambda'] * loss_IL
            # shape: (aug_batch,)
            loss.sum().backward()

            optimizer.step()

        # Return
        ###############################################
        aug_reward = incumbent_reward.reshape(aug_factor, batch_size)
        # shape: (augmentation, batch)

        no_aug_score = -aug_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = aug_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()
