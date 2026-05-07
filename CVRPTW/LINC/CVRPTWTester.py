
import torch

import os
import math
import time
from logging import getLogger

from CVRPTWEnv import CVRPTWEnv as Env
from CVRPTWModel import CANDIDATE_FEATURE_INDEX, CVRPTWModel as Model

from utils.utils import *
import itertools
from torch.optim import Adam as Optimizer

class CVRPTWTester:
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

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        merged_model_params = dict(model_params)
        merged_model_params.update(checkpoint.get('model_params', {}))
        self.model_params = merged_model_params

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)
        self.env.enable_candidate_features = bool(self.model_params.get('use_candidate_features', False))
        use_fused_candidate_features = bool(
            self.env.enable_candidate_features
            and self.model_params.get('candidate_scorer_type', 'quotient_lite') == 'quotient_lite'
        )
        self.env.use_fused_candidate_features = use_fused_candidate_features
        self.env.use_selected_candidate_features = bool(
            self.env.enable_candidate_features and not use_fused_candidate_features
        )
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.corrector_params = self._resolve_corrector_params()

        # utility
        self.time_estimator = TimeEstimator()

        self.binary_string_pool = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=self.model_params['z_dim'])])

    def _sample_z_vectors(self, batch_size, aug_factor, starting_points, z_sample_size, z_dim):
        mode = str(self.model_params.get('z_sampling_mode', 'random'))
        if mode == 'polynet_binary_vectors':
            k = min(int(self.model_params.get('polynet_k', z_sample_size)), 2 ** int(z_dim))
            pool = self.binary_string_pool[:k]
            repeat = math.ceil(int(z_sample_size) / max(1, k))
            base = pool.repeat(repeat, 1)[:z_sample_size]
            z = base[None, None, :, :].expand(
                batch_size * aug_factor,
                starting_points,
                z_sample_size,
                z_dim,
            )
            return z.transpose(1, 2).reshape(batch_size * aug_factor, starting_points * z_sample_size, z_dim)

        z_idx = torch.multinomial(
            (torch.ones(batch_size * aug_factor * starting_points, 2 ** z_dim) / 2 ** z_dim),
            z_sample_size,
            replacement=z_sample_size > 2 ** z_dim,
        )
        z = self.binary_string_pool[z_idx].reshape(batch_size * aug_factor, starting_points, z_sample_size, z_dim)
        return z.transpose(1, 2).reshape(batch_size * aug_factor, starting_points * z_sample_size, z_dim)

    def _resolve_corrector_params(self):
        cfg = self.tester_params.get('corrector_params', {})
        enabled = bool(self.model_params.get('use_learned_corrector', False)) and bool(cfg.get('enable', True))
        return {
            'enable': enabled,
            'interval_steps': int(cfg.get('interval_steps', 16)),
            'rounds': int(cfg.get('rounds', 1)),
            'min_selected_count': int(cfg.get('min_selected_count', 8)),
        }

    def run(self):
        start = time.perf_counter()
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        rows = []

        test_num_episode = self.tester_params['test_episodes']
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            if not self.tester_params['EAS_params']['enable']:
                batch_result = self._test_one_batch(batch_size)
            else:
                batch_result = self._search_one_batch(batch_size)

            if len(batch_result) == 3:
                score, aug_score, batch_rows = batch_result
            else:
                score, aug_score = batch_result
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
        peak_memory = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        return {
            "episodes": int(test_num_episode),
            "score_mean": float(score_AM.avg),
            "aug_score_mean": float(aug_score_AM.avg),
            "z_samples": int(self.tester_params["test_z_sample_size"]),
            "aug_factor": int(self.tester_params["aug_factor"]) if self.tester_params["augmentation_enable"] else 1,
            "eval_type": self.model_params["eval_type"],
            "peak_memory_mb": peak_memory,
            "batch_size": int(self.tester_params["test_batch_size"]),
            "mean_distance": float(aug_score_AM.avg),
            "elapsed_sec": float(time.perf_counter() - start),
            "rows": rows,
        }

    def _test_one_batch(self, batch_size):
        if self.corrector_params['enable']:
            return self._test_one_batch_with_corrector(batch_size)

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
            self.env.load_problems(batch_size, rollout_size, self.device, aug_factor)
            reset_state, _, _ = self.env.reset()

            # Sample z vectors only for PolyNet/LINC. POMO checkpoints do not use the PolyNet residual.
            if getattr(self.model.decoder, 'use_poly_residual', True):
                z = self._sample_z_vectors(batch_size, aug_factor, starting_points, z_sample_size, z_dim)
            else:
                z = None

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
                "no_aug_distance": float(no_aug_values[idx].item()),
                "distance": float(aug_values[idx].item()),
            }
            for idx in range(batch_size)
        ]

        return no_aug_values.mean().item(), aug_values.mean().item(), rows

    def _empty_history_buffers(self, batch_size, rollout_size):
        step_feature_dim = int(self.model.corrector_step_feature_dim)
        return {
            'actions': torch.zeros((batch_size, rollout_size, 0), dtype=torch.long, device=self.device),
            'valid': torch.zeros((batch_size, rollout_size, 0), dtype=torch.bool, device=self.device),
            'step_features': torch.zeros((batch_size, rollout_size, 0, step_feature_dim), dtype=torch.float32, device=self.device),
            'path_cost': torch.zeros((batch_size, rollout_size), dtype=torch.float32, device=self.device),
        }

    def _gather_selected_candidate_features(self, state, selected):
        if state.candidate_features is None:
            return torch.zeros(
                (selected.size(0), selected.size(1), len(CANDIDATE_FEATURE_INDEX)),
                dtype=torch.float32,
                device=selected.device,
            )
        return state.candidate_features[state.BATCH_IDX, state.ROLLOUT_IDX, selected].to(dtype=torch.float32)

    def _compute_transition_distance(self, prev_node, selected):
        if prev_node is None:
            return torch.zeros_like(selected, dtype=torch.float32)
        return self.env.distance_matrix[self.env.BATCH_IDX, prev_node, selected].to(dtype=torch.float32)

    def _build_corrector_step_features(self, selected, selected_candidate_features, load_after, time_after, prev_node, step_index):
        if prev_node is None:
            route_start = selected.gt(0)
        else:
            route_start = selected.gt(0) & prev_node.eq(0)

        extras = torch.stack(
            (
                load_after.to(dtype=torch.float32),
                time_after.to(dtype=torch.float32),
                selected.eq(0).to(dtype=torch.float32),
                route_start.to(dtype=torch.float32),
                torch.full_like(load_after, float(step_index) / max(1.0, float(self.env.problem_size * 2))),
            ),
            dim=-1,
        )
        step_features = torch.cat((selected_candidate_features.to(dtype=torch.float32), extras), dim=-1)
        target_dim = int(self.model.corrector_step_feature_dim)
        if step_features.size(-1) < target_dim:
            pad = torch.zeros(
                (*step_features.shape[:-1], target_dim - step_features.size(-1)),
                dtype=step_features.dtype,
                device=step_features.device,
            )
            step_features = torch.cat((step_features, pad), dim=-1)
        elif step_features.size(-1) > target_dim:
            step_features = step_features[..., :target_dim]
        return step_features

    def _append_history_step(self, history, selected, step_features, valid_mask, transition_distance):
        valid_mask = valid_mask.to(dtype=torch.bool)
        selected_store = torch.where(valid_mask, selected, torch.zeros_like(selected))
        history['actions'] = torch.cat((history['actions'], selected_store[:, :, None]), dim=2)
        history['valid'] = torch.cat((history['valid'], valid_mask[:, :, None]), dim=2)
        history['step_features'] = torch.cat(
            (history['step_features'], (step_features * valid_mask[:, :, None].to(dtype=step_features.dtype))[:, :, None, :]),
            dim=2,
        )
        history['path_cost'] = history['path_cost'] + transition_distance.to(dtype=torch.float32) * valid_mask.to(dtype=torch.float32)
        return history

    def _sample_greedy_corrector(self, count_logits, step_logits, removable_mask):
        batch_size, rollout_size, seq_len = step_logits.shape
        max_removals = int(count_logits.size(-1) - 1)
        removable_counts = removable_mask.sum(dim=2)
        valid_count_mask = torch.arange(max_removals + 1, device=step_logits.device)[None, None, :] <= removable_counts[:, :, None]
        masked_count_logits = count_logits.masked_fill(~valid_count_mask, float('-inf'))
        sampled_counts = masked_count_logits.argmax(dim=-1)

        flat_step_logits = step_logits.reshape(batch_size * rollout_size, seq_len)
        flat_work_mask = removable_mask.reshape(batch_size * rollout_size, seq_len).clone()
        flat_removed = torch.zeros_like(flat_work_mask)
        flat_counts = sampled_counts.reshape(-1)

        for remove_idx in range(max_removals):
            active_rows = torch.nonzero(flat_counts > remove_idx, as_tuple=False).squeeze(1)
            if active_rows.numel() == 0:
                break
            masked_logits = flat_step_logits[active_rows].masked_fill(~flat_work_mask[active_rows], float('-inf'))
            picked = masked_logits.argmax(dim=1)
            flat_removed[active_rows, picked] = True
            flat_work_mask[active_rows, picked] = False

        return flat_removed.reshape(batch_size, rollout_size, seq_len)

    def _truncate_history_before_first_removal(self, history, removed_mask):
        valid_mask = history['valid']
        batch_size, rollout_size, seq_len = valid_mask.shape
        device = valid_mask.device

        position_idx = torch.arange(seq_len, device=device)[None, None, :].expand(batch_size, rollout_size, seq_len)
        removed_pos = torch.where(removed_mask, position_idx, torch.full_like(position_idx, seq_len))
        first_removed = removed_pos.min(dim=2).values
        no_removal = ~removed_mask.any(dim=2)
        first_removed = torch.where(no_removal, valid_mask.sum(dim=2), first_removed)

        prefix_mask = valid_mask & (position_idx < first_removed[:, :, None])
        max_keep = int(first_removed.max().item()) if first_removed.numel() > 0 else 0
        flat_prefix = prefix_mask.reshape(batch_size * rollout_size, seq_len)
        flat_actions = history['actions'].reshape(batch_size * rollout_size, seq_len)

        kept_actions = torch.zeros((batch_size * rollout_size, max_keep), dtype=torch.long, device=self.device)
        kept_valid = torch.zeros((batch_size * rollout_size, max_keep), dtype=torch.bool, device=self.device)
        if max_keep > 0:
            row_idx, time_idx = torch.nonzero(flat_prefix, as_tuple=True)
            kept_actions[row_idx, time_idx] = flat_actions[row_idx, time_idx]
            kept_valid[row_idx, time_idx] = True

        return (
            kept_actions.reshape(batch_size, rollout_size, max_keep),
            kept_valid.reshape(batch_size, rollout_size, max_keep),
            first_removed,
        )

    def _replay_prefix_history(self, kept_actions, kept_valid):
        history = self._empty_history_buffers(self.env.batch_size, self.env.rollout_size)
        self.env.reset()
        state, _, _ = self.env.pre_step()

        if kept_actions.size(2) == 0:
            return state, history, False

        done = False
        for step_idx in range(kept_actions.size(2)):
            active_mask = kept_valid[:, :, step_idx]
            if not active_mask.any():
                break

            selected = kept_actions[:, :, step_idx]
            selected_candidate_features = self._gather_selected_candidate_features(state, selected)
            prev_node = state.current_node
            transition_distance = self._compute_transition_distance(prev_node, selected)
            state, _, done = self.env.step_with_mask(selected, active_mask)
            step_features = self._build_corrector_step_features(
                selected,
                selected_candidate_features,
                self.env.load,
                self.env.time / self.env.grid_size_tensor[:, None],
                prev_node,
                self.env.selected_count,
            )
            history = self._append_history_step(history, selected, step_features, active_mask, transition_distance)

        return state, history, bool(done)

    def _test_one_batch_with_corrector(self, batch_size):
        if self.model_params['force_first_move']:
            raise NotImplementedError("Learned corrector evaluation does not support force_first_move yet.")

        z_sample_size = self.tester_params['test_z_sample_size']
        z_dim = self.model_params['z_dim']
        amp_inference = self.tester_params['amp_inference']
        device = "cuda" if self.tester_params['use_cuda'] else "cpu"
        greedy_action_selection = self.model_params['eval_type'] == 'argmax'
        rollout_size = z_sample_size

        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, rollout_size, self.device, aug_factor)
            reset_state, _, _ = self.env.reset()

            if getattr(self.model.decoder, 'use_poly_residual', True):
                z = self._sample_z_vectors(batch_size, aug_factor, 1, z_sample_size, z_dim)
            else:
                z = None
            self.model.pre_forward(reset_state, z)

            history = self._empty_history_buffers(self.env.batch_size, self.env.rollout_size)
            correction_round = 0
            steps_since_correction = 0
            state, _, done = self.env.pre_step()

            while True:
                with torch.amp.autocast(device_type=device, enabled=amp_inference):
                    while not done:
                        selected, _ = self.model(state, greedy_action_selection)

                        selected_candidate_features = self._gather_selected_candidate_features(state, selected)
                        prev_node = state.current_node
                        transition_distance = self._compute_transition_distance(prev_node, selected)
                        state, _, done = self.env.step(selected)
                        done = bool(done)
                        step_features = self._build_corrector_step_features(
                            selected,
                            selected_candidate_features,
                            self.env.load,
                            self.env.time / self.env.grid_size_tensor[:, None],
                            prev_node,
                            self.env.selected_count,
                        )
                        history = self._append_history_step(
                            history,
                            selected,
                            step_features,
                            torch.ones_like(selected, dtype=torch.bool),
                            transition_distance,
                        )
                        steps_since_correction += 1

                        removable_mask = history['valid'] & history['actions'].ne(0)
                        enough_history = history['actions'].size(2) >= int(self.corrector_params['min_selected_count'])
                        should_interrupt = (
                            correction_round < int(self.corrector_params['rounds'])
                            and enough_history
                            and bool(removable_mask.any().item())
                            and (steps_since_correction >= int(self.corrector_params['interval_steps']) or done)
                        )
                        if should_interrupt:
                            break

                removable_mask = history['valid'] & history['actions'].ne(0)
                enough_history = history['actions'].size(2) >= int(self.corrector_params['min_selected_count'])
                should_correct = (
                    correction_round < int(self.corrector_params['rounds'])
                    and enough_history
                    and bool(removable_mask.any().item())
                    and (steps_since_correction >= int(self.corrector_params['interval_steps']) or done)
                )

                if should_correct:
                    count_logits, step_logits = self.model.score_correction(
                        history['actions'],
                        history['step_features'],
                        history['valid'],
                        removable_mask,
                    )
                    removed_mask = self._sample_greedy_corrector(count_logits, step_logits, removable_mask)
                    kept_actions, kept_valid, _ = self._truncate_history_before_first_removal(history, removed_mask)
                    state, history, done = self._replay_prefix_history(kept_actions, kept_valid)
                    correction_round += 1
                    steps_since_correction = 0
                    continue

                if done:
                    break

        final_reward = -history['path_cost']
        aug_reward = final_reward.reshape(aug_factor, batch_size, rollout_size)
        max_pomo_reward, _ = aug_reward.max(dim=2)
        no_aug_score = -max_pomo_reward[0, :].float().mean()
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
        aug_score = -max_aug_pomo_reward.float().mean()
        return no_aug_score.item(), aug_score.item()



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
        self.env.load_problems(batch_size, rollout_size, self.device, aug_factor)
        reset_state, _, _ = self.env.reset()

        # Sample z vectors
        def sample_z_vectors():
            if getattr(self.model.decoder, 'use_poly_residual', True):
                z_idx = torch.multinomial((torch.ones(batch_size * aug_factor * starting_points, 2 ** z_dim) / 2 ** z_dim),
                                          z_sample_size, replacement=z_sample_size > 2**z_dim)
                z = self.binary_string_pool[z_idx].reshape(batch_size * aug_factor, starting_points, z_sample_size, z_dim)
                z = z.transpose(1, 2).reshape(batch_size * aug_factor, rollout_size, z_dim)
            else:
                z = None
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
