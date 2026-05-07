import math
import os
import itertools
from logging import getLogger

import numpy as np
import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
from utils.utils import *


class CVRPTrainer:
    def __init__(self, run_params, env_params, model_params, optimizer_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        self.model = Model(**self.model_params)
        if self.trainer_params['compile_model']:
            self.model = torch.compile(self.model)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.scaler = torch.cuda.amp.GradScaler()

        self.start_epoch = 1
        self.advantage_params = self._resolve_advantage_params()
        self.current_advantage_params = dict(self.advantage_params)

        if run_params["name"] is not None:
            model_load_path = os.path.join("result", "run_" + run_params["name"], "checkpoint_latest.pt")
            if os.path.exists(model_load_path):
                checkpoint = torch.load(model_load_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                self.start_epoch = 1 + checkpoint['epoch']
                self.result_log.set_raw_data(checkpoint['result_log'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.scheduler.last_epoch = self.start_epoch - 1
                self.logger.info('Resuming named training run!')

        if trainer_params['model_load']['enable'] and self.start_epoch == 1:
            model_load_path = '{path}/checkpoint-{epoch}.pt'.format(**trainer_params['model_load'])
            checkpoint = torch.load(model_load_path, map_location=self.device)
            self._load_model_state_compatible(checkpoint['model_state_dict'])
            self.logger.info('Saved Model Loaded!')

        self.time_estimator = TimeEstimator()
        self.binary_string_pool = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=model_params['z_dim'])])

    def _get_train_num_episode(self):
        if self.model_params['force_first_move']:
            solutions_per_instance = self.trainer_params["K"] * self.env_params['problem_size']
        else:
            solutions_per_instance = self.trainer_params["K"]
        return int(self.trainer_params['train_num_rollouts'] / solutions_per_instance)

    def _get_train_batches_per_epoch(self):
        train_num_episode = self._get_train_num_episode()
        train_batch_size = int(self.trainer_params['train_batch_size'])
        return max(int(math.ceil(float(train_num_episode) / float(train_batch_size))), 1)

    def _resolve_module_slow_start_progress(self, epoch, batch_in_epoch=0):
        slow_epochs = int(self.trainer_params.get('module_slow_start_epochs', 0))
        if slow_epochs <= 0:
            return 1.0

        by_batch = bool(self.trainer_params.get('module_slow_start_by_batch', True))
        if by_batch:
            batches_per_epoch = self._get_train_batches_per_epoch()
            total_batches = max(int(slow_epochs * batches_per_epoch), 1)
            global_batch_idx = max((int(epoch) - 1) * batches_per_epoch + int(batch_in_epoch), 0)
            if total_batches == 1:
                return 1.0
            return min(max(global_batch_idx / float(total_batches - 1), 0.0), 1.0)

        if slow_epochs == 1:
            return 1.0
        return min(max((int(epoch) - 1) / float(slow_epochs - 1), 0.0), 1.0)

    def _apply_encoder_freeze_schedule(self, epoch):
        freeze_until_epoch = int(self.trainer_params.get('freeze_encoder_until_epoch', 0))
        freeze_layer_count = int(self.trainer_params.get('freeze_encoder_layer_count', 0))
        freeze_early = freeze_until_epoch > 0 and int(epoch) <= freeze_until_epoch

        encoder = getattr(self.model, 'encoder', None)
        if encoder is None:
            return {'freeze_early': False, 'freeze_layer_count': 0}

        for module_name in ('embedding_depot', 'embedding_node'):
            module = getattr(encoder, module_name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = not freeze_early

        layers = getattr(encoder, 'layers', None)
        if layers is not None:
            for layer_idx, layer in enumerate(layers):
                should_train = (not freeze_early) or (layer_idx >= freeze_layer_count)
                for param in layer.parameters():
                    param.requires_grad = should_train

        return {
            'freeze_early': bool(freeze_early),
            'freeze_layer_count': int(freeze_layer_count),
        }

    def _load_model_state_compatible(self, checkpoint_state_dict):
        model_state = self.model.state_dict()
        compatible_state = {
            key: value
            for key, value in checkpoint_state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        self.model.load_state_dict(compatible_state, strict=False)

    def _resolve_advantage_params(self):
        advantage_params = self.trainer_params.get('advantage_params', {})
        return {
            'mode': advantage_params.get('mode', 'matched_soft_top1'),
            'tau': float(advantage_params.get('tau', 1.0)),
            'reward_scale': float(advantage_params.get('reward_scale', 1.0)),
            'normalize_adv': bool(advantage_params.get('normalize_adv', False)),
            'eps': float(advantage_params.get('eps', 1e-6)),
            'rollout_mask_mode': advantage_params.get('rollout_mask_mode', 'best_only'),
            'tau_start': float(advantage_params.get('tau_start', advantage_params.get('tau', 1.0))),
            'tau_end': float(advantage_params.get('tau_end', advantage_params.get('tau', 1.0))),
            'tau_anneal_ratio': float(advantage_params.get('tau_anneal_ratio', 0.0)),
            'best_only_start_ratio': float(advantage_params.get('best_only_start_ratio', 0.0)),
        }

    def _resolve_effective_advantage_params(self, epoch, batch_in_epoch=0):
        effective = dict(self.advantage_params)
        if epoch is None or self.trainer_params['epochs'] <= 0:
            effective['tau'] = effective['tau_end']
            return effective

        if bool(self.trainer_params.get('advantage_schedule_by_batch', True)):
            batches_per_epoch = self._get_train_batches_per_epoch()
            total_batches = max(int(self.trainer_params['epochs']) * batches_per_epoch, 1)
            global_batch_idx = max((int(epoch) - 1) * batches_per_epoch + int(batch_in_epoch), 0)
            if total_batches == 1:
                progress = 1.0
            else:
                progress = min(max(global_batch_idx / float(total_batches - 1), 0.0), 1.0)
        else:
            progress = min(max(epoch / self.trainer_params['epochs'], 0.0), 1.0)
        anneal_ratio = effective['tau_anneal_ratio']
        if anneal_ratio > 0:
            anneal_progress = min(progress / anneal_ratio, 1.0)
            tau_start = effective['tau_start']
            tau_end = effective['tau_end']
            effective['tau'] = tau_start * (tau_end / tau_start) ** anneal_progress
        else:
            effective['tau'] = effective['tau_end']

        if effective['rollout_mask_mode'] == 'best_only' and progress < effective['best_only_start_ratio']:
            effective['rollout_mask_mode'] = 'dense'
        return effective

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')
            module_progress = self._resolve_module_slow_start_progress(epoch, batch_in_epoch=0)
            if hasattr(self.model, 'set_module_slow_start_progress'):
                self.model.set_module_slow_start_progress(module_progress)
            freeze_state = self._apply_encoder_freeze_schedule(epoch)
            runtime_state = self.model.get_module_slow_start_state() if hasattr(self.model, 'get_module_slow_start_state') else {}
            runtime_state = dict(runtime_state)
            runtime_state['module_progress'] = float(module_progress)
            runtime_state.update(freeze_state)
            self.logger.info('Module State: %s', runtime_state)
            self.current_advantage_params = self._resolve_effective_advantage_params(epoch, batch_in_epoch=0)
            self.logger.info(
                "Advantage Mode: %s, tau=%.4f, rollout_mask=%s"
                % (
                    self.current_advantage_params['mode'],
                    self.current_advantage_params['tau'],
                    self.current_advantage_params['rollout_mask_mode'],
                )
            )

            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            self.scheduler.step()

            if self.trainer_params.get('enable_epoch_validation', True):
                self.logger.info("Starting validation")
                self.logger.info("Greedy:")
                self._validate(greedy_construction=True)
                self.logger.info("Sampling:")
                self._validate(greedy_construction=False)
                self.logger.info("Sampling Aug:")
                self._validate(greedy_construction=False, use_augmentation=True)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']

            self.logger.info("Saving trained_model")
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'result_log': self.result_log.get_raw_data(),
                'z_dim': self.model_params['z_dim'],
                'force_first_move': self.model_params["force_first_move"],
                'model_params': self.model_params,
                'advantage_params': self.advantage_params,
            }
            torch.save(checkpoint_dict, '{}/checkpoint_latest.pt'.format(self.result_folder, epoch))
            if all_done or (epoch % model_save_interval) == 0:
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self._get_train_num_episode()

        episode = 0
        loop_cnt = 0
        batch_in_epoch = 0
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            self.current_advantage_params = self._resolve_effective_advantage_params(epoch, batch_in_epoch=batch_in_epoch)

            if hasattr(self.model, 'set_module_slow_start_progress'):
                module_progress = self._resolve_module_slow_start_progress(epoch, batch_in_epoch=batch_in_epoch)
                self.model.set_module_slow_start_progress(module_progress)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size
            batch_in_epoch += 1

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    progress_log = self._resolve_module_slow_start_progress(epoch, batch_in_epoch=batch_in_epoch - 1)
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f},  ModuleProgress: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg, progress_log))

        progress_log = self._resolve_module_slow_start_progress(epoch, batch_in_epoch=max(batch_in_epoch - 1, 0))
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  ModuleProgress: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg, progress_log))
        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):
        K = self.trainer_params['K']
        z_dim = self.model_params['z_dim']
        amp_training = self.trainer_params['amp_training']
        device = "cuda" if self.trainer_params['use_cuda'] else "cpu"

        if self.model_params['force_first_move']:
            starting_points = self.env_params['problem_size']
            rollout_size = starting_points * K
        else:
            starting_points = 1
            rollout_size = K

        self.model.train()
        self.env.load_problems(batch_size, rollout_size)
        reset_state, _, _ = self.env.reset()

        z = self.sample_z_vectors(batch_size, starting_points, z_dim, K, rollout_size)
        self.model.pre_forward(reset_state, z)

        prob_list = []
        state, reward, done = self.env.pre_step()

        with torch.amp.autocast(device_type=device, enabled=amp_training):
            while not done:
                selected, prob = self.model(state)
                state, reward, done = self.env.step(selected)
                prob_list.append(prob)

        reward_pop = reward.reshape(batch_size, K, -1)
        advantage = self._compute_advantage(reward_pop, self.current_advantage_params).reshape(batch_size, -1)
        prob_list = torch.stack(prob_list, dim=2)
        log_prob = prob_list.log().sum(dim=2)

        mask = self._compute_rollout_mask(reward_pop, self.current_advantage_params)
        if mask is not None:
            log_prob = log_prob * mask.reshape(batch_size, -1)

        loss = -advantage * log_prob
        loss_mean = loss.mean()

        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        self.model.zero_grad(set_to_none=True)
        if not amp_training:
            loss_mean.backward()
            self.optimizer.step()
        else:
            self.scaler.scale(loss_mean).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return score_mean.item(), loss_mean.item()

    def _compute_advantage(self, reward_pop, advantage_params):
        rollout_dim = 2 if self.model_params['force_first_move'] else 1
        mode = advantage_params['mode']
        if mode == 'group_mean':
            advantage = reward_pop - reward_pop.mean(dim=rollout_dim, keepdim=True)
        elif mode == 'matched_soft_top1':
            costs = -reward_pop
            advantage = self._matched_soft_top1_advantage(costs, rollout_dim, advantage_params)
        else:
            raise ValueError(f"Unsupported advantage mode: {mode}")

        if advantage_params['normalize_adv']:
            advantage = advantage / (advantage.std() + advantage_params['eps'])
        return advantage

    def _matched_soft_top1_advantage(self, costs, rollout_dim, advantage_params):
        tau = advantage_params['tau']
        reward_scale = advantage_params['reward_scale']
        eps = advantage_params['eps']
        if tau <= 0:
            raise ValueError("advantage_params['tau'] must be > 0 for matched_soft_top1")
        if reward_scale <= 0:
            raise ValueError("advantage_params['reward_scale'] must be > 0 for matched_soft_top1")

        c_hat = costs / reward_scale
        group_size = c_hat.size(rollout_dim)
        if group_size <= 1:
            raise ValueError("matched_soft_top1 requires at least two rollouts in the active advantage dimension")

        scaled = -c_hat / tau
        max_scaled = scaled.max(dim=rollout_dim, keepdim=True).values
        exp_scaled = torch.exp(scaled - max_scaled)
        exp_sum = exp_scaled.sum(dim=rollout_dim, keepdim=True)
        loo_exp_sum = (exp_sum - exp_scaled).clamp_min(eps)

        log_mean_all = torch.logsumexp(scaled, dim=rollout_dim, keepdim=True) - math.log(group_size)
        log_mean_loo = max_scaled + torch.log(loo_exp_sum) - math.log(group_size - 1)

        m_all = -tau * log_mean_all
        m_loo = -tau * log_mean_loo
        return (group_size - 1) * (m_loo - m_all)

    def _compute_rollout_mask(self, reward_pop, advantage_params):
        if advantage_params['rollout_mask_mode'] == 'dense':
            return None
        costs = -reward_pop
        rollout_dim = 2 if self.model_params['force_first_move'] else 1
        best_idx = costs.argsort(dim=rollout_dim).argsort(dim=rollout_dim)
        return (best_idx < 1).to(reward_pop.dtype)

    def _validate(self, greedy_construction=False, use_augmentation=False):
        val_num_episode = self.trainer_params['val_episodes']
        z_sample_size = self.trainer_params['val_z_sample_size']
        z_dim = self.model_params['z_dim']
        batch_size = self.trainer_params['val_batch_size']

        if self.model_params['force_first_move']:
            starting_points = self.env_params['problem_size']
            rollout_size = starting_points * z_sample_size
        else:
            starting_points = 1
            rollout_size = z_sample_size

        aug_factor = 8 if use_augmentation else 1
        if use_augmentation:
            batch_size = max(1, batch_size // aug_factor)

        self.model.eval()
        val_env = Env(**self.env_params)
        if self.trainer_params['validation_data_load']['enable']:
            val_env.use_saved_problems(self.trainer_params['validation_data_load']['filename'], 'cpu')

        costs = torch.zeros(size=(0, starting_points, z_sample_size * aug_factor))
        mean_log_prob = []
        episode = 0
        while episode < val_num_episode:
            remaining = val_num_episode - episode
            batch_size = min(batch_size, remaining)
            val_env.load_problems(batch_size, rollout_size, aug_factor)
            episode += batch_size

            with torch.no_grad():
                reset_state, _, _ = val_env.reset()
                z = self.sample_z_vectors(batch_size * aug_factor, starting_points, z_dim, z_sample_size, rollout_size)
                self.model.pre_forward(reset_state, z)

                state, reward, done = val_env.pre_step()
                prob_list = []
                while not done:
                    selected, prob = self.model(state, greedy_construction)
                    state, reward, done = val_env.step(selected)
                    prob_list.append(prob)

                reward = reward.reshape(aug_factor, batch_size, z_sample_size, starting_points).transpose(0, 1)
                reward = reward.reshape(batch_size, aug_factor * z_sample_size, starting_points).transpose(1, 2)
                costs = torch.cat((costs, -reward), dim=0)
                prob_list = torch.stack(prob_list, dim=2)
                mean_log_prob.append(prob_list.log().sum(2).mean().item())

        sorted_costs = costs.sort(dim=2).values
        unique_counts = torch.ones_like(sorted_costs[:, :, 0], dtype=torch.float32)
        unique_counts += (sorted_costs[:, :, 1:] != sorted_costs[:, :, :-1]).sum(dim=2).to(dtype=torch.float32)
        mean_unique = unique_counts.mean() / (aug_factor * z_sample_size)
        cost_best = costs.min(dim=2)[0].mean()
        cost_pomo = costs.min(dim=2)[0].min(dim=1)[0].mean()
        self.logger.info(
            f'Log prob: {np.array(mean_log_prob).mean():.4f} Percentage of unique costs: {mean_unique:.3f} Costs (mean, best, best pomo): {costs.mean():.4f} {cost_best:.4f} {cost_pomo:.4f}')

    def sample_z_vectors(self, batch_size, starting_points, z_dim, z_sample_size, rollout_size):
        if 2 ** z_dim == rollout_size:
            z = self.binary_string_pool[None].expand(batch_size, rollout_size, z_dim)
        else:
            z_idx = torch.multinomial(
                (torch.ones(batch_size * starting_points, 2 ** z_dim) / 2 ** z_dim),
                z_sample_size,
                replacement=z_sample_size > 2 ** z_dim,
            )
            z = self.binary_string_pool[z_idx].reshape(batch_size, starting_points, z_sample_size, z_dim)
            z = z.reshape(batch_size, rollout_size, z_dim)
        return z.to(self.device)
