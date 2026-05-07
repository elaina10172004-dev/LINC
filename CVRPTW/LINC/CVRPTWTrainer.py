
import json
import math
import os
import random
from pathlib import Path
from logging import getLogger

import torch
import vrptw_data  # noqa: F401  Ensures torch.load can unpickle local shard instances.

from CVRPTWEnv import CVRPTWEnv as Env, _accurate_cdist
from CVRPTWModel import CANDIDATE_FEATURE_INDEX, CVRPTWModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *
import itertools


class CVRPTWTrainer:
    def __init__(self,
                 run_params,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        if self.trainer_params['compile_model']:
            self.model = torch.compile(self.model)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.scaler = torch.cuda.amp.GradScaler()

        # Restore
        self.start_epoch = 1
        self.corrector_params = self._resolve_corrector_params()
        if run_params["name"] is not None:
            # try to resume named training run
            model_load_path = os.path.join("result", "run_" + run_params["name"], "checkpoint_latest.pt")
            if os.path.exists(model_load_path):
                checkpoint = torch.load(model_load_path, map_location=self.device)
                self._load_model_state_compatible(checkpoint['model_state_dict'], strict=not self.corrector_params['enable'])
                self.start_epoch = 1 + checkpoint['epoch']
                self.result_log.set_raw_data(checkpoint['result_log'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.scheduler.last_epoch = self.start_epoch - 1
                self.logger.info('Resuming named training run!')

        if trainer_params['model_load']['enable'] and self.start_epoch == 1:
            # start new training run from POMO base model or continue PolyNet training
            model_load_path = '{path}/checkpoint-{epoch}.pt'.format(**trainer_params['model_load'])
            checkpoint = torch.load(model_load_path, map_location=self.device)
            if "decoder.poly_layer_1.weight" in checkpoint['model_state_dict'].keys():
                # Loaded model is PolyNet model
                # self.start_epoch = 1 + checkpoint['epoch']
                self._load_model_state_compatible(checkpoint['model_state_dict'], strict=not self.corrector_params['enable'])
            else:
                self._load_model_state_compatible(checkpoint['model_state_dict'], strict=False)
            self.logger.info('Saved Model Loaded!')

        # utility
        self.time_estimator = TimeEstimator()

        self.binary_string_pool = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=model_params['z_dim'])])
        self.problem_size_list = self._resolve_problem_size_list()
        self.distribution_list = self._resolve_distribution_list()
        self.use_fixed_task_batches = self._should_use_fixed_task_batches()
        self.advantage_params = self._resolve_advantage_params()
        self.train_shards_params = self._resolve_train_shards_params()
        self.train_batch_size_schedule = self._resolve_size_schedule(
            self.trainer_params.get('train_batch_size_schedule', []),
            self.trainer_params['train_batch_size'],
            key_name='batch_size',
        )
        self.train_subbatch_size_schedule = self._resolve_size_schedule(
            self.trainer_params.get('train_subbatch_size_schedule', []),
            None,
            key_name='subbatch_size',
        )
        self.use_train_shards = self.train_shards_params['enable']
        self.train_shard_files = self._list_train_shards(self.train_shards_params['shard_dir']) if self.use_train_shards else []
        self.train_shard_cache_dir = None
        if self.use_train_shards and self.train_shard_files:
            self.train_shard_cache_dir = self.train_shard_files[0].parent / f".tensor_cache_p{int(self.env_params['problem_size'])}"
            self.train_shard_cache_dir.mkdir(parents=True, exist_ok=True)
        self.best_train_score = self._resolve_initial_best_train_score()
        self.best_train_score_epoch = self._resolve_initial_best_train_score_epoch()
        self._last_candidate_aux_metrics = None

    def _resolve_advantage_params(self):
        advantage_params = self.trainer_params.get('advantage_params', {})
        return {
            'mode': advantage_params.get('mode', 'group_mean'),
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

    def _resolve_corrector_params(self):
        cfg = self.trainer_params.get('corrector_params', {})
        enabled = bool(self.model_params.get('use_learned_corrector', False)) and bool(cfg.get('enable', True))
        return {
            'enable': enabled,
            'interval_steps': int(cfg.get('interval_steps', 16)),
            'rounds': int(cfg.get('rounds', 1)),
            'min_selected_count': int(cfg.get('min_selected_count', 8)),
            'lambda': float(cfg.get('lambda', 1.0)),
            'count_temperature': float(cfg.get('count_temperature', 1.0)),
            'step_temperature': float(cfg.get('step_temperature', 1.0)),
            'greedy_eval': bool(cfg.get('greedy_eval', True)),
            'max_removals': int(self.model_params.get('corrector_max_removals', cfg.get('max_removals', 4))),
        }

    def _load_model_state_compatible(self, checkpoint_state_dict, strict):
        if strict:
            self.model.load_state_dict(checkpoint_state_dict, strict=True)
            return

        model_state = self.model.state_dict()
        compatible_state = {
            key: value
            for key, value in checkpoint_state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        self.model.load_state_dict(compatible_state, strict=False)

    def _resolve_train_shards_params(self):
        shard_params = self.trainer_params.get('train_shards_load', {})
        return {
            'enable': bool(shard_params.get('enable', False)),
            'shard_dir': shard_params.get('shard_dir', None),
            'shuffle_each_epoch': bool(shard_params.get('shuffle_each_epoch', True)),
        }

    def _resolve_size_schedule(self, schedule_items, default_value, key_name):
        normalized = []
        for item in schedule_items or []:
            if not isinstance(item, dict):
                continue
            end_epoch = int(item.get('end_epoch', 0))
            value = item.get(key_name, None)
            if end_epoch <= 0 or value is None:
                continue
            normalized.append({
                'end_epoch': end_epoch,
                key_name: int(value),
            })
        normalized.sort(key=lambda x: x['end_epoch'])
        if default_value is not None:
            normalized.append({
                'end_epoch': math.inf,
                key_name: int(default_value),
            })
        return normalized

    def _resolve_initial_best_train_score(self):
        if self.result_log.has_key('train_score'):
            scores = self.result_log.get('train_score')
            if isinstance(scores, list):
                return float(min(scores))
            return float(scores)
        return math.inf

    def _resolve_initial_best_train_score_epoch(self):
        if self.result_log.has_key('train_score'):
            xs, ys = self.result_log.getXY('train_score')
            if isinstance(ys, list):
                best_idx = min(range(len(ys)), key=lambda i: ys[i])
                return int(xs[best_idx])
            return int(xs)
        return 0

    def _build_candidate_feature_checkpoint_metadata(self):
        return {
            'selected_candidate_feature_names': list(self.model_params.get('selected_candidate_feature_names', [])),
            'relative_candidate_feature_names': list(self.model_params.get('relative_candidate_feature_names', [])),
            'zero_depot_relative_features': bool(self.model_params.get('zero_depot_relative_features', False)),
        }

    def _verify_candidate_feature_checkpoint_metadata(self, checkpoint_dict):
        expected = self._build_candidate_feature_checkpoint_metadata()
        for key, value in expected.items():
            if checkpoint_dict.get(key) != value:
                raise AssertionError(f"Checkpoint metadata mismatch for {key}: {checkpoint_dict.get(key)!r} != {value!r}")

    def _empty_candidate_aux_epoch_stats(self):
        return {
            'sample_weight': 0,
            'feasible_ratio_sum': 0.0,
            'feasible_ratio_count': 0,
            'gate_mean_sum': 0.0,
            'gate_std_sum': 0.0,
            'gate_count': 0,
        }

    def _update_candidate_aux_epoch_stats(self, stats, metrics, batch_size):
        if not metrics:
            return
        if metrics.get('feasible_ratio') is not None:
            stats['feasible_ratio_sum'] += float(metrics['feasible_ratio']) * batch_size
            stats['feasible_ratio_count'] += batch_size
        if metrics.get('gate_mean') is not None and metrics.get('gate_std') is not None:
            stats['gate_mean_sum'] += float(metrics['gate_mean']) * batch_size
            stats['gate_std_sum'] += float(metrics['gate_std']) * batch_size
            stats['gate_count'] += batch_size
        stats['sample_weight'] += batch_size

    def _format_candidate_aux_epoch_stats(self, stats):
        parts = []
        if stats['feasible_ratio_count'] > 0:
            parts.append(f"FeasRatio:{stats['feasible_ratio_sum'] / stats['feasible_ratio_count']:.4f}")
        if stats['gate_count'] > 0:
            parts.append(
                "Gate:{:.4f}/{:.4f}".format(
                    stats['gate_mean_sum'] / stats['gate_count'],
                    stats['gate_std_sum'] / stats['gate_count'],
                )
            )
        return "  " + "  ".join(parts) if parts else ""

    def _get_scheduled_scalar(self, schedule, epoch, key_name, fallback):
        if epoch is None:
            return int(fallback)
        for item in schedule:
            if int(epoch) <= item['end_epoch']:
                return int(item[key_name])
        return int(fallback)

    def _get_train_batch_size(self, epoch):
        return self._get_scheduled_scalar(
            self.train_batch_size_schedule,
            epoch,
            'batch_size',
            self.trainer_params['train_batch_size'],
        )

    def _get_train_subbatch_by_size(self, epoch):
        default_sizes = self.trainer_params['train_subbatch_by_size']
        if not self.train_subbatch_size_schedule:
            return list(default_sizes)
        scheduled_value = self._get_scheduled_scalar(
            self.train_subbatch_size_schedule,
            epoch,
            'subbatch_size',
            default_sizes[0],
        )
        return [scheduled_value for _ in default_sizes]

    def _estimate_train_batches_per_epoch(self, epoch, train_num_episode=None):
        if self.use_train_shards:
            num_episodes = int(train_num_episode) if train_num_episode is not None else int(self._load_shard_epoch_data(epoch)['num_instances'])
            return max(int(math.ceil(float(num_episodes) / float(self._get_train_batch_size(epoch)))), 1)

        if self.use_fixed_task_batches:
            batch_single_by_size = self.trainer_params['batch_single_by_size']
            train_subbatch_by_size = self._get_train_subbatch_by_size(epoch)
            total_batches = 0
            for size_idx in range(len(self.problem_size_list)):
                task_batch = int(batch_single_by_size[size_idx])
                max_subbatch = int(train_subbatch_by_size[size_idx])
                total_batches += int(math.ceil(float(task_batch) / float(max_subbatch))) * len(self.distribution_list)
            return max(total_batches, 1)

        base_problem_size = int(self.problem_size_list[0])
        if self.model_params['force_first_move']:
            solutions_per_instance = self.trainer_params["K"] * base_problem_size
        else:
            solutions_per_instance = self.trainer_params["K"]
        train_num_episode = int(self.trainer_params['train_num_rollouts'] / solutions_per_instance)
        return max(int(math.ceil(float(train_num_episode) / float(self._get_train_batch_size(epoch)))), 1)

    def _resolve_effective_advantage_params(self, epoch, batch_in_epoch=0, batches_per_epoch=None):
        effective = dict(self.advantage_params)
        if epoch is None or self.trainer_params['epochs'] <= 0:
            effective['tau'] = effective['tau_end']
            return effective

        if bool(self.trainer_params.get('advantage_schedule_by_batch', True)):
            batches_per_epoch = int(batches_per_epoch or self._estimate_train_batches_per_epoch(epoch))
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

    def _resolve_problem_size_list(self):
        if 'problem_size_list' in self.env_params:
            return [int(x) for x in self.env_params['problem_size_list']]
        problem_size = self.env_params['problem_size']
        if isinstance(problem_size, (list, tuple)):
            return [int(x) for x in problem_size]
        return [int(problem_size)]

    def _resolve_distribution_list(self):
        if 'distribution_list' in self.env_params:
            return list(self.env_params['distribution_list'])
        distribution = self.env_params.get('distribution', {})
        return [distribution.get('data_type', 'uniform')]

    def _should_use_fixed_task_batches(self):
        if self.trainer_params.get('train_shards_load', {}).get('enable', False):
            return False
        if self.trainer_params.get('epoch_volume_rule', '') != 'fixed_task_batches':
            return False
        if self.trainer_params['train_data_load']['enable']:
            self.logger.info("epoch_volume_rule=fixed_task_batches is ignored when train_data_load is enabled.")
            return False
        return True

    def _list_train_shards(self, shard_dir):
        if not shard_dir:
            raise ValueError("train_shards_load.enable=True requires a non-empty shard_dir")

        shard_root = Path(shard_dir)
        if not shard_root.is_absolute():
            shard_root = (Path.cwd() / shard_root).resolve()
        if not shard_root.exists():
            raise FileNotFoundError(f"Shard directory not found: {shard_root}")

        shard_files = []
        manifest_path = shard_root / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for item in manifest.get("files", []):
                    rel = item.get("file")
                    if not rel:
                        continue
                    file_path = shard_root / rel
                    if file_path.exists():
                        shard_files.append(file_path)
            except Exception:
                shard_files = []

        if not shard_files:
            shard_files = sorted(shard_root.glob("shard_*.pt"))

        if not shard_files:
            raise FileNotFoundError(f"No shard_*.pt files found in {shard_root}")

        self.logger.info("Using shard training data from %s (%d shards)", str(shard_root), len(shard_files))
        return shard_files

    def _load_shard_epoch_data(self, epoch):
        shard_idx = (int(epoch) - 1) % len(self.train_shard_files)
        shard_file = self.train_shard_files[shard_idx]
        batch_data = self._load_or_build_shard_tensor_data(shard_file)
        if self.train_shards_params['shuffle_each_epoch']:
            batch_data = self._shuffle_tensor_batch(batch_data, epoch=epoch, shard_idx=shard_idx)
        batch_data['shard_name'] = shard_file.name
        return batch_data

    def _load_or_build_shard_tensor_data(self, shard_file):
        cache_path = None
        if self.train_shard_cache_dir is not None:
            cache_path = self.train_shard_cache_dir / f"{shard_file.stem}.tensor.pt"
            if cache_path.exists():
                loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
                if isinstance(loaded, dict) and 'num_instances' in loaded:
                    return self._ensure_tensor_batch_cpu(loaded)

        loaded = torch.load(shard_file, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and 'num_instances' in loaded:
            batch_data = loaded
        elif isinstance(loaded, list):
            batch_data = self._convert_instances_to_tensors(loaded, int(self.env_params['problem_size']))
        else:
            raise TypeError(f"Expected list or tensor dict in shard file, got {type(loaded)}")

        batch_data = self._ensure_tensor_batch_cpu(batch_data)

        if cache_path is not None and not cache_path.exists():
            tmp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
            try:
                torch.save(batch_data, tmp_path)
                os.replace(tmp_path, cache_path)
            except Exception:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
        return batch_data

    def _ensure_tensor_batch_cpu(self, batch_data):
        normalized = {}
        for key, value in batch_data.items():
            if torch.is_tensor(value):
                normalized[key] = value.detach().to(device='cpu')
            else:
                normalized[key] = value
        return normalized

    def _shuffle_tensor_batch(self, batch_data, epoch, shard_idx):
        num_instances = int(batch_data['num_instances'])
        generator = torch.Generator(device='cpu')
        generator.manual_seed(10_000_019 * int(epoch) + int(shard_idx))
        order = torch.randperm(num_instances, generator=generator, device='cpu')
        shuffled = {}
        for key, value in batch_data.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == num_instances:
                local_order = order if value.device.type == 'cpu' else order.to(value.device)
                shuffled[key] = value.index_select(0, local_order)
            else:
                shuffled[key] = value
        return shuffled

    def _convert_instances_to_tensors(self, instances, problem_size):
        usable = []
        for inst in instances:
            customers = list(getattr(inst, 'customers', []))
            if len(customers) >= problem_size:
                usable.append(inst)

        if not usable:
            raise RuntimeError(f"No usable instances with at least {problem_size} customers.")

        depot_xy = []
        node_xy = []
        node_demand = []
        node_tw = []
        service_duration = []
        capacity = []
        depot_tw = []
        travel_time_scale = []
        grid_size_values = []

        for inst in usable:
            depot = getattr(inst, 'depot')
            customers = list(getattr(inst, 'customers'))[:problem_size]
            service_values = {float(c.service_time) for c in customers}
            if len(service_values) != 1:
                raise ValueError(f"Instance {getattr(inst, 'name', '<unnamed>')} has mixed service times.")

            depot_xy.append([[float(depot.x), float(depot.y)]])
            node_xy.append([[float(c.x), float(c.y)] for c in customers])
            node_demand.append([float(c.demand) for c in customers])
            node_tw.append([[float(c.ready_time), float(c.due_time)] for c in customers])
            service_duration.append(float(customers[0].service_time))
            capacity.append(float(getattr(inst, 'capacity')))
            depot_tw.append([[float(depot.ready_time), float(depot.due_time)]])
            travel_time_scale.append(float(getattr(inst, 'travel_time_scale', 1.0)))

            local_max_coord = max(
                float(depot.x),
                float(depot.y),
                max(float(c.x) for c in customers),
                max(float(c.y) for c in customers),
            )
            instance_grid_size = getattr(inst, 'grid_size', None)
            if instance_grid_size is None:
                instance_grid_size = float(max(100.0, math.ceil(local_max_coord)))
            grid_size_values.append(float(instance_grid_size))

        grid_size = float(max(grid_size_values))
        return {
            'depot_xy': torch.tensor(depot_xy, dtype=torch.float32),
            'node_xy': torch.tensor(node_xy, dtype=torch.float32),
            'node_demand': torch.tensor(node_demand, dtype=torch.float32),
            'node_tw': torch.tensor(node_tw, dtype=torch.float32),
            'depot_tw': torch.tensor(depot_tw, dtype=torch.float32),
            'service_t': torch.tensor(service_duration, dtype=torch.float32),
            'capacity': torch.tensor(capacity, dtype=torch.float32),
            'travel_time_scale': torch.tensor(travel_time_scale, dtype=torch.float32),
            'grid_size': grid_size,
            'num_instances': len(usable),
        }

    def _load_tensor_batch_into_env(self, batch_data, start_idx, batch_size, rollout_size):
        end_idx = start_idx + batch_size
        depot_xy = batch_data['depot_xy'][start_idx:end_idx].to(self.device)
        node_xy = batch_data['node_xy'][start_idx:end_idx].to(self.device)
        node_demand = batch_data['node_demand'][start_idx:end_idx].to(self.device)
        node_tw = batch_data['node_tw'][start_idx:end_idx].to(self.device)
        capacity = batch_data['capacity'][start_idx:end_idx].to(self.device)
        service_t = batch_data['service_t'][start_idx:end_idx].to(self.device)[:, None]
        depot_tw = batch_data.get('depot_tw', None)
        if depot_tw is not None:
            depot_tw = depot_tw[start_idx:end_idx].to(self.device)
        travel_time_scale = batch_data.get('travel_time_scale', None)
        if travel_time_scale is None:
            travel_time_scale = torch.ones((batch_size,), dtype=torch.float32, device=self.device)
        else:
            travel_time_scale = travel_time_scale[start_idx:end_idx].to(self.device)

        self.env.batch_size = batch_size
        self.env.rollout_size = rollout_size
        self.env.problem_size = int(node_xy.size(1))
        self.env.grid_size_tensor = self.env._normalize_grid_size(
            batch_data['grid_size'],
            batch_size,
            self.device,
        )
        self.env.grid_size = float(self.env.grid_size_tensor.float().mean().item())
        self.env.service_t = service_t
        self.env.travel_time_scale = travel_time_scale

        self.env.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        self.env.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        if depot_tw is None:
            depot_tw = torch.tensor([0.0, float('inf')], dtype=torch.float32, device=self.device)[None, None].expand(batch_size, 1, 2)
        self.env.depot_node_tw = torch.cat((depot_tw, node_tw), dim=1)
        self.env.distance_matrix = _accurate_cdist(self.env.depot_node_xy, self.env.depot_node_xy)
        self.env.travel_time_matrix = self.env.distance_matrix * travel_time_scale[:, None, None]
        self.env._prepare_candidate_static_cache()
        self.env.BATCH_IDX = torch.arange(batch_size, device=self.device)[:, None].expand(batch_size, rollout_size)
        self.env.ROLLOUT_IDX = torch.arange(rollout_size, device=self.device)[None, :].expand(batch_size, rollout_size)
        self.env.depot_node_demand = self.env.depot_node_demand / capacity[:, None]

        grid_xy_scale = self.env.grid_size_tensor[:, None, None]
        self.env.reset_state.depot_xy = depot_xy / grid_xy_scale
        self.env.reset_state.node_xy = node_xy / grid_xy_scale
        self.env.reset_state.node_demand = self.env.depot_node_demand[:, 1:]
        self.env.reset_state.node_tw = node_tw / grid_xy_scale
        self.env.step_state.BATCH_IDX = self.env.BATCH_IDX
        self.env.step_state.ROLLOUT_IDX = self.env.ROLLOUT_IDX

    def _is_full_gradient_epoch(self, epoch):
        ratio = float(self.trainer_params.get('early_full_grad_ratio', 0.0))
        if ratio <= 0:
            return False
        early_end_epoch = max(1, math.floor(self.trainer_params['epochs'] * ratio))
        return epoch <= early_end_epoch

    def run(self):
        self.time_estimator.reset(self.start_epoch)

        # Load training data
        if self.use_train_shards and self.trainer_params['train_data_load']['enable']:
            self.logger.info("train_data_load is ignored because train_shards_load is enabled.")
        elif self.trainer_params['train_data_load']['enable']:
            self.env.use_saved_problems(self.trainer_params['train_data_load']['filename'], 'cpu')

        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            effective_advantage_params = self._resolve_effective_advantage_params(
                epoch,
                batch_in_epoch=0,
                batches_per_epoch=self._estimate_train_batches_per_epoch(epoch),
            )
            self.logger.info(
                "Advantage Mode: %s, tau=%.4f, rollout_mask=%s"
                % (
                    effective_advantage_params['mode'],
                    effective_advantage_params['tau'],
                    effective_advantage_params['rollout_mask_mode'],
                )
            )
            self.logger.info(
                "Train Batch Schedule: batch_size=%d, subbatch_size=%s",
                self._get_train_batch_size(epoch),
                self._get_train_subbatch_by_size(epoch),
            )

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            # LR Decay
            self.scheduler.step()

            if self.trainer_params.get('enable_epoch_validation', True):
                # Validate
                self.logger.info("Starting validation")
                self.logger.info("Greedy:")
                self._validate(greedy_construction=True)
                self.logger.info("Sampling:")
                self._validate(greedy_construction=False)
                self.logger.info("Sampling Aug:")
                self._validate(greedy_construction=False, use_augmentation=True)


            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            is_new_best = train_score < self.best_train_score
            if is_new_best:
                self.best_train_score = float(train_score)
                self.best_train_score_epoch = int(epoch)

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
                'corrector_params': self.corrector_params,
                'best_train_score': self.best_train_score,
                'best_train_score_epoch': self.best_train_score_epoch,
            }
            checkpoint_dict.update(self._build_candidate_feature_checkpoint_metadata())
            self._verify_candidate_feature_checkpoint_metadata(checkpoint_dict)
            latest_save_interval = max(1, int(self.trainer_params['logging'].get('latest_save_interval', model_save_interval)))
            save_latest_on_best = bool(self.trainer_params['logging'].get('save_latest_on_best', True))
            save_best_checkpoint = bool(self.trainer_params['logging'].get('save_best_checkpoint', True))
            should_save_latest = all_done or (epoch % latest_save_interval) == 0 or (save_latest_on_best and is_new_best)
            should_save_epoch = all_done or (epoch % model_save_interval) == 0

            if should_save_latest or should_save_epoch or (is_new_best and save_best_checkpoint):
                self.logger.info("Saving trained_model")

            if should_save_latest:
                torch.save(checkpoint_dict, '{}/checkpoint_latest.pt'.format(self.result_folder, epoch))
            if should_save_epoch:
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))
            if is_new_best and save_best_checkpoint:
                torch.save(checkpoint_dict, '{}/checkpoint_best_train_score.pt'.format(self.result_folder))
                self.logger.info(
                    "New best mean_score: epoch=%d score=%.4f",
                    self.best_train_score_epoch,
                    self.best_train_score,
                )

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        if self.use_train_shards:
            return self._train_one_epoch_shards(epoch)
        if self.use_fixed_task_batches:
            return self._train_one_epoch_fixed_task_batches(epoch)
        return self._train_one_epoch_rollouts(epoch)

    def _train_one_epoch_shards(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        aux_stats = self._empty_candidate_aux_epoch_stats()
        epoch_data = self._load_shard_epoch_data(epoch)
        train_num_episode = int(epoch_data['num_instances'])
        batches_per_epoch = self._estimate_train_batches_per_epoch(epoch, train_num_episode=train_num_episode)
        episode = 0
        loop_cnt = 0
        batch_in_epoch = 0

        self.logger.info("Epoch %3d: shard=%s instances=%d", epoch, epoch_data['shard_name'], train_num_episode)

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(int(self._get_train_batch_size(epoch)), remaining)

            avg_score, avg_loss = self._train_one_batch(
                batch_size=batch_size,
                epoch=epoch,
                tensor_data=epoch_data,
                tensor_start_idx=episode,
                batch_in_epoch=batch_in_epoch,
                batches_per_epoch=batches_per_epoch,
            )
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            self._update_candidate_aux_epoch_stats(aux_stats, self._last_candidate_aux_metrics, batch_size)
            episode += batch_size
            batch_in_epoch += 1

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}{}'
                        .format(
                            epoch,
                            episode,
                            train_num_episode,
                            100. * episode / train_num_episode,
                            score_AM.avg,
                            loss_AM.avg,
                            self._format_candidate_aux_epoch_stats(aux_stats),
                        )
                    )

        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}{}  Shard:{}'
            .format(
                epoch,
                100. * episode / train_num_episode,
                score_AM.avg,
                loss_AM.avg,
                self._format_candidate_aux_epoch_stats(aux_stats),
                epoch_data['shard_name'],
            )
        )
        return score_AM.avg, loss_AM.avg

    def _train_one_epoch_rollouts(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        aux_stats = self._empty_candidate_aux_epoch_stats()
        base_problem_size = int(self.problem_size_list[0])

        if self.model_params['force_first_move']:
            solutions_per_instance = self.trainer_params["K"] * base_problem_size
        else:
            solutions_per_instance = self.trainer_params["K"]

        # Number of batches per epoch
        train_num_episode = int(self.trainer_params['train_num_rollouts'] / solutions_per_instance)

        episode = 0
        loop_cnt = 0
        batches_per_epoch = self._estimate_train_batches_per_epoch(epoch)
        batch_in_epoch = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self._get_train_batch_size(epoch), remaining)

            avg_score, avg_loss = self._train_one_batch(
                batch_size,
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                batches_per_epoch=batches_per_epoch,
            )
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            self._update_candidate_aux_epoch_stats(aux_stats, self._last_candidate_aux_metrics, batch_size)

            episode += batch_size
            batch_in_epoch += 1

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}{}'
                        .format(
                            epoch,
                            episode,
                            train_num_episode,
                            100. * episode / train_num_episode,
                            score_AM.avg,
                            loss_AM.avg,
                            self._format_candidate_aux_epoch_stats(aux_stats),
                        )
                    )

        # Log Once, for each epoch
        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}{}'
            .format(
                epoch,
                100. * episode / train_num_episode,
                score_AM.avg,
                loss_AM.avg,
                self._format_candidate_aux_epoch_stats(aux_stats),
            )
        )

        return score_AM.avg, loss_AM.avg

    def _train_one_epoch_fixed_task_batches(self, epoch):
        batch_single_by_size = self.trainer_params['batch_single_by_size']
        train_subbatch_by_size = self._get_train_subbatch_by_size(epoch)
        aux_stats = self._empty_candidate_aux_epoch_stats()
        if len(batch_single_by_size) != len(self.problem_size_list):
            raise ValueError("Length mismatch: batch_single_by_size vs problem_size_list")
        if len(train_subbatch_by_size) != len(self.problem_size_list):
            raise ValueError("Length mismatch: train_subbatch_by_size vs problem_size_list")

        num_tasks = len(self.problem_size_list) * len(self.distribution_list)
        total_instances = sum(batch_single_by_size) * len(self.distribution_list)

        task_score_sum = 0.0
        task_loss_sum = 0.0
        task_cnt = 0
        processed = 0
        batches_per_epoch = self._estimate_train_batches_per_epoch(epoch)
        batch_in_epoch = 0

        for size_idx, problem_size in enumerate(self.problem_size_list):
            task_batch = int(batch_single_by_size[size_idx])
            max_subbatch = int(train_subbatch_by_size[size_idx])
            if task_batch <= 0 or max_subbatch <= 0:
                raise ValueError("batch_single_by_size/train_subbatch_by_size must be positive")

            for distribution in self.distribution_list:
                remaining = task_batch
                this_task_score = 0.0
                this_task_loss = 0.0

                while remaining > 0:
                    subbatch = min(remaining, max_subbatch)
                    avg_score, avg_loss = self._train_one_batch(
                        batch_size=subbatch,
                        epoch=epoch,
                        problem_size=problem_size,
                        distribution=distribution,
                        batch_in_epoch=batch_in_epoch,
                        batches_per_epoch=batches_per_epoch,
                    )
                    self._update_candidate_aux_epoch_stats(aux_stats, self._last_candidate_aux_metrics, subbatch)
                    this_task_score += avg_score * subbatch
                    this_task_loss += avg_loss * subbatch
                    processed += subbatch
                    remaining -= subbatch
                    batch_in_epoch += 1

                this_task_score /= task_batch
                this_task_loss /= task_batch
                task_score_sum += this_task_score
                task_loss_sum += this_task_loss
                task_cnt += 1

        score_mean = task_score_sum / max(task_cnt, 1)
        loss_mean = task_loss_sum / max(task_cnt, 1)
        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}{}  Tasks:{:d}'
            .format(
                epoch,
                100. * processed / max(total_instances, 1),
                score_mean,
                loss_mean,
                self._format_candidate_aux_epoch_stats(aux_stats),
                num_tasks,
            )
        )
        return score_mean, loss_mean

    def _train_one_batch(self, batch_size, epoch=None, problem_size=None, distribution=None, tensor_data=None, tensor_start_idx=0, batch_in_epoch=0, batches_per_epoch=None):
        K = self.trainer_params['K']
        z_dim = self.model_params['z_dim']
        amp_training = self.trainer_params['amp_training']
        device = "cuda" if self.trainer_params['use_cuda'] else "cpu"
        effective_advantage_params = self._resolve_effective_advantage_params(
            epoch,
            batch_in_epoch=batch_in_epoch,
            batches_per_epoch=batches_per_epoch,
        )
        self._last_candidate_aux_metrics = None

        if tensor_data is not None:
            active_problem_size = int(tensor_data['node_xy'].size(1))
        else:
            active_problem_size = int(problem_size) if problem_size is not None else int(self.env.problem_size)
        full_gradient_epoch = self._is_full_gradient_epoch(epoch) if epoch is not None else False

        if self.corrector_params['enable']:
            return self._train_one_batch_with_corrector(
                batch_size=batch_size,
                epoch=epoch,
                problem_size=problem_size,
                distribution=distribution,
                tensor_data=tensor_data,
                tensor_start_idx=tensor_start_idx,
                full_gradient_epoch=full_gradient_epoch,
                active_problem_size=active_problem_size,
                K=K,
                z_dim=z_dim,
                amp_training=amp_training,
                device=device,
                effective_advantage_params=effective_advantage_params,
            )

        if self.model_params['force_first_move']:
            starting_points = active_problem_size
            rollout_size = starting_points * K
        else:
            starting_points = 1
            rollout_size = K


        # Prep
        ###############################################
        self.model.train()
        if tensor_data is not None:
            self._load_tensor_batch_into_env(tensor_data, tensor_start_idx, batch_size, rollout_size)
        else:
            self.env.load_problems(
                batch_size,
                rollout_size,
                self.device,
                problem_size=active_problem_size,
                distribution=distribution,
            )
        reset_state, _, _ = self.env.reset()

        # Sample z vectors
        z = self.sample_z_vectors(batch_size, starting_points, z_dim, K, rollout_size)

        self.model.pre_forward(reset_state, z)

        prob_list = []
        # shape: (batch, rollout, 0~problem)
        feasible_ratio_sum = 0.0
        feasible_ratio_count = 0
        gate_mean_sum = 0.0
        gate_std_sum = 0.0
        gate_count = 0
        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        with torch.amp.autocast(device_type=device, enabled=amp_training):
            while not done:
                selected, prob = self.model(state)
                candidate_aux = getattr(self.model, 'last_candidate_aux_metrics', None)
                if candidate_aux is not None:
                    feasible_ratio = candidate_aux.get('feasible_ratio')
                    if feasible_ratio is not None:
                        feasible_ratio_sum += float(feasible_ratio.float().mean().detach().cpu())
                        feasible_ratio_count += 1
                    gate_mean = candidate_aux.get('gate_mean')
                    gate_std = candidate_aux.get('gate_std')
                    if gate_mean is not None and gate_std is not None:
                        gate_mean_sum += float(gate_mean.detach().cpu())
                        gate_std_sum += float(gate_std.detach().cpu())
                        gate_count += 1
                # shape: (batch, rollout)
                state, reward, done = self.env.step(selected)
                prob_list.append(prob)

        # POMO Loss
        reward_pop = reward.reshape(batch_size, K, -1)
        advantage = self._compute_advantage(reward_pop, effective_advantage_params).reshape(batch_size, -1)
        # shape: (batch, rollout)
        prob_list = torch.stack(prob_list, dim=2)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, rollout)

        mask = self._compute_rollout_mask(reward_pop, full_gradient_epoch, effective_advantage_params)
        if mask is not None:
            log_prob = log_prob * mask.reshape(batch_size, -1)

        policy_loss = (-(advantage * log_prob)).mean()
        loss_mean = policy_loss
        loss = - advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, rollout)

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad(set_to_none=True)

        if not amp_training:
            loss_mean.backward()
            self.optimizer.step()
        else:
            self.scaler.scale(loss_mean).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self._last_candidate_aux_metrics = {
            'feasible_ratio': (feasible_ratio_sum / feasible_ratio_count) if feasible_ratio_count > 0 else None,
            'gate_mean': (gate_mean_sum / gate_count) if gate_count > 0 else None,
            'gate_std': (gate_std_sum / gate_count) if gate_count > 0 else None,
        }
        return score_mean.item(), loss_mean.item()

    def _empty_history_buffers(self, batch_size, rollout_size):
        step_feature_dim = int(self.model.corrector_step_feature_dim if getattr(self.model, 'corrector', None) is not None else len(self.model_params.get('selected_candidate_feature_names', [])) + 5)
        device = self.device
        return {
            'actions': torch.zeros((batch_size, rollout_size, 0), dtype=torch.long, device=device),
            'valid': torch.zeros((batch_size, rollout_size, 0), dtype=torch.bool, device=device),
            'step_features': torch.zeros((batch_size, rollout_size, 0, step_feature_dim), dtype=torch.float32, device=device),
            'path_cost': torch.zeros((batch_size, rollout_size), dtype=torch.float32, device=device),
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
        current_dim = int(step_features.size(-1))
        if current_dim < target_dim:
            pad = torch.zeros(
                (*step_features.shape[:-1], target_dim - current_dim),
                dtype=step_features.dtype,
                device=step_features.device,
            )
            step_features = torch.cat((step_features, pad), dim=-1)
        elif current_dim > target_dim:
            step_features = step_features[..., :target_dim]
        return step_features

    def _append_history_step(self, history, selected, step_features, valid_mask, transition_distance):
        valid_mask = valid_mask.to(dtype=torch.bool)
        selected_store = torch.where(valid_mask, selected, torch.zeros_like(selected))
        step_features_store = step_features * valid_mask[:, :, None].to(dtype=step_features.dtype)
        history['actions'] = torch.cat((history['actions'], selected_store[:, :, None]), dim=2)
        history['valid'] = torch.cat((history['valid'], valid_mask[:, :, None]), dim=2)
        history['step_features'] = torch.cat((history['step_features'], step_features_store[:, :, None, :]), dim=2)
        history['path_cost'] = history['path_cost'] + transition_distance.to(dtype=torch.float32) * valid_mask.to(dtype=torch.float32)
        return history

    def _sample_corrector_decisions(self, count_logits, step_logits, removable_mask, greedy=False):
        batch_size, rollout_size, seq_len = step_logits.shape
        max_removals = int(count_logits.size(-1) - 1)
        device = step_logits.device

        removable_counts = removable_mask.sum(dim=2)
        valid_count_mask = torch.arange(max_removals + 1, device=device)[None, None, :] <= removable_counts[:, :, None]
        masked_count_logits = count_logits.masked_fill(~valid_count_mask, float('-inf'))
        count_log_prob = torch.zeros((batch_size, rollout_size), dtype=torch.float32, device=device)

        if greedy:
            sampled_counts = masked_count_logits.argmax(dim=-1)
        else:
            count_temp = max(float(self.corrector_params['count_temperature']), 1e-6)
            count_probs = torch.softmax(masked_count_logits / count_temp, dim=-1)
            sampled_counts = count_probs.reshape(-1, max_removals + 1).multinomial(1).reshape(batch_size, rollout_size)
            count_log_prob = torch.log(
                count_probs.gather(dim=2, index=sampled_counts[:, :, None]).squeeze(2).clamp_min(1e-12)
            )

        flat_step_logits = step_logits.reshape(batch_size * rollout_size, seq_len)
        flat_work_mask = removable_mask.reshape(batch_size * rollout_size, seq_len).clone()
        flat_removed = torch.zeros_like(flat_work_mask)
        flat_counts = sampled_counts.reshape(-1)
        flat_log_prob = count_log_prob.reshape(-1)

        for remove_idx in range(max_removals):
            active_rows = torch.nonzero(flat_counts > remove_idx, as_tuple=False).squeeze(1)
            if active_rows.numel() == 0:
                break

            masked_step_logits = flat_step_logits[active_rows].masked_fill(~flat_work_mask[active_rows], float('-inf'))
            if greedy:
                picked = masked_step_logits.argmax(dim=1)
            else:
                step_temp = max(float(self.corrector_params['step_temperature']), 1e-6)
                step_probs = torch.softmax(masked_step_logits / step_temp, dim=1)
                picked = step_probs.multinomial(1).squeeze(1)
                flat_log_prob[active_rows] = flat_log_prob[active_rows] + torch.log(
                    step_probs.gather(dim=1, index=picked[:, None]).squeeze(1).clamp_min(1e-12)
                )

            flat_removed[active_rows, picked] = True
            flat_work_mask[active_rows, picked] = False

        removed_mask = flat_removed.reshape(batch_size, rollout_size, seq_len)
        log_prob = flat_log_prob.reshape(batch_size, rollout_size)
        return removed_mask, sampled_counts, log_prob

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

        kept_actions = kept_actions.reshape(batch_size, rollout_size, max_keep)
        kept_valid = kept_valid.reshape(batch_size, rollout_size, max_keep)
        return kept_actions, kept_valid, first_removed

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

    def _train_one_batch_with_corrector(
        self,
        *,
        batch_size,
        epoch,
        problem_size,
        distribution,
        tensor_data,
        tensor_start_idx,
        full_gradient_epoch,
        active_problem_size,
        K,
        z_dim,
        amp_training,
        device,
        effective_advantage_params,
    ):
        self._last_candidate_aux_metrics = None
        if self.model_params['force_first_move']:
            raise NotImplementedError("Learned corrector training does not support force_first_move yet.")

        rollout_size = K
        starting_points = 1
        self.model.train()

        if tensor_data is not None:
            self._load_tensor_batch_into_env(tensor_data, tensor_start_idx, batch_size, rollout_size)
        else:
            self.env.load_problems(
                batch_size,
                rollout_size,
                self.device,
                problem_size=active_problem_size,
                distribution=distribution,
            )
        reset_state, _, _ = self.env.reset()

        z = self.sample_z_vectors(batch_size, starting_points, z_dim, K, rollout_size)
        self.model.pre_forward(reset_state, z)

        history = self._empty_history_buffers(self.env.batch_size, self.env.rollout_size)
        filler_log_prob = torch.zeros((self.env.batch_size, self.env.rollout_size), dtype=torch.float32, device=self.device)
        corrector_log_prob = torch.zeros_like(filler_log_prob)

        correction_round = 0
        steps_since_correction = 0
        state, _, done = self.env.pre_step()

        while True:
            with torch.amp.autocast(device_type=device, enabled=amp_training):
                while not done:
                    selected, prob = self.model(state)

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
                    valid_mask = torch.ones_like(selected, dtype=torch.bool)
                    history = self._append_history_step(history, selected, step_features, valid_mask, transition_distance)
                    filler_log_prob = filler_log_prob + prob.clamp_min(1e-12).log()
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
                removed_mask, _, sampled_log_prob = self._sample_corrector_decisions(
                    count_logits,
                    step_logits,
                    removable_mask,
                    greedy=False,
                )
                corrector_log_prob = corrector_log_prob + sampled_log_prob
                kept_actions, kept_valid, _ = self._truncate_history_before_first_removal(history, removed_mask)
                state, history, done = self._replay_prefix_history(kept_actions, kept_valid)
                correction_round += 1
                steps_since_correction = 0
                continue

            if done:
                break

        final_reward = -history['path_cost']
        reward_pop = final_reward.reshape(batch_size, K, -1)
        advantage = self._compute_advantage(reward_pop, effective_advantage_params).reshape(batch_size, -1)

        combined_log_prob = filler_log_prob + float(self.corrector_params['lambda']) * corrector_log_prob
        mask = self._compute_rollout_mask(reward_pop, full_gradient_epoch, effective_advantage_params)
        if mask is not None:
            combined_log_prob = combined_log_prob * mask.reshape(batch_size, -1)

        policy_loss = (-(advantage * combined_log_prob)).mean()
        max_pomo_reward, _ = final_reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        self.model.zero_grad(set_to_none=True)
        if not amp_training:
            policy_loss.backward()
            self.optimizer.step()
        else:
            self.scaler.scale(policy_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return score_mean.item(), policy_loss.item()

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

        # tau -> inf recovers the current group-mean baseline.
        # tau -> 0 approaches top-1 / elite credit assignment.
        m_all = -tau * log_mean_all
        m_loo = -tau * log_mean_loo
        return (group_size - 1) * (m_loo - m_all)

    def _compute_rollout_mask(self, reward_pop, full_gradient_epoch, advantage_params):
        if full_gradient_epoch or advantage_params['rollout_mask_mode'] == 'dense':
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
        val_problem_size = int(self.problem_size_list[0])

        if self.model_params['force_first_move']:
            starting_points = val_problem_size
            rollout_size = starting_points * z_sample_size
        else:
            starting_points = 1
            rollout_size = z_sample_size

        if use_augmentation:
            aug_factor = 8
            batch_size = max(1, batch_size // aug_factor)
        else:
            aug_factor = 1

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
            val_env.load_problems(batch_size, rollout_size, self.device, aug_factor)
            episode += batch_size

            with torch.no_grad():
                reset_state, _, _ = val_env.reset()

                # Sample z vectors
                z = self.sample_z_vectors(batch_size * aug_factor, starting_points, z_dim, z_sample_size, rollout_size)

                self.model.pre_forward(reset_state, z)

                # POMO Rollout
                ###############################################
                state, reward, done = val_env.pre_step()
                prob_list = []
                while not done:
                    selected, prob = self.model(state, greedy_construction)
                    # shape: (batch, rollout)
                    state, reward, done = val_env.step(selected)
                    prob_list.append(prob)

                reward = reward.reshape(aug_factor, batch_size, z_sample_size, starting_points).transpose(0, 1)
                reward = reward.reshape(batch_size, aug_factor*z_sample_size, starting_points).transpose(1, 2)
                costs = torch.cat((costs, -reward), dim=0)
                prob_list = torch.stack(prob_list, dim=2)
                mean_log_prob.append(prob_list.log().sum(2).mean().item())

        sorted_costs = costs.sort(dim=2).values
        unique_counts = torch.ones_like(sorted_costs[:, :, 0], dtype=torch.float32)
        unique_counts += (sorted_costs[:, :, 1:] != sorted_costs[:, :, :-1]).sum(dim=2).to(dtype=torch.float32)
        mean_unique = unique_counts.mean()/(aug_factor*z_sample_size)
        cost_best = costs.min(dim=2)[0].mean()
        cost_pomo = costs.min(dim=2)[0].min(dim=1)[0].mean()
        self.logger.info(
            f'Log prob: {np.array(mean_log_prob).mean():.4f} Percentage of unique costs: {mean_unique:.3f} Costs (mean, best, best pomo): {costs.mean():.4f} {cost_best:.4f} {cost_pomo:.4f}')


    def sample_z_vectors(self, batch_size, starting_points, z_dim, z_sample_size, rollout_size):

        if 2**z_dim == rollout_size:
            z = self.binary_string_pool[None].expand(batch_size, rollout_size, z_dim)
        else:
            z_idx = torch.multinomial((torch.ones(batch_size * starting_points, 2**z_dim) / 2**z_dim),
                                  z_sample_size, replacement=z_sample_size > 2**z_dim)
            z = self.binary_string_pool[z_idx].reshape(batch_size, starting_points, z_sample_size, z_dim)
            z = z.transpose(1, 2).reshape(batch_size, rollout_size, z_dim)
        return z
