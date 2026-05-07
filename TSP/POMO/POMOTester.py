"""POMO TSP sampling tester."""
import time

import torch
from logging import getLogger
from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from utils import *

class POMOTester:
    def __init__(self, env_params, model_params, tester_params):
        self.tester_params = tester_params
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()
        USE_CUDA = tester_params['use_cuda']
        if USE_CUDA:
            torch.cuda.set_device(tester_params['cuda_device_num'])
            self.device = torch.device('cuda', tester_params['cuda_device_num'])
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.env = Env(**env_params)
        self.model = Model(**model_params)
        ml = tester_params['model_load']
        ckpt = torch.load('{path}/checkpoint-{epoch}.pt'.format(**ml), map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.time_estimator = TimeEstimator()
        self.raw_saved_problems = None
        self.metric = "euclidean"

    def run(self):
        start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.time_estimator.reset()
        score_AM, aug_score_AM = AverageMeter(), AverageMeter()
        rows = []
        total, bs = self.tester_params['test_episodes'], self.tester_params['test_batch_size']
        z, aug = self.tester_params['test_z_sample_size'], self.tester_params['aug_factor'] if self.tester_params['augmentation_enable'] else 1
        if self.tester_params['test_data_load']['enable']:
            self.env.use_pkl_saved_problems(self.tester_params['test_data_load']['filename'], total)
            raw = self.env.saved_problems.float()
            coord_scale = raw.amax(dim=(1, 2)).clamp_min(1.0)
            if bool((coord_scale > 2.0).any().item()):
                self.raw_saved_problems = raw
                self.env.saved_problems = raw / coord_scale[:, None, None]
                self.metric = "tsplib_euc2d"
        ep = 0
        while ep < total:
            b = min(bs, total - ep)
            s, ag, batch_rows = self._test_one_batch(b, z, aug)
            for row in batch_rows:
                row = dict(row)
                row["instance"] = int(ep + row["instance"])
                rows.append(row)
            score_AM.update(s, b); aug_score_AM.update(ag, b)
            ep += b
            el, rm = self.time_estimator.get_est_string(ep, total)
            self.logger.info(f"episode {ep}/{total}, Elapsed[{el}], Remain[{rm}], score:{s:.3f}, aug_score:{ag:.3f}")
        self.logger.info(" *** Test Done *** ")
        self.logger.info(f" NO-AUG SCORE: {score_AM.avg:.4f} ")
        self.logger.info(f" AUGMENTATION SCORE: {aug_score_AM.avg:.4f} ")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        else:
            peak_memory = None
        return {
            "episodes": int(total),
            "batch_size": int(bs),
            "score_mean": float(score_AM.avg),
            "aug_score_mean": float(aug_score_AM.avg),
            "mean_cost": float(aug_score_AM.avg),
            "elapsed_sec": float(time.perf_counter() - start),
            "peak_memory_mb": peak_memory,
            "pomo_size": int(self.env.pomo_size),
            "aug_factor": int(aug),
            "eval_type": self.tester_params.get('eval_type', 'greedy'),
            "metric": self.metric,
            "rows": rows,
        }

    def _test_one_batch(self, batch_size, z_sample_size, aug_factor):
        device = "cuda" if self.tester_params['use_cuda'] else "cpu"
        amp = self.tester_params.get('amp_inference', True)
        self.model.eval()
        with torch.no_grad():
            raw_batch = None
            if self.raw_saved_problems is not None:
                raw_batch = self.raw_saved_problems[self.env.saved_index:self.env.saved_index + batch_size]
            self.env.load_problems(batch_size, z_sample_size, aug_factor)
            rs, _, _ = self.env.reset()
            self.model.pre_forward(rs)
            state, reward, done = self.env.pre_step()
            with torch.amp.autocast(device_type=device, enabled=amp):
                while not done:
                    selected, _ = self.model(
                        state,
                        eval_type=self.tester_params.get('eval_type', 'greedy'),
                    )
                    state, reward, done = self.env.step(selected)
        if raw_batch is not None:
            tours = self.env.selected_node_list[:, :, :self.env.problem_size]
            raw_aug = raw_batch.repeat(int(aug_factor), 1, 1)
            cost = self._tsplib_euc2d_cost(raw_aug, tours).reshape(int(aug_factor), batch_size, z_sample_size)
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
        aug_reward = reward.reshape(aug_factor, batch_size, z_sample_size)
        max_pr, _ = aug_reward.max(dim=2)
        no_aug_values = -max_pr[0, :].float()
        max_apr, _ = max_pr.max(dim=0)
        aug_values = -max_apr.float()
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
