"""POMO CVRP sampling tester."""
import time

import torch
from logging import getLogger
from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
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

    def run(self):
        start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.time_estimator.reset()
        score_AM, aug_score_AM = AverageMeter(), AverageMeter()
        total, bs = self.tester_params['test_episodes'], self.tester_params['test_batch_size']
        z, aug = self.tester_params['test_z_sample_size'], self.tester_params['aug_factor'] if self.tester_params['augmentation_enable'] else 1
        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)
        ep = 0
        while ep < total:
            b = min(bs, total - ep)
            s, ag = self._test_one_batch(b, aug)
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
        }

    def _test_one_batch(self, batch_size, aug_factor):
        device = "cuda" if self.tester_params['use_cuda'] else "cpu"
        amp = self.tester_params.get('amp_inference', True)
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
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
        pomo = self.env.pomo_size
        aug_reward = reward.reshape(aug_factor, batch_size, pomo)
        max_pr, _ = aug_reward.max(dim=2)
        no_aug = -max_pr[0, :].float().mean()
        max_apr, _ = max_pr.max(dim=0)
        aug_score = -max_apr.float().mean()
        return no_aug.item(), aug_score.item()
