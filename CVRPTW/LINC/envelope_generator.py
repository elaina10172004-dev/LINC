"""Benchmark-agnostic broad-envelope CVRPTW generator.

This module defines ``EnvelopeCVRPTW`` / ``ECVRPTW``:

- it does not read Solomon benchmark files;
- it does not fit Solomon statistics or empirical tables;
- it does not use benchmark-conditioned caches or histograms;
- it only samples from explicit procedural rules.

Solomon-like structures may appear as special cases of this broad envelope,
but they are not the generation target and are never used as templates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SPACE_COMPONENTS = ("cluster", "uniform", "corridor", "outlier")
WIDTH_COMPONENTS = ("narrow", "medium", "loose")
PHASE_COMPONENTS = ("cluster", "radial", "angular", "random")


@dataclass(frozen=True)
class EnvelopeCVRPTWConfig:
    """Configuration for the benchmark-agnostic envelope generator."""

    grid_size_range: tuple[float, float] = (80.0, 250.0)
    horizon_ratio_range: tuple[float, float] = (10.0, 28.0)
    service_ratio_range: tuple[float, float] = (0.06, 0.14)
    alpha_range: tuple[float, float] = (0.9, 1.2)

    cluster_count_range: tuple[int, int] = (2, 8)
    sigma_c_range: tuple[float, float] = (0.03, 0.12)
    sigma_p_range: tuple[float, float] = (0.02, 0.10)
    corridor_halfwidth_range: tuple[float, float] = (0.015, 0.06)
    outlier_ratio_range: tuple[float, float] = (0.02, 0.20)
    constrained_ratio_range: tuple[float, float] = (0.25, 0.90)
    route_density_target_range: tuple[float, float] = (8.0, 24.0)
    capacity_noise_range: tuple[float, float] = (0.85, 1.15)
    w_min_ratio_range: tuple[float, float] = (0.02, 0.08)

    depot_position: tuple[float, float] = (0.5, 0.5)
    horizon_start: float = 0.0
    unbounded_depot_due: bool = False
    center_low: float = 0.10
    center_high: float = 0.90
    outlier_min_distance: float = 0.28
    outlier_resample_limit: int = 256

    space_mixture_concentration: tuple[float, float, float, float] = (1.2, 1.2, 0.8, 0.5)
    width_mixture_concentration: tuple[float, float, float] = (1.0, 1.0, 1.0)
    phase_weight_concentration: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    narrow_beta_a_range: tuple[float, float] = (1.5, 3.0)
    narrow_beta_b_range: tuple[float, float] = (10.0, 18.0)
    medium_beta_a_range: tuple[float, float] = (2.5, 5.0)
    medium_beta_b_range: tuple[float, float] = (4.0, 8.0)
    loose_beta_a_range: tuple[float, float] = (4.0, 8.0)
    loose_beta_b_range: tuple[float, float] = (1.5, 4.0)

    unconstrained_full_prob: float = 0.5
    unconstrained_start_frac_range: tuple[float, float] = (0.0, 0.08)
    unconstrained_end_frac_range: tuple[float, float] = (0.90, 1.0)

    hard_slice_prob: float = 0.25
    hard_constrained_ratio_range: tuple[float, float] = (0.70, 0.95)
    hard_horizon_ratio_range: tuple[float, float] = (10.0, 18.0)
    hard_grid_size_range: tuple[float, float] = (120.0, 250.0)
    hard_service_ratio_range: tuple[float, float] = (0.08, 0.14)
    hard_outlier_ratio_range: tuple[float, float] = (0.10, 0.30)
    hard_corridor_bias_range: tuple[float, float] = (0.20, 0.45)
    hard_cluster_bias_range: tuple[float, float] = (0.20, 0.45)
    hard_loose_width_weight_max: float = 0.10
    hard_narrow_width_weight_min: float = 0.45
    hard_medium_width_weight_min: float = 0.20
    hard_alpha_range: tuple[float, float] = (1.0, 1.2)
    hard_w_min_ratio_range: tuple[float, float] = (0.02, 0.06)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None = None) -> "EnvelopeCVRPTWConfig":
        if mapping is None:
            return cls()
        data = dict(mapping)
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {key: value for key, value in data.items() if key in valid_keys}
        return cls(**filtered)


class EnvelopeCVRPTWGenerator:
    """Broad-envelope, benchmark-agnostic CVRPTW generator."""

    def __init__(self, config: EnvelopeCVRPTWConfig | None = None, seed: int | None = None):
        self.config = config or EnvelopeCVRPTWConfig()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset_seed(self, seed: int | None) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def sample_instance(
        self,
        problem_size: int,
        seed: int | None = None,
        return_metadata: bool = True,
    ) -> dict[str, Any]:
        rng = self._rng if seed is None else np.random.default_rng(seed)
        batch = self._sample_batch_impl(
            batch_size=1,
            problem_size=problem_size,
            rng=rng,
            return_metadata=return_metadata,
        )
        return self._squeeze_batch(batch)

    def sample_batch(
        self,
        batch_size: int,
        problem_size: int,
        return_metadata: bool = False,
    ) -> dict[str, Any]:
        return self._sample_batch_impl(
            batch_size=batch_size,
            problem_size=problem_size,
            rng=self._rng,
            return_metadata=return_metadata,
        )

    def export_dataset(
        self,
        output_path: str | Path,
        num_samples: int,
        problem_size: int,
        batch_size: int = 128,
        include_metadata: bool = False,
    ) -> Path:
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        chunks: list[dict[str, Any]] = []
        generated = 0
        while generated < num_samples:
            current_batch = min(batch_size, num_samples - generated)
            chunks.append(
                self.sample_batch(
                    batch_size=current_batch,
                    problem_size=problem_size,
                    return_metadata=include_metadata,
                )
            )
            generated += current_batch

        dataset = self._concat_batches(chunks, include_metadata=include_metadata)
        dataset["dataset_name"] = "EnvelopeCVRPTW"
        dataset["dataset_alias"] = "ECVRPTW"
        dataset["generator_name"] = self.__class__.__name__
        dataset["benchmark_agnostic"] = True
        dataset["generator_config"] = asdict(self.config)
        if include_metadata and dataset["metadata"] is not None:
            dataset["metadata"] = self._sanitize_for_torch_save(dataset["metadata"])
        torch.save(dataset, output)
        return output

    def validate_instance(self, instance: Mapping[str, Any], atol: float = 1e-6) -> dict[str, Any]:
        batch = {
            "depot_xy": self._ensure_batch_tensor(instance["depot_xy"], min_dim=3),
            "depot_tw": self._ensure_batch_tensor(
                instance.get("depot_tw", instance.get("depot_horizon")),
                min_dim=3,
            ),
            "node_xy": self._ensure_batch_tensor(instance["node_xy"], min_dim=3),
            "node_demand": self._ensure_batch_tensor(instance["node_demand"], min_dim=2),
            "node_tw": self._ensure_batch_tensor(instance["node_tw"], min_dim=3),
            "capacity": self._ensure_batch_tensor(instance["capacity"], min_dim=1),
            "service_t": self._ensure_batch_tensor(instance["service_t"], min_dim=2),
            "travel_time_scale": self._ensure_batch_tensor(instance["travel_time_scale"], min_dim=1),
        }
        if "metadata" in instance and instance["metadata"] is not None:
            batch["metadata"] = [instance["metadata"]]
        return validate_envelope_batch(batch, atol=atol)

    def validate_batch(self, batch: Mapping[str, Any], atol: float = 1e-6) -> dict[str, Any]:
        return validate_envelope_batch(batch, atol=atol)

    def _sample_batch_impl(
        self,
        *,
        batch_size: int,
        problem_size: int,
        rng: np.random.Generator,
        return_metadata: bool,
    ) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if problem_size <= 0:
            raise ValueError("problem_size must be positive")

        depot_xy = np.zeros((batch_size, 1, 2), dtype=np.float32)
        depot_tw = np.zeros((batch_size, 1, 2), dtype=np.float32)
        node_xy = np.zeros((batch_size, problem_size, 2), dtype=np.float32)
        node_demand = np.zeros((batch_size, problem_size), dtype=np.float32)
        node_tw = np.zeros((batch_size, problem_size, 2), dtype=np.float32)
        capacity = np.zeros((batch_size,), dtype=np.float32)
        service_t = np.zeros((batch_size, 1), dtype=np.float32)
        travel_time_scale = np.zeros((batch_size,), dtype=np.float32)
        grid_size = np.zeros((batch_size,), dtype=np.float32)
        family = []
        metadata: list[dict[str, Any]] = []

        for row_idx in range(batch_size):
            sample = self._sample_numpy_instance(problem_size=problem_size, rng=rng)
            depot_xy[row_idx] = sample["depot_xy"]
            depot_tw[row_idx] = sample["depot_tw"]
            node_xy[row_idx] = sample["node_xy"]
            node_demand[row_idx] = sample["node_demand"]
            node_tw[row_idx] = sample["node_tw"]
            capacity[row_idx] = sample["capacity"]
            service_t[row_idx, 0] = sample["service_t"]
            travel_time_scale[row_idx] = sample["travel_time_scale"]
            grid_size[row_idx] = sample["grid_size"]
            family.append(sample["family"])
            if return_metadata:
                metadata.append(sample["metadata"])

        return {
            "depot_xy": torch.from_numpy(depot_xy),
            "depot_tw": torch.from_numpy(depot_tw),
            "depot_horizon": torch.from_numpy(depot_tw[:, 0, :]),
            "node_xy": torch.from_numpy(node_xy),
            "node_demand": torch.from_numpy(node_demand),
            "node_tw": torch.from_numpy(node_tw),
            "capacity": torch.from_numpy(capacity),
            "service_t": torch.from_numpy(service_t),
            "service_duration": torch.from_numpy(service_t.copy()),
            "travel_time_scale": torch.from_numpy(travel_time_scale),
            "grid_size": torch.from_numpy(grid_size),
            "scale": torch.from_numpy(grid_size.copy()),
            "family": family,
            "metadata": metadata if return_metadata else None,
        }

    def _sample_numpy_instance(self, *, problem_size: int, rng: np.random.Generator) -> dict[str, Any]:
        latent = self._sample_latent(rng)
        node_xy_norm, spatial_info = self._sample_node_layout(problem_size=problem_size, latent=latent, rng=rng)
        node_xy = (node_xy_norm * latent["grid_size"]).astype(np.float32)
        depot_xy = np.asarray(latent["depot_position_abs"], dtype=np.float32)[None, :]
        node_demand = rng.integers(1, 11, size=(problem_size,), endpoint=False).astype(np.float32)
        capacity = self._sample_capacity(node_demand=node_demand, latent=latent, rng=rng)
        travel_to_depot, lower_bound, upper_bound = self._compute_feasible_bounds(
            node_xy=node_xy,
            depot_xy=depot_xy[0],
            service_t=float(latent["service_t"]),
            horizon_end=float(latent["horizon_end"]),
            travel_time_scale=float(latent["alpha"]),
        )
        phase = self._sample_phase(
            node_xy_norm=node_xy_norm,
            spatial_info=spatial_info,
            latent=latent,
            rng=rng,
        )
        constrained_mask = rng.random(problem_size) < float(latent["constrained_ratio"])
        node_tw, width_ratio = self._sample_time_windows(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            phase=phase,
            constrained_mask=constrained_mask,
            latent=latent,
            rng=rng,
        )
        family = self._classify_family(latent["space_weights"])
        metadata = {
            "latent": self._serialize_latent(latent),
            "is_hard": bool(latent["is_hard"]),
            "component_ids": spatial_info["component_ids"].astype(np.int64),
            "cluster_ids": spatial_info["cluster_ids"].astype(np.int64),
            "constrained_mask": constrained_mask.astype(bool),
            "window_width": (node_tw[:, 1] - node_tw[:, 0]).astype(np.float32),
            "feasible_span": (upper_bound - lower_bound).astype(np.float32),
            "width_ratio": width_ratio.astype(np.float32),
            "travel_from_depot": travel_to_depot.astype(np.float32),
            "family": family,
        }
        return {
            "depot_xy": depot_xy,
            "depot_tw": np.asarray(
                [[self.config.horizon_start, np.inf if self.config.unbounded_depot_due else latent["horizon_end"]]],
                dtype=np.float32,
            ),
            "node_xy": node_xy,
            "node_demand": node_demand,
            "node_tw": node_tw,
            "capacity": np.float32(capacity),
            "service_t": np.float32(latent["service_t"]),
            "service_duration": np.float32(latent["service_t"]),
            "travel_time_scale": np.float32(latent["alpha"]),
            "grid_size": np.float32(latent["grid_size"]),
            "family": family,
            "metadata": metadata,
        }

    def _sample_latent(self, rng: np.random.Generator) -> dict[str, Any]:
        cfg = self.config
        is_hard = bool(rng.random() < cfg.hard_slice_prob)
        grid_size = float(rng.uniform(*(cfg.hard_grid_size_range if is_hard else cfg.grid_size_range)))
        horizon_ratio = float(rng.uniform(*(cfg.hard_horizon_ratio_range if is_hard else cfg.horizon_ratio_range)))
        service_ratio = float(rng.uniform(*(cfg.hard_service_ratio_range if is_hard else cfg.service_ratio_range)))
        alpha = float(rng.uniform(*(cfg.hard_alpha_range if is_hard else cfg.alpha_range)))
        cluster_count = int(rng.integers(cfg.cluster_count_range[0], cfg.cluster_count_range[1] + 1))
        sigma_c = float(rng.uniform(*cfg.sigma_c_range))
        sigma_p = float(rng.uniform(*cfg.sigma_p_range))
        corridor_halfwidth = float(rng.uniform(*cfg.corridor_halfwidth_range))
        outlier_ratio = float(rng.uniform(*(cfg.hard_outlier_ratio_range if is_hard else cfg.outlier_ratio_range)))
        constrained_ratio = float(rng.uniform(*(cfg.hard_constrained_ratio_range if is_hard else cfg.constrained_ratio_range)))
        route_density_target = float(rng.uniform(*cfg.route_density_target_range))
        capacity_noise = float(rng.uniform(*cfg.capacity_noise_range))
        w_min_ratio = float(rng.uniform(*(cfg.hard_w_min_ratio_range if is_hard else cfg.w_min_ratio_range)))

        cluster_bias = float(rng.uniform(*cfg.hard_cluster_bias_range)) if is_hard else 0.0
        corridor_bias = float(rng.uniform(*cfg.hard_corridor_bias_range)) if is_hard else 0.0

        space_weights = self._sample_space_weights(
            rng=rng,
            outlier_ratio=outlier_ratio,
            is_hard=is_hard,
            cluster_bias=cluster_bias,
            corridor_bias=corridor_bias,
        )
        width_weights = self._sample_width_weights(rng=rng, is_hard=is_hard)
        phase_weights = self._sample_phase_weights(rng=rng, is_hard=is_hard)
        width_beta_params = self._sample_width_beta_params(rng=rng, is_hard=is_hard)

        cluster_centers = rng.uniform(cfg.center_low, cfg.center_high, size=(cluster_count, 2)).astype(np.float32)
        cluster_phase = rng.uniform(0.1, 0.9, size=(cluster_count,)).astype(np.float32)
        cluster_phase.sort()
        corridor_theta = float(rng.uniform(0.0, 2.0 * np.pi))
        corridor_direction = np.asarray([np.cos(corridor_theta), np.sin(corridor_theta)], dtype=np.float32)
        corridor_normal = np.asarray([-corridor_direction[1], corridor_direction[0]], dtype=np.float32)
        corridor_center = rng.uniform(0.15, 0.85, size=(2,)).astype(np.float32)
        corridor_length = float(rng.uniform(0.55, 1.20))

        return {
            "grid_size": grid_size,
            "is_hard": is_hard,
            "horizon_end": float(cfg.horizon_start + horizon_ratio * grid_size),
            "horizon_ratio": horizon_ratio,
            "service_ratio": service_ratio,
            "service_t": float(service_ratio * grid_size),
            "alpha": alpha,
            "cluster_count": cluster_count,
            "sigma_c": sigma_c,
            "sigma_p": sigma_p,
            "corridor_halfwidth": corridor_halfwidth,
            "outlier_ratio": outlier_ratio,
            "constrained_ratio": constrained_ratio,
            "route_density_target": route_density_target,
            "capacity_noise": capacity_noise,
            "w_min_ratio": w_min_ratio,
            "w_min_abs": float(w_min_ratio * grid_size),
            "cluster_bias": cluster_bias,
            "corridor_bias": corridor_bias,
            "space_weights": space_weights,
            "width_weights": width_weights,
            "phase_weights": phase_weights,
            "width_beta_params": width_beta_params,
            "cluster_centers": cluster_centers,
            "cluster_phase": cluster_phase,
            "corridor_direction": corridor_direction,
            "corridor_normal": corridor_normal,
            "corridor_center": corridor_center,
            "corridor_length": corridor_length,
            "depot_position_abs": (
                float(cfg.depot_position[0] * grid_size),
                float(cfg.depot_position[1] * grid_size),
            ),
            "depot_position_norm": np.asarray(cfg.depot_position, dtype=np.float32),
        }

    def _sample_space_weights(
        self,
        *,
        rng: np.random.Generator,
        outlier_ratio: float,
        is_hard: bool,
        cluster_bias: float,
        corridor_bias: float,
    ) -> np.ndarray:
        cfg = self.config
        base = rng.dirichlet(np.asarray(cfg.space_mixture_concentration, dtype=np.float64)).astype(np.float32)
        target_outlier = float(np.clip(outlier_ratio, 0.02, 0.75))
        outlier_weight = float(np.clip(0.45 * float(base[3]) + 0.55 * target_outlier, 0.02, 0.75))
        non_outlier = base[:3].astype(np.float64)
        if is_hard:
            non_outlier[0] *= 1.0 + cluster_bias
            non_outlier[2] *= 1.0 + corridor_bias
            non_outlier[1] *= 0.60 + 0.25 * float(rng.random())
        non_outlier_sum = float(non_outlier.sum())
        if non_outlier_sum <= 0.0:
            non_outlier = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
            non_outlier_sum = 3.0
        non_outlier = non_outlier / non_outlier_sum * (1.0 - outlier_weight)
        weights = np.concatenate((non_outlier.astype(np.float32), np.asarray([outlier_weight], dtype=np.float32)))
        weights /= weights.sum()
        return weights.astype(np.float32)

    def _sample_width_weights(self, *, rng: np.random.Generator, is_hard: bool) -> np.ndarray:
        cfg = self.config
        if not is_hard:
            return rng.dirichlet(np.asarray(cfg.width_mixture_concentration, dtype=np.float64)).astype(np.float32)

        narrow_min = cfg.hard_narrow_width_weight_min
        medium_min = cfg.hard_medium_width_weight_min
        loose_max = cfg.hard_loose_width_weight_max
        loose = float(rng.uniform(0.0, loose_max))
        remaining = 1.0 - loose
        extra = max(0.0, remaining - narrow_min - medium_min)
        split = float(rng.random())
        narrow = narrow_min + extra * split
        medium = medium_min + extra * (1.0 - split)
        weights = np.asarray([narrow, medium, loose], dtype=np.float32)
        weights /= weights.sum()
        return weights

    def _sample_phase_weights(self, *, rng: np.random.Generator, is_hard: bool) -> np.ndarray:
        cfg = self.config
        base = rng.dirichlet(np.asarray(cfg.phase_weight_concentration, dtype=np.float64)).astype(np.float32)
        if not is_hard:
            return base
        random_weight = float(rng.uniform(0.05, 0.18))
        structured = base[:3].astype(np.float64)
        structured[0] *= 1.15
        structured[1] *= 1.10
        structured[2] *= 1.10
        structured_sum = float(structured.sum())
        if structured_sum <= 0.0:
            structured = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
            structured_sum = 3.0
        structured = structured / structured_sum * (1.0 - random_weight)
        weights = np.concatenate((structured.astype(np.float32), np.asarray([random_weight], dtype=np.float32)))
        weights /= weights.sum()
        return weights.astype(np.float32)

    def _sample_width_beta_params(self, *, rng: np.random.Generator, is_hard: bool) -> dict[str, tuple[float, float]]:
        cfg = self.config
        if not is_hard:
            return {
                "narrow": (
                    float(rng.uniform(*cfg.narrow_beta_a_range)),
                    float(rng.uniform(*cfg.narrow_beta_b_range)),
                ),
                "medium": (
                    float(rng.uniform(*cfg.medium_beta_a_range)),
                    float(rng.uniform(*cfg.medium_beta_b_range)),
                ),
                "loose": (
                    float(rng.uniform(*cfg.loose_beta_a_range)),
                    float(rng.uniform(*cfg.loose_beta_b_range)),
                ),
            }
        return {
            "narrow": (
                float(rng.uniform(1.2, 2.5)),
                float(rng.uniform(13.0, 22.0)),
            ),
            "medium": (
                float(rng.uniform(2.0, 4.0)),
                float(rng.uniform(6.0, 10.0)),
            ),
            "loose": (
                float(rng.uniform(2.5, 4.5)),
                float(rng.uniform(4.0, 8.0)),
            ),
        }

    def _sample_node_layout(
        self,
        *,
        problem_size: int,
        latent: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        component_ids = rng.choice(len(SPACE_COMPONENTS), size=(problem_size,), p=latent["space_weights"]).astype(np.int64)
        cluster_ids = np.full((problem_size,), fill_value=-1, dtype=np.int64)
        node_xy = np.zeros((problem_size, 2), dtype=np.float32)

        for idx, component_id in enumerate(component_ids):
            component = SPACE_COMPONENTS[int(component_id)]
            if component == "cluster":
                cluster_id = int(rng.integers(0, latent["cluster_count"]))
                cluster_ids[idx] = cluster_id
                center = latent["cluster_centers"][cluster_id]
                node_xy[idx] = self._sample_truncated_gaussian(center=center, sigma=float(latent["sigma_c"]), rng=rng)
            elif component == "uniform":
                node_xy[idx] = rng.uniform(0.0, 1.0, size=(2,)).astype(np.float32)
            elif component == "corridor":
                node_xy[idx] = self._sample_corridor_point(latent=latent, rng=rng)
            else:
                node_xy[idx] = self._sample_outlier_point(latent=latent, rng=rng)

        return node_xy, {
            "component_ids": component_ids,
            "cluster_ids": cluster_ids,
        }

    def _sample_truncated_gaussian(self, *, center: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
        for _ in range(self.config.outlier_resample_limit):
            point = center + rng.normal(0.0, sigma, size=(2,))
            if np.all(point >= 0.0) and np.all(point <= 1.0):
                return point.astype(np.float32)
        return np.clip(point, 0.0, 1.0).astype(np.float32)

    def _sample_corridor_point(self, *, latent: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
        t = float(rng.uniform(-0.5, 0.5) * latent["corridor_length"])
        lateral = float(rng.normal(0.0, latent["corridor_halfwidth"]))
        point = latent["corridor_center"] + t * latent["corridor_direction"] + lateral * latent["corridor_normal"]
        return np.clip(point, 0.0, 1.0).astype(np.float32)

    def _sample_outlier_point(self, *, latent: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
        depot = latent["depot_position_norm"]
        min_dist = self.config.outlier_min_distance
        point = rng.uniform(0.0, 1.0, size=(2,))
        for _ in range(self.config.outlier_resample_limit):
            point = rng.uniform(0.0, 1.0, size=(2,))
            depot_dist = np.linalg.norm(point - depot)
            if depot_dist < min_dist:
                continue
            if latent["cluster_count"] > 0:
                center_dist = np.linalg.norm(point[None, :] - latent["cluster_centers"], axis=1)
                if np.any(center_dist < min_dist):
                    continue
            return point.astype(np.float32)
        return np.clip(point, 0.0, 1.0).astype(np.float32)

    def _sample_capacity(
        self,
        *,
        node_demand: np.ndarray,
        latent: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> float:
        mean_demand = float(node_demand.mean())
        base_capacity = float(latent["route_density_target"] * mean_demand * latent["capacity_noise"])
        return float(max(node_demand.max() + 1.0, base_capacity))

    def _compute_feasible_bounds(
        self,
        *,
        node_xy: np.ndarray,
        depot_xy: np.ndarray,
        service_t: float,
        horizon_end: float,
        travel_time_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        euclidean = np.linalg.norm(node_xy.astype(np.float32) - depot_xy[None, :].astype(np.float32), axis=1)
        travel_to_depot = (travel_time_scale * euclidean).astype(np.float32)
        lower_bound = travel_to_depot
        upper_bound = (horizon_end - travel_to_depot - service_t).astype(np.float32)
        return travel_to_depot, lower_bound, upper_bound

    def _sample_phase(
        self,
        *,
        node_xy_norm: np.ndarray,
        spatial_info: Mapping[str, np.ndarray],
        latent: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        depot = latent["depot_position_norm"]
        radial = np.linalg.norm(node_xy_norm - depot[None, :], axis=1) / np.sqrt(2.0)
        angular = (np.arctan2(node_xy_norm[:, 1] - depot[1], node_xy_norm[:, 0] - depot[0]) + np.pi) / (2.0 * np.pi)
        random_phase = rng.uniform(0.0, 1.0, size=(node_xy_norm.shape[0],)).astype(np.float32)
        cluster_phase = random_phase.copy()
        clustered_mask = spatial_info["cluster_ids"] >= 0
        if clustered_mask.any():
            cluster_phase[clustered_mask] = latent["cluster_phase"][spatial_info["cluster_ids"][clustered_mask]]
        weights = latent["phase_weights"]
        phase = (
            weights[0] * cluster_phase
            + weights[1] * radial
            + weights[2] * angular
            + weights[3] * random_phase
        )
        phase = phase + rng.normal(0.0, latent["sigma_p"], size=(node_xy_norm.shape[0],))
        return np.clip(phase.astype(np.float32), 0.0, 1.0)

    def _sample_time_windows(
        self,
        *,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        phase: np.ndarray,
        constrained_mask: np.ndarray,
        latent: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        problem_size = int(lower_bound.shape[0])
        full_due = float(latent["horizon_end"] - latent["service_t"])
        node_tw = np.zeros((problem_size, 2), dtype=np.float32)
        width_ratio = np.ones((problem_size,), dtype=np.float32)

        unconstrained_idx = np.nonzero(~constrained_mask)[0]
        if unconstrained_idx.size > 0:
            use_full = rng.random(unconstrained_idx.size) < self.config.unconstrained_full_prob
            start = np.zeros((unconstrained_idx.size,), dtype=np.float32)
            end = np.full((unconstrained_idx.size,), fill_value=full_due, dtype=np.float32)
            if (~use_full).any():
                start_lo, start_hi = self.config.unconstrained_start_frac_range
                end_lo, end_hi = self.config.unconstrained_end_frac_range
                relaxed_idx = np.nonzero(~use_full)[0]
                start[relaxed_idx] = (
                    rng.uniform(start_lo, start_hi, size=(relaxed_idx.size,)).astype(np.float32) * full_due
                )
                end[relaxed_idx] = (
                    rng.uniform(end_lo, end_hi, size=(relaxed_idx.size,)).astype(np.float32) * full_due
                )
                end = np.maximum(end, start)
            node_tw[unconstrained_idx, 0] = start
            node_tw[unconstrained_idx, 1] = end
            width_ratio[unconstrained_idx] = (end - start) / max(full_due, 1e-6)

        constrained_idx = np.nonzero(constrained_mask)[0]
        if constrained_idx.size == 0:
            return node_tw, width_ratio

        lower_c = lower_bound[constrained_idx]
        upper_c = upper_bound[constrained_idx]
        span_c = np.maximum(upper_c - lower_c, 0.0)
        center = lower_c + phase[constrained_idx] * span_c

        width_component_ids = rng.choice(len(WIDTH_COMPONENTS), size=(constrained_idx.size,), p=latent["width_weights"])
        omega = np.zeros((constrained_idx.size,), dtype=np.float32)
        for comp_idx, comp_name in enumerate(WIDTH_COMPONENTS):
            mask = width_component_ids == comp_idx
            if not mask.any():
                continue
            a, b = latent["width_beta_params"][comp_name]
            omega[mask] = rng.beta(a, b, size=(int(mask.sum()),)).astype(np.float32)

        width = np.maximum(float(latent["w_min_abs"]), omega * span_c).astype(np.float32)
        start = np.maximum(lower_c, center - 0.5 * width)
        end = np.minimum(upper_c, center + 0.5 * width)
        collapsed = end < start
        if collapsed.any():
            midpoint = np.clip(center[collapsed], lower_c[collapsed], upper_c[collapsed])
            start[collapsed] = midpoint
            end[collapsed] = midpoint

        node_tw[constrained_idx, 0] = start
        node_tw[constrained_idx, 1] = end
        width_ratio[constrained_idx] = (end - start) / np.maximum(span_c, 1e-6)
        return node_tw, width_ratio

    def _classify_family(self, space_weights: np.ndarray) -> str:
        dominant = SPACE_COMPONENTS[int(np.argmax(space_weights))]
        if dominant == "cluster" and float(space_weights[0]) >= 0.5:
            return "cluster-heavy"
        if dominant == "uniform" and float(space_weights[1]) >= 0.5:
            return "uniform-heavy"
        if dominant == "corridor" and float(space_weights[2]) >= 0.35:
            return "corridor-heavy"
        return "mixed-envelope"

    def _serialize_latent(self, latent: Mapping[str, Any]) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for key, value in latent.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, dict):
                serialized[key] = {
                    inner_key: list(inner_value) if isinstance(inner_value, tuple) else inner_value
                    for inner_key, inner_value in value.items()
                }
            elif isinstance(value, tuple):
                serialized[key] = list(value)
            else:
                serialized[key] = value
        return serialized

    def _sanitize_for_torch_save(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: self._sanitize_for_torch_save(inner_value) for key, inner_value in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_for_torch_save(item) for item in value]
        return value

    def _concat_batches(self, batches: Sequence[Mapping[str, Any]], include_metadata: bool) -> dict[str, Any]:
        tensor_keys = (
            "depot_xy",
            "depot_tw",
            "depot_horizon",
            "node_xy",
            "node_demand",
            "node_tw",
            "capacity",
            "service_t",
            "service_duration",
            "travel_time_scale",
            "grid_size",
            "scale",
        )
        merged = {
            key: torch.cat([torch.as_tensor(batch[key]) for batch in batches], dim=0)
            for key in tensor_keys
        }
        merged["family"] = [item for batch in batches for item in batch["family"]]
        merged["metadata"] = [item for batch in batches for item in (batch["metadata"] or [])] if include_metadata else None
        return merged

    def _squeeze_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        squeezed: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                squeezed[key] = value[0] if value.ndim > 0 else value
            elif key == "family":
                squeezed[key] = value[0]
            elif key == "metadata":
                squeezed[key] = value[0] if value else None
            else:
                squeezed[key] = value
        return squeezed

    def _ensure_batch_tensor(self, value: Any, min_dim: int) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        while tensor.ndim < min_dim:
            tensor = tensor.unsqueeze(0)
        return tensor


def validate_envelope_batch(batch: Mapping[str, Any], atol: float = 1e-6) -> dict[str, Any]:
    depot_xy = torch.as_tensor(batch["depot_xy"], dtype=torch.float32)
    depot_tw = torch.as_tensor(batch.get("depot_tw", batch.get("depot_horizon")), dtype=torch.float32)
    node_xy = torch.as_tensor(batch["node_xy"], dtype=torch.float32)
    node_demand = torch.as_tensor(batch["node_demand"], dtype=torch.float32)
    node_tw = torch.as_tensor(batch["node_tw"], dtype=torch.float32)
    capacity = torch.as_tensor(batch["capacity"], dtype=torch.float32)
    service_t = torch.as_tensor(batch["service_t"], dtype=torch.float32).reshape(-1, 1)
    travel_time_scale = torch.as_tensor(batch.get("travel_time_scale", 1.0), dtype=torch.float32).reshape(-1)

    if depot_xy.ndim != 3 or depot_xy.size(-1) != 2:
        raise ValueError("depot_xy must have shape (batch, 1, 2)")
    if depot_tw.ndim == 2:
        depot_tw = depot_tw[:, None, :]
    if depot_tw.ndim != 3 or depot_tw.size(-1) != 2:
        raise ValueError("depot_tw/depot_horizon must have shape (batch, 1, 2)")
    if node_xy.ndim != 3 or node_xy.size(-1) != 2:
        raise ValueError("node_xy must have shape (batch, n, 2)")
    if node_tw.ndim != 3 or node_tw.size(-1) != 2:
        raise ValueError("node_tw must have shape (batch, n, 2)")
    if not bool((node_demand > 0).all()):
        raise ValueError("All node_demand values must be > 0")
    if not bool((capacity > 0).all()):
        raise ValueError("capacity must be > 0")
    if not bool((node_tw[..., 0] <= node_tw[..., 1] + atol).all()):
        raise ValueError("Every time window must satisfy ready <= due")

    ready = node_tw[..., 0]
    due = node_tw[..., 1]
    horizon_start = depot_tw[:, 0, 0:1]
    horizon_end = depot_tw[:, 0, 1:2]
    full_due = horizon_end - service_t

    if not bool((ready >= horizon_start - atol).all()):
        raise ValueError("Time-window starts must be >= depot_horizon_start")
    if not bool((due <= full_due + max(atol, 1e-3)).all()):
        raise ValueError("Time-window ends must be <= depot_horizon_end - service_t")

    depot = depot_xy[:, 0:1, :]
    euclidean = torch.linalg.vector_norm(node_xy - depot, dim=-1)
    travel = euclidean * travel_time_scale[:, None]
    lower_bound = travel
    upper_bound = horizon_end - travel - service_t

    metadata = batch.get("metadata")
    constrained_mask = None
    hard_mask = None
    if metadata:
        constrained_mask = torch.as_tensor(
            np.stack([np.asarray(item["constrained_mask"], dtype=bool) for item in metadata], axis=0)
        )
        hard_mask = torch.as_tensor(
            np.asarray([bool(item.get("is_hard", item.get("latent", {}).get("is_hard", False))) for item in metadata], dtype=bool)
        )
        if constrained_mask.any():
            if not bool((ready[constrained_mask] >= lower_bound[constrained_mask] - atol).all()):
                raise ValueError("Constrained windows must satisfy the feasible lower bound")
            if not bool((due[constrained_mask] <= upper_bound[constrained_mask] + atol).all()):
                raise ValueError("Constrained windows must satisfy the feasible upper bound")

    width = due - ready
    span = upper_bound - lower_bound
    width_ratio = width / torch.clamp(span, min=1e-6)
    grid = torch.as_tensor(batch["grid_size"], dtype=torch.float32).reshape(-1)
    mean_constrained_ratio = float(constrained_mask.float().mean().item()) if constrained_mask is not None else -1.0
    hard_count = int(hard_mask.sum().item()) if hard_mask is not None else 0
    easy_count = int((~hard_mask).sum().item()) if hard_mask is not None else int(node_xy.size(0))
    hard_mean_constrained_ratio = -1.0
    easy_mean_constrained_ratio = -1.0
    hard_mean_width_ratio = -1.0
    easy_mean_width_ratio = -1.0
    if constrained_mask is not None and hard_mask is not None:
        if hard_mask.any():
            hard_mean_constrained_ratio = float(constrained_mask[hard_mask].float().mean().item())
            hard_mean_width_ratio = float(width_ratio[hard_mask].mean().item())
        easy_mask = ~hard_mask
        if easy_mask.any():
            easy_mean_constrained_ratio = float(constrained_mask[easy_mask].float().mean().item())
            easy_mean_width_ratio = float(width_ratio[easy_mask].mean().item())
    return {
        "batch_size": int(node_xy.size(0)),
        "problem_size": int(node_xy.size(1)),
        "hard_count": hard_count,
        "easy_count": easy_count,
        "mean_width": float(width.mean().item()),
        "mean_width_ratio": float(width_ratio.mean().item()),
        "mean_feasible_span": float(span.mean().item()),
        "mean_constrained_ratio": mean_constrained_ratio,
        "mean_horizon_ratio": float((horizon_end / grid[:, None]).mean().item()),
        "mean_service_ratio": float((service_t[:, 0] / grid).mean().item()),
        "mean_distance_scale": float(travel_time_scale.mean().item()),
        "mean_grid_size": float(grid.mean().item()),
        "constrained_count": int(constrained_mask.sum().item()) if constrained_mask is not None else -1,
        "unconstrained_count": int((~constrained_mask).sum().item()) if constrained_mask is not None else -1,
        "hard_mean_constrained_ratio": hard_mean_constrained_ratio,
        "easy_mean_constrained_ratio": easy_mean_constrained_ratio,
        "hard_mean_width_ratio": hard_mean_width_ratio,
        "easy_mean_width_ratio": easy_mean_width_ratio,
    }


BenchmarkAgnosticCVRPTWGenerator = EnvelopeCVRPTWGenerator


__all__ = [
    "BenchmarkAgnosticCVRPTWGenerator",
    "EnvelopeCVRPTWConfig",
    "EnvelopeCVRPTWGenerator",
    "validate_envelope_batch",
]
