import numpy as np
import torch


def _get_demand_scaler(problem_size):
    if 20 <= problem_size < 50:
        return 30
    if 50 <= problem_size < 100:
        return 40
    if problem_size >= 100:
        return 30 + problem_size // 5
    raise NotImplementedError(f"Unsupported problem_size: {problem_size}")


def _sample_locations(batch_size, problem_size, distribution):
    data_type = distribution.get('data_type', 'uniform')

    if data_type == 'uniform':
        depot_xy = torch.rand(size=(batch_size, 1, 2))
        node_xy = torch.rand(size=(batch_size, problem_size, 2))
        return depot_xy, node_xy

    if data_type == 'cluster':
        n_cluster = int(distribution.get('n_cluster', 3))
        lower = float(distribution.get('lower', 0.2))
        upper = float(distribution.get('upper', 0.8))
        std = float(distribution.get('std', 0.07))

        centers = lower + (upper - lower) * np.random.rand(batch_size, n_cluster, 2)
        depot_xy_list = []
        node_xy_list = []

        points_per_cluster = (problem_size + 1) // n_cluster
        for b_idx in range(batch_size):
            coords = torch.zeros(problem_size + 1, 2)
            for c_idx in range(n_cluster):
                start = points_per_cluster * c_idx
                end = points_per_cluster * (c_idx + 1) if c_idx < n_cluster - 1 else problem_size + 1
                if end <= start:
                    continue
                cluster_center = torch.tensor(centers[b_idx, c_idx], dtype=torch.float32)
                coords[start:end] = torch.randn(end - start, 2) * std + cluster_center

            coords = coords.clamp(0.0, 1.0)
            depot_idx = torch.randint(low=0, high=problem_size + 1, size=(1,)).item()
            depot_xy_list.append(coords[depot_idx].view(1, 1, 2))
            node_coords = torch.cat((coords[:depot_idx], coords[depot_idx + 1:]), dim=0)
            node_xy_list.append(node_coords.unsqueeze(0))

        depot_xy = torch.cat(depot_xy_list, dim=0)
        node_xy = torch.cat(node_xy_list, dim=0)
        return depot_xy, node_xy

    if data_type == 'mixed':
        n_cluster_mix = int(distribution.get('n_cluster_mix', 1))
        lower = float(distribution.get('lower', 0.2))
        upper = float(distribution.get('upper', 0.8))
        std = float(distribution.get('std', 0.07))

        centers = lower + (upper - lower) * np.random.rand(batch_size, n_cluster_mix, 2)
        depot_xy = torch.rand(size=(batch_size, 1, 2))
        node_xy_list = []

        mutate_count = problem_size // 2
        points_per_cluster = mutate_count // max(1, n_cluster_mix)
        for b_idx in range(batch_size):
            coords = torch.rand(problem_size, 2)
            mutate_idx = torch.randperm(problem_size)[:mutate_count]
            for c_idx in range(n_cluster_mix):
                start = points_per_cluster * c_idx
                end = points_per_cluster * (c_idx + 1) if c_idx < n_cluster_mix - 1 else mutate_count
                if end <= start:
                    continue
                idx = mutate_idx[start:end]
                cluster_center = torch.tensor(centers[b_idx, c_idx], dtype=torch.float32)
                coords[idx] = torch.randn(end - start, 2) * std + cluster_center
            node_xy_list.append(coords.clamp(0.0, 1.0).unsqueeze(0))

        node_xy = torch.cat(node_xy_list, dim=0)
        return depot_xy, node_xy

    raise NotImplementedError(f"Unsupported data_type: {data_type}")


def get_random_problems(batch_size, problem_size, distribution=None):
    if distribution is None:
        distribution = {'data_type': 'uniform'}

    grid_size = float(distribution.get('grid_size', 1000))
    depot_xy, node_xy = _sample_locations(batch_size, problem_size, distribution)
    depot_xy = depot_xy * grid_size
    node_xy = node_xy * grid_size

    demand_scaler = _get_demand_scaler(problem_size)
    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)).float()
    capacity = torch.full(size=(batch_size,), fill_value=float(demand_scaler), dtype=torch.float32)

    return depot_xy, node_xy, node_demand, capacity


def get_random_problems_from_data(depot_xy, node_xy, node_demand, augment=True):
    service_window = 2400
    service_duration = 50
    time_window_size = 500

    batch_size = node_xy.shape[0]
    problem_size = node_xy.shape[1]

    traveling_time = torch.linalg.vector_norm((depot_xy - node_xy).float(), dim=-1)
    tw_start_min = torch.ceil(traveling_time) + 1
    tw_end_max = service_window - torch.ceil(traveling_time + service_duration) - 1
    tw_center = tw_start_min + torch.round((tw_end_max - tw_start_min) * torch.rand(batch_size, problem_size))

    tw_start = tw_center - time_window_size // 2
    tw_end = tw_center + time_window_size // 2
    tw_start = torch.clamp(tw_start, min=tw_start_min)
    tw_end = torch.clamp(tw_end, max=tw_end_max)

    node_tw = torch.stack([tw_start, tw_end], dim=-1).int()
    depot_tw = torch.IntTensor([[0, service_window]]).repeat(batch_size, 1)
    return depot_xy, node_xy, node_demand, depot_tw, node_tw, service_duration


def augment_xy_data_by_8_fold(xy_data, grid_size):
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    grid = torch.as_tensor(grid_size, dtype=xy_data.dtype, device=xy_data.device)
    if grid.ndim == 0:
        grid = grid.view(1, 1, 1)
    elif grid.ndim == 1:
        grid = grid[:, None, None]
    else:
        raise ValueError("grid_size must be a scalar or a (batch,) tensor")

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((grid - x, y), dim=2)
    dat3 = torch.cat((x, grid - y), dim=2)
    dat4 = torch.cat((grid - x, grid - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((grid - y, x), dim=2)
    dat7 = torch.cat((y, grid - x), dim=2)
    dat8 = torch.cat((grid - y, grid - x), dim=2)

    return torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
