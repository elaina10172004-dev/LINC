import torch


def get_random_problems_dummy(batch_size, problem_size):
    service_window = 24000
    service_duration = 1


    # Sample locations
    depot_xy = torch.randint(0, 1000, size=(batch_size, 1, 2))
    node_xy = torch.randint(0, 1000, size=(batch_size, problem_size, 2))


    # Sample time windows start
    tw_start = torch.zeros(batch_size, problem_size)


    tw_end = torch.ones(batch_size, problem_size) * 24000

    node_tw = torch.stack([tw_start, tw_end], dim=-1)
    depot_tw = torch.IntTensor([[0, service_window]]).repeat(batch_size, 1)



    # Scale node demand based on vehicle capacity
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size))
    capacity = torch.ones(batch_size) * demand_scaler

    return depot_xy, node_xy, node_demand, depot_tw, node_tw, service_duration, capacity

def get_random_problems_from_data(depot_xy, node_xy, node_demand, augment=True):
    service_window = 2400 # v1: 2300 v2: 2400
    service_duration = 50 # v1: 100 v2: 50
    time_window_size = 500

    batch_size = node_xy.shape[0]
    problem_size = node_xy.shape[1]


    # Distance from the nodes to the depot
    traveling_time = torch.linalg.vector_norm((depot_xy - node_xy).float(), dim=-1)

    # TW start needs to be feasibly reachable directly from depot
    tw_start_min = torch.ceil(traveling_time) + 1

    # TW end needs to be early enough to perform service and return to depot until end of service window
    tw_end_max = service_window - torch.ceil(traveling_time + service_duration) - 1

    # Sample time windows center
    tw_center = tw_start_min + torch.round((tw_end_max - tw_start_min) * torch.rand(batch_size, problem_size))

    # Define time window start and end
    tw_start = tw_center - time_window_size // 2
    tw_end = tw_center + time_window_size // 2

    tw_start = torch.clamp(tw_start, min=tw_start_min)
    tw_end = torch.clamp(tw_end, max=tw_end_max)


    node_tw = torch.stack([tw_start, tw_end], dim=-1).int()
    depot_tw = torch.IntTensor([[0, service_window]]).repeat(batch_size, 1)


    return depot_xy, node_xy, node_demand, depot_tw, node_tw, service_duration

def augment_xy_data_by_8_fold(xy_data, grid_size):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((grid_size - x, y), dim=2)
    dat3 = torch.cat((x, grid_size - y), dim=2)
    dat4 = torch.cat((grid_size - x, grid_size - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((grid_size - y, x), dim=2)
    dat7 = torch.cat((y, grid_size - x), dim=2)
    dat8 = torch.cat((grid_size - y, grid_size - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data