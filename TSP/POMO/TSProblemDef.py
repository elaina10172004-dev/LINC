import torch


def get_random_problems(batch_size, problem_size):
    return torch.rand(size=(batch_size, problem_size, 2))


def augment_xy_data_by_8_fold(problems):
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]

    return torch.cat(
        (
            torch.cat((x, y), dim=2),
            torch.cat((1 - x, y), dim=2),
            torch.cat((x, 1 - y), dim=2),
            torch.cat((1 - x, 1 - y), dim=2),
            torch.cat((y, x), dim=2),
            torch.cat((1 - y, x), dim=2),
            torch.cat((y, 1 - x), dim=2),
            torch.cat((1 - y, 1 - x), dim=2),
        ),
        dim=0,
    )
