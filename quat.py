import torch

# Reference: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_invert

def quat_mul(a: torch.Tensor, b: torch.Tensor):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quat_inv(quat: torch.Tensor):
    coeff = torch.tensor([1, -1, -1, -1], device = quat.device)
    # broadcast
    return quat * coeff

def quat_rot(rot_params: torch.Tensor, point: torch.Tensor):
    # rot_params: (N_batch, N_rot, N_sample, 4)
    # point: (N_batch, N_rot, N_sample, 3)
    # Get a column of zeros
    zeros = point.new_zeros(point.shape[ :-1] + (1, ))
    point_quat = torch.cat((zeros, point), dim = -1)
    new_point = quat_mul(quat_mul(rot_params, point_quat), quat_inv(rot_params))

    return new_point[..., 1: ]
