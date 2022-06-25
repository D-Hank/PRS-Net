import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

import quat

from typing import List, Tuple

from settings import *

class Network(nn.Module):
    def __init__(self, in_channel_0 = IN_CHANNEL_0, out_channel_0 = OUT_CHANNEL_0, num_layers = NUM_CONV_LAYERS):
        super(Network, self).__init__()

        in_channel = in_channel_0
        out_channel = out_channel_0
        self.num_layers = num_layers
        self.cnn_layers = nn.ModuleList([])
        for i in range(0, num_layers):
            self.cnn_layers.append(torch.nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1),
                nn.MaxPool3d(kernel_size = 2),
                nn.LeakyReLU(negative_slope = 0.2, inplace = True)
            ))
            in_channel = out_channel
            out_channel *= 2

    def forward(self, cnn_in: torch.Tensor):
        # Sequential/iterate
        cnn_out = None
        for layer in self.cnn_layers:
            cnn_out = layer(cnn_in)
            cnn_in = cnn_out

        return cnn_out

class MLPHead(nn.Module):
    def __init__(self, in_channel = 64, num_plane = 3, num_rot = 3, activation = nn.LeakyReLU(0.2, True)):
        super(MLPHead, self).__init__()
        self.num_plane = num_plane
        self.num_rot = num_rot
        self.in_channel = in_channel

        # Depth-wise convolution: parallel linear layers
        __group = (num_plane + num_rot)
        __in = in_channel * __group
        __mid = in_channel // 2 * __group
        __out = in_channel // 4 * __group
        self.mlp_layers = nn.Sequential(
            nn.Conv1d(
                in_channels = __in,
                out_channels = __mid,
                kernel_size = 1,
                groups = __group
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(
                in_channels = __mid,
                out_channels = __out,
                kernel_size = 1,
                groups = __group
            ),
            nn.LeakyReLU(0.2, True)
        )

        self.last_layer = nn.Conv1d(
            in_channels = __out,
            out_channels = 4 * __group,
            kernel_size = 1,
            groups = __group
        )

    def init_params(self, bias):
        rot_part = torch.rand(4 * self.num_rot, self.in_channel // 4, 1) / 10.0 - 0.05
        plane_part = torch.rand(4 * self.num_plane, self.in_channel // 4, 1) / 10.0 - 0.05
        self.last_layer.weight.data = torch.cat((plane_part, rot_part), dim = 0)
        self.last_layer.bias.data = bias

    def forward(self, feature: torch.Tensor):
        # feature: (N_batch, 64) -> (N_batch, 64*6, 1)
        input_vec = feature.unsqueeze(-1).repeat(1, self.num_plane + self.num_rot, 1)
        # out: (N_batch, 4*6, 1) -> (N_batch, 6, 4)
        output_vec = self.last_layer(self.mlp_layers(input_vec)).reshape(-1, self.num_plane + self.num_rot, 4)
        plane_out, rot_out = torch.split(output_vec, [self.num_plane, self.num_rot], 1)
        # (N_batch, 3, 4)
        # For plane, we only normalize a,b,c
        plane_out = plane_out / (torch.norm(plane_out[ : , : , : 3], p = 2, dim = 2) + 1e-12).unsqueeze(-1).expand(-1, -1, 4)
        rot_out = functional.normalize(rot_out, p = 2, dim = 2, eps = 1e-12)

        return plane_out, rot_out

class PRSModel(nn.Module):
    def init_params(self):
        init_angle = np.sin(np.pi / 2)
        mlp_bias = torch.tensor(
            [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,

                0, init_angle, 0, 0,
                0, 0, init_angle, 0,
                0, 0, 0, init_angle
            ]
        ).to(torch.float)
        self.mlp_head.init_params(mlp_bias)

    def __init__(
        self,
        device = GPU,
        in_channel_0 = IN_CHANNEL_0,
        out_channel_0 = OUT_CHANNEL_0,
        num_conv_layers = NUM_CONV_LAYERS,
        num_plane = NUM_PLANE,
        num_rot = NUM_ROT,
        num_sample = N_SAMPLE,
        batch_size = BATCH_SIZE,
        voxel_size = VOXEL_SIZE,
        lower_bound = LOWER_BOUND,
        upper_bound = UPPER_BOUND,
        grid_size = GRID_SIZE,
        regular_weight = REGULAR_WEIGHT,
        ):
        super(PRSModel, self).__init__()
        self.num_plane = num_plane
        self.num_rot = num_rot
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.voxel_size = voxel_size
        self.lower_bound = torch.tensor(lower_bound).to(device)
        self.upper_bound = torch.tensor(upper_bound).to(device)
        self.grid_size = torch.tensor(grid_size).to(device)
        self.regular_weight = regular_weight
        self.device = device

        # (3, 1)
        # point order: (x, y, z), voxel index: (z, y, x)
        # maps: p=(1, 0, 0) -> idx=1*1024 + 0*32 + 0*1 -> closest[1024]
        self.serialize_coeff = torch.tensor([[voxel_size * voxel_size], [voxel_size], [1]]).to(torch.float).to(device)

        self.network = Network(in_channel_0, out_channel_0, num_conv_layers)
        self.mlp_head = MLPHead(out_channel_0 * (2 ** (num_conv_layers - 1)), num_plane, num_rot)

        self.init_params()

    def forward(self, voxel: torch.Tensor):
        net_out = self.network(voxel)
        # net_out: (N_batch, 64, 1, 1, 1)
        plane_params, rot_params = self.mlp_head(net_out[ : , : , 0, 0, 0])

        return plane_params, rot_params

    # Get the serialized coordinates
    def point_to_vox(self, transformed_points: torch.Tensor) -> torch.Tensor:
        # points: (N_batch, N_sym, N_sample, 3)
        # [0, 1) -> 0, ..., [31, 32) -> 31
        # Reduce each point from interval to endpoint to get its grid center
        points_floor = torch.round(transformed_points).clamp(0, self.voxel_size - 1)
        # Broadcast multiplication: (N_batch, N_sym, N_sample, 3) * (3, 1) -> (N_batch, N_sym, N_sample, 1)
        # (N_batch, N_sym, N_sample)
        return torch.matmul(points_floor, self.serialize_coeff).to(torch.int64).squeeze()

    def reflect_trans(self, sampled_points: torch.Tensor, plane_params: torch.Tensor) -> torch.Tensor:
        # sampled_points: (N_batch, N_plane, N_sample, 3)
        # plane_params: (N_batch, N_plane, 4) -> (N_batch, N_plane, N_sample, 4)
        plane_params = plane_params.unsqueeze(2).expand(-1, -1, self.num_sample, -1)
        # norm_vec: (N_batch, N_plane, N_sample, 3), offset: (N_batch, N_plane, N_sample, 1)
        norm_vec, plane_offset = plane_params.split((3, 1), dim = -1)
        # foreach batch, plane and sample
        # (N_batch, N_plane, N_sample, 1, 3) * (N_batch, N_plane, N_sample, 3, 1) -> (N_batch, N_plane, N_sample, 1, 1)
        dot = torch.matmul(sampled_points.unsqueeze(3), norm_vec.unsqueeze(4))

        # (N_batch, N_plane, 1, 3) -> (N_batch, N_plane, 1, 1), copy to save calculation
        norm_square = torch.sum(torch.square(norm_vec[ : , : , 0, : ]), dim = -1, keepdim = True).unsqueeze(-1)
        # (N_batch, N_plane, N_sample, 1)
        scale = (plane_offset + dot[ : , : , : , : , 0]) / (norm_square + 1e-12)
        # (N_batch, N_plane, N_sample, 3)
        shift = -2.0 * scale * norm_vec
        # (N_batch, N_plane, N_sample, 3)
        reflect_points = shift + sampled_points

        return reflect_points

    def rotate_trans(self, sampled_points: torch.Tensor, rot_params: torch.Tensor) -> torch.Tensor:
        # sampled_points: (N_batch, N_rot, N_sample, 3)
        # rot_params: (N_batch, N_rot, 4) -> (N_batch, N_rot, N_sample, 4)
        rot_params = rot_params.unsqueeze(2).expand(-1, -1, self.num_sample, -1)

        # (N_batch, N_rot, N_sample, 3)
        return quat.quat_rot(rot_params = rot_params, point = sampled_points)

    # push coordinates in voxel space out to mesh/sampling space
    def push_out(self, sampled_points: torch.Tensor) -> torch.Tensor:
        # sampled_points: (N_batch, N_sample, 3)
        cube_coord = (sampled_points + self.grid_size / 2) / self.voxel_size
        mesh_coord = cube_coord + self.lower_bound

        return mesh_coord

    # After transformation, pull mesh coordinates back into voxel space
    def pull_back(self, trans_points: torch.Tensor) -> torch.Tensor:
        # trans_points: (N_batch, N_sym, N_sample, 3)
        cube_coord = trans_points - self.lower_bound
        vox_coord = cube_coord * self.voxel_size - self.grid_size / 2

        return vox_coord

    def dist_loss(
        self,
        sampled_points: torch.Tensor,
        plane_params: torch.Tensor,
        rot_params: torch.Tensor,
        closest_pt: torch.Tensor
        ) -> List[torch.Tensor]:
        # voxel: (N_batch, voxel_size, v_size, v_size)
        # sampled_points: (N_batch, N_sample, 3)
        # plane_params: (N_batch, N_plane, 4)
        # rot_params: (N_batch, N_rot, 4)
        # closest: (N_batch, v_size ^ 3, 3), closest point of each grid center

        # (N_batch, N_sample, 3) -> (N_batch, N_plane, N_sample, 3)
        plane_sample = sampled_points.unsqueeze(1).expand(-1, self.num_plane, -1, -1)
        # (N_batch, N_sample, 3) -> (N_batch, N_rot, N_sample, 3)
        rot_sample = sampled_points.unsqueeze(1).expand(-1, self.num_rot, -1, -1)

        # mesh space -> mesh space
        # Save for later use
        self.ref_trans_points = self.reflect_trans(plane_sample, plane_params)
        self.rot_trans_points = self.rotate_trans(rot_sample, rot_params)

        # Voxel coordinates of sample points: mesh space -> voxel space
        ref_trans_vox = self.pull_back(self.ref_trans_points)
        rot_trans_vox = self.pull_back(self.rot_trans_points)

        # Serialized voxel coordinates, that is, voxel indices of transformed points
        # reflected: (N_batch, N_plane, N_sample, 3) -> (N_batch, N_plane, N_sample)
        reflected_vox = self.point_to_vox(ref_trans_vox)
        # rotated: (N_batch, N_rot, N_sample, 3) -> (N_batch, N_rot, N_sample)
        rotated_vox = self.point_to_vox(rot_trans_vox)

        # foreach batch, plane/rot, sampled_points, use its voxel to get closest-point
        # (N_batch, N_plane, N_sample) -> (N_batch, N_plane, N_sample, 3)
        reflected_idx = reflected_vox.unsqueeze(-1).expand(-1, -1, -1, 3)
        # (N_batch, N_rot, N_sample) -> (N_batch, N_plane, N_sample, 3)
        rotated_idx = rotated_vox.unsqueeze(-1).expand(-1, -1, -1, 3)

        # Get coordinates of closest point
        # Reshape to extract a vector
        # (N_batch, V*V*V, 3) -> (N_batch, 1, V*V*V, 3)
        closest_pt = closest_pt.unsqueeze(1)
        # (N_batch, N_sym, N_sample, 3)
        # voxel space -> mesh space
        # Save for later use
        self.reflected_closest = torch.gather(closest_pt.expand(-1, self.num_plane, -1, -1), dim = 2, index = reflected_idx)
        self.rotated_closest = torch.gather(closest_pt.expand(-1, self.num_rot, -1, -1), dim = 2, index = rotated_idx)

        # Compute distance in mesh space, we use the squared distance (mse) here instead of plain distance
        # (N_batch, N_plane, N_sample, 3)
        ref_dist = functional.mse_loss(self.ref_trans_points, self.reflected_closest, reduction = 'none')
        rot_dist = functional.mse_loss(self.rot_trans_points, self.rotated_closest, reduction = 'none')
        # foreach plane and batch, sum along sample and axis
        # (N_batch, N_plane, N_sample, 3) -> (N_batch, N_plane)
        # Note: for sample, to take mean or to take sum?
        ref_dist = torch.sum(ref_dist, dim = (2, 3))
        rot_dist = torch.sum(rot_dist, dim = (2, 3))

        return ref_dist, rot_dist

    def regular_loss(self, plane_or_rot_vec) -> torch.Tensor:
        # plane_or_rot_vec: (N_batch, N_sym=3, 3)
        M_t = functional.normalize(plane_or_rot_vec, p = 2, dim = -1)
        M = M_t.transpose(1, 2)
        # (N_batch, 3, N_sym) * (N_batch, N_sym, 3) -> (N_batch, 3, 3)
        # Broadcast subtraction
        A_or_B = torch.matmul(M, M_t) - torch.eye(3, 3, device = M.device)

        # foreach batch, sum along N_sym and 3
        norm = torch.sum(torch.square(A_or_B), dim = (1, 2))
        # (N_batch)
        return norm

    def get_loss(
        self,
        sampled_points: torch.Tensor,
        plane_params: torch.Tensor,
        rot_params: torch.Tensor,
        closest_pt_idx: torch.Tensor
        ) -> Tuple[torch.Tensor, dict]:
        # sampled_points: (N_batch, N_sample, 3)
        # plane/rot_params: (N_batch, N_sym=3, 4)
        # closest: (N_batch, v_size^3)

        # dist_loss: (N_batch, N_sym=3) -> 1
        ref_dist_loss, rot_dist_loss = self.dist_loss(sampled_points, plane_params, rot_params, closest_pt_idx)
        total_ref_dist = torch.sum(ref_dist_loss) / self.batch_size
        total_rot_dist = torch.sum(rot_dist_loss) / self.batch_size

        # reg_loss: (N_batch, ) -> 1
        ref_reg_loss = self.regular_loss(plane_params[ : , : , : 3])
        rot_reg_loss = self.regular_loss(rot_params[ : , : , 1 : ])
        total_ref_reg = torch.mean(ref_reg_loss)
        total_rot_reg = torch.mean(rot_reg_loss)

        total_loss = total_ref_dist + total_rot_dist + self.regular_weight * (total_ref_reg + total_rot_reg)

        # Save different loss for both train and test
        loss_dict = {
            "total_loss": total_loss,
            "total_ref_dist": total_ref_dist,
            "total_rot_dist": total_rot_dist,
            "total_ref_reg": total_ref_reg,
            "total_rot_reg": total_rot_reg,
            "each_ref_dist": ref_dist_loss,
            "each_rot_dist": rot_dist_loss
        }

        return total_loss, loss_dict

    # Compute the mutual dihedral_angle
    # Treat rot as plane
    def dihedral_angle(self, vec_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
        # vec: (N_batch, 3)
        # Already normalized, so no need to do it again
        # (N_batch, 1, 3) * (N_batch, 3, 1) -> (N_batch, 1, 1) -> (N_batch)
        return torch.abs(torch.matmul(vec_a.unsqueeze(1), vec_b.unsqueeze(2)).squeeze())

    def validate_remove(
        self,
        plane_params: torch.Tensor,
        rot_params: torch.Tensor,
        each_ref_dist: torch.Tensor,
        each_rot_dist: torch.Tensor,
        loss_threshold: float,
        angle_threshold: float
        ) -> List[torch.Tensor]:
        # plane/rot_params: (N_batch, N_sym=3, 4)
        # each_ref/rot_dist: (N_batch, N_sym=3)
        # mask: (N_batch, N_sym)
        plane_loss_mask = torch.where(each_ref_dist <= loss_threshold, 1, 0)
        rot_loss_mask = torch.where(each_rot_dist <= loss_threshold, 1, 0)

        # Note: In test stage, the dataset is mostly small, so we simply use a loop to check
        # 1: keep, 0: drop
        plane_angle_mask = torch.ones(self.batch_size, self.num_plane)
        for i in range(0, self.num_plane):
            for j in range(i + 1, self.num_plane):
                # (N_batch, )
                plane_cos = self.dihedral_angle(plane_params[ : , i, : 3], plane_params[ : , j, : 3])
                # For each batch, check if this (i,j) pair is bad
                # (N_bad, )
                bad_plane_pair = torch.nonzero(plane_cos >= angle_threshold)
                # (N_bad, )
                higher_ij = torch.where(each_ref_dist[bad_plane_pair, i] >= each_ref_dist[bad_plane_pair, j], i, j)
                plane_angle_mask[bad_plane_pair, higher_ij] = 0

        rot_angle_mask = torch.ones(self.batch_size, self.num_rot)
        for i in range(0, self.num_rot):
            for j in range(i + 1, self.num_rot):
                rot_cos = self.dihedral_angle(rot_params[ : , i, : 3], rot_params[ : , j, : 3])
                # Too close
                bad_rot_pair = torch.nonzero(rot_cos >= angle_threshold)
                higher_ij = torch.where(each_rot_dist[bad_rot_pair, i] >= each_rot_dist[bad_rot_pair, j], i, j)
                rot_angle_mask[bad_rot_pair, higher_ij] = 0

        mask_dict = {
            "plane_sdl": plane_loss_mask,
            "rot_sdl": rot_loss_mask,
            "plane_angle": plane_angle_mask,
            "rot_angle": rot_angle_mask
        }

        return mask_dict
