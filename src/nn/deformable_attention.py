"""
Pure PyTorch Multi-Scale Deformable Attention
=============================================
Drop-in replacement for mmcv.ops.MultiScaleDeformableAttention.
No CUDA compilation needed — compatible with Windows, Linux, macOS.

Based on Deformable DETR (Zhu et al., ICLR 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                         sampling_locations, attention_weights):
    """Pure PyTorch multi-scale deformable attention.

    Args:
        value:            [B, sum(H*W), n_head, C]
        value_spatial_shapes: [n_levels, 2] — (H, W) per level
        sampling_locations:   [B, Nq, n_head, n_levels, n_points, 2]
        attention_weights:    [B, Nq, n_head, n_levels, n_points]

    Returns:
        output: [B, Nq, n_head*C]
    """
    B, Nq, n_head, n_levels, n_points, _ = sampling_locations.shape
    _, _, _, C = value.shape

    split_shape = [(h * w).item() for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)

    # Convert sampling locations from [0,1] to [-1,1] for grid_sample
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        h, w = h.item(), w.item()
        value_l = value_list[level]  # [B, H*W, n_head, C]

        # [B, H*W, n_head, C] → [B*H*W, n_head, C]
        B_, N_, nH_, C_ = value_l.shape
        value_l = value_l.reshape(B_, h, w, nH_, C_)  # [B, H, W, n_head, C]
        value_l = value_l.permute(0, 3, 4, 1, 2)  # [B, n_head, C, H, W]
        value_l = value_l.reshape(B_ * n_head, C_, h, w)

        # Sampling grid: [B, Nq, n_head, n_levels, n_points, 2]
        # → [B, n_head, Nq, n_levels, n_points, 2]
        # → [B * n_head, Nq, n_points, 2]
        sampling_grid_l = sampling_grids[:, :, :, level]  # [B, Nq, n_head, n_points, 2]
        sampling_grid_l = sampling_grid_l.permute(0, 2, 1, 3, 4)  # [B, n_head, Nq, n_points, 2]
        sampling_grid_l = sampling_grid_l.reshape(B * n_head, Nq, n_points, 2)

        sampling_value_l = F.grid_sample(
            value_l, sampling_grid_l,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )  # [B*n_head, C, Nq, n_points]

        sampling_value_l = sampling_value_l.reshape(
            B, n_head, C, Nq, n_points
        ).permute(0, 3, 1, 4, 2)  # [B, Nq, n_head, n_points, C]

        sampling_value_list.append(sampling_value_l)

    # Stack levels: [B, Nq, n_head, n_levels, n_points, C]
    sampling_values = torch.stack(sampling_value_list, dim=-3)

    # Apply attention weights: [B, Nq, n_head, n_levels, n_points, C]
    attention_weights = attention_weights.unsqueeze(-1)  # [B, Nq, n_head, n_levels, n_points, 1]
    output = (sampling_values * attention_weights).sum(dim=4).sum(dim=3)
    # After sum(dim=4): [B, Nq, n_head, n_levels, C]
    # After sum(dim=3): [B, Nq, n_head, C]

    output = output.permute(0, 1, 3, 2).reshape(B, Nq, n_head * C)
    return output


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module (pure PyTorch).

    Drop-in replacement for mmcv.ops.MultiScaleDeformableAttention.

    Args:
        embed_dims:  feature dimension (d_model)
        num_heads:   number of attention heads
        num_levels:  number of feature pyramid levels
        num_points:  number of sampling points per head per level
    """

    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

        # Initialize reference points as uniform grid
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = (grid_init / grid_init.abs().max() * 0.5 + 0.5)
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        grid_init = grid_init.reshape(-1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init)

    def forward(self, query, value, spatial_shapes, level_start_index, key_padding_mask=None):
        """Forward pass.

        Args:
            query:           [B, Nq, C]
            value:           [B, sum(H*W), C]
            spatial_shapes:  [n_levels, 2]
            level_start_index: [n_levels]

        Returns:
            out: [B, Nq, C]
        """
        B, Nq, C = query.shape
        _, Nv, _ = value.shape
        n_levels = spatial_shapes.shape[0]

        # Value projection
        value = self.value_proj(value)  # [B, Nv, C]
        # Split into heads
        value = value.view(B, Nv, self.num_heads, C // self.num_heads)

        # Sampling offsets
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            B, Nq, self.num_heads, n_levels, self.num_points, 2
        )

        # Attention weights
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            B, Nq, self.num_heads, n_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            B, Nq, self.num_heads, n_levels, self.num_points
        )

        # Generate reference points
        if spatial_shapes.dim() == 1:
            spatial_shapes = spatial_shapes.unsqueeze(0)
            spatial_shapes = spatial_shapes.expand(n_levels, 2)

        offset_normalizer = torch.stack(
            [spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1
        ).float()  # [n_levels, 2]

        # Build reference points for ALL levels, then gather per query
        ref_points_all = []
        for lvl in range(n_levels):
            h, w = spatial_shapes[lvl]
            h, w = h.item(), w.item()
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, device=query.device),
                torch.linspace(0.5, w - 0.5, w, device=query.device),
                indexing='ij'
            )
            ref_yx = torch.stack([grid_y, grid_x], dim=-1)
            ref_yx = ref_yx.reshape(1, -1, 2)  # [1, H*W, 2]
            ref_yx = ref_yx / offset_normalizer[lvl].view(1, 1, 2)
            ref_points_all.append(ref_yx)
        ref_points = torch.cat(ref_points_all, dim=1)  # [1, sum(H*W), 2]

        # For self-attention: each position is its own reference
        # Expand reference points for all heads, levels, points
        # ref_points: [B, Nq, 2] → [B, Nq, n_head, n_levels, n_points, 2]
        B = query.shape[0]
        query_ref = ref_points[:, :, None, None, None, :].expand(
            B, Nq, self.num_heads, n_levels, self.num_points, 2
        ).clone()

        sampling_locations = query_ref + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)
        return output
