import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed
import torch.nn.functional as F
from mmdit import DismantledBlock


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    scaling_factor=None,
    offset=None,
):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    if scaling_factor is not None:
        grid = grid / scaling_factor
    if offset is not None:
        grid = grid - offset

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


class PositionalPatchEmbedder(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        pos_embed_scaling_factor: Optional[float] = None,
        pos_embed_offset: Optional[float] = None,
        pos_embed_max_size: Optional[int] = None,
    ):
        super().__init__()
        self.patcher = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            strict_img_size=pos_embed_max_size is None,
        )

        self.pos_embed_max_size = pos_embed_max_size
        pos_embed_grid_size = (
            int(self.patcher.num_patches**0.5)
            if pos_embed_max_size is None
            else pos_embed_max_size
        )
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim,
            pos_embed_grid_size,
            scaling_factor=pos_embed_scaling_factor,
            offset=pos_embed_offset,
        )
        self.pos_embed: torch.Tensor
        self.register_buffer(
            "pos_embed", torch.from_numpy(pos_embed).float()[None, ...]
        )

    def cropped_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        if self.pos_embed_max_size is None:
            return self.pos_embed
        p: int = self.patcher.patch_size[0]
        h, w = hw
        # patched size
        h = h // p
        w = w // p
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(
            self.pos_embed,
            "1 (h w) c -> 1 h w c",
            h=self.pos_embed_max_size,
            w=self.pos_embed_max_size,
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, "1 h w c -> 1 (h w) c")
        return spatial_pos_embed

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.patcher(x) + self.cropped_pos_embed(x.shape[-2:])


# Copied from src/diffusers/models/embeddings.py:27
def get_time_proj_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_time_proj_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class TextProjection(nn.Module):

    def __init__(self, in_features, hidden_size, out_features=None):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(
            in_features=in_features, out_features=hidden_size, bias=True
        )
        self.act_1 = nn.SiLU()
        self.linear_2 = nn.Linear(
            in_features=hidden_size, out_features=out_features, bias=True
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# copied from src/diffusers/models/embeddings.py:543
class TimestepProjection(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, sample_proj_bias)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepProjection(
            in_channels=256, time_embed_dim=embedding_dim
        )
        self.text_embedder = TextProjection(pooled_projection_dim, embedding_dim)

    def forward(self, timestep: Tensor, pooled_projection: Tensor):
        timesteps_proj: Tensor = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=pooled_projection.dtype)
        )

        pooled_projection = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projection

        return conditioning


class ControlNetEmbedder(nn.Module):

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        attention_head_dim: int,
        num_attention_heads: int,
        pooled_projection_dim: int,
        joint_attention_dim: int,
        caption_projection_dim: int,
        num_layers: int,
        pos_embed_max_size: Optional[int] = None,
        resize_cond_if_needed: bool = False,
        use_pre_embed_x: bool = False,
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.inner_dim,
            strict_img_size=pos_embed_max_size is None,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
        )
        self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)
        self.transformer_blocks = nn.ModuleList(
            DismantledBlock(
                hidden_size=self.inner_dim, num_heads=num_attention_heads, qkv_bias=True
            )
            for _ in range(num_layers)
        )
        self.resize_cond_if_needed = resize_cond_if_needed
        self.use_pre_embed_x = use_pre_embed_x

        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            controlnet_block = nn.Linear(self.inner_dim, self.inner_dim)
            self.controlnet_blocks.append(controlnet_block)

        self.pos_embed_input = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.inner_dim,
            strict_img_size=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        controlnet_cond: Tensor,
        conditioning_scale: int = 1,
        encoder_hidden_states: Optional[Tensor] = None,
        pooled_projections: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        if self.use_pre_embed_x:
            hidden_states = self.pos_embed(hidden_states)

        if pooled_projections is None:
            pooled_projections = torch.zeros_like(encoder_hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        hidden_states = hidden_states + self.pos_embed_input(controlnet_cond)

        block_res_samples = ()

        for block in self.transformer_blocks:
            out = block(hidden_states, temb)
            hidden_states = out
            block_res_samples += (hidden_states,)

        controlnet_block_res_samples = ()
        for block_res_sample, controlnet_block in zip(
            block_res_samples, self.controlnet_blocks
        ):
            block_res_sample = controlnet_block(block_res_sample)
            controlnet_block_res_samples = controlnet_block_res_samples + (
                block_res_sample,
            )

        # scale the controlnet outputs
        controlnet_block_res_samples = [
            sample * conditioning_scale for sample in controlnet_block_res_samples
        ]
        return {
            "hidden_states": hidden_states,
            "controlnet_block_res_samples": controlnet_block_res_samples,
        }
