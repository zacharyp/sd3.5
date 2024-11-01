import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from mmditx import DismantledBlock, PatchEmbed, VectorEmbedder, TimestepEmbedder

class ControlNetEmbedder(nn.Module):

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        attention_head_dim: int,
        num_attention_heads: int,
        adm_in_channels: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
        pos_embed_max_size: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = num_attention_heads * attention_head_dim
        self.x_embedder = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.hidden_size,
            strict_img_size=pos_embed_max_size is None,
        )

        self.t_embedder = TimestepEmbedder(self.hidden_size, dtype=dtype, device=device)
        self.y_embedder = VectorEmbedder(
            adm_in_channels, self.hidden_size, dtype, device
        )

        self.transformer_blocks = nn.ModuleList(
            DismantledBlock(
                hidden_size=self.hidden_size, num_heads=num_attention_heads, qkv_bias=True
            )
            for _ in range(num_layers)
        )

        # self.use_y_embedder = pooled_projection_dim != self.time_text_embed.text_embedder.linear_1.in_features
        # TODO double check this logic when 8b
        self.use_y_embedder = True

        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            controlnet_block = nn.Linear(self.hidden_size, self.hidden_size)
            self.controlnet_blocks.append(controlnet_block)

        self.pos_embed_input = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.hidden_size,
            strict_img_size=False,
        )

    def forward(
        self,
        x: Tensor,
        x_cond: Tensor,
        y: Tensor,
        scale: int = 1,
        timesteps: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:

        x = self.x_embedder(x)
        timesteps = timesteps * 1000
        c = self.t_embedder(timesteps, dtype=x.dtype)
        if y is not None and self.y_embedder is not None:
            y = self.y_embedder(y)
            c = c + y

        x = x + self.pos_embed_input(x_cond)

        block_out = ()

        for block in self.transformer_blocks:
            out = block(x, c)
            block_out += (out,)

        x_out = ()
        for out, controlnet_block in zip(
            block_out, self.controlnet_blocks
        ):
            out = controlnet_block(out)
            x_out = x_out + (out,)

        # scale the controlnet outputs
        x_out = [sample * scale for sample in x_out]
        return x_out
