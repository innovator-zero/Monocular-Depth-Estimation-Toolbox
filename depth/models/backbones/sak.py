import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from ..builder import BACKBONES


class Adapter(nn.Module):
    """
    Original Adapter module.
    :param int input_dim: Input dimension.
    :param int output_dim: Output dimension.
    :param int down_ratio: Adapter down ratio.
    """

    def __init__(self, input_dim: int, down_ratio: int, output_dim: int = None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        hidden_dim = int(input_dim // down_ratio)
        self.down = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, output_dim)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        x = x * self.scale
        return x + residual


class GlobalRouter(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, x, hw_shape):
        H, W = hw_shape
        x = self.module(x)  # B, L, C'
        # Global pooling
        x = torch.mean(x, dim=1)  # B, C'
        # B, C' -> B, C', H, W
        x = x.reshape(x.shape[0], x.shape[1], 1, 1).expand(-1, -1, H, W)
        return x


class MoE(nn.Module):

    def __init__(
        self,
        router_type: str,
        input_dim: int,
        tea_names: list,
        noisy_gating: bool = True,
    ):
        super().__init__()
        self.tea_names = tea_names
        self.noisy_gating = noisy_gating

        if noisy_gating:
            router_dim = (len(tea_names) + 1) * 2
        else:
            router_dim = len(tea_names) + 1

        if router_type == "global":
            router_fx = GlobalRouter
        else:
            raise NotImplementedError

        self.router = router_fx(input_dim, router_dim)

    def forward(self, vit_fea, stu_fea_dict, hw_shape):
        H, W = hw_shape
        B = vit_fea.shape[0]

        logits = self.router(vit_fea, hw_shape)  # B, num_experts(*2), H, W

        if self.noisy_gating:
            clean_logits, raw_noise_stddev = logits.chunk(2, dim=1)
            if self.training:
                noise_stddev = F.softplus(raw_noise_stddev) + 1e-2
                eps = torch.randn_like(clean_logits)
                noisy_logits = clean_logits + eps * noise_stddev
                logits = noisy_logits
            else:
                logits = clean_logits

        probs = F.softmax(logits, dim=1)  # B, num_experts, H, W

        # Get feature of each expert
        fea_list = [vit_fea.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()]
        for tea_name in self.tea_names:  # maintain order
            stu_fea = stu_fea_dict[tea_name]
            fea_list.append(stu_fea.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())

        assert len(fea_list) == probs.shape[1]
        fea = torch.stack(fea_list, dim=1)  # B, num_experts, C, H, W
        fea = fea * probs.unsqueeze(2)  # B, num_experts, C, H, W
        fea = fea.sum(dim=1)  # B, C, H, W

        return fea


@BACKBONES.register_module()
class SAK(BaseModule):
    """
    Vision Transformer with Multi-Teacher Adapters.
    :param int img_size: Input image size.
    :param str vit_name: Vision Transformer name.
    :param dict teacher_dims: Dict of teacher dimensions.
    :param int down_ratio: Adapter down ratio.
    :param bool freeze_vit: Freeze Vision Transformer.
    """

    def __init__(
        self,
        img_size: tuple,
        tea_dims: dict,
        down_ratio: int = 4,
        router_type: str = "global",
        freeze: bool = True,
        out_indices: list = [2, 5, 8, 11],
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        from .myvit import vit_base_patch16_384

        self.vit = vit_base_patch16_384(img_size=img_size, pretrained=False, dynamic_img_size=True)
        self.vit.norm = None

        self.embed_dim = self.vit.embed_dim
        self.out_indices = out_indices
        # self.fea_size = (img_size[0] // 16, img_size[1] // 16)

        self.tea_names = tea_dims.keys()

        # Define multi-teacher adapters
        self.mt_adapters = nn.ModuleDict()
        for tea_name in self.tea_names:
            self.mt_adapters[tea_name] = nn.ModuleList()
            self.mt_adapters[tea_name].append(Adapter(self.embed_dim, down_ratio))  # for patch embed

        self.out_norms = nn.ModuleDict()

        # Define multi-teacher MoEs for each task output
        if router_type == "none":
            self.moes = None
        else:
            self.moes = nn.ModuleDict()

        for i in range(len(self.vit.blocks)):
            # A group of adapters for each block
            for tea_name in self.tea_names:
                self.mt_adapters[tea_name].append(Adapter(self.embed_dim, down_ratio))

            if i in self.out_indices:
                # A group of output norms for each adapter
                for tea_name in self.tea_names:
                    self.out_norms[str(i) + "_" + tea_name] = nn.LayerNorm(self.embed_dim)

                # A group of MoEs for each task
                if router_type != "none":
                    self.moes[str(i)] = MoE(
                        "global",
                        self.embed_dim,
                        self.tea_names,
                        True,
                    )

        # Init weights for adapters and aligners
        self.mt_adapters.apply(self._init_weights)
        self.out_norms.apply(self._init_weights)

        if self.moes:
            self.moes.apply(self._init_weights)

        if freeze:
            for name, param in self.named_parameters():
                if not "moe" in name:
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out_feas_list = []

        # Forward pass
        B, _, height, width = x.shape
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        residuals_dict = {}  # dict of residuals for each adapter
        for tea_name in self.tea_names:
            residuals_dict[tea_name] = self.mt_adapters[tea_name][0](x)

        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)

            # Get adapter features for each teacher
            for tea_name in self.tea_names:
                residuals_dict[tea_name] = self.mt_adapters[tea_name][i + 1](residuals_dict[tea_name] + x)

            if i in self.out_indices:
                stu_fea_dict = {}
                # dict of student features in this block {tea_name: [B, L, C]}
                for tea_name in self.tea_names:
                    # Student out feature = residual
                    stu_fea = residuals_dict[tea_name][:, 1:]
                    stu_fea = self.out_norms[str(i) + "_" + tea_name](stu_fea)
                    stu_fea_dict[tea_name] = stu_fea

                # Get ViT feature tokens
                cls_token = x[:, 0]
                vit_fea = x[:, 1:]

                hw_shape = self.vit.patch_embed.dynamic_feat_size((height, width))
                if self.moes is not None:
                    # Get MoE output features for each task
                    out_fea = self.moes[str(i)](vit_fea, stu_fea_dict, hw_shape)
                else:
                    out_fea = stu_fea_dict["dinov2"]
                    out_fea = out_fea.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2).contiguous()

                out_feas_list.append((out_fea, cls_token))

        return out_feas_list
