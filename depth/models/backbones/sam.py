from mmcv.runner import BaseModule

from ..builder import BACKBONES


from functools import partial
from typing import Dict, Tuple

import timm
import torch
import torch.nn.functional as F


def preprocess(x: torch.Tensor, new_size: int) -> torch.Tensor:
    # Resize
    oldh, oldw = x.shape[-2:]
    scale = new_size * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    x = F.interpolate(x, (newh, neww), mode="bicubic", align_corners=False)

    # Pad
    padh = new_size - newh
    padw = new_size - neww
    x = F.pad(x, (0, padw, 0, padh))

    return x


def _convert_sam(
    state_dict: Dict[str, torch.Tensor],
    new_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    for k, v in state_dict.items():
        if k == "pos_embed":
            # Resize pos embedding
            if v.shape[1:3] != new_state_dict[k].shape[1:3]:
                v = F.interpolate(
                    v.permute(0, 3, 1, 2),
                    size=new_state_dict[k].shape[1:3],
                    mode="bicubic",
                    antialias=False,
                )
                v = v.permute(0, 2, 3, 1)
        elif "rel_pos" in k:
            # Interpolate rel pos if needed.
            max_rel_dist = new_state_dict[k].shape[0]
            if v.shape[0] != max_rel_dist:
                v = F.interpolate(
                    v.reshape(1, v.shape[0], -1).permute(0, 2, 1),
                    size=max_rel_dist,
                    mode="linear",
                )
                v = v.reshape(-1, max_rel_dist).permute(1, 0)
        out_dict[k] = v
    return out_dict


@BACKBONES.register_module()
class SAM(BaseModule):
    def __init__(
        self,
        backbone_type: str,
        img_size: Tuple[int, int],
        out_indices: Tuple[int],
        resize: bool = True,
        freeze: bool = True,
        pretrain: bool = True,
    ):
        super().__init__()
        timm_name = "samvit_base_patch16"
        log = f"{backbone_type}: Loading {timm_name} backbone from timm, "

        if "sam" in backbone_type and resize:
            log += "Resize image to 1024x1024, "
            img_size = 1024
            self.preprocess = partial(preprocess, new_size=img_size)
        else:
            self.preprocess = None

        if "sam" in backbone_type and not resize:
            timm_pretrain = False
        else:
            timm_pretrain = pretrain

        self.model = timm.create_model(
            timm_name, img_size=img_size, num_classes=0, global_pool="", pretrained=timm_pretrain
        )
        self.embed_dim = self.model.embed_dim

        # Load pretrained weight for SAM
        if "sam" in backbone_type and not resize:
            log += "Loading SAM weights for different image size, "
            state_dict = timm.create_model(
                timm_name,
                num_classes=0,
                global_pool="",
                pretrained=True,
            ).state_dict()
            state_dict = _convert_sam(state_dict, self.model.state_dict())
            self.model.load_state_dict(state_dict)

        # Remove useless layers
        if "dinov2" in backbone_type or "lip" in backbone_type or "vit" in backbone_type:
            self.model.norm = None
        elif "sam" in backbone_type:
            self.model.neck = None

        if "sam" in backbone_type or "swin" in backbone_type:
            self.model.forward = partial(
                self.model.forward_intermediates,
                indices=out_indices,
                norm=False,
                intermediates_only=True,
            )
        else:
            self.model.forward = partial(
                self.model.forward_intermediates,
                indices=out_indices,
                return_prefix_tokens=False,
                norm=False,
                intermediates_only=True,
            )

        if freeze:
            log += "Freeze backbone"
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            log += "Tune backbone"

        print(log)

    def forward(self, x: torch.Tensor):
        if self.preprocess is not None:
            x = self.preprocess(x=x)
        return self.model(x)
