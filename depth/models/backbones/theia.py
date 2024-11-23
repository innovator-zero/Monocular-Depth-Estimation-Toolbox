from typing import Tuple

import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import BACKBONES


@BACKBONES.register_module()
class Theia(BaseModule):

    def __init__(
        self,
        backbone_type: str,
        out_indices: Tuple[int],
        freeze: bool = True,
    ):
        super().__init__()
        model_name = "theaiinstitute/theia-base-patch16-224-cdiv"
        log = f"{backbone_type}: Loading {model_name} backbone from huggingface, "

        self.out_indices = out_indices

        if "siglip" in backbone_type:
            from transformers import SiglipVisionModel

            self.model = SiglipVisionModel.from_pretrained(model_name)
            self.model.vision_model.head = nn.Identity()
            self.model.vision_model.post_layernorm = nn.Identity()
            self.embed_dim = self.model.config.hidden_size
        elif "theia" in backbone_type:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).backbone.model
            self.model.layernorm = nn.Identity()
            self.embed_dim = self.model.config.hidden_size

        else:
            raise NotImplementedError

        if freeze:
            log += "Freeze backbone"
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            log += "Tune backbone"

        print(log)

    def forward(self, x: torch.Tensor):
        B, _, height, width = x.shape
        patch_size = self.model.config.patch_size
        H, W = height // patch_size, width // patch_size

        y = self.model(x, output_hidden_states=True, interpolate_pos_encoding=True)
        hidden_states = y.hidden_states[1:]  # remove patch embeddings

        features = []
        for i in self.out_indices:
            if hidden_states[i].shape[1] != H * W:
                cls_token = hidden_states[i][:, 0]
                fea = hidden_states[i][:, 1:]
            else:
                fea = hidden_states[i]

            fea = fea.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            features.append((fea, cls_token))

        return features
