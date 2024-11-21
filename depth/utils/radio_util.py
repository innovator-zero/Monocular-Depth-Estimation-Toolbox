from functools import partial

import torch

from depth.models import build_depther
from einops import rearrange


def radio_forward(x, fx, patch_size):
    output = fx(x)
    bb_summary, bb_features = output["backbone"]
    # dino_summary, dino_features = output["dino_v2"]
    bb_summary = bb_summary[:, -768:]
    bb_features = rearrange(
        bb_features, "b (h w) d -> b d h w", h=x.shape[-2] // patch_size, w=x.shape[-1] // patch_size
    )

    return [(bb_features, bb_summary)]


def create_depther(cfg, backbone_model, train=True):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    if train:
        depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)
    else:
        depther = build_depther(cfg.model, test_cfg=test_cfg)

    depther.backbone.forward = partial(radio_forward, fx=backbone_model.forward, patch_size=backbone_model._patch_size)

    return depther


def build_radio_depther(cfg, size, train=True):
    backbone_name = f"radio_v2.5-{size[0]}"

    backbone_model = torch.hub.load(
        "NVlabs/RADIO",
        "radio_model",
        version=backbone_name,
        progress=True,
        skip_validation=True,
        adaptor_names="dino_v2",
    )
    # backbone_model.input_conditioner = torch.nn.Identity()
    backbone_model.make_preprocessor_external()

    backbone_model.eval()
    for param in backbone_model.parameters():
        param.requires_grad = False
    backbone_model.cuda()

    model = create_depther(
        cfg,
        backbone_model=backbone_model,
        train=train,
    )

    return model
