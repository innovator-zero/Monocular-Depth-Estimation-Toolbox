def cal_params(model, logger):
    stu_params = 0
    stu_backbone_params = 0
    stu_trainable_params = 0
    stu_mt_adapter_params = 0
    stu_mt_aligner_params = 0
    stu_moe_params = 0
    stu_head_params = 0

    for name, param in model.named_parameters():
        stu_params += param.numel()
        if param.requires_grad:
            stu_trainable_params += param.numel()

        if "vit" in name:
            stu_backbone_params += param.numel()
        elif "mt_adapters" in name:
            stu_mt_adapter_params += param.numel()
        elif "mt_aligners" in name:
            stu_mt_aligner_params += param.numel()
        elif "moes" in name:
            stu_moe_params += param.numel()
        elif "head" in name:
            stu_head_params += param.numel()

    logger.info("--- Number of parameters ---")
    logger.info(f"All:      {stu_params/1e6:>10.2f}M")
    logger.info(f"Trainable:    {stu_trainable_params/1e6:>10.2f}M")
    logger.info(f"Backbone:     {stu_backbone_params/1e6:>10.2f}M")
    logger.info(f"MT Adapters:  {stu_mt_adapter_params/1e6:>10.2f}M")
    logger.info(f"MT Aligners:  {stu_mt_aligner_params/1e6:>10.2f}M")
    logger.info(f"MoEs:         {stu_moe_params/1e6:>10.2f}M")
    logger.info(f"Heads:        {stu_head_params/1e6:>10.2f}M")

    return stu_params
