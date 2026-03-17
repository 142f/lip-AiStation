import os

import torch


def strip_state_dict_prefixes(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        name = key
        while name.startswith("module.") or name.startswith("_orig_mod."):
            if name.startswith("module."):
                name = name[7:]
            elif name.startswith("_orig_mod."):
                name = name[10:]
        cleaned[name] = value
    return cleaned


def set_ablation_env(opt):
    os.environ["LIPFD_NO_INNOV"] = "1" if getattr(opt, "no_innov", False) else "0"
    os.environ["LIPFD_NO_MODALITY_BIAS"] = "1" if getattr(opt, "no_modality_bias", False) else "0"
    os.environ["LIPFD_NO_ATTN_BIAS"] = "1" if getattr(opt, "no_attn_bias", False) else "0"
    os.environ["LIPFD_NO_SE_FUSION"] = "1" if getattr(opt, "no_se_fusion", False) else "0"
    os.environ["LIPFD_NO_RESIDUAL_CLS"] = "1" if getattr(opt, "no_residual_cls", False) else "0"
    os.environ["REGION_NO_PE"] = "1" if (getattr(opt, "no_region_pe", False) or getattr(opt, "no_region_innov", False)) else "0"
    os.environ["REGION_NO_SE"] = "1" if (getattr(opt, "no_region_se", False) or getattr(opt, "no_region_innov", False)) else "0"


def load_checkpoint(model, ckpt_path, allow_partial_load=False, prefer_ema=True):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if (
        prefer_ema
        and isinstance(checkpoint, dict)
        and "model_ema" in checkpoint
        and isinstance(checkpoint["model_ema"], dict)
    ):
        state_dict = checkpoint["model_ema"]
        source = "model_ema"
    elif isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
        source = "model"
    else:
        state_dict = checkpoint
        source = "raw"

    state_dict = strip_state_dict_prefixes(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        detail = (
            f"Checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}. "
            f"missing_head={missing[:8]} unexpected_head={unexpected[:8]}"
        )
        if allow_partial_load:
            print(f"[Warning] {detail}")
        else:
            raise RuntimeError(detail)

    return {
        "source": source,
        "missing": missing,
        "unexpected": unexpected,
    }
