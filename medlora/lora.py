from typing import List
import torch
from peft import LoraConfig, get_peft_model

# Target the encoder linear maps only (qkv/proj and MLP linears)
TARGETS = ["qkv","proj","linear1","linear2"]

def apply_lora_to_swin_encoder(model, r=8, alpha=16, dropout=0.0, bias="none"):
    """
    Freezes the whole model, injects LoRA into model.swinViT (encoder) for selected Linear layers,
    then unfreezes decoder & seg head. Returns the model with PEFT injected.
    """
    # 1) Freeze everything
    for p in model.parameters(): p.requires_grad = False

    # 2) Inject LoRA into the encoder ONLY
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias=bias,
        target_modules=TARGETS, task_type="FEATURE_EXTRACTION"
    )
    model.swinViT = get_peft_model(model.swinViT, cfg)

    # 3) Unfreeze decoder + output head (everything not under 'swinViT.')
    for name, p in model.named_parameters():
        if not name.startswith("swinViT."):
            p.requires_grad = True
        else:
            # keep base encoder frozen; LoRA weights inside swinViT are already trainable
            if "lora_" in name: p.requires_grad = True
    return model
