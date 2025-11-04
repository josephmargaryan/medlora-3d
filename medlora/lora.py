# medlora/lora.py (force LoRA on every nn.Linear and nn.Conv3d, including head)
from __future__ import annotations
from typing import Tuple, List
import torch.nn as nn
from peft import LoraConfig, get_peft_model

_INCLUDE_TYPES: Tuple[type, ...] = (nn.Linear, nn.Conv3d)

def _collect_target_module_names(model: nn.Module) -> List[str]:
    names: List[str] = []
    for name, mod in model.named_modules():
        if name and isinstance(mod, _INCLUDE_TYPES):
            names.append(name)
    return names

def apply_lora(
    model: nn.Module,
    *,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> nn.Module:
    target_modules = _collect_target_module_names(model)
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,   # all Linear + Conv3d (head included)
        modules_to_save=None,            # nothing excluded / fully-trainable
    )
    return get_peft_model(model, cfg)
