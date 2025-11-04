# medlora/lora.py
from __future__ import annotations
from typing import Tuple, List
import torch.nn as nn
from peft import LoraConfig, get_peft_model

# Apply LoRA to every Linear + Conv3d leaf module in the network (encoder, decoder, head).
_INCLUDE_TYPES: Tuple[type, ...] = (nn.Linear, nn.Conv3d)


def _collect_target_module_names(model: nn.Module) -> List[str]:
    """
    Collect fully-qualified names for all leaf modules that are instances of
    the types in _INCLUDE_TYPES.
    """
    names: List[str] = []
    for name, mod in model.named_modules():
        if not name:
            continue
        # treat only leaf modules to avoid wrapping parents
        if isinstance(mod, _INCLUDE_TYPES) and len(list(mod.children())) == 0:
            names.append(name)
    if not names:
        raise RuntimeError(
            "No target modules found for LoRA. Check your model or supported layer types."
        )
    return names


def apply_lora(
    model: nn.Module,
    *,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> nn.Module:
    """
    Wrap ALL nn.Linear + nn.Conv3d layers with PEFT LoRA. No exclusions.
    (Base weights frozen; only LoRA adapters are trainable.)
    """
    target_modules = _collect_target_module_names(model)

    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,  
        modules_to_save=None,  # Just apply it to all target modules
    )
    peft_model = get_peft_model(model, cfg)
    return peft_model
