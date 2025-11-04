# medlora/lora.py
from __future__ import annotations
import math
import torch
import torch.nn as nn

TARGETS = ("qkv", "proj", "linear1", "linear2")

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear")
        self.base = base
        self.r = int(r); self.alpha = int(alpha)
        self.scaling = self.alpha / max(1, self.r)
        in_f, out_f = base.in_features, base.out_features
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.zeros(self.r, in_f, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(out_f, self.r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)); nn.init.zeros_(self.B)
        self.A._is_lora_param = True  # for accounting
        self.B._is_lora_param = True
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        for p in self.base.parameters(): p.requires_grad = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        dx = (self.drop(x) @ self.A.t()) @ self.B.t()
        return y + self.scaling * dx

def _inject_lora_linear(module: nn.Module, target_names=TARGETS, r=8, alpha=16, dropout=0.0) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(t in name for t in target_names):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout)); replaced += 1
        else:
            replaced += _inject_lora_linear(child, target_names, r, alpha, dropout)
    return replaced

def apply_lora_to_encoder(model, r=8, alpha=16, dropout=0.0):
    """
    LoRA-on-encoder + Full-FT-on-decoder:
      • Freeze all base weights.
      • Inject LoRA only into encoder (qkv/proj/linear1/linear2).
      • Train ONLY LoRA A/B inside encoder (encoder base stays frozen).
      • UNFREEZE the entire decoder + seg head (full FT on decoder & head).
    """
    for p in model.parameters(): p.requires_grad = False
    n_wrapped = _inject_lora_linear(model.swinViT, TARGETS, r=r, alpha=alpha, dropout=dropout)
    print(f"[LoRA] Wrapped {n_wrapped} Linear layers in encoder (r={r}, alpha={alpha}).")
    # LoRA trainable; keep wrapped base Linear frozen
    for m in model.swinViT.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad_(True); m.B.requires_grad_(True)
            for p in m.base.parameters(): p.requires_grad_(False)
    # Full-FT on decoder + head
    for name, p in model.named_parameters():
        if not name.startswith("swinViT."):  # everything outside encoder
            p.requires_grad = True
    return model
