# medlora/lora.py
from __future__ import annotations
import math
import torch
import torch.nn as nn

# Target only the Linear layers used in attention & MLP inside the Swin encoder
TARGETS = ("qkv", "proj", "linear1", "linear2")


class LoRALinear(nn.Module):
    """
    Wrap an nn.Linear with a rank-r LoRA branch:
        y = W x + (alpha/r) * B(A x)
    Base (W, b) are frozen; only A, B are trainable. Forward signature unchanged.
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / max(1, self.r)

        in_f, out_f = base.in_features, base.out_features
        dev  = base.weight.device
        dt   = base.weight.dtype

        # LoRA factors on SAME device/dtype as the base Linear
        self.A = nn.Parameter(torch.zeros(self.r, in_f, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(out_f, self.r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # mark so we can count them cleanly later
        self.A._is_lora_param = True  # type: ignore[attr-defined]
        self.B._is_lora_param = True  # type: ignore[attr-defined]

        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Freeze base weights
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        dx = (self.drop(x) @ self.A.t()) @ self.B.t()
        return y + self.scaling * dx


def _inject_lora_linear(module: nn.Module, target_names=TARGETS, r=8, alpha=16, dropout=0.0) -> int:
    """
    Recursively replace nn.Linear children whose attribute name contains any of target_names.
    Returns count of layers wrapped.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(t in name for t in target_names):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
        else:
            replaced += _inject_lora_linear(child, target_names, r, alpha, dropout)
    return replaced


def apply_lora_to_encoder(model, r=8, alpha=16, dropout=0.0):
    """
    Freeze everything; inject LoRA into encoder Linear layers (qkv/proj/linear1/linear2);
    then unfreeze ONLY:
      • LoRA A/B tensors in the encoder
      • final segmentation head ('out.*') so classes can adapt

    Decoder base weights remain frozen for true parameter efficiency.
    """
    # 1) Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # 2) Inject LoRA into the Swin encoder only
    n_wrapped = _inject_lora_linear(model.swinViT, TARGETS, r=r, alpha=alpha, dropout=dropout)
    print(f"[LoRA] Wrapped {n_wrapped} Linear layers in encoder (r={r}, alpha={alpha}).")

    # 3) Ensure LoRA A/B are trainable
    for m in model.swinViT.modules():
        if isinstance(m, LoRALinear):
            for p in m.parameters():
                p.requires_grad = True

    # 4) Unfreeze final segmentation head
    for name, p in model.named_parameters():
        if name.startswith("out."):
            p.requires_grad = True

    return model
