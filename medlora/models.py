import re, torch, os
from pathlib import Path
from monai.apps import download_url
from monai.networks.nets import SwinUNETR, UNETR
from .constants import SSL_URL, SSL_PATH

def build_swin_unetr(in_ch, out_ch, use_v2=False, feature_size=48, use_checkpoint=True):
    try:
        return SwinUNETR(spatial_dims=3, in_channels=in_ch, out_channels=out_ch,
                         feature_size=feature_size, use_checkpoint=use_checkpoint, use_v2=use_v2)
    except TypeError:
        return SwinUNETR(spatial_dims=3, in_channels=in_ch, out_channels=out_ch,
                         feature_size=feature_size, use_checkpoint=use_checkpoint)

def build_unetr(in_ch, out_ch, img_size=(96,96,96)):
    try:
        return UNETR(spatial_dims=3, img_size=img_size,
                     in_channels=in_ch, out_channels=out_ch,
                     feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
                     norm_name="instance", res_block=True, qkv_bias=True, dropout_rate=0.0)
    except TypeError:
        return UNETR(spatial_dims=3,
                     in_channels=in_ch, out_channels=out_ch,
                     feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
                     norm_name="instance", res_block=True, qkv_bias=True, dropout_rate=0.0)

def ensure_ssl(path=SSL_PATH, url=SSL_URL):
    if not Path(path).exists(): download_url(url, path)
    return path

@torch.no_grad()
def load_ct_ssl_encoder(model: SwinUNETR, path=SSL_PATH):
    ensure_ssl(path)
    blob = torch.load(path, map_location="cpu")
    for k in ("state_dict","model","weights","params","net"):
        if isinstance(blob, dict) and k in blob and isinstance(blob[k], dict):
            blob = blob[k]; break
    enc_sd = model.swinViT.state_dict()  # RELATIVE keys (no prefix)
    def rel_map(k: str) -> str:
        for p in ("module.","backbone.","encoder.","swin_vit.","swinViT.","network."):
            if k.startswith(p): k = k[len(p):]
        # collapse layersX.a.b.* -> layersX.a.*
        k = re.sub(r"^(layers[1-4])\.([0-9]+)\.([0-9]+)\.", r"\1.\2.", k)
        return k
    to_load = {}
    for ck, cv in blob.items():
        if not isinstance(cv, torch.Tensor): continue
        rel = rel_map(ck)
        if rel in enc_sd and enc_sd[rel].shape == cv.shape: to_load[rel] = cv
    if to_load:
        enc_new = enc_sd.copy(); enc_new.update(to_load)
        model.swinViT.load_state_dict(enc_new, strict=False)
    return len(to_load), len(enc_sd)
