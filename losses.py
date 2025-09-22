import torch

# ---------- Scale-invariant log (SILog) ----------
# Implementación típica usada en SOTA monocular con lambda = 0.85 (BTS, AdaBins, etc.)
# pred/target en metros, >0; mask selecciona píxeles válidos
def silog_loss(pred, target, mask, eps=1e-3):
    d = torch.log(pred.clamp(min=eps)) - torch.log(target.clamp(min=eps))
    d = d[mask]
    if d.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    mu = d.mean()
    return (d*d).mean() - 0.85*(mu*mu)  # factor 0.85 habitual

