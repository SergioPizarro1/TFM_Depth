# utils/metrics.py
import torch

def _valid_mask(gt, min_depth=1e-3, max_depth=1e6):
    return (gt > min_depth) & (gt < max_depth) & torch.isfinite(gt)

def abs_rel(pred, gt):
    mask = _valid_mask(gt)
    return (torch.abs(pred[mask] - gt[mask]) / gt[mask]).mean().item()

def rmse(pred, gt):
    mask = _valid_mask(gt)
    return torch.sqrt(((pred[mask]-gt[mask])**2).mean()).item()

def rmse_log(pred, gt):
    mask = _valid_mask(gt) & (pred > 0) & (gt > 0)
    return torch.sqrt(((torch.log(pred[mask]) - torch.log(gt[mask]))**2).mean()).item()

def delta_accuracy(pred, gt, thresh=1.25):
    mask = _valid_mask(gt) & (pred > 0)
    ratio = torch.max(pred[mask] / gt[mask], gt[mask] / pred[mask])
    return (ratio < thresh).float().mean().item()

def evaluate_batch(pred, gt):
    """pred, gt: [B,1,H,W] (m). Devuelve dict con métricas promedio del batch."""
    m = {}
    m["absrel"]   = abs_rel(pred, gt)
    m["rmse"]     = rmse(pred, gt)
    m["rmse_log"] = rmse_log(pred, gt)
    m["delta1"]   = delta_accuracy(pred, gt, 1.25)
    m["delta2"]   = delta_accuracy(pred, gt, 1.25**2)
    m["delta3"]   = delta_accuracy(pred, gt, 1.25**3)
    return m
