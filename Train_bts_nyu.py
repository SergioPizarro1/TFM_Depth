# train_bts_nyu.py
import os
import math
import argparse
from datetime import datetime
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

from NYU_dataset import make_nyu_csv_loaders
from losses import silog_loss
from metrics import evaluate_batch
from plots import save_triplet


# ===========================
#  UTILIDADES
# ===========================
"""
def has_kwarg(model, name: str) -> bool:
    #Comprueba si el forward del modelo acepta un kwarg concreto (p.ej., 'focal').
    try:
        sig = inspect.signature(model.forward)
        return name in sig.parameters
    except Exception: 
        return False
"""

def freeze_first_blocks_and_bn(model: nn.Module):
    # Congelar conv1/bn1/layer1/layer2 del ResNet base
    if hasattr(model, "encoder") and hasattr(model.encoder, "base_model"):
        bm = model.encoder.base_model
        for name in ["conv1", "bn1", "layer1", "layer2"]:
            if hasattr(bm, name):
                for p in getattr(bm, name).parameters():
                    p.requires_grad = False

    # Congelar todos los BatchNorm de TODO el modelo
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def poly_decay_lambda(step, total_steps, base_lr, end_lr, power=0.9):
    if total_steps <= 0:
        return 1.0
    step = min(step, total_steps)
    return ((base_lr - end_lr) * ((1 - step / total_steps) ** power) + end_lr) / base_lr


# ===========================
#  VALIDACIÓN
# ===========================

@torch.no_grad()
def validate(model, loader, device, max_depth, save_dir=None, save_n=6, focal=None, amp=True):
    model.eval()
    agg = {"absrel": 0.0, "rmse": 0.0, "rmse_log": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
    steps = 0
    #accepts_focal = has_kwarg(model, "focal")

    for i, (rgb, gt, f) in enumerate(loader):
        rgb, gt = rgb.to(device, non_blocking=True), gt.to(device, non_blocking=True)

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=amp):
            focal = torch.full((rgb.size(0),), args.focal, device=rgb.device, dtype=rgb.dtype)
            d8, d4, d2, d1, pred = model(rgb, focal = focal)

            # Asegura tamaño [B,1,H,W] igual al GT
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)

        m = evaluate_batch(pred, gt)
        for k in agg: agg[k] += m[k]
        steps += 1

        # Guardado de tripletas
        if save_dir and i * loader.batch_size < save_n:
            bs = rgb.size(0)
            for b in range(min(bs, max(0, save_n - i * bs))):
                out_path = os.path.join(save_dir, f"val_{i:04d}_{b}.png")
                save_triplet(rgb[b], pred[b], gt[b], out_path, vmax=max_depth)

    if steps > 0:
        for k in agg: agg[k] /= steps
    return agg


# ===========================
#  ENTRENAMIENTO
# ===========================

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data ----
    train_loader, val_loader = make_nyu_csv_loaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        image_size=(args.input_height, args.input_width),
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_depth=args.max_depth,
        augment_train=True,
    )

    # ---- Modelo ----
    # IMPORTA TU BTS REAL AQUÍ (ajusta nombres/kwargs si difieren)
    # Debe devolver la predicción final [B,1,H,W] en metros (sin tuplas ni dicts).
    from types import SimpleNamespace
    from BTS_test import BtsModel  # tu clase real

    # Construimos el objeto params que tu modelo espera
    params = SimpleNamespace(
        encoder   = args.encoder,        # 'resnet50_bts', etc.
        max_depth = args.max_depth,      # 10.0 para NYU
        bts_size  = args.bts_size,       # 512 por defecto
        dataset   = 'nyu'                # IMPORTANTE: para evitar la rama 'kitti'
    )

    model = BtsModel(params).to(device)

    if args.freeze_stem_bn:
        freeze_first_blocks_and_bn(model)

    # ===== Inicialización neutra de la cabeza de profundidad =====
    # Queremos que la salida de la conv final sea ~0 al inicio -> sigmoid(0)=0.5
    with torch.no_grad():
        head = model.decoder.get_depth[0]  # es la nn.Conv2d final
        head.weight.zero_()                # bias no existe (bias=False)
    
    print("Parámetros entrenables:", count_trainable(model))

    # DataParallel opcional
    if torch.cuda.device_count() > 1 and not args.no_dp:
        model = nn.DataParallel(model)

    # ---- Opt + LR (poly decay) ----
    total_steps = args.epochs * len(train_loader)
    base_lr = args.lr
    end_lr = args.end_lr if args.end_lr >= 0 else base_lr * 0.1

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=base_lr, weight_decay=args.weight_decay)
    lr_lambda = lambda s: poly_decay_lambda(s, total_steps, base_lr, end_lr, power=args.poly_power)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu", enabled=args.amp)

    # ---- Reanudar ----
    start_epoch, global_step = 0, 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt.get("model", ckpt)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        if "opt" in ckpt: optimizer.load_state_dict(ckpt["opt"])
        if "sched" in ckpt: scheduler.load_state_dict(ckpt["sched"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        print(f"=> Resumido desde {args.resume} (epoch {start_epoch})")

    # ---- Eval-only ----
    if args.eval_only:
        save_dir = os.path.join(args.output_dir, "eval_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)
        metrics = validate(model, val_loader, device, args.max_depth,
                           save_dir if args.save_imgs else None, save_n=12,
                           focal=args.focal, amp=args.amp)
        print("[Eval]", metrics)
        return

    # ---- Loop ----
    best_delta1 = 0.0

    history = {"train_silog": [], "val_absrel": [], "val_rmse": [], "val_rmse_log": [],
           "val_delta1": [], "val_delta2": [], "val_delta3": []}
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    #accepts_focal = has_kwarg(model, "focal")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # acumulador de pérdida media por época
        running_loss, running_count = 0.0, 0

        save_dir_train = os.path.join(args.output_dir, f"train_ep{epoch:02d}")
        os.makedirs(save_dir_train, exist_ok=True)
        saved_train = 0
        save_train_N = 8            # guarda hasta 8 tripletas por época
        save_train_every = 800

        for it, (rgb, gt, f) in enumerate(train_loader):
            rgb, gt = rgb.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            valid = (gt > 1e-3) & (gt < args.max_depth)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=args.amp):
                focal = torch.full((rgb.size(0),), args.focal, device=rgb.device, dtype=rgb.dtype)
                d8, d4, d2, d1, pred = model(rgb, focal=focal)

                # Alinear tamaño al GT (por si el modelo devuelve downsample)
                if pred.shape[-2:] != gt.shape[-2:]:
                    pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
                if args.save_imgs and saved_train < save_train_N and (it % save_train_every == 0):
                    # guardamos primera muestra del batch
                    out_path = os.path.join(save_dir_train, f"train_e{epoch:02d}_it{it:04d}.png")
                    # IMPORTANT: detach() para no arrastrar el grafo
                    save_triplet(rgb[0].detach(), pred[0].detach(), gt[0].detach(),
                                 out_path, vmax=args.max_depth)
                    saved_train += 1

                loss = silog_loss(pred, gt, valid)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            scheduler.step()

            # acumular para media de época
            running_loss += loss.item() * rgb.size(0)
            running_count += rgb.size(0)

            if (it % args.log_every) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"[{epoch:02d}/{args.epochs}] it {it:04d}/{len(train_loader)} "
                      f"loss {loss.item():.4f} | lr {cur_lr:.2e}")

        # ===== Fin de época: imprimir loss media y métricas de validación =====
        train_loss_epoch = running_loss / max(1, running_count)
        save_dir = os.path.join(args.output_dir, f"val_ep{epoch:02d}")
        os.makedirs(save_dir, exist_ok=True)
        val_metrics = validate(model, val_loader, device, args.max_depth,
                               save_dir if args.save_imgs else None, save_n=12,
                               focal=args.focal, amp=args.amp)

        # Imprime resumen por época (lo que pedías)
        print(f"[Epoch {epoch:02d}] train_silog: {train_loss_epoch:.4f} | "
              f"val_absrel: {val_metrics['absrel']:.4f} | "
              f"val_rmse: {val_metrics['rmse']:.4f} | "
              f"val_rmse_log: {val_metrics['rmse_log']:.4f} | "
              f"val_delta1: {val_metrics['delta1']:.4f} | "
              f"val_delta2: {val_metrics['delta2']:.4f} | "
              f"val_delta3: {val_metrics['delta3']:.4f}")

        # Guardados
        is_best = val_metrics["delta1"] > best_delta1
        if is_best:
            best_delta1 = val_metrics["delta1"]

        history["train_silog"].append(train_loss_epoch)
        for k in ["absrel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]:
            history[f"val_{k}"].append(val_metrics[k])
        
        # Dibujar y guardar curvas (loss + métricas principales)
        import matplotlib.pyplot as plt
        # 1) Pérdida (train_silog)
        plt.figure()
        plt.plot(history["train_silog"], label="train_silog")
        plt.xlabel("epoch"); plt.ylabel("loss")
        plt.title("SiLog (train)")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "loss_train_silog.png"), dpi=120)
        plt.close()

        # 2) Métricas val (AbsRel + δ1)
        plt.figure()
        plt.plot(history["val_absrel"], label="val_absrel")
        plt.plot(history["val_delta1"], label="val_delta1")
        plt.xlabel("epoch"); plt.ylabel("value")
        plt.title("Val metrics")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "val_metrics.png"), dpi=120)
        plt.close()
        
        ckpt = {
            "epoch": epoch + 1,
            "step": global_step,
            "model": model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
            "opt": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "metrics": val_metrics,
            "args": vars(args)
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
            print("=> Nuevo BEST por δ1, guardado.")


# ===========================
#  ARGPARSE
# ===========================

def get_parser():
    p = argparse.ArgumentParser(description="Entrenamiento BTS (NYU) con SILog")
    # Datos
    p.add_argument("--train_csv", type=str, default="./Data/nyu_data/data/nyu2_train.csv")
    p.add_argument("--val_csv",   type=str, default="./Data/nyu_data/data/nyu2_test.csv")
    p.add_argument("--input_height", type=int, default=480)
    p.add_argument("--input_width",  type=int, default=640)
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--focal", type=float, default=518.8579, help="NYU fx en píxeles (si tu modelo la usa)")

    # Modelo
    p.add_argument("--encoder", type=str, default="resnet50_bts")
    p.add_argument("--bts_size", type=int, default=512)
    p.add_argument("--no_dp", action="store_true", help="No usar DataParallel")

    # Opt / Schedule (poly decay)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--end_lr", type=float, default=-1, help="Si <0, se usa 0.1*lr al final del poly decay")
    p.add_argument("--poly_power", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--amp", action="store_true", default=True, help="Mixed precision (AMP)")

    # Congelado estilo BTS
    p.add_argument("--freeze_stem_bn", action="store_true", default=True,
                   help="Congela conv1/layer1/layer2 + todos los BN (warmup)")

    # Misc
    p.add_argument("--output_dir", type=str, default="./runs/nyu_bts")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--save_imgs", action="store_true", default=True)
    p.add_argument("--log_every", type=int, default=50)

    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    if os.name == "nt":
        torch.multiprocessing.set_start_method("spawn", force=True)
    train(args)
