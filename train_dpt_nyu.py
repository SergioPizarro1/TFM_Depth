# Train_dpt_nyu.py
import os, time, csv, argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from NYU_dataset import make_nyu_csv_loaders
from losses import silog_loss
from metrics import evaluate_batch
from plots import save_triplet
from DPT_test import DPTDepthModel


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def validate(model, loader, device, max_depth, min_depth, save_dir=None, save_n=6, amp=True):
    """Evalúa MÉTRICAS y también la pérdida SiLog en validación."""
    model.eval()
    agg = {"absrel":0.0,"rmse":0.0,"rmse_log":0.0,"delta1":0.0,"delta2":0.0,"delta3":0.0}
    loss_sum, steps = 0.0, 0

    for i, (rgb, gt, f) in enumerate(loader):
        rgb, gt = rgb.to(device, non_blocking=True), gt.to(device, non_blocking=True)

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=amp):
            pred = model(rgb)  # [B,1,H,W]
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)

            valid = (gt > min_depth) & (gt < max_depth)
            vloss = silog_loss(pred, gt, valid)

        loss_sum += float(vloss)
        m = evaluate_batch(pred, gt)
        for k in agg: agg[k] += m[k]
        steps += 1

        # guardar algunas tripletas
        if save_dir and i * loader.batch_size < save_n:
            bs = rgb.size(0)
            for b in range(min(bs, max(0, save_n - i * bs))):
                out_path = os.path.join(save_dir, f"val_{i:04d}_{b}.png")
                save_triplet(rgb[b], pred[b], gt[b], out_path, vmax=max_depth)

    if steps > 0:
        for k in agg: agg[k] /= steps
        val_silog = loss_sum / steps
    else:
        val_silog = 0.0
    return agg, val_silog


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Datos
    train_loader, val_loader = make_nyu_csv_loaders(
        train_csv=args.train_csv, val_csv=args.val_csv,
        image_size=(args.input_height, args.input_width),
        batch_size=args.batch_size, num_workers=args.workers,
        max_depth=args.max_depth, augment_train=True
    )

    # ---- Modelo
    model = DPTDepthModel(
        features=args.features,
        pretrained_vit=True,
        min_depth=args.min_depth, max_depth=args.max_depth
    ).to(device)

    print("Parámetros entrenables:", count_trainable(model))

    if torch.cuda.device_count() > 1 and not args.no_dp:
        model = nn.DataParallel(model)

    # ---- Optimizador con dos grupos de LR (constantes, como DPT)
    net = model.module if isinstance(model, nn.DataParallel) else model
    back_params = list(net.pretrained.model.parameters())
    dec_params  = []
    dec_params += list(net.scratch.parameters())
    dec_params += list(net.refine1.parameters()) + list(net.refine2.parameters()) + list(net.refine3.parameters()) + list(net.refine4.parameters())
    dec_params += list(net.head.parameters())

    back_set = set(id(p) for p in back_params)
    dec_params = [p for p in dec_params if id(p) not in back_set]

    optimizer = optim.AdamW(
        [{"params": back_params, "lr": args.lr_backbone},
         {"params": dec_params,  "lr": args.lr_decoder}]
    )
    scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu", enabled=args.amp)

    # ---- Resume
    start_epoch, global_step = 0, 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt.get("model", ckpt)
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(state)
        if "opt" in ckpt: optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        print(f"=> Resumido desde {args.resume} (epoch {start_epoch})")

    # ---- Eval-only
    if args.eval_only:
        save_dir = os.path.join(args.output_dir, "eval")
        os.makedirs(save_dir, exist_ok=True)
        metrics, val_silog = validate(model, val_loader, device, args.max_depth, args.min_depth,
                                      save_dir if args.save_imgs else None, save_n=12, amp=args.amp)
        print("[Eval]", metrics, "val_silog:", f"{val_silog:.4f}")
        return

    # ---- CSV
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_silog","val_silog","val_absrel","val_rmse","val_rmse_log",
                        "val_delta1","val_delta2","val_delta3","epoch_time_sec","gpu_peak_mb"])

    best_delta1 = 0.0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

        running_loss, running_count = 0.0, 0

        # --- guardado train
        save_dir_train = os.path.join(args.output_dir, f"train_ep{epoch:02d}")
        os.makedirs(save_dir_train, exist_ok=True)
        saved_train, save_train_N, save_train_every = 0, 8, 800

        for it, (rgb, gt, f) in enumerate(train_loader):
            rgb, gt = rgb.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            valid = (gt > args.min_depth) & (gt < args.max_depth)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=args.amp):
                pred = model(rgb)
                if pred.shape[-2:] != gt.shape[-2:]:
                    pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
                loss = silog_loss(pred, gt, valid)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            global_step += 1
            running_loss += loss.item() * rgb.size(0)
            running_count += rgb.size(0)

            if args.save_imgs and saved_train < save_train_N and (it % save_train_every == 0):
                outp = os.path.join(save_dir_train, f"train_e{epoch:02d}_it{it:04d}.png")
                save_triplet(rgb[0].detach(), pred[0].detach(), gt[0].detach(),
                             outp, vmax=args.max_depth)
                saved_train += 1

            if (it % args.log_every) == 0:
                lrs = [pg['lr'] for pg in optimizer.param_groups]
                print(f"[{epoch:02d}/{args.epochs}] it {it:04d}/{len(train_loader)} "
                      f"loss {loss.item():.4f} | lr_back {lrs[0]:.2e} | lr_dec {lrs[1]:.2e}")

        train_silog = running_loss / max(1, running_count)

        # --- Validación (métricas + pérdida)
        save_dir = os.path.join(args.output_dir, f"val_ep{epoch:02d}")
        os.makedirs(save_dir, exist_ok=True)
        val_metrics, val_silog = validate(model, val_loader, device,
                                          args.max_depth, args.min_depth,
                                          save_dir if args.save_imgs else None,
                                          save_n=12, amp=args.amp)

        epoch_time = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated()/(1024**2) if torch.cuda.is_available() else 0.0

        print(f"[Epoch {epoch:02d}] train_silog: {train_silog:.4f} | val_silog: {val_silog:.4f} | "
              f"val_absrel: {val_metrics['absrel']:.4f} | val_rmse: {val_metrics['rmse']:.4f} | "
              f"val_rmse_log: {val_metrics['rmse_log']:.4f} | val_delta1: {val_metrics['delta1']:.4f} | "
              f"val_delta2: {val_metrics['delta2']:.4f} | val_delta3: {val_metrics['delta3']:.4f} | "
              f"time: {epoch_time:.1f}s | gpu_peak: {peak_mb:.0f}MB")

        # CSV
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_silog:.6f}", f"{val_silog:.6f}",
                        f"{val_metrics['absrel']:.6f}", f"{val_metrics['rmse']:.6f}",
                        f"{val_metrics['rmse_log']:.6f}", f"{val_metrics['delta1']:.6f}",
                        f"{val_metrics['delta2']:.6f}", f"{val_metrics['delta3']:.6f}",
                        f"{epoch_time:.3f}", f"{peak_mb:.1f}"])

        # Guardados
        is_best = val_metrics["delta1"] > best_delta1
        if is_best: best_delta1 = val_metrics["delta1"]

        ckpt = {
            "epoch": epoch + 1, "step": global_step,
            "model": model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
            "opt": optimizer.state_dict(),
            "metrics": val_metrics, "args": vars(args)
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
            print("=> Nuevo BEST por δ1, guardado.")


def get_parser():
    p = argparse.ArgumentParser(description="Entrenamiento DPT (NYU) con SiLog")
    # Datos
    p.add_argument("--train_csv", type=str, default="./Data/nyu_data/data/nyu2_train.csv")
    p.add_argument("--val_csv",   type=str, default="./Data/nyu_data/data/nyu2_test.csv")
    p.add_argument("--input_height", type=int, default=480)
    p.add_argument("--input_width",  type=int, default=640)
    p.add_argument("--min_depth", type=float, default=1e-3)
    p.add_argument("--max_depth", type=float, default=10.0)
    # Modelo
    p.add_argument("--features", type=int, default=256)
    p.add_argument("--vit_from_scratch", action="store_true")
    p.add_argument("--no_dp", action="store_true")
    p.add_argument("--amp", action="store_true", default=True)
    # Opt (LRS FIJOS como en DPT)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_decoder",  type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    # Misc
    p.add_argument("--output_dir", type=str, default="./runs/nyu_dpt")
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
