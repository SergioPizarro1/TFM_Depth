# losses.py
import torch
import torch.nn.functional as F

# ---------- SSIM (en intensidad) ----------
# Entrada: x,y en [0,1], shape [B,1,H,W] o [B,3,H,W]
def ssim(x, y, C1=0.01**2, C2=0.03**2, kernel_size=3):
    pad = kernel_size // 2
    mu_x = F.avg_pool2d(x, kernel_size, 1, pad)
    mu_y = F.avg_pool2d(y, kernel_size, 1, pad)

    sigma_x  = F.avg_pool2d(x*x, kernel_size, 1, pad) - mu_x*mu_x
    sigma_y  = F.avg_pool2d(y*y, kernel_size, 1, pad) - mu_y*mu_y
    sigma_xy = F.avg_pool2d(x*y, kernel_size, 1, pad) - mu_x*mu_y

    ssim_n = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    ssim_d = (mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x + sigma_y + C2)
    ssim_map = ssim_n / (ssim_d + 1e-7)
    return torch.clamp((1 - ssim_map) / 2, 0, 1)

# ---------- Gradiente (suavidad guiada por bordes) ----------
def gradient_x(img):  # [B,1,H,W]
    return img[:, :, :, 1:] - img[:, :, :, :-1]
def gradient_y(img):
    return img[:, :, 1:, :] - img[:, :, :-1, :]

def edge_aware_smoothness(depth, image):
    # depth en metros, image normalizada [-?], usa magnitud de gradiente de imagen para atenuar
    dx = gradient_x(depth)
    dy = gradient_y(depth)
    wx = torch.exp(-torch.mean(torch.abs(gradient_x(image)), 1, keepdim=True))
    wy = torch.exp(-torch.mean(torch.abs(gradient_y(image)), 1, keepdim=True))
    return (wx[:, :, :, :-1]*torch.abs(dx[:, :, :, :-1])).mean() + \
           (wy[:, :, :-1, :]*torch.abs(dy[:, :, :-1, :])).mean()

# ---------- BerHu (L1 inversa “robusta”) ----------
def berhu_loss(pred, target, mask):
    diff = torch.abs(pred - target)
    c = 0.2 * diff[mask].max().clamp(min=1e-6)
    l1 = diff.clone()
    l2 = (diff*diff + c*c) / (2*c)
    berhu = torch.where(diff <= c, l1, l2)
    return berhu[mask].mean()

# ---------- Scale-invariant log RMSE (SILog) ----------
# Implementación típica usada en SOTA monocular (BTS, AdaBins, etc.)
# pred/target en metros, >0; mask selecciona píxeles válidos
def silog_loss(pred, target, mask, eps=1e-6):
    d = torch.log(pred.clamp(min=eps)) - torch.log(target.clamp(min=eps))
    d = d[mask]
    if d.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    mu = d.mean()
    return (d*d).mean() - 0.85*(mu*mu)  # factor 0.85 habitual

# ---------- Composición típica ----------
class DepthLoss:
    """
    L = w_silog * SILog + w_berhu * BerHu + w_smooth * smooth(image)
    SSIM no suele aplicarse directo a depth; si quieres usarlo, úsalo sobre mapas normalizados [0,1].
    """
    def __init__(self, w_silog=1.0, w_berhu=0.2, w_smooth=0.001):
        self.w_silog = w_silog
        self.w_berhu = w_berhu
        self.w_smooth = w_smooth

    def __call__(self, pred, gt, img, mask):
        # pred, gt: [B,1,H,W] en metros. img: [B,3,H,W] normalizada
        l = 0.0
        if self.w_silog:
            l = l + self.w_silog * silog_loss(pred, gt, mask)
        if self.w_berhu:
            l = l + self.w_berhu * berhu_loss(pred, gt, mask)
        if self.w_smooth:
            l = l + self.w_smooth * edge_aware_smoothness(pred, img)
        return l
