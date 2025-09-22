import torch
import torch.nn as nn
import timm
import types
from typing import List, Tuple
import math
import torch.nn.functional as F

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook

class Slice(nn.Module):
    """Descarta los tokens especiales del principio."""
    # [B, 1+N, C] -> [B, N, C]
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    """Suma (broadcast) el CLS a todos los tokens de parches."""
    # [B, 1+N, C] -> [B, N, C]
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        readout = x[:, 0] # [B, C]
        return x[:, self.start_index :] + readout.unsqueeze(1) # [B, N, C]


class ProjectReadout(nn.Module):
    """Concat CLS con cada token y proyecta 2C->C (fusión aprendida)."""
    # [B, 1+N, C] -> [B, N, C]
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :]) # [B, N, C]
        features = torch.cat((x[:, self.start_index :], readout), -1) # [B, N, 2C]

        return self.project(features) # [B, N, C]
    
class Transpose(nn.Module):
    """Capa 'transpose' para usar en Sequential."""
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x
    
class Interpolate(nn.Module):
    """Capa 'interpolate' para usar en Sequential."""
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
                x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x
    
def forward_vit(pretrained, x):
    """
    Forward del ViT, aplica embeddning posicional redimensionado,
    activa postprocesado correspondiente a cada skip-connection.
    """
    b, c, h, w = x.shape

    _ = pretrained.model.forward_flex(x)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(2,torch.Size([h // pretrained.model.patch_size[1],w // pretrained.model.patch_size[0]])),
    )

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    """
    Redimensiona el positional embedding de la malla de parches a (gs_h, gs_w).
    posemb: [1, T+N_old, C]  ->  [1, T+N_new, C]
    """
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],  # [1, T, C]
        posemb[0, self.start_index :],  # [N_old, C]
    )

    gs_old = int(math.sqrt(len(posemb_grid))) # asume malla cuadrada en pretrain (24x24 para 384x384 con patch 16)

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2) # [1, C, gs_old, gs_old]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear", align_corners=False) # [1, C, gs_h, gs_w]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1) # [1, N_new, C]

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1) # [1, T+N_new, C]

    return posemb

def forward_flex(self, x):
    """
    Forward del ViT adaptado a entrada arbitraria HxW.
    x: [B,3,H,W] -> secuencia: [B, N+T, C]
    """
    b, c, h, w = x.shape

    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2) # [B, N, C]

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, C] 
        dist_token = self.dist_token.expand(B, -1, -1) # [B, 1, C]
        x = torch.cat((cls_tokens, dist_token, x), dim=1) # [B, 2+N, C]
    else:
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
        x = torch.cat((cls_tokens, x), dim=1) # [B, 1+N, C]

    x = x + pos_embed # [B, 1+N, C]
    x = self.pos_drop(x) # [B, 1+N, C]

    for blk in self.blocks:
        x = blk(x) # [B, 1+N, C]

    x = self.norm(x) # [B, 1+N, C]

    return x 

def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index) for _ in features]
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index) for _ in features]
    elif use_readout == "project":
        readout_oper = [ProjectReadout(vit_features, start_index) for _ in features]
    else:
        assert (False), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper

def _make_vit_b16_backbone(
    model,
    features=[96, 192, 384, 768], # [256, 512, 1024, 1024],
    size=[384, 384],
    hooks=[2, 5, 8, 11], #[5, 11, 17, 23],
    vit_features=768, # 1024,
    use_readout="project",
    start_index=1,
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0], # [B, N, 768]
        Transpose(1, 2), # [B, 768, N]
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), # [B, 768, H/16, W/16]
        nn.Conv2d(in_channels=vit_features, out_channels=features[0], kernel_size=1,stride=1, padding=0), # [B, 96, H/16, W/16]
        nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=4, stride=4, padding=0, 
                           bias=True, dilation=1, groups=1), # [B, 96, H/4, W/4]
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1], # [B, N, 768]
        Transpose(1, 2), # [B, 768, N]
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), # [B, 768, H/16, W/16]
        nn.Conv2d(in_channels=vit_features, out_channels=features[1], kernel_size=1, stride=1, padding=0), # [B, 192, H/16, W/16]
        nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2, padding=0, 
                           bias=True, dilation=1, groups=1), # [B, 192, H/8, W/8]
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2], # [B, N, 768]
        Transpose(1, 2), # [B, 768, N]
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), # [B, 768, H/16, W/16]
        nn.Conv2d(in_channels=vit_features, out_channels=features[2], kernel_size=1, stride=1, padding=0), # [B, 384, H/16, W/16]
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3], # [B, N, 768]
        Transpose(1, 2), # [B, 768, N]
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), # [B, 768, H/16, W/16]
        nn.Conv2d(in_channels=vit_features, out_channels=features[3], kernel_size=1, stride=1, padding=0), # [B, 768, H/16, W/16]
        nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=2, padding=1), # [B, 768, H/32, W/32]
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    """Se añaden métodos al modelo para poder usarlos sin cambiar librería timm."""  
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)

    return pretrained

def _make_pretrained_vitl16_384(
    pretrained=True, use_readout="project", hooks=None
):
    model = timm.create_model("vit_base_patch16_384", pretrained=pretrained) # ViT-B/16
    #model = timm.create_model("vit_large_patch16_384", pretrained=pretrained) # ViT-L/16

    hooks = [2, 5, 8, 11] if hooks == None else hooks # ViT-B/16
    #hooks = [5, 11, 17, 23] if hooks == None else hooks # ViT-L/16
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768], # [128, 256, 512, 1024], # Usar si ViT-B/16
        hooks=hooks,
        vit_features=768, # 1024, # Usar si ViT-B/16 
        use_readout=use_readout,
    )

class Scratch(nn.Module):
    """LLeva cada skip a un número fijo de características (out_features)."""
    def __init__(self, in_channels: List[int], out_features: int):
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.layer1_rn = nn.Conv2d(c1, out_features, kernel_size=3, stride=1, padding=1, bias=True) # [B,F,H/4,W/4]
        self.layer2_rn = nn.Conv2d(c2, out_features, kernel_size=3, stride=1, padding=1, bias=True) # [B,F,H/8,W/8]
        self.layer3_rn = nn.Conv2d(c3, out_features, kernel_size=3, stride=1, padding=1, bias=True) # [B,F,H/16,W/16]
        self.layer4_rn = nn.Conv2d(c4, out_features, kernel_size=3, stride=1, padding=1, bias=True) # [B,F,H/32,W/32]

class ResidualConvUnit(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True),
        )
    def forward(self, x):                    # [B,F,H,W] -> [B,F,H,W]
        return x + self.block(x)
    
class FeatureFusionBlock(nn.Module):
    """
    Fusiona un skip tras aplicarle RCU. Vuelve a aplicar y upsamplea x2.
    """
    def __init__(self, features: int):
        super().__init__()
        self.res1 = ResidualConvUnit(features)
        self.res2 = ResidualConvUnit(features)
    def forward(self, x, skip=None):         # x:[B,F,H,W], skip opcional:[B,F,H,W]
        out = x
        if skip is not None:
            out = out + self.res1(skip)      # suma tras RCU para estabilizar
        out = self.res2(out)
        out = F.interpolate(out, scale_factor=2.0, mode="bilinear", align_corners=True)
        return out
    
class DPTDepthModel(nn.Module):
    def __init__(self,
                 features: int = 256,
                 pretrained_vit: bool = True,
                 min_depth: float = 1e-3,
                 max_depth: float = 10.0):
        """
        Modelo DPT (Dense Prediction Transformer) para estimación de profundidad.
        features: ancho F tras 'scratch' (256))
        pretrained_vit: cargar pesos ImageNet de timm en el ViT
        """
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        # Encoder ViT-L/16 + hooks
        self.pretrained = _make_pretrained_vitl16_384(
            pretrained=pretrained_vit, use_readout="project"
        )

        # Pirámide del ViT (tras forward_vit) tendrá canales [256,512,1024,1024]
        self.scratch = Scratch([96, 192, 384, 768], out_features=features) # Usar si ViT-B/16
        #self.scratch = Scratch([256,512,1024,1024], out_features=features) # Usar si ViT-L/16

        # Bloques de fusión (top-down)
        self.refine4 = FeatureFusionBlock(features)  # 12->24
        self.refine3 = FeatureFusionBlock(features)  # 24->48
        self.refine2 = FeatureFusionBlock(features)  # 48->96
        self.refine1 = FeatureFusionBlock(features)  # 96->192

        # Head de depth 
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1), # [B,F/2,H/2,W/2]
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True), # [B,F/2,H,W]
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, 32, kernel_size=1, stride=1, padding=0), # [B,32,H,W]
            nn.ReLU(inplace=True),  
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), # [B,1,H,W]
            nn.Sigmoid()
        )

    @torch.no_grad()
    def _clamp_depth(self, d: torch.Tensor):
        return torch.clamp(d, self.min_depth, self.max_depth)

    def forward(self, x: torch.Tensor):                   # x: [B,3,H,W]
        B, C, H, W = x.shape

        # 1) Encoder ViT -> 4 mapas 2D piramidales
        l1, l2, l3, l4 = forward_vit(self.pretrained, x)
        # l1:[B,96,H/4,W/4], l2:[B,192,H/8,W/8], l3:[B,384,H/16,W/16], l4:[B,768,H/32,W/32]

        # 2) Normalizar canales a F
        z1 = self.scratch.layer1_rn(l1)                   # [B,F,H/4,W/4]
        z2 = self.scratch.layer2_rn(l2)                   # [B,F,H/8,W/8]
        z3 = self.scratch.layer3_rn(l3)                   # [B,F,H/16,W/16]
        z4 = self.scratch.layer4_rn(l4)                   # [B,F,H/32,W/32]

        # 3) Fusión top-down
        y4 = self.refine4(z4)                             # [B,F,H/16,W/16]
        y3 = self.refine3(y4, z3)                         # [B,F,H/8,W/8]
        y2 = self.refine2(y3, z2)                         # [B,F,H/4,W/4]
        y1 = self.refine1(y2, z1)                         # [B,F,H/2,W/2]

        # 4) Head a resolución de entrada
        out = self.head(y1)             # [B,1,H,W] (0,1)
        depth = self.min_depth + (self.max_depth - self.min_depth) * out  # [B,1,H,W] (min,max) (m)
        return depth