import torch
from torch import nn
from .conv import Conv
# from .transformer import LayerNorm2d
# from .common_utils_mbyolo import autopad

# class SimpleStem(nn.Module):
#     """YOLOv8 的初始下采样 Stem，替代 Focus，对应 P2/4（stride 4）"""

#     def __init__(self, c1, c2, k=3):
#         super().__init__()
#         # 两次 stride 2 叠加实现 stride 4
#         self.stem = nn.Sequential(
#             Conv(c1, c2 // 2, k=3, s=2),
#             Conv(c2 // 2, c2, k=3, s=2),
#         )

#     def forward(self, x):
#         return self.stem(x)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SimpleStem(nn.Module):
    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        self.hidden_dims = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, self.hidden_dims, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU(),
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


# class VisionClueMerge(nn.Module):
#     """Mamba-YOLO 风格的下采样 + 通道扩展模块，stride 2"""

#     def __init__(self, c1, c2):
#         super().__init__()
#         self.conv = Conv(c1, c2, k=3, s=2)

#     def forward(self, x):
#         return self.conv(x)

class VisionClueMerge(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)

        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        y = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)
        return self.pw_linear(y)


class StarBlock(nn.Module):
    """
    可替换 C2f 的 StarBlock 模块。
    输入: [B, C1, H, W]
    输出: [B, C2, H, W]
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5,
        layer_scale_init_value=1e-6,
        mode="star"
    ):
        super().__init__()

        self.mode = mode
        self.c1 = c1
        self.c2 = c2

        # channel projection: c1 -> c2
        self.proj_in = (
            nn.Identity()
            if c1 == c2
            else nn.Conv2d(c1, c2, kernel_size=1)
        )

        self.norm = nn.LayerNorm(c2)

        self.dwconv = nn.Conv2d(
            c2, c2, kernel_size=7, padding=3, groups=c2
        )

        self.linear_f = nn.Linear(c2, 6 * c2)
        self.act = nn.GELU()
        self.linear_g = nn.Linear(3 * c2, c2)

        # layer scale
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones(c2),
                requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):

        # input projection
        x = self.proj_in(x)
        shortcut = x

        # LN: BCHW -> BHWC
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        # DWConv: BHWC -> BCHW
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)

        # Linear: BCHW -> BHWC
        x = x.permute(0, 2, 3, 1)
        x = self.linear_f(x)

        # star operation: split into two 3*c2, act(x1) * x2 or act(x1) + x2
        x1, x2 = x.chunk(2, dim=-1)

        if self.mode == "sum":
            x = self.act(x1) + x2
        else:
            x = self.act(x1) * x2

        x = self.linear_g(x)

        # back to BCHW
        x = x.permute(0, 3, 1, 2)

        # residual + layer scale
        if self.gamma is not None:
            x = shortcut + self.gamma.view(1, -1, 1, 1) * x
        else:
            x = shortcut + x

        return x
    
class StarNetBlock(nn.Module):
    """StarNet block – 更轻量的星运算模块，适配 parse_model (c1, c2) 接口"""
    def __init__(self, c1, c2, expansion=2, use_layer_scale=True, layer_scale_init=1e-6):
        super().__init__()
        inner_dim = int(c2 * expansion)

        # c1 → c2 投影（parse_model 必传 c1/c2）
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1)
        # c2 → inner_dim 扩展（expansion）
        self.proj_in = nn.Identity() if c2 == inner_dim else nn.Conv2d(c2, inner_dim, 1)

        self.norm = nn.LayerNorm(inner_dim)
        self.dwconv = nn.Conv2d(inner_dim, inner_dim, kernel_size=7, padding=3, groups=inner_dim)
        self.linear_f = nn.Linear(inner_dim, 2 * inner_dim)
        self.act = nn.GELU()
        self.linear_g = nn.Linear(inner_dim, c2)

        self.gamma = nn.Parameter(layer_scale_init * torch.ones(c2), requires_grad=True) if use_layer_scale else None

    def forward(self, x):
        x = self.proj(x)                        # [B, c2, H, W]
        shortcut = x
        x = self.proj_in(x)                     # [B, inner, H, W]

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear_f(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.linear_g(x)                    # B,H,W,c2
        x = x.permute(0, 3, 1, 2)              # B,c2,H,W

        if self.gamma is not None:
            x = shortcut + self.gamma.view(1, -1, 1, 1) * x
        else:
            x = shortcut + x
        return x