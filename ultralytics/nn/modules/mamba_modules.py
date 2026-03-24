import torch
import torch.nn as nn
from mamba_ssm import Mamba

class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):  # 关键修改：expansion -> expand
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,  # 使用 expand 参数名
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # --- 补丁 1：CPU 环境下直接返回，跳过 Mamba 算子 ---
        if not x.is_cuda:
            return x 
            
        b, c, h, w = x.shape
        # 展平: (B, C, H, W) -> (B, L, C)
        x = x.permute(0, 2, 3, 1).reshape(b, -1, c).contiguous()
        x = self.norm(x)
        
        # Mamba 核心处理 (仅限 GPU)
        x = self.mamba(x)
        
        # 还原: (B, L, C) -> (B, C, H, W)
        return x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

class VSSBlock(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, 1, 1)
        # 支持 n 次堆叠
        self.ss2d = nn.Sequential(*(SS2D(d_model=c2) for _ in range(n)))
        self.cv2 = nn.Conv2d(c2, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # --- 补丁 2：Stride 检查时的防御性转发 ---
        if not x.is_cuda:
            # 保证输出 Shape 为 (B, c2, H, W)，不影响 Stride 计算
            return self.cv2(self.cv1(x)) 
            
        out = self.cv1(x)
        out = self.ss2d(out)
        out = self.cv2(out)  # 修复空格 typo (ou t -> out)
        return x + out if self.add else out
class LG_VSSBlock(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        self.c = c2
        # 1x1 卷积调整通道
        self.cv1 = nn.Conv2d(c1, c2, 1, 1)
        
        # 局部分支: 3x3 深度可分离卷积，强化像素级感知
        self.local_branch = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        
        # 全局分支: 沿用 SS2D 逻辑
        self.global_branch = nn.Sequential(*(SS2D(d_model=c2) for _ in range(n)))
        
        # 融合层: 拼接后还原
        self.cv2 = nn.Conv2d(c2 * 2, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        if not x.is_cuda: # 保持 CPU 兼容性补丁
            out = self.cv1(x)
            return self.cv2(torch.cat([out, out], dim=1))
            
        x_hid = self.cv1(x)
        l_feat = self.local_branch(x_hid)
        g_feat = self.global_branch(x_hid)
        
        out = self.cv2(torch.cat([l_feat, g_feat], dim=1))
        return x + out if self.add else out
    
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torch.utils.checkpoint import checkpoint

class OctoSS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        b, c, h, w = x.shape
        out_sum = torch.zeros_like(x)
        
        if not x.is_cuda:
            return x
            
        L = h * w
        # 1. 准备方向特征
        x_h = x.permute(0, 2, 3, 1).reshape(b, L, c)
        x_v = x.permute(0, 3, 2, 1).reshape(b, L, c)
        
        # 针对对角线：如果不是正方形，强制对齐后再 roll
        x_tlbr = torch.stack([torch.roll(x[:, :, i, :], shifts=i, dims=-1) for i in range(h)], dim=2).permute(0, 2, 3, 1).reshape(b, L, c)
        x_trbl = torch.stack([torch.roll(x[:, :, i, :], shifts=-i, dims=-1) for i in range(h)], dim=2).permute(0, 2, 3, 1).reshape(b, L, c)

        tasks = [
            (x_h, 'h', False), (x_h, 'h', True),
            (x_v, 'v', False), (x_v, 'v', True),
            (x_tlbr, 'tlbr', False), (x_tlbr, 'tlbr', True),
            (x_trbl, 'trbl', False), (x_trbl, 'trbl', True)
        ]

        for feat, direct, flip in tasks:
            cur_feat = feat.flip(1) if flip else feat
            res = self.mamba(self.norm(cur_feat))
            if flip:
                res = res.flip(1)
            
            # --- ✨ 核心修复：动态维度还原 ---
            if direct == 'v':
                # 纵向扫描时，Mamba 看到的序列是 W * H，还原时要先 view 成 (W, H)
                res = res.view(b, w, h, c).permute(0, 3, 2, 1) # 直接还原为 (B, C, H, W)
            else:
                # 其他扫描（横向、对角线）默认是 H * W
                res = res.view(b, h, w, c).permute(0, 3, 1, 2)
            
            # 还原对角线偏移
            if direct == 'tlbr':
                for i in range(h):
                    res[:, :, i, :] = torch.roll(res[:, :, i, :], shifts=-i, dims=-1)
            elif direct == 'trbl':
                for i in range(h):
                    res[:, :, i, :] = torch.roll(res[:, :, i, :], shifts=i, dims=-1)
            
            # --- 🛡️ 安全检查：二次对齐 (解决 136 vs 80 的最后防线) ---
            if res.shape != out_sum.shape:
                import torch.nn.functional as F
                res = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
            
            out_sum += res
            del res

        del x_h, x_v, x_tlbr, x_trbl
        return out_sum / 8
class OctoLG_VSSBlock(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, 1, 1)
        self.local_branch = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        self.global_branch = nn.Sequential(*(OctoSS2D(d_model=c2) for _ in range(n)))
        self.cv2 = nn.Conv2d(c2 * 2, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        def _inner_forward(x_in):
            x_hid = self.cv1(x_in)
            l_feat = self.local_branch(x_hid)
            g_feat = self.global_branch(x_hid)
            return self.cv2(torch.cat([l_feat, g_feat], dim=1))

        # 训练时使用重计算以压榨显存空间
        if self.training and x.is_cuda:
            res = checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            res = _inner_forward(x)

        return x + res if self.add else res