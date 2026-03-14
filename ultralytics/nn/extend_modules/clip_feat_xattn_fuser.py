import torch
import torch.nn as nn
from ultralytics.nn.extend_modules.clip_visual_extractor import CLIPVisualExtractor
from ultralytics.nn.modules.conv import Conv
from typing import Union, List, Tuple

class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=8, ffn_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * ffn_ratio)

        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, q, k, v):

        # ----- Cross Attention -----
        q2 = self.norm1(q)
        attn_out, _ = self.attn(q2, k, v)

        x = q + attn_out

        # ----- FFN -----
        x2 = self.norm2(x)
        x = x + self.ffn(x2)

        return x

class CLIPFeatXattnFuser(nn.Module):

    def __init__(self, channels, clip_size, device,
                 clip_model=None, clip_image_preprocess=None,
                 num_heads=8):

        super().__init__()

        self.extractor = CLIPVisualExtractor(
            clip_size, device, clip_model, clip_image_preprocess
        )

        self.clip_dim = self.extractor.channels
        self.channels = channels

        # Q projection
        self.q_proj = nn.ModuleList([
            nn.Conv2d(c, self.clip_dim, 1)
            for c in channels
        ])

        # output projection
        self.out_proj = nn.ModuleList([
            nn.Conv2d(self.clip_dim, c, 1)
            for c in channels
        ])

        # cross-attention blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(self.clip_dim, num_heads)
            for _ in channels
        ])

        # CLIP token projection
        self.k_proj = nn.Linear(self.clip_dim, self.clip_dim)
        self.v_proj = nn.Linear(self.clip_dim, self.clip_dim)

    def forward(self, x):

        img = x[0]
        feats = x[1:]

        cls_token, spatial_tokens = self.extractor.get_spatial_feats(image=img, return_cls_token=True) # cls: [B, D], spatial: [B, D, H, W]
        clip_tokens = torch.cat([cls_token.unsqueeze(dim=-1), spatial_tokens.flatten(-2)], dim=-1).permute(0, 2, 1).contiguous() # [B, D, N] -> [B, N, D]

        K = self.k_proj(clip_tokens)
        V = self.v_proj(clip_tokens)

        outs = []

        for feat, q_proj, block, out_proj in zip(
            feats,
            self.q_proj,
            self.blocks,
            self.out_proj
        ):

            B,C,H,W = feat.shape

            q = q_proj(feat)
            q = q.flatten(2).transpose(1,2)

            # Cross attention block
            q = block(q, K, V)

            q = q.transpose(1,2).reshape(
                B, self.clip_dim, H, W
            )

            out = feat + out_proj(q)

            outs.append(out)

        return outs


if __name__ == "__main__":
    device = 'cuda:0'

    def exception_shutdown_decorator(func):
        def newfunc(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(e)
                raise e
                exit(0)
        return newfunc
    
    @exception_shutdown_decorator
    def __clip_grad_test__():
        size = 'ViT-B/32'
        B = 4
        raw_H, raw_W = 640, 640
        layers = [3, 4, 5]
        channels = [256, 512, 1024]
        feats = [
            torch.randn(B, c, h, w).to(device)
            for c, h, w in [
                [c, (raw_H - 1) // 2 ** l + 1, (raw_W - 1) // 2 ** l + 1]
                for c, l in zip(channels, layers)
            ]
        ]
        image = torch.randn(B, 3, raw_H, raw_W).to(device)

        enhancer = CLIPFeatXattnFuser(channels, size, device).to(device)
        outputs = enhancer([image] + feats)
        l = sum([(e**2).sum() for e in outputs])
        l.backward()
        for name, param in enhancer.named_parameters():
            # if param.grad is not None or param.requires_grad:
            if param.grad is not None:
                print(name, param.requires_grad, param.grad is not None)
        print(f'Max Memory: {torch.cuda.max_memory_allocated() / 1024**2} MB')
    
    __clip_grad_test__()