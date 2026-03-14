import torch
import torch.nn as nn
from typing import List, Tuple, Union
from ultralytics.nn.extend_modules.clip_visual_extractor import CLIPVisualExtractor
from ultralytics.nn.modules.conv import Conv


class FiLMGenerator(nn.Module):
    """
    Generate gamma and beta from conditioning embedding
    """

    def __init__(self, cond_dim, feat_dim, hidden_dim=None):
        super().__init__()

        hidden_dim = hidden_dim or cond_dim

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim * 2)
        )

        # zero init (important for stability)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cond):

        gamma_beta = self.mlp(cond)

        gamma, beta = gamma_beta.chunk(2, dim=-1)

        return gamma, beta


class CLIPFeatFiLMFuser(nn.Module):

    def __init__(
        self,
        channels,
        clip_size: str,
        device: torch.device,
        clip_model=None,
        clip_image_preprocess=None
    ):
        super().__init__()

        self.extractor = CLIPVisualExtractor(
            clip_size,
            device,
            clip_model,
            clip_image_preprocess
        )

        self.clip_dim = self.extractor.channels
        self.channels = channels

        # FiLM generators for each scale
        self.film_generators = nn.ModuleList([
            FiLMGenerator(self.clip_dim, c)
            for c in channels
        ])

        # optional post conv
        self.post_convs = nn.ModuleList([
            Conv(c, c, 1, 1)
            for c in channels
        ])

    def forward(self, x: List[torch.Tensor]) -> Union[torch.Tensor, Tuple]:

        img = x[0]
        feats = x[1:]

        # CLIP tokens
        clip_global, _ = self.extractor.get_spatial_feats(image=img, return_cls_token=True) # [B, D]

        outs = []

        for feat, film, conv in zip(
            feats,
            self.film_generators,
            self.post_convs
        ):

            gamma, beta = film(clip_global) # [B, C]

            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)

            out = feat * (1 + gamma) + beta

            out = out + conv(out)

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

        enhancer = CLIPFeatFiLMFuser(channels, size, device).to(device)
        outputs = enhancer([image] + feats)
        l = sum([(e**2).sum() for e in outputs])
        l.backward()
        for name, param in enhancer.named_parameters():
            if param.grad is not None:
            # if param.grad is not None or param.requires_grad:
                print(name, param.requires_grad, param.grad is not None)
        print(f'Max Memory: {torch.cuda.max_memory_allocated() / 1024**2} MB')
    
    __clip_grad_test__()