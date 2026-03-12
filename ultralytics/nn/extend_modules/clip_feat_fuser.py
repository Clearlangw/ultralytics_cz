import torch
import torch.nn as nn
from ultralytics.nn.extend_modules.clip_enhancer import CLIPEnhancer
from ultralytics.nn.modules.conv import Conv
from typing import Union, List, Tuple

class CLIPFeatFuser(nn.Module):
    def __init__(self, channels, clip_size: str, device: torch.device, clip_model=None, clip_image_preprocess=None) -> None:
        super().__init__()
        self.enhancer = CLIPEnhancer(channels, clip_size, device, clip_model, clip_image_preprocess)
        self.feat_proj = nn.ModuleList([
            Conv(c, c, 3, 1)
            for c in channels
        ])
        self.clip_proj = nn.ModuleList([
            Conv(c, c, 3, 1)
            for c in channels
        ])
    
    def forward(self, x: List[torch.Tensor]) -> Union[torch.Tensor, Tuple]:
        feats = x[1:]
        clip_feats = self.enhancer(x)

        feats = [conv(e) for conv, e in zip(self.feat_proj, feats)]
        clip_feats = [conv(e) for conv, e in zip(self.clip_proj, clip_feats)]
        
        fused_feats = [f1 + f2 for f1, f2 in zip(feats, clip_feats)]
        return fused_feats


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

        enhancer = CLIPFeatFuser(channels, size, device).to(device)
        outputs = enhancer([image] + feats)
        l = sum([(e**2).sum() for e in outputs])
        l.backward()
        for name, param in enhancer.named_parameters():
            if param.grad is not None or param.requires_grad:
                print(name, param.requires_grad, param.grad is not None)
        print(f'Max Memory: {torch.cuda.max_memory_allocated() / 1024**2} MB')
    
    __clip_grad_test__()