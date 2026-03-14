import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ultralytics.utils import checks
from ultralytics.utils.torch_utils import smart_inference_mode
from typing import Union, List, Tuple
import types
try:
    import clip
except ImportError:
    checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
    import clip
from ultralytics.nn.modules import Conv

from ultralytics.nn.extend_modules.clip_visual_extractor import CLIPVisualExtractor

class CLIPEnhancer(nn.Module):
    def __init__(self, channels, clip_size: str, device: torch.device, clip_model=None, clip_image_preprocess=None) -> None:
        """将 CLIP Visual Spatial 特征对齐到指定特征图 shape 上；
        
        forward 输入第一个参数为 image tensor，后续参数为特征图。
        """
        super().__init__()
        self.extractor = CLIPVisualExtractor(clip_size, device, clip_model, clip_image_preprocess)
        self.extractor.eval()
        self.channels = [channels] if isinstance(channels, int) else channels
        self.clip_channels = self.extractor.channels
        self.convs = nn.ModuleList([
            Conv(self.clip_channels, c, 1, 1)
            for c in self.channels
        ])

    def train(self, mode):
        self.training = mode
        self.convs.train(mode)
        self.extractor.eval()
        return self

    def forward(self, x: List[torch.Tensor]) -> Union[torch.Tensor, Tuple]:
        images = x[0]
        feats = x[1:]
        clip_feats = self.extractor.get_spatial_feats(images)
        spatial_aligned_clip_feats = self.extractor.align_to_sizes(clip_feats, [e.shape[-2:] for e in feats])
        spatial_aligned_clip_feats = [e.clone() for e in spatial_aligned_clip_feats]
        aligned_clip_feats = [
            conv(e)
            for conv, e in zip(self.convs, spatial_aligned_clip_feats)
        ]
        return aligned_clip_feats



if __name__ == "__main__":
    from PIL import Image
    image = Image.open("/irip/liaozhixuan_2023/yolo/ultralytics/workplace/kitchen.webp")
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
    def __clip_visual_extractor_unitest__(size, device, model=None, image_preprocess=None):
        extractor = CLIPVisualExtractor(size=size, device=device, model=model, image_preprocess=image_preprocess).to(device)
        spatial = extractor.get_spatial_feats(image)
        targets = [(14,14), (28,28), (56,56)]
        aligned = extractor.align_to_sizes(spatial, targets)
        shapes = [e.shape for e in aligned]
        print(spatial.shape, shapes)
    @exception_shutdown_decorator
    def __clip_visual_extractor_tensor_input_unitest__(size, device, model=None, image_preprocess=None):
        extractor = CLIPVisualExtractor(size=size, device=device, model=model, image_preprocess=image_preprocess).to(device)
        from torchvision import transforms
        timage = transforms.ToTensor()(image).unsqueeze(dim=0).to(device)
        spatial = extractor.get_spatial_feats(timage)
        targets = [(14,14), (28,28), (56,56)]
        aligned = extractor.align_to_sizes(spatial, targets)
        shapes = [e.shape for e in aligned]
        print(spatial.shape, shapes)
    def __clip_visual_extractor_test__():
        __clip_visual_extractor_tensor_input_unitest__('ViT-B/32', device)
        __clip_visual_extractor_unitest__('ViT-B/32', device)
        __clip_visual_extractor_unitest__('RN50', device)
        __clip_visual_extractor_unitest__('', device, *clip.load('ViT-B/32', device=device))
        __clip_visual_extractor_unitest__('', device, *clip.load('RN50', device=device))

    @exception_shutdown_decorator
    def __clip_enhancer_unitest__(size, device, model=None, image_preprocess=None):
        B = 4
        raw_H, raw_W = 1024, 1024
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

        enhancer = CLIPEnhancer(channels, size, device, model, image_preprocess).to(device)
        outputs = enhancer([image] + feats)
        shapes1 = [e.shape for e in feats]
        shapes2 = [e.shape for e in outputs]
        print(shapes1, shapes2)
    def __clip_enhancer_test__():
        __clip_enhancer_unitest__('ViT-B/32', device)
        __clip_enhancer_unitest__('RN50', device)

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

        enhancer = CLIPEnhancer(channels, size, device).to(device)
        outputs = enhancer([image] + feats)
        l = sum([(e**2).sum() for e in outputs])
        l.backward()
        for name, param in enhancer.named_parameters():
            if param.grad is not None or param.requires_grad:
                print(name, param.requires_grad, param.grad is not None)
        print(f'Max Memory: {torch.cuda.max_memory_allocated() / 1024**2} MB')
    
    def __clip_enhancer_alltest():
        __clip_visual_extractor_test__()
        __clip_enhancer_test__()
        __clip_grad_test__()
    __clip_enhancer_alltest()