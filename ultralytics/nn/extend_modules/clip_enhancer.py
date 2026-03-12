import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ultralytics.utils import checks
from ultralytics.utils.torch_utils import smart_inference_mode
from typing import Union, List, Tuple
try:
    import clip
except ImportError:
    checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
    import clip
from ultralytics.nn.modules import Conv


class CLIPVisualExtractor(nn.Module):
    def __init__(self, size: str, device: torch.device, model=None, image_preprocess=None) -> None:
        """用于提取 CLIP Vis Enc 的 Spatial 特征，可以用 size 初始化，也可以复用 model/image_preprocess。

        get_spatial_feats 输入图片，获取 CLIP 视觉 Spatial 特征；
        align_to_sizes 输入 CLIP 视觉 Spatial 特征与 HW 列表，输出空间上 Aligned 特征。
        """
        super().__init__()
        # self.model, self.image_preprocess = clip.load(size, device=device)
        if (model is not None) and (image_preprocess is not None):
            self.model, self.image_preprocess = model, image_preprocess
        else:
            self.model, self.image_preprocess = clip.load(size, device=device)
        self.to(device)
        self.device = device
        self.eval()
        self.hook_register()
        self.input_resolution = self.model.visual.input_resolution
        self.device = device

    def train(self, mode):
        self.training = mode
        return self

    def hook_register(self):
        self._spatial_hidden = None
        from clip.model import ModifiedResNet, VisionTransformer
        if isinstance(self.model.visual, VisionTransformer):
            def hook(module, input, output):
                x = output # LND
                x = x.permute(1, 2, 0)[:, :, 1:] # LND -> NDL
                N, D, L = x.shape
                grid = int(L ** 0.5)
                self._spatial_hidden = x.reshape(N, D, grid, grid) # NDL -> NDHW
            self.model.visual.transformer.resblocks[-1].register_forward_hook(hook)
            self.channels = 768
        elif isinstance(self.model.visual, ModifiedResNet):
            def hook(module, input, output):
                self._spatial_hidden = output
            self.model.visual.layer4[-1].register_forward_hook(hook)
            self.channels = 2048
        else:
            raise NotImplementedError("unacceptable visual extractor type")

    @smart_inference_mode()
    def get_spatial_feats(self, image: Union[Image.Image, torch.Tensor], dtype: torch.dtype = torch.float32, return_cls_token: bool = False) -> torch.Tensor:
        self._spatial_hidden = None
        image_device = image.device if isinstance(image, torch.Tensor) else 'cpu'
        same_device_image = image.to(self.device) if isinstance(image, torch.Tensor) else image
        cls_token = self._encode_image(same_device_image, dtype).to(image_device).clone()
        self._spatial_hidden = self._spatial_hidden.to(dtype).to(image_device).clone()
        if return_cls_token:
            return cls_token, self._spatial_hidden
        else:
            return self._spatial_hidden
    
    @smart_inference_mode()
    def align_to_sizes(self, spatial_feats, sizes):
        """按 sizes 列表对齐 spatial_feats 到多个尺度."""
        aligned_feats = []
        for (h, w) in sizes:
            # 使用双线性插值
            aligned_feats.append(
                torch.nn.functional.interpolate(spatial_feats, size=(h, w), mode='bilinear', align_corners=False)
            )
        return aligned_feats

    @smart_inference_mode()
    def _encode_image(self, image: Union[Image.Image, torch.Tensor], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode images into normalized feature vectors.

        This method processes image inputs through the CLIP model to generate feature vectors, which are then
        normalized to unit length. These normalized vectors can be used for text-image similarity comparisons.

        Args:
            image (PIL.Image | torch.Tensor): Image input as a PIL Image or preprocessed tensor. If a PIL Image is
                provided, it will be converted to a tensor using the model's image preprocessing function.
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized image feature vectors with unit length (L2 norm = 1).

        Examples:
            >>> from ultralytics.nn.text_model import CLIP
            >>> from PIL import Image
            >>> clip_model = CLIP("ViT-B/32", device="cuda")
            >>> image = Image.open("path/to/image.jpg")
            >>> image_tensor = clip_model.image_preprocess(image).unsqueeze(0).to("cuda")
            >>> features = clip_model.encode_image(image_tensor)
            >>> features.shape
            torch.Size([1, 512])
        """
        if isinstance(image, Image.Image):
            image = self.image_preprocess(image).unsqueeze(0).to(self.device)
        else:
            image = F.interpolate(image, size=self.input_resolution, mode='bilinear', align_corners=False)
        img_feats = self.model.encode_image(image).to(dtype)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        return img_feats

class CLIPEnhancer(nn.Module):
    def __init__(self, channels, clip_size: str, device: torch.device, clip_model=None, clip_image_preprocess=None) -> None:
        """将 CLIP Visual Spatial 特征对齐到指定特征图 shape 上；
        
        forward 输入第一个参数为 image tensor，后续参数为特征图。
        """
        super().__init__()
        self.extractor = CLIPVisualExtractor(clip_size, device, clip_model, clip_image_preprocess).requires_grad_(False)
        self.channels = [channels] if isinstance(channels, int) else channels
        self.clip_channels = self.extractor.channels
        self.convs = nn.ModuleList([
            Conv(self.clip_channels, c, 1, 1)
            for c in self.channels
        ])

    def forward(self, x: List[torch.Tensor]) -> Union[torch.Tensor, Tuple]:
        images = x[0]
        feats = x[1:]
        with torch.no_grad():
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