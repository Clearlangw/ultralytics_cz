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


from clip.model import ModifiedResNet, VisionTransformer
def _VisionTransformer_forward(self: VisionTransformer, x: torch.Tensor):
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat(
        [
            self.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ],
        dim=1,
    )  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)

    x = x.permute(1, 2, 0) # LND -> NDL
    
    cls_token = x[:, :, 0]
    x = x[:, :, 1:]
    N, D, L = x.shape
    grid = int(L ** 0.5)
    x = x.reshape(N, D, grid, grid) # NDL -> NDHW

    return cls_token, x

def _ModifiedResNet_forward(self: ModifiedResNet, x: torch.Tensor):
    def stem(x):
        """Forward pass through the network stem, applying convolutions, batch normalization, ReLU activations, and
        average pooling.
        """
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    cls_token = self.attnpool(x)

    return cls_token, x
VisionTransformer.forward = _VisionTransformer_forward
ModifiedResNet.forward = _ModifiedResNet_forward


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

    def hook_register(self):
        self._spatial_hidden = None
        from clip.model import ModifiedResNet, VisionTransformer
        if isinstance(self.model.visual, VisionTransformer):
            self.channels = 768
        elif isinstance(self.model.visual, ModifiedResNet):
            self.channels = 2048
        else:
            raise NotImplementedError("unacceptable visual extractor type")

    @torch.no_grad()
    def get_spatial_feats(self, image: Union[Image.Image, torch.Tensor], dtype: torch.dtype = torch.float32, return_cls_token: bool = False) -> torch.Tensor:
        image_device = getattr(image, "device", None) or 'cpu'
        dtype = getattr(image, "dtype", None) or dtype
        same_device_image = image.to(self.device) if isinstance(image, torch.Tensor) else image
        cls_token, spatial_tokens = self._encode_image(same_device_image, dtype)
        cls_token = cls_token.to(image_device).clone()
        spatial_tokens = spatial_tokens.to(image_device).clone()
        if return_cls_token:
            return cls_token, spatial_tokens
        else:
            return spatial_tokens
    
    @torch.no_grad()
    def align_to_sizes(self, spatial_feats, sizes):
        """按 sizes 列表对齐 spatial_feats 到多个尺度."""
        aligned_feats = []
        for (h, w) in sizes:
            # 使用双线性插值
            aligned_feats.append(
                torch.nn.functional.interpolate(spatial_feats, size=(h, w), mode='bilinear', align_corners=False).clone()
            )
        return aligned_feats

    @torch.no_grad()
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
            image = F.interpolate(image, size=self.input_resolution, mode='bilinear', align_corners=False).to(torch.float32)
        cls_token, spatial_tokens = self.model.to(torch.float32).encode_image(image)
        cls_token, spatial_tokens = cls_token.to(dtype), spatial_tokens.to(dtype)
        cls_token = cls_token / cls_token.norm(p=2, dim=-1, keepdim=True)
        return cls_token, spatial_tokens


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
    
    
    def __clip_enhancer_alltest():
        __clip_visual_extractor_test__()
    __clip_enhancer_alltest()