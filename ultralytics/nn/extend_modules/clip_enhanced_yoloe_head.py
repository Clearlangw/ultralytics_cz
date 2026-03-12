import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.extend_modules.clip_feat_fuser import CLIPFeatFuser
from ultralytics.nn.modules.conv import Conv, DWConv
from typing import Union, List, Tuple
from ultralytics.nn.modules.head import YOLOEDetect
import copy
from ultralytics.utils.torch_utils import fuse_conv_and_bn, smart_inference_mode
from ultralytics.nn.modules.block import BNContrastiveHead
from ultralytics.nn.tasks import LRPCHead
import math


class CLIPEnhancedYOLOEDetect(YOLOEDetect):
    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max=16, end2end=False, ch: tuple = (),
        clip_size: str='ViT-B/32', device: torch.device=('cuda' if torch.cuda.is_available() else 'cpu'), clip_model=None, clip_image_preprocess=None
    ):
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)

        self.fuser = CLIPFeatFuser(ch, clip_size, device, clip_model, clip_image_preprocess)

        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.enhanced_cv3 = copy.deepcopy(self.cv3)
        self.enhanced_cv4 = copy.deepcopy(self.cv4)
        if end2end:
            self.one2one_enhanced_cv3 = copy.deepcopy(self.enhanced_cv3)  # overwrite with new enhanced_head
            self.one2one_enhanced_cv4 = copy.deepcopy(self.enhanced_cv4)  # overwrite with new enhanced_head

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor = None):
        if self.is_fused:
            return 
        if txt_feats is None:
            self.enhanced_cv3 = self.enhanced_cv4 = None
            return 
        super().fuse(txt_feats)
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        self._fuse_tp(txt_feats, self.enhanced_cv3, self.enhanced_cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_enhanced_cv3, self.one2one_enhanced_cv4)

    def get_vpe(self, x: List[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """Get visual prompt embeddings with spatial awareness."""
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward(self, x: List[torch.Tensor]) -> Union[torch.Tensor, Tuple]:
        """Process features with class prompt embeddings to generate detections."""
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x[:4])
            # return self.forward_lrpc(x[:3])
        return super().forward(x)

    def forward_lrpc(self, x: List[torch.Tensor]) -> Union[torch.Tensor, Tuple]:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        enhanced_x, x = self.fuser(x), x[1:]
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        enhanced_cv3 = self.enhanced_cv3 if not self.end2end else self.one2one_enhanced_cv3
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            enhanced_cls_feat = enhanced_cv3[i](enhanced_x[i])
            box, score, idx = self.lrpc[i](
                enhanced_cls_feat,
                # cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=x, index=torch.cat(index))
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.enhanced_cv3, contrastive_head=self.enhanced_cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_enhanced_cv3, contrastive_head=self.one2one_enhanced_cv4)

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        """Concatenates and returns predicted bounding boxes, class probabilities, and contrastive scores."""
        assert len(x) == 5, f"Expected 5 features including 1 raw image, 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        enhanced_x, x = self.fuser(x), x[1:]
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [contrastive_head[i](cls_head[i](enhanced_x[i]), x[-1]).reshape(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        return dict(boxes=boxes, scores=scores, feats=enhanced_x[:3])
