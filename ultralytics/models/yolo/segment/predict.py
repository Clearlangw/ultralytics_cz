# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class SegmentationPredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a segmentation model.

    This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
    prediction results.

    Attributes:
        args (dict): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO segmentation model.
        batch (list): Current batch of images being processed.

    Methods:
        postprocess: Apply non-max suppression and process segmentation detections.
        construct_results: Construct a list of result objects from predictions.
        construct_result: Construct a single result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo26n-seg.pt", source=ASSETS)
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the SegmentationPredictor with configuration, overrides, and callbacks.

        This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
        prediction results.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """Apply non-max suppression and process segmentation detections for each image in the input batch.

        Args:
            preds (tuple): Model predictions, containing bounding boxes, scores, classes, and mask coefficients.
            img (torch.Tensor): Input image tensor in model format, with shape (B, C, H, W).
            orig_imgs (list | torch.Tensor | np.ndarray): Original image or batch of images.

        Returns:
            (list): List of Results objects containing the segmentation predictions for each image in the batch. Each
                Results object includes both bounding boxes and segmentation masks.

        Examples:
            >>> predictor = SegmentationPredictor(overrides=dict(model="yolo26n-seg.pt"))
            >>> results = predictor.postprocess(preds, img, orig_img)
        """
        # Extract protos - tuple if PyTorch model or array if exported
        # 来自forward的输出得到tuple，y是_inference得到的张量，preds是forward_head的输出字典
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        # 新增：提取属性分数和属性数量
        preds_dict = preds[1] if isinstance(preds[0], tuple) else preds[0]
        #attr_scores = preds_dict.get('attr_scores', None)  # shape: (batch, na, num_boxes)
        na = preds_dict.get('na', 0)  # 属性数量
        # 调用父类 postprocess，并传入属性信息
        return super().postprocess(preds[0], img, orig_imgs, protos=protos, na=na)


    def construct_results(self, preds, img, orig_imgs, protos, na=0):
        """Construct a list of result objects from the predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.
            protos (torch.Tensor): Prototype masks tensor with shape (B, C, H, W).
            na (int): Number of attributes (Phase 3).

        Returns:
            (list[Results]): List of result objects containing the original images, image paths, class names, bounding
                boxes, and masks.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path, proto, na=na)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto, na=0):
        """Construct a single result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
            proto (torch.Tensor): The prototype masks.
            na (int): Number of attributes (Phase 3).

        Returns:
            (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
        """
        # Extract attribute scores if present (Phase 3)
        # NMS 输出格式：boxes(4) + scores(1) + class(1) + attr_scores(na) + mask_coeff(nm)
        attr_scores = None
        nm = proto.shape[1] if proto is not None else 0
        # print(pred)
        # import
        if pred.shape[0] > 0 and na > 0:
            # 属性分数在位置 [6:6+na]
            attr_scores = pred[:, 6:6+na]
            # 重新组织 pred：只保留 boxes + scores + class + mask_coeff
            # 需要将 mask_coeff 从 [6+na:] 移到 [6:]
            import torch
            pred = torch.cat([pred[:, :6], pred[:, 6+na:]], dim=1)
        
        if pred.shape[0] == 0:  # save empty boxes
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # NHW
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # NHW
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0  # only keep predictions with masks
            if not all(keep):  # most predictions have masks
                pred, masks = pred[keep], masks[keep]  # indexing is slow
                if attr_scores is not None:
                    attr_scores = attr_scores[keep]
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks, attr_scores=attr_scores)
