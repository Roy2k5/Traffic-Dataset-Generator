"""
YOLOv8 detection plugin (ultralytics).

Wraps a YOLOv8 .pt weight file as a post-processing plugin that draws
bounding boxes onto the rendered RGB image.

Usage:
    from models.yolo_wrapper import YOLODetectionPlugin
    from models.model_wrapper import PostProcessor

    pp = PostProcessor()
    pp.add(YOLODetectionPlugin('models/weights/best.pt', conf=0.4))

    annotated = pp.run(rgb_image)   # numpy H×W×3 uint8
"""

import numpy as np
from .model_wrapper import ModelPlugin, draw_detection_boxes

# Class names must match training labels (exporter uses car/tree/lamppost)
DEFAULT_NAMES = {0: 'car', 1: 'tree', 2: 'lamppost'}


class YOLODetectionPlugin(ModelPlugin):
    """
    Object detection post-processor using YOLOv8 (ultralytics).

    Parameters
    ----------
    model_path : str
        Path to the .pt weight file.
    conf : float
        Confidence threshold (default 0.25).
    iou : float
        NMS IoU threshold (default 0.45).
    class_names : dict
        Optional override {class_id: label_str}.
    """

    TASK = "detection"

    def __init__(self, model_path: str, conf=0.25, iou=0.45,
                 class_names: dict | None = None, **kwargs):
        super().__init__(model_path, **kwargs)
        self.conf        = conf
        self.iou         = iou
        self.class_names = class_names or DEFAULT_NAMES
        self._model      = None

    def load(self):
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)
        self.loaded = True
        print(f"[YOLODetectionPlugin] Loaded '{self.model_path}'")

    def annotate(self, image: np.ndarray) -> np.ndarray:
        """Run YOLOv8 inference and draw bounding boxes on *image* (RGB)."""
        self.ensure_loaded()
        # ultralytics expects BGR or RGB – we pass RGB with source=image
        results = self._model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        'label': self.class_names.get(cls_id, str(cls_id)),
                        'score': float(box.conf[0]),
                        'bbox':  [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                        'class_id': cls_id,
                    })
        return draw_detection_boxes(image, detections)


class YOLOSegmentationPlugin(ModelPlugin):
    """
    Instance segmentation post-processor using YOLOv8-seg (ultralytics).

    Parameters
    ----------
    model_path : str  Path to .pt seg weight.
    conf : float      Confidence threshold.
    alpha : float     Mask blend alpha (0=transparent, 1=opaque).
    """

    TASK = "segmentation"

    def __init__(self, model_path: str, conf=0.25, alpha=0.45,
                 class_names: dict | None = None, **kwargs):
        super().__init__(model_path, **kwargs)
        self.conf        = conf
        self.alpha       = alpha
        self.class_names = class_names or DEFAULT_NAMES
        self._model      = None

    def load(self):
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)
        self.loaded = True
        print(f"[YOLOSegmentationPlugin] Loaded '{self.model_path}'")

    def annotate(self, image: np.ndarray) -> np.ndarray:
        from .model_wrapper import draw_segmentation_masks
        self.ensure_loaded()
        results = self._model.predict(source=image, conf=self.conf, verbose=False)
        masks_data = []
        if results and results[0].masks is not None:
            for i, m in enumerate(results[0].masks.data):
                mask_np = m.cpu().numpy().astype(bool)
                # Resize mask to image size if needed
                import cv2
                if mask_np.shape != image.shape[:2]:
                    mask_np = cv2.resize(
                        mask_np.astype(np.uint8),
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                cls_id = int(results[0].boxes.cls[i])
                masks_data.append({
                    'mask': mask_np,
                    'label': self.class_names.get(cls_id, str(cls_id)),
                    'class_id': cls_id,
                })
        annotated = draw_segmentation_masks(image, masks_data, self.alpha)
        # Also draw boxes on top
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                dets.append({'label': self.class_names.get(cls_id, str(cls_id)),
                             'score': float(box.conf[0]),
                             'bbox': [int(x1),int(y1),int(x2-x1),int(y2-y1)]})
        return draw_detection_boxes(annotated, dets)
