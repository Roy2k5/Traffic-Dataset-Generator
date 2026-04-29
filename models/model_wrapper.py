"""
Post-processing pipeline module.

Architecture:
    Rendered RGB (numpy H×W×3)
        │
        ▼
    PostProcessor.run(image)
        │  ← chains multiple ModelPlugin instances
        ▼
    Annotated image (numpy H×W×3)
        │
        ▼
    Display / save

Each ModelPlugin receives an image, annotates it in-place or returns a new one,
then passes the result to the next plugin.  Plugins can be detection, segmentation,
depth-colorisation, classification, etc.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import cv2


# ── Base plugin ───────────────────────────────────────────────────────────────

class ModelPlugin(ABC):
    """
    Abstract base for any model that post-processes a rendered image.

    Subclasses override:
        load()      – load weights / initialise resources
        annotate()  – consume an RGB image, return annotated RGB image
    """

    # Human-readable task type – used by UI to label the overlay
    TASK: str = "generic"

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.config     = kwargs
        self.loaded     = False

    def ensure_loaded(self):
        if not self.loaded:
            self.load()

    @abstractmethod
    def load(self):
        """Load model weights."""
        ...

    @abstractmethod
    def annotate(self, image: np.ndarray) -> np.ndarray:
        """
        Annotate *image* (RGB uint8 H×W×3) and return annotated copy.
        Must NOT modify the input array in-place (make a copy first).
        """
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.model_path}')"


# ── Post-processing pipeline ──────────────────────────────────────────────────

class PostProcessor:
    """
    Chains multiple ModelPlugin instances.

    Usage:
        pp = PostProcessor()
        pp.add(YOLODetectionPlugin('models/weights/best.pt'))
        pp.add(SomeSegPlugin('models/weights/seg.pt'))

        annotated = pp.run(rgb_image)   # runs all plugins in order
    """

    def __init__(self):
        self._plugins: list[ModelPlugin] = []

    def add(self, plugin: ModelPlugin) -> "PostProcessor":
        plugin.ensure_loaded()
        self._plugins.append(plugin)
        return self   # fluent API

    def remove(self, plugin: ModelPlugin):
        self._plugins.remove(plugin)

    def clear(self):
        self._plugins.clear()

    @property
    def plugins(self):
        return list(self._plugins)

    def run(self, image: np.ndarray) -> np.ndarray:
        """
        Pass *image* through every plugin in order.
        Returns the final annotated image.
        """
        result = image.copy()
        for plugin in self._plugins:
            result = plugin.annotate(result)
        return result

    def __len__(self):
        return len(self._plugins)

    def __repr__(self):
        return f"PostProcessor({self._plugins})"


# ── Drawing helpers ────────────────────────────────────────────────────────────

# Palette of 20 distinct BGR colours for class colouring
_PALETTE = [
    (56, 159, 245), (73, 218, 73),  (237, 106,  38), (214,  47, 214),
    (56, 245, 227), (214,  47,  81), (181, 237,  73), (245,  56, 130),
    (255, 195,  56), (56, 123, 245), (245, 186,  56), (109,  56, 245),
    (56, 245, 131), (245,  56, 214), (56, 209, 245), (130, 245,  56),
    (245,  56,  56), (56,  56, 245), (245, 245,  56), (109, 245, 181),
]

def palette_color(idx: int) -> tuple:
    return _PALETTE[idx % len(_PALETTE)]


def draw_detection_boxes(image: np.ndarray,
                          detections: list[dict],
                          label_key='label',
                          score_key='score',
                          bbox_key='bbox',
                          thickness=2) -> np.ndarray:
    """
    Draw bounding boxes on *image* (RGB).  detections is a list of dicts with
    keys: label (str), score (float), bbox ([x,y,w,h] in pixels).
    Returns a new annotated image.
    """
    out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i, det in enumerate(detections):
        x, y, w, h = [int(v) for v in det.get(bbox_key, [0,0,0,0])]
        color = palette_color(i)
        cv2.rectangle(out, (x, y), (x+w, y+h), color, thickness)
        label = det.get(label_key, '?')
        score = det.get(score_key, 0.0)
        text  = f"{label} {score:.2f}" if score else label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x, y-th-6), (x+tw+4, y), color, -1)
        cv2.putText(out, text, (x+2, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def draw_segmentation_masks(image: np.ndarray,
                             masks: list[dict],
                             alpha=0.45) -> np.ndarray:
    """
    Blend semantic/instance segmentation masks onto *image* (RGB).
    masks is a list of dicts: {'mask': H×W bool array, 'label': str, 'class_id': int}.
    """
    out = image.copy().astype(np.float32)
    for seg in masks:
        m     = seg.get('mask')
        cid   = seg.get('class_id', 0)
        color = np.array(palette_color(cid)[::-1], dtype=np.float32)  # BGR→RGB
        if m is not None and m.any():
            out[m] = out[m] * (1 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
