import numpy as np
import os, json
from PIL import Image
from render.renderer import MASK_PALETTE, _PALETTE_SIZE

# Only these types contribute to object-detection labels
DETECTION_TYPES = {'car', 'tree', 'lamppost'}
CATEGORIES = {'car': 0, 'tree': 1, 'lamppost': 2}


class Exporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        for sub in ('rgb', 'depth', 'mask', 'yolo'):
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": v, "name": k} for k, v in CATEGORIES.items()]
        }
        self._annot_id = 1

    # ── Mask lookup ──────────────────────────────────────────────────────────
    def _get_bbox(self, mask, obj_id):
        """Return [x,y,w,h] bounding box of pixels matching obj_id's palette colour."""
        r, g, b = MASK_PALETTE[obj_id % _PALETTE_SIZE]
        hit = ((mask[:, :, 0] == r) & (mask[:, :, 1] == g) & (mask[:, :, 2] == b))
        if not np.any(hit):
            return None
        rows = np.any(hit, axis=1)
        cols = np.any(hit, axis=0)
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]

    # ── Main export entry ────────────────────────────────────────────────────
    def export(self, frame_idx, rgb, depth, mask, objects, label_fmt='Both'):
        h, w = rgb.shape[:2]

        Image.fromarray(rgb  ).save(os.path.join(self.output_dir, 'rgb',   f"{frame_idx:04d}.png"))
        Image.fromarray(depth).save(os.path.join(self.output_dir, 'depth', f"{frame_idx:04d}.png"))
        Image.fromarray(mask ).save(os.path.join(self.output_dir, 'mask',  f"{frame_idx:04d}.png"))

        if label_fmt in ('COCO', 'Both'):
            self.coco_data['images'].append({
                "id": frame_idx, "file_name": f"{frame_idx:04d}.png",
                "width": w, "height": h
            })

        yolo_lines = []
        for obj in objects:
            if obj['type'] not in DETECTION_TYPES:
                continue
            bbox = self._get_bbox(mask, obj['id'])
            if bbox is None:
                continue
            x, y, bw, bh = bbox
            if bw < 2 or bh < 2:
                continue   # skip near-invisible detections
            cat_id = CATEGORIES[obj['type']]

            if label_fmt in ('COCO', 'Both'):
                self.coco_data['annotations'].append({
                    "id": self._annot_id, "image_id": frame_idx,
                    "category_id": cat_id,
                    "bbox": bbox, "area": bw * bh, "iscrowd": 0
                })
                self._annot_id += 1

            if label_fmt in ('YOLO', 'Both'):
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                yolo_lines.append(
                    f"{cat_id} {xc:.6f} {yc:.6f} {bw/w:.6f} {bh/h:.6f}")

        if label_fmt in ('YOLO', 'Both') and yolo_lines:
            with open(os.path.join(self.output_dir, 'yolo', f"{frame_idx:04d}.txt"), 'w') as f:
                f.write("\n".join(yolo_lines))

    def save_coco(self):
        with open(os.path.join(self.output_dir, 'coco_annotations.json'), 'w') as f:
            json.dump(self.coco_data, f, indent=2)
