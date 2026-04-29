"""
test_yolov5.py
==============
Evaluate YOLOv5 on the test split via ultralytics model.val().

The test images are expected at:
    output/yolov5_data/images/test/
    output/yolov5_data/labels/test/

If not present, this script builds the test split automatically
(same logic as train_yolov5.py but for test indices 1000–1199).

Output written to: src/test/yolov5_results.txt

Run from project root:
    python src/test/test_yolov5.py
"""

import os
import sys
import shutil
import yaml
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from ultralytics import YOLO

# ── Paths & Config ────────────────────────────────────────────────────────────
DATA_ROOT   = os.path.join(ROOT, 'output', 'train')
YOLO_DATA   = os.path.join(ROOT, 'output', 'yolov5_data')
BEST_WEIGHTS = os.path.join(ROOT, 'src', 'logs', 'yolov5', 'run', 'weights', 'best.pt')
RESULT_PATH = os.path.join(ROOT, 'src', 'test', 'yolov5_results.txt')

CLASS_NAMES = {0: 'car', 1: 'tree', 2: 'lamppost'}
TEST_START, TEST_END = 1000, 1200


# ── Build test split directories if missing ───────────────────────────────────
def setup_test_split():
    img_dir = os.path.join(YOLO_DATA, 'images', 'test')
    lbl_dir = os.path.join(YOLO_DATA, 'labels', 'test')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    rgb_src  = os.path.join(DATA_ROOT, 'rgb')
    yolo_src = os.path.join(DATA_ROOT, 'yolo')

    print('Preparing test split …')
    for idx in range(TEST_START, TEST_END):
        img_dst = os.path.join(img_dir, f'{idx:04d}.png')
        lbl_dst = os.path.join(lbl_dir, f'{idx:04d}.txt')
        if not os.path.isfile(img_dst):
            shutil.copy2(os.path.join(rgb_src,  f'{idx:04d}.png'), img_dst)
        if not os.path.isfile(lbl_dst):
            src = os.path.join(yolo_src, f'{idx:04d}.txt')
            if os.path.isfile(src):
                shutil.copy2(src, lbl_dst)
            else:
                open(lbl_dst, 'w').close()


def update_yaml_with_test():
    yaml_path = os.path.join(YOLO_DATA, 'data.yaml')
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    data['test'] = 'images/test'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def main():
    if not os.path.isfile(BEST_WEIGHTS):
        print(f'ERROR: Best weights not found: {BEST_WEIGHTS}')
        print('Please run  python src/train/train_yolov5.py  first.')
        sys.exit(1)

    setup_test_split()
    yaml_path = update_yaml_with_test()

    print(f'Loading weights: {BEST_WEIGHTS}')
    model = YOLO(BEST_WEIGHTS)

    print('Evaluating on test split …')
    results = model.val(
        data   = yaml_path,
        split  = 'test',
        imgsz  = 640,
        batch  = 16,
        device = '0' if torch.cuda.is_available() else 'cpu',
        verbose= True,
    )

    # ── Extract metrics ───────────────────────────────────────────────────────
    box = results.box if hasattr(results, 'box') else results

    map50    = float(box.map50)
    map_5095 = float(box.map)
    precision = float(box.mp)   # mean precision
    recall    = float(box.mr)   # mean recall

    # per-class AP (if available)
    per_class_lines = []
    try:
        maps = box.maps           # list of AP50 per class
        for i, name in CLASS_NAMES.items():
            per_class_lines.append(f'  {name:<12}: {maps[i]*100:.2f}%')
    except Exception:
        per_class_lines = ['  (per-class breakdown not available)']

    lines = [
        '=== YOLOv5 Test Results ===',
        f'Dataset: {TEST_END - TEST_START} images (index 1000-1199)',
        f'Weights: {BEST_WEIGHTS}',
        '',
        'Per-class AP@0.5:',
        *per_class_lines,
        '',
        f'mAP@0.5     : {map50*100:.2f}%',
        f'mAP@0.5:0.95: {map_5095*100:.2f}%',
        f'Precision   : {precision*100:.2f}%',
        f'Recall      : {recall*100:.2f}%',
    ]

    report = '\n'.join(lines)
    print('\n' + report)

    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(f'\nResults saved to: {RESULT_PATH}')


if __name__ == '__main__':
    main()
