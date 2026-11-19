"""Move the Tiny-ImageNet validation images into one folder per class, as torchvision's
ImageFolder expects. Run once after extracting the dataset to data/archive/tiny-imagenet-200.

    python scripts/prepare_tiny_imagenet.py
"""

import os

VAL_DIR = './data/archive/tiny-imagenet-200/val'


def main():
    with open(os.path.join(VAL_DIR, 'val_annotations.txt')) as f:
        rows = [line.strip().split('\t')[:2] for line in f if line.strip()]
    moved = 0
    for image, label in rows:
        src = os.path.join(VAL_DIR, 'images', image)
        if os.path.exists(src):
            os.makedirs(os.path.join(VAL_DIR, label), exist_ok=True)
            os.rename(src, os.path.join(VAL_DIR, label, image))
            moved += 1
    if os.path.isdir(os.path.join(VAL_DIR, 'images')) and not os.listdir(os.path.join(VAL_DIR, 'images')):
        os.rmdir(os.path.join(VAL_DIR, 'images'))
    print(f'moved {moved} images')


if __name__ == '__main__':
    main()
