import os
import cv2
import numpy as np
from pathlib import Path
import random

def extract_patches(src_dir, dst_dir, patch_size=640, patches_per_image=5):
    src = Path(src_dir)
    dst = Path(dst_dir)

    for split in ["train", "val", "test"]:
        n = patches_per_image if split == "train" else 1

        for cls in ["baluchari", "maheshwari", "negammam", "phulkari"]:
            in_dir  = src / split / cls
            out_dir = dst / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            images = list(in_dir.glob("*.[jp][pn]g")) + list(in_dir.glob("*.jpeg"))
            saved = 0

            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None: continue

                h, w = img.shape[:2]

                # STEP 1: Resize to standard size first
                # 1280px longest side — zoom consistent across all images
                scale = 1280 / max(h, w)
                img = cv2.resize(img, (int(w*scale), int(h*scale)))
                h, w = img.shape[:2]

                # STEP 2: Center 70% mein crop
                margin_x = int(w * 0.15)
                margin_y = int(h * 0.15)

                got = 0
                attempts = 0
                while got < n and attempts < 20:
                    attempts += 1
                    x = random.randint(margin_x, max(margin_x+1, w - patch_size - margin_x))
                    y = random.randint(margin_y, max(margin_y+1, h - patch_size - margin_y))
                    patch = img[y:y+patch_size, x:x+patch_size]

                    if patch.std() < 15:
                        continue

                    cv2.imwrite(str(out_dir / f"{img_path.stem}_p{got}.jpg"), patch)
                    got += 1
                    saved += 1

            print(f"{split}/{cls}: {len(images)} imgs → {saved} patches")

extract_patches("data", "data_patched")