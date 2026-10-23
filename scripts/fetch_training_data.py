"""
MarineVision - Fast Training Split Downloader via Direct HTTP.
Downloads a balanced, representative selection of 300-350 training images & labels
directly from Hugging Face into data_raw/train.
"""

import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi

REPO_ID = "rehan9599/drishti-sss"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

# Target 60 balanced samples per group = 300 images + labels
PER_GROUP = 60


def download_url(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True, "already exists"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, 'wb') as f:
                f.write(r.content)
            return True, "ok"
        else:
            return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


def main():
    print(f"Listing training files from {REPO_ID}...")
    api = HfApi()
    all_files = set(api.list_repo_files(REPO_ID, repo_type='dataset'))

    train_imgs = [f for f in all_files if f.startswith('train/images/')]
    from collections import defaultdict
    grouped = defaultdict(list)
    for f in sorted(train_imgs):
        p = os.path.basename(f).split('_')[0]
        grouped[p].append(f)

    tasks = []
    dest_img_dir = os.path.join('data_raw', 'train', 'images')
    dest_lbl_dir = os.path.join('data_raw', 'train', 'labels')

    print(f"Selecting up to {PER_GROUP} samples per group:")
    for prefix, files in sorted(grouped.items()):
        chosen = files[:PER_GROUP]
        print(f"  {prefix}: {len(chosen)} images")
        for img_rel in chosen:
            img_url = f"{BASE_URL}/{img_rel}"
            img_dst = os.path.join(dest_img_dir, os.path.basename(img_rel))
            tasks.append((img_url, img_dst))

            base = os.path.splitext(os.path.basename(img_rel))[0]
            lbl_rel = f"train/labels/{base}.txt"
            if lbl_rel in all_files:
                lbl_url = f"{BASE_URL}/{lbl_rel}"
                lbl_dst = os.path.join(dest_lbl_dir, f"{base}.txt")
                tasks.append((lbl_url, lbl_dst))
            else:
                lbl_upper = f"train/labels/{base}.TXT"
                if lbl_upper in all_files:
                    lbl_url = f"{BASE_URL}/{lbl_upper}"
                    lbl_dst = os.path.join(dest_lbl_dir, f"{base}.txt")
                    tasks.append((lbl_url, lbl_dst))

    print(f"\nStarting download of {len(tasks)} files across 16 threads...")
    start_t = time.time()
    success_count = 0

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(download_url, url, dst): (url, dst) for url, dst in tasks}
        for fut in as_completed(futures):
            ok, msg = fut.result()
            if ok:
                success_count += 1
            if success_count % 50 == 0 or success_count == len(tasks):
                print(f"  Downloaded: {success_count}/{len(tasks)} files...")

    elapsed = round(time.time() - start_t, 2)
    print(f"\nFinished in {elapsed}s. {success_count}/{len(tasks)} files saved to data_raw/train.")


if __name__ == '__main__':
    main()
