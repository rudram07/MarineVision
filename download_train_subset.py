"""
Helper script to download a stratified, balanced training split from rehan9599/drishti-sss
into data_raw/train/images and data_raw/train/labels.
"""

import os
import time
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "rehan9599/drishti-sss"
SAMPLES_PER_GROUP = 70  # 70 * 5 groups = 350 balanced training images

def download_file(hf_path, dest_dir):
    try:
        local_cache = hf_hub_download(REPO_ID, hf_path, repo_type='dataset')
        fname = os.path.basename(hf_path)
        dest_path = os.path.join(dest_dir, fname)
        os.makedirs(dest_dir, exist_ok=True)
        if not os.path.exists(dest_path):
            import shutil
            shutil.copyfile(local_cache, dest_path)
        return True, hf_path
    except Exception as e:
        return False, f"{hf_path}: {e}"

def main():
    print(f"Connecting to Hugging Face dataset {REPO_ID}...")
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type='dataset')

    train_imgs = [f for f in all_files if f.startswith('train/images/')]
    grouped = collections.defaultdict(list)
    for f in train_imgs:
        fname = os.path.basename(f)
        prefix = fname.split('_')[0]
        grouped[prefix].append(f)

    # Select balanced images and their corresponding label files
    selected_img_files = []
    selected_lbl_files = []

    for prefix, files in sorted(grouped.items()):
        chosen = files[:SAMPLES_PER_GROUP]
        for img_f in chosen:
            selected_img_files.append(img_f)
            base = os.path.splitext(os.path.basename(img_f))[0]
            lbl_f = f"train/labels/{base}.txt"
            if lbl_f in all_files:
                selected_lbl_files.append(lbl_f)
            else:
                # Some labels might be uppercase or background
                lbl_upper = f"train/labels/{base}.TXT"
                if lbl_upper in all_files:
                    selected_lbl_files.append(lbl_upper)

    print(f"Selected {len(selected_img_files)} images and {len(selected_lbl_files)} labels across groups:")
    for prefix, files in grouped.items():
        print(f"  {prefix}: {min(SAMPLES_PER_GROUP, len(files))} / {len(files)}")

    dest_train_imgs = os.path.join('data_raw', 'train', 'images')
    dest_train_lbls = os.path.join('data_raw', 'train', 'labels')
    os.makedirs(dest_train_imgs, exist_ok=True)
    dest_train_lbls_path = os.path.join('data_raw', 'train', 'labels')
    os.makedirs(dest_train_lbls_path, exist_ok=True)

    start_time = time.time()
    total_to_download = len(selected_img_files) + len(selected_lbl_files)
    print(f"\nDownloading {total_to_download} files using 16 concurrent workers...")

    tasks = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for f in selected_img_files:
            tasks.append(executor.submit(download_file, f, dest_train_imgs))
        for f in selected_lbl_files:
            tasks.append(executor.submit(download_file, f, dest_train_lbls))

        completed = 0
        for future in as_completed(tasks):
            success, info = future.result()
            completed += 1
            if completed % 50 == 0 or completed == total_to_download:
                print(f"  Progress: {completed}/{total_to_download} files downloaded...")

    elapsed = round(time.time() - start_time, 2)
    print(f"\nSuccessfully downloaded training split in {elapsed}s to data_raw/train/")

if __name__ == '__main__':
    main()
