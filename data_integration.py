import os
import shutil
import random

bccd_bcd_dir = 'path_to_bccd_bcd_dataset'
pbc_dir = 'path_to_pbc_dataset'
raabinwbc_dir = 'path_to_raabinwbc_dataset'

output_dir = 'path_to_combined_dataset'
os.makedirs(output_dir, exist_ok=True)

def copy_and_rename_images(src_dir, dst_dir, start_idx):
    images = os.listdir(src_dir)
    for idx, img_file in enumerate(images):
        new_name = f'image_{start_idx + idx:04d}.jpg'
        shutil.copy(os.path.join(src_dir, img_file), os.path.join(dst_dir, new_name))
    return start_idx + len(images)

def combine_datasets():
    start_idx = 0
    start_idx = copy_and_rename_images(bccd_bcd_dir, output_dir, start_idx)
    start_idx = copy_and_rename_images(pbc_dir, output_dir, start_idx)
    start_idx = copy_and_rename_images(raabinwbc_dir, output_dir, start_idx)

    all_images = os.listdir(output_dir)
    random.shuffle(all_images)

    for idx, img_file in enumerate(all_images):
        old_path = os.path.join(output_dir, img_file)
        new_path = os.path.join(output_dir, f'BloodImage_{idx + 1:04d}.jpg')
        os.rename(old_path, new_path)

combine_datasets()


