import os
import shutil
import random

dataset_dir = 'path_to_combined_dataset'
train_dir = 'path_to_train_set'
val_dir = 'path_to_val_set'
test_dir = 'path_to_test_set'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

all_files = os.listdir(dataset_dir)
random.shuffle(all_files)

total_samples = len(all_files)
train_size = int(0.7 * total_samples)
val_size = int(0.2 * total_samples)
test_size = total_samples - train_size - val_size

train_files = all_files[:train_size]
val_files = all_files[train_size:train_size + val_size]
test_files = all_files[train_size + val_size:]

def move_files(files, src_dir, dst_dir):
    for file in files:
        shutil.move(os.path.join(src_dir, file), os.path.join(dst_dir, file))

move_files(train_files, dataset_dir, train_dir)
move_files(val_files, dataset_dir, val_dir)
move_files(test_files, dataset_dir, test_dir)

