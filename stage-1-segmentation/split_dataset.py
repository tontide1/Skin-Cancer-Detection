#!/usr/bin/env python3
"""
Script chia dataset ISIC 2018 theo tỷ lệ 70/15/15 (train/val/test)
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import numpy as np

# Set random seed để reproducible
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Cấu hình đường dẫn
BASE_DIR = Path(__file__).parent / "data"
INPUT_DIR = BASE_DIR / "ISIC2018_Task1_Input"
GROUNDTRUTH_DIR = BASE_DIR / "ISIC2018_Task1_GroundTruth"

# Thư mục đích
OUTPUT_BASE = BASE_DIR
TRAIN_IMG_DIR = OUTPUT_BASE / "ISIC2018_Task1-2_Training_Input"
TRAIN_MASK_DIR = OUTPUT_BASE / "ISIC2018_Task1_Training_GroundTruth"

VAL_IMG_DIR = OUTPUT_BASE / "ISIC2018_Task1-2_Validation_Input"
VAL_MASK_DIR = OUTPUT_BASE / "ISIC2018_Task1_Validation_GroundTruth"

TEST_IMG_DIR = OUTPUT_BASE / "ISIC2018_Task1-2_Test_Input"
TEST_MASK_DIR = OUTPUT_BASE / "ISIC2018_Task1_Test_GroundTruth"

# Tỷ lệ chia
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_directories():
    """Tạo các thư mục đích"""
    dirs = [
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        TEST_IMG_DIR, TEST_MASK_DIR
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")

def get_image_list():
    """Lấy danh sách tất cả các file ảnh"""
    image_files = sorted(list(INPUT_DIR.glob("*.jpg")))
    
    # Lọc ra những file có mask tương ứng
    valid_images = []
    for img_path in image_files:
        img_name = img_path.stem  # ISIC_0000000
        mask_path = GROUNDTRUTH_DIR / f"{img_name}_segmentation.png"
        
        if mask_path.exists():
            valid_images.append(img_name)
        else:
            print(f"⚠ Warning: Missing mask for {img_name}")
    
    return valid_images

def split_dataset(image_names):
    """Chia dataset thành train/val/test"""
    # Chia train và temp (val + test)
    train_names, temp_names = train_test_split(
        image_names,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    # Chia temp thành val và test
    val_names, test_names = train_test_split(
        temp_names,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    return train_names, val_names, test_names

def copy_files(image_names, img_dest_dir, mask_dest_dir, split_name):
    """Copy files vào thư mục tương ứng"""
    print(f"\n📁 Copying {split_name} set ({len(image_names)} files)...")
    
    for i, img_name in enumerate(image_names, 1):
        # Copy image
        src_img = INPUT_DIR / f"{img_name}.jpg"
        dst_img = img_dest_dir / f"{img_name}.jpg"
        shutil.copy2(src_img, dst_img)
        
        # Copy mask
        src_mask = GROUNDTRUTH_DIR / f"{img_name}_segmentation.png"
        dst_mask = mask_dest_dir / f"{img_name}_segmentation.png"
        shutil.copy2(src_mask, dst_mask)
        
        if i % 100 == 0:
            print(f"  Copied {i}/{len(image_names)} files...")
    
    print(f"✓ Completed {split_name} set!")

def print_statistics(train_names, val_names, test_names, total):
    """In thống kê"""
    print("\n" + "=" * 60)
    print("DATASET SPLIT STATISTICS")
    print("=" * 60)
    print(f"Total images: {total}")
    print(f"\nTrain set: {len(train_names)} images ({len(train_names)/total*100:.1f}%)")
    print(f"Val set:   {len(val_names)} images ({len(val_names)/total*100:.1f}%)")
    print(f"Test set:  {len(test_names)} images ({len(test_names)/total*100:.1f}%)")
    print("=" * 60)

def main():
    print("=" * 60)
    print("ISIC 2018 DATASET SPLITTING")
    print("=" * 60)
    print(f"Train : Val : Test = {TRAIN_RATIO:.0%} : {VAL_RATIO:.0%} : {TEST_RATIO:.0%}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 60)
    
    # 1. Tạo thư mục
    print("\n[Step 1/4] Creating directories...")
    create_directories()
    
    # 2. Lấy danh sách ảnh
    print("\n[Step 2/4] Getting image list...")
    image_names = get_image_list()
    print(f"✓ Found {len(image_names)} valid image-mask pairs")
    
    # 3. Chia dataset
    print("\n[Step 3/4] Splitting dataset...")
    train_names, val_names, test_names = split_dataset(image_names)
    print_statistics(train_names, val_names, test_names, len(image_names))
    
    # 4. Copy files
    print("\n[Step 4/4] Copying files...")
    copy_files(train_names, TRAIN_IMG_DIR, TRAIN_MASK_DIR, "Train")
    copy_files(val_names, VAL_IMG_DIR, VAL_MASK_DIR, "Validation")
    copy_files(test_names, TEST_IMG_DIR, TEST_MASK_DIR, "Test")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ DATASET SPLIT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nOutput directories:")
    print(f"  Train Images: {TRAIN_IMG_DIR}")
    print(f"  Train Masks:  {TRAIN_MASK_DIR}")
    print(f"  Val Images:   {VAL_IMG_DIR}")
    print(f"  Val Masks:    {VAL_MASK_DIR}")
    print(f"  Test Images:  {TEST_IMG_DIR}")
    print(f"  Test Masks:   {TEST_MASK_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
