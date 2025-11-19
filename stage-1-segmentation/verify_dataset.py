#!/usr/bin/env python3
"""
Script kiểm tra tính toàn vẹn của dataset ISIC 2018 sau khi chia
Đảm bảo mỗi file ảnh đều có file groundtruth tương ứng
"""

import os
from pathlib import Path
from collections import defaultdict

# Cấu hình đường dẫn
BASE_DIR = Path(__file__).parent / "data"

# Dataset gốc
ORIGINAL_INPUT_DIR = BASE_DIR / "ISIC2018_Task1_Input"
ORIGINAL_GROUNDTRUTH_DIR = BASE_DIR / "ISIC2018_Task1_GroundTruth"

# Dataset đã chia
SPLITS = {
    "train": {
        "input": BASE_DIR / "ISIC2018_Task1-2_Training_Input",
        "groundtruth": BASE_DIR / "ISIC2018_Task1_Training_GroundTruth"
    },
    "val": {
        "input": BASE_DIR / "ISIC2018_Task1-2_Validation_Input",
        "groundtruth": BASE_DIR / "ISIC2018_Task1_Validation_GroundTruth"
    },
    "test": {
        "input": BASE_DIR / "ISIC2018_Task1-2_Test_Input",
        "groundtruth": BASE_DIR / "ISIC2018_Task1_Test_GroundTruth"
    }
}

class DatasetVerifier:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
    
    def check_original_dataset(self):
        """Kiểm tra dataset gốc"""
        print("\n" + "=" * 70)
        print("CHECKING ORIGINAL DATASET")
        print("=" * 70)
        
        if not ORIGINAL_INPUT_DIR.exists():
            self.errors.append(f"❌ Original input directory not found: {ORIGINAL_INPUT_DIR}")
            return False
        
        if not ORIGINAL_GROUNDTRUTH_DIR.exists():
            self.errors.append(f"❌ Original groundtruth directory not found: {ORIGINAL_GROUNDTRUTH_DIR}")
            return False
        
        # Lấy danh sách file ảnh
        image_files = list(ORIGINAL_INPUT_DIR.glob("*.jpg"))
        self.stats['original_images'] = len(image_files)
        print(f"✓ Found {len(image_files)} images in original dataset")
        
        # Kiểm tra từng ảnh có mask không
        missing_masks = []
        for img_path in image_files:
            img_name = img_path.stem
            mask_path = ORIGINAL_GROUNDTRUTH_DIR / f"{img_name}_segmentation.png"
            
            if not mask_path.exists():
                missing_masks.append(img_name)
        
        if missing_masks:
            self.errors.append(f"❌ Found {len(missing_masks)} images without masks in original dataset")
            for name in missing_masks[:5]:  # Hiển thị 5 cái đầu
                print(f"  Missing mask for: {name}")
            if len(missing_masks) > 5:
                print(f"  ... and {len(missing_masks) - 5} more")
            return False
        
        self.stats['original_masks'] = len(image_files)
        print(f"✓ All {len(image_files)} images have corresponding masks")
        return True
    
    def check_split_dataset(self, split_name, split_dirs):
        """Kiểm tra một split cụ thể"""
        print(f"\n📁 Checking {split_name.upper()} set...")
        
        input_dir = split_dirs['input']
        groundtruth_dir = split_dirs['groundtruth']
        
        # Kiểm tra thư mục tồn tại
        if not input_dir.exists():
            self.errors.append(f"❌ {split_name} input directory not found: {input_dir}")
            return False
        
        if not groundtruth_dir.exists():
            self.errors.append(f"❌ {split_name} groundtruth directory not found: {groundtruth_dir}")
            return False
        
        # Lấy danh sách file
        image_files = sorted(list(input_dir.glob("*.jpg")))
        mask_files = sorted(list(groundtruth_dir.glob("*_segmentation.png")))
        
        num_images = len(image_files)
        num_masks = len(mask_files)
        
        self.stats[f'{split_name}_images'] = num_images
        self.stats[f'{split_name}_masks'] = num_masks
        
        print(f"  Images: {num_images}")
        print(f"  Masks:  {num_masks}")
        
        # Kiểm tra số lượng khớp
        if num_images != num_masks:
            self.errors.append(f"❌ {split_name}: Number of images ({num_images}) != masks ({num_masks})")
        
        # Kiểm tra từng ảnh có mask tương ứng
        missing_masks = []
        extra_masks = []
        
        image_names = {img.stem for img in image_files}
        mask_names = {mask.stem.replace('_segmentation', '') for mask in mask_files}
        
        # Ảnh không có mask
        missing_masks = image_names - mask_names
        # Mask không có ảnh
        extra_masks = mask_names - image_names
        
        if missing_masks:
            self.errors.append(f"❌ {split_name}: {len(missing_masks)} images without masks")
            for name in list(missing_masks)[:5]:
                print(f"    Missing mask for: {name}")
            if len(missing_masks) > 5:
                print(f"    ... and {len(missing_masks) - 5} more")
        
        if extra_masks:
            self.warnings.append(f"⚠ {split_name}: {len(extra_masks)} masks without images")
            for name in list(extra_masks)[:5]:
                print(f"    Extra mask for: {name}")
            if len(extra_masks) > 5:
                print(f"    ... and {len(extra_masks) - 5} more")
        
        if not missing_masks and not extra_masks:
            print(f"  ✓ All images have corresponding masks!")
            return True
        
        return len(missing_masks) == 0
    
    def check_duplicates(self):
        """Kiểm tra trùng lặp giữa các split"""
        print("\n" + "=" * 70)
        print("CHECKING FOR DUPLICATES BETWEEN SPLITS")
        print("=" * 70)
        
        # Lấy danh sách tên file từ mỗi split
        train_images = {f.stem for f in SPLITS['train']['input'].glob("*.jpg")} if SPLITS['train']['input'].exists() else set()
        val_images = {f.stem for f in SPLITS['val']['input'].glob("*.jpg")} if SPLITS['val']['input'].exists() else set()
        test_images = {f.stem for f in SPLITS['test']['input'].glob("*.jpg")} if SPLITS['test']['input'].exists() else set()
        
        # Kiểm tra overlap
        train_val_overlap = train_images & val_images
        train_test_overlap = train_images & test_images
        val_test_overlap = val_images & test_images
        
        has_duplicates = False
        
        if train_val_overlap:
            self.errors.append(f"❌ Found {len(train_val_overlap)} duplicates between train and val")
            print(f"❌ Train-Val overlap: {len(train_val_overlap)} files")
            for name in list(train_val_overlap)[:5]:
                print(f"  {name}")
            has_duplicates = True
        
        if train_test_overlap:
            self.errors.append(f"❌ Found {len(train_test_overlap)} duplicates between train and test")
            print(f"❌ Train-Test overlap: {len(train_test_overlap)} files")
            for name in list(train_test_overlap)[:5]:
                print(f"  {name}")
            has_duplicates = True
        
        if val_test_overlap:
            self.errors.append(f"❌ Found {len(val_test_overlap)} duplicates between val and test")
            print(f"❌ Val-Test overlap: {len(val_test_overlap)} files")
            for name in list(val_test_overlap)[:5]:
                print(f"  {name}")
            has_duplicates = True
        
        if not has_duplicates:
            print("✓ No duplicates found between splits!")
        
        return not has_duplicates
    
    def check_completeness(self):
        """Kiểm tra tất cả file gốc đã được chia hết chưa"""
        print("\n" + "=" * 70)
        print("CHECKING DATASET COMPLETENESS")
        print("=" * 70)
        
        if not ORIGINAL_INPUT_DIR.exists():
            print("⚠ Cannot check completeness: original directory not found")
            return True
        
        # Lấy tất cả file gốc
        original_images = {f.stem for f in ORIGINAL_INPUT_DIR.glob("*.jpg")}
        
        # Lấy tất cả file đã chia
        split_images = set()
        for split_name, split_dirs in SPLITS.items():
            if split_dirs['input'].exists():
                split_images.update({f.stem for f in split_dirs['input'].glob("*.jpg")})
        
        # Kiểm tra thiếu
        missing_in_splits = original_images - split_images
        extra_in_splits = split_images - original_images
        
        print(f"Original dataset: {len(original_images)} images")
        print(f"Split dataset:    {len(split_images)} images")
        
        if missing_in_splits:
            self.warnings.append(f"⚠ {len(missing_in_splits)} images from original not in any split")
            print(f"\n⚠ Missing in splits: {len(missing_in_splits)} files")
            for name in list(missing_in_splits)[:10]:
                print(f"  {name}")
            if len(missing_in_splits) > 10:
                print(f"  ... and {len(missing_in_splits) - 10} more")
        
        if extra_in_splits:
            self.errors.append(f"❌ {len(extra_in_splits)} images in splits not in original")
            print(f"\n❌ Extra in splits: {len(extra_in_splits)} files")
            for name in list(extra_in_splits)[:10]:
                print(f"  {name}")
        
        if not missing_in_splits and not extra_in_splits:
            print("✓ All original images are included exactly once!")
        
        return len(extra_in_splits) == 0
    
    def print_summary(self):
        """In tóm tắt kết quả"""
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)
        
        # Statistics
        if self.stats:
            print("\n📊 Dataset Statistics:")
            if 'original_images' in self.stats:
                print(f"  Original: {self.stats['original_images']} images")
            
            total_split = 0
            for split in ['train', 'val', 'test']:
                if f'{split}_images' in self.stats:
                    count = self.stats[f'{split}_images']
                    total_split += count
                    if 'original_images' in self.stats and self.stats['original_images'] > 0:
                        percent = (count / self.stats['original_images']) * 100
                        print(f"  {split.capitalize():6s}: {count:4d} images ({percent:5.1f}%)")
            
            if total_split > 0 and 'original_images' in self.stats:
                print(f"  {'Total':6s}: {total_split:4d} images")
        
        # Errors
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  {error}")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        # Final verdict
        print("\n" + "=" * 70)
        if not self.errors:
            print("✅ VERIFICATION PASSED - Dataset is ready to use!")
        else:
            print("❌ VERIFICATION FAILED - Please fix the errors above!")
        print("=" * 70)
    
    def run(self):
        """Chạy tất cả các kiểm tra"""
        print("=" * 70)
        print("ISIC 2018 DATASET VERIFICATION")
        print("=" * 70)
        
        # 1. Kiểm tra dataset gốc (optional)
        self.check_original_dataset()
        
        # 2. Kiểm tra từng split
        print("\n" + "=" * 70)
        print("CHECKING SPLIT DATASETS")
        print("=" * 70)
        
        for split_name, split_dirs in SPLITS.items():
            self.check_split_dataset(split_name, split_dirs)
        
        # 3. Kiểm tra trùng lặp
        self.check_duplicates()
        
        # 4. Kiểm tra đầy đủ
        self.check_completeness()
        
        # 5. Tóm tắt
        self.print_summary()
        
        return len(self.errors) == 0


def main():
    verifier = DatasetVerifier()
    success = verifier.run()
    
    # Exit code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
