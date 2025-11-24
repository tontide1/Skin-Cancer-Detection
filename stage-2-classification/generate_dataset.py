"""
generate_dataset.py

Pipeline: Low-Res Localization, High-Res Extraction
----------------------------------------------------
Chiến lược xử lý ảnh y tế kích thước lớn bằng cách:
1. Downscale ảnh gốc xuống 256x256 → Inference với Segmentation Model
2. Upscale Mask 256x256 lên kích thước gốc (giữ nguyên tọa độ)
3. Tính Bounding Box trên Mask upscaled
4. Crop ảnh gốc với độ phân giải nguyên bản 100%

"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from PIL import Image
from tqdm.auto import tqdm
import argparse

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for dataset generation pipeline"""
    
    # Paths
    MODEL_PATH = Path("/home/tontide1/coding/deep_learning/Skin-Cancer-Detection/stage-1-segmentation/output/final_resnet34_unet/results/best_model.pth")
    INPUT_DIR = Path("data/raw/ISIC_2019_Training_Input")
    OUTPUT_DIR = Path("data/processed/cropped_lesions")
    
    # Model Configuration (from isic-2018-segmentation.py)
    MODEL_CONFIG = {
        'encoder_name': 'resnet34',
        'encoder_weights': 'imagenet',
        'decoder_attention_type': 'scse',
        'in_channels': 3,
        'classes': 1,
    }
    
    # Processing Configuration
    INFERENCE_SIZE = (256, 256)  # Size for segmentation inference
    PADDING = 30  # Padding around bounding box (pixels)
    MASK_THRESHOLD = 0.5  # Threshold for binary mask
    MIN_CONTOUR_AREA = 100  # Minimum contour area to filter noise
    FORCE_SQUARE_CROP = True  # Force square crop to prevent aspect ratio distortion
    CENTER_CROP_FALLBACK = True  # Use center crop as fallback when mask not found
    CENTER_CROP_SIZE = 512  # Size for center crop fallback
    
    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MIXED_PRECISION = True
    
    # Data augmentation (for inference - only resize + normalize)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_segmentation_model(model_path: Path, config: Dict, device: str) -> nn.Module:
    """
    Load pretrained segmentation model from checkpoint
    
    Args:
        model_path: Path to .pth checkpoint file
        config: Model configuration dictionary
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Loaded model in eval mode
    """
    print(f" Loading segmentation model from: {model_path}")
    
    # Create model architecture
    model = smp.Unet(
        encoder_name=config['encoder_name'],
        encoder_weights=None,  # We'll load from checkpoint
        decoder_attention_type=config['decoder_attention_type'],
        in_channels=config['in_channels'],
        classes=config['classes'],
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle DataParallel wrapper if present
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' prefix from DataParallel
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f" Model loaded successfully!")
    print(f"   Checkpoint info: Epoch {checkpoint.get('epoch', 'N/A')}, Val Dice: {checkpoint.get('val_dice', 'N/A'):.4f}")
    
    return model


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def get_inference_transform(input_size: Tuple[int, int], 
                           mean: list, 
                           std: list) -> A.Compose:
    """
    Get Albumentations transform for inference (resize + normalize only)
    
    Args:
        input_size: Target size (H, W) for model input
        mean: Normalization mean values
        std: Normalization std values
    
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(*input_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def preprocess_image(image_path: Path, 
                    transform: A.Compose) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    Load and preprocess image for model inference
    
    Args:
        image_path: Path to input image
        transform: Albumentations transform pipeline
    
    Returns:
        Tuple of (preprocessed_tensor, original_image_rgb, original_size)
    """
    # Load image in RGB (OpenCV loads in BGR by default)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_size = (image_rgb.shape[0], image_rgb.shape[1])  # (H, W)
    
    # Apply transform
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image']
    
    return image_tensor, image_rgb, original_size


# ============================================================================
# SEGMENTATION INFERENCE
# ============================================================================

@torch.no_grad()
def predict_mask(model: nn.Module, 
                image_tensor: torch.Tensor, 
                device: str,
                threshold: float = 0.5,
                use_amp: bool = True) -> np.ndarray:
    """
    Run segmentation inference on preprocessed image
    
    Args:
        model: Loaded segmentation model
        image_tensor: Preprocessed image tensor (C, H, W)
        device: Device to run inference on
        threshold: Threshold for binary mask
        use_amp: Whether to use mixed precision (faster on GPU)
    
    Returns:
        Binary mask as numpy array (H, W) with values {0, 1}
    """
    # Add batch dimension and move to device
    image_batch = image_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    
    # Inference with optional mixed precision
    with autocast(enabled=use_amp):
        logits = model(image_batch)  # (1, 1, H, W)
    
    # Convert to probability
    probs = torch.sigmoid(logits)
    
    # Binarize mask
    mask = (probs > threshold).float()
    
    # Convert to numpy: (1, 1, H, W) → (H, W)
    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
    
    return mask_np


# ============================================================================
# MASK PROCESSING & BOUNDING BOX EXTRACTION
# ============================================================================

def upscale_mask(mask_lowres: np.ndarray, 
                original_size: Tuple[int, int]) -> np.ndarray:
    """
    Upscale low-resolution mask to original image size
    
    Args:
        mask_lowres: Binary mask from model (256x256)
        original_size: Target size (H, W)
    
    Returns:
        Upscaled binary mask with same size as original image
    
    Note:
        Uses INTER_NEAREST to preserve binary values {0, 1}
    """
    # Multiply by 255 for proper interpolation, then threshold back
    mask_upscaled = cv2.resize(
        mask_lowres * 255, 
        (original_size[1], original_size[0]),  # OpenCV uses (W, H)
        interpolation=cv2.INTER_NEAREST
    )
    
    # Ensure binary values
    mask_upscaled = (mask_upscaled > 127).astype(np.uint8)
    
    return mask_upscaled


def compute_bbox_from_mask(mask: np.ndarray, 
                          padding: int = 0,
                          min_area: int = 100,
                          force_square: bool = True) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute bounding box from binary mask using contour detection
    
    Args:
        mask: Binary mask (H, W) with values {0, 1}
        padding: Padding to add around bbox (pixels)
        min_area: Minimum contour area to filter noise
        force_square: If True, expand bbox to square shape (prevents distortion)
    
    Returns:
        Bounding box (x, y, w, h) or None if no valid contour found
    
    Note:
        Returns coordinates in (x, y, w, h) format suitable for cropping.
        If force_square=True, w==h (square crop to prevent aspect ratio distortion).
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Filter tiny contours (noise)
    if cv2.contourArea(largest_contour) < min_area:
        return None
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Convert to (x1, y1, x2, y2) for easier manipulation
    img_h, img_w = mask.shape
    x1 = x - padding
    y1 = y - padding
    x2 = x + w + padding
    y2 = y + h + padding
    
    # Make square if requested (Critical: Prevents aspect ratio distortion)
    if force_square:
        current_w = x2 - x1
        current_h = y2 - y1
        target_size = max(current_w, current_h)  # Use larger dimension
        
        # Calculate center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Expand to square around center
        x1 = int(center_x - target_size / 2)
        y1 = int(center_y - target_size / 2)
        x2 = int(center_x + target_size / 2)
        y2 = int(center_y + target_size / 2)
    
    # Clamp coordinates to image boundaries (Improved boundary check)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    
    # Convert back to (x, y, w, h)
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    
    # Final validation: ensure non-zero area
    if w <= 0 or h <= 0:
        return None
    
    return (x, y, w, h)


# ============================================================================
# HIGH-RES CROPPING
# ============================================================================

def compute_center_crop_bbox(image_shape: Tuple[int, int], 
                            crop_size: int) -> Tuple[int, int, int, int]:
    """
    Compute bounding box for center crop (fallback strategy)
    
    Args:
        image_shape: Image shape (H, W)
        crop_size: Desired crop size (will be square)
    
    Returns:
        Bounding box (x, y, w, h) for center crop
    
    Note:
        This is used as fallback when mask detection fails.
        Prevents data loss by cropping the center region instead of discarding image.
    """
    img_h, img_w = image_shape
    
    # Use minimum dimension if crop_size is larger than image
    actual_crop_size = min(crop_size, img_h, img_w)
    
    # Calculate center coordinates
    center_x = img_w // 2
    center_y = img_h // 2
    
    # Calculate crop coordinates
    x = max(0, center_x - actual_crop_size // 2)
    y = max(0, center_y - actual_crop_size // 2)
    
    # Ensure we don't exceed image boundaries
    if x + actual_crop_size > img_w:
        x = img_w - actual_crop_size
    if y + actual_crop_size > img_h:
        y = img_h - actual_crop_size
    
    return (x, y, actual_crop_size, actual_crop_size)


def crop_high_res_lesion(image_rgb: np.ndarray, 
                        bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop lesion region from original high-resolution image
    
    Args:
        image_rgb: Original RGB image (H, W, 3)
        bbox: Bounding box (x, y, w, h)
    
    Returns:
        Cropped image region with original resolution
    """
    x, y, w, h = bbox
    cropped = image_rgb[y:y+h, x:x+w]
    return cropped


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_single_image(image_path: Path,
                        model: nn.Module,
                        transform: A.Compose,
                        config: Config,
                        save_visualization: bool = False) -> Optional[Dict]:
    """
    Process a single image through the entire pipeline
    
    Args:
        image_path: Path to input image
        model: Loaded segmentation model
        transform: Preprocessing transform
        config: Configuration object
        save_visualization: Whether to save visualization (for debugging)
    
    Returns:
        Dictionary with processing results or None if failed
    """
    try:
        # Step 1: Load & Preprocess (Downscale to 256x256)
        image_tensor, image_rgb, original_size = preprocess_image(image_path, transform)
        
        # Step 2: Inference (Get 256x256 mask)
        mask_lowres = predict_mask(
            model, 
            image_tensor, 
            config.DEVICE,
            threshold=config.MASK_THRESHOLD,
            use_amp=config.MIXED_PRECISION
        )
        
        # Step 3: Upscale mask to original size
        mask_highres = upscale_mask(mask_lowres, original_size)
        
        # Step 4: Compute bounding box with square crop (prevents distortion)
        bbox = compute_bbox_from_mask(
            mask_highres, 
            padding=config.PADDING,
            min_area=config.MIN_CONTOUR_AREA,
            force_square=config.FORCE_SQUARE_CROP
        )
        
        # Step 4.5: Fallback strategy if mask detection fails
        used_fallback = False
        if bbox is None:
            if config.CENTER_CROP_FALLBACK:
                # Use center crop instead of discarding the image
                bbox = compute_center_crop_bbox(original_size, config.CENTER_CROP_SIZE)
                used_fallback = True
                print(f"  Using center crop fallback for: {image_path.name}")
            else:
                print(f" No valid lesion found in: {image_path.name}")
                return None
        
        # Step 5: Crop high-res lesion
        cropped_lesion = crop_high_res_lesion(image_rgb, bbox)
        
        # Prepare result metadata
        result = {
            'image_name': image_path.stem,
            'original_size': original_size,
            'bbox': bbox,
            'cropped_size': (cropped_lesion.shape[0], cropped_lesion.shape[1]),
            'cropped_image': cropped_lesion,
            'used_fallback': used_fallback,
            'is_square': bbox[2] == bbox[3] if bbox else False,
        }
        
        # Optional: Save visualization for debugging
        if save_visualization:
            save_debug_visualization(image_rgb, mask_highres, bbox, image_path.stem, config)
        
        return result
        
    except Exception as e:
        print(f" Error processing {image_path.name}: {str(e)}")
        return None


def save_debug_visualization(image: np.ndarray,
                            mask: np.ndarray,
                            bbox: Tuple[int, int, int, int],
                            image_name: str,
                            config: Config):
    """Save visualization for debugging (optional)"""
    debug_dir = config.OUTPUT_DIR / 'debug_visualizations'
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Create overlay
    overlay = image.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    # Draw bounding box
    x, y, w, h = bbox
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Save
    output_path = debug_dir / f"{image_name}_debug.jpg"
    overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), overlay_bgr)


def process_dataset(config: Config, 
                   save_visualizations: bool = False,
                   max_images: Optional[int] = None):
    """
    Process entire dataset through the pipeline
    
    Args:
        config: Configuration object
        save_visualizations: Whether to save debug visualizations
        max_images: Maximum number of images to process (for testing)
    """
    print("="*70)
    print(" STARTING DATASET GENERATION PIPELINE")
    print("="*70)
    print(f"\n Input directory: {config.INPUT_DIR}")
    print(f" Output directory: {config.OUTPUT_DIR}")
    print(f"  Device: {config.DEVICE}")
    print(f" Inference size: {config.INFERENCE_SIZE}")
    print(f" Padding: {config.PADDING}px")
    
    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_segmentation_model(config.MODEL_PATH, config.MODEL_CONFIG, config.DEVICE)
    
    # Get preprocessing transform
    transform = get_inference_transform(
        config.INFERENCE_SIZE, 
        config.NORMALIZE_MEAN, 
        config.NORMALIZE_STD
    )
    
    # Get list of images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(config.INPUT_DIR.glob(ext)))
    
    if max_images is not None:
        image_paths = image_paths[:max_images]
    
    print(f"\n📊 Found {len(image_paths)} images to process")
    
    # Process images
    results = []
    failed_count = 0
    
    print("\n🔄 Processing images...")
    for image_path in tqdm(image_paths, desc="Processing"):
        result = process_single_image(
            image_path,
            model,
            transform,
            config,
            save_visualization=save_visualizations
        )
        
        if result is not None:
            # Save cropped image
            output_path = config.OUTPUT_DIR / f"{result['image_name']}.jpg"
            cropped_bgr = cv2.cvtColor(result['cropped_image'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), cropped_bgr)
            
            # Save metadata (without image array)
            metadata = {k: v for k, v in result.items() if k != 'cropped_image'}
            results.append(metadata)
        else:
            failed_count += 1
    
    # Calculate fallback statistics
    fallback_count = sum(1 for r in results if r.get('used_fallback', False))
    square_count = sum(1 for r in results if r.get('is_square', False))
    
    # Save processing report
    report = {
        'total_images': len(image_paths),
        'successful': len(results),
        'failed': failed_count,
        'fallback_used': fallback_count,
        'square_crops': square_count,
        'config': {
            'inference_size': config.INFERENCE_SIZE,
            'padding': config.PADDING,
            'mask_threshold': config.MASK_THRESHOLD,
            'min_contour_area': config.MIN_CONTOUR_AREA,
            'force_square_crop': config.FORCE_SQUARE_CROP,
            'center_crop_fallback': config.CENTER_CROP_FALLBACK,
            'center_crop_size': config.CENTER_CROP_SIZE,
        },
        'results': results
    }
    
    report_path = config.OUTPUT_DIR / 'processing_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETED")
    print("="*70)
    print(f"\n Summary:")
    print(f"   Total images: {len(image_paths)}")
    print(f"    Successful: {len(results)} ({len(results)/len(image_paths)*100:.1f}%)")
    print(f"    Failed: {failed_count} ({failed_count/len(image_paths)*100:.1f}%)")
    print(f"    Fallback used: {fallback_count} ({fallback_count/len(image_paths)*100:.1f}%)")
    print(f"    Square crops: {square_count} ({square_count/len(results)*100:.1f}% of successful)")
    print(f"\n Output:")
    print(f"   Cropped images: {config.OUTPUT_DIR}")
    print(f"   Report: {report_path}")
    
    if save_visualizations:
        print(f"   Debug visualizations: {config.OUTPUT_DIR / 'debug_visualizations'}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(
        description="Generate cropped lesion dataset using segmentation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images
  python generate_dataset.py

  # Process with debug visualizations
  python generate_dataset.py --visualize

  # Test on first 10 images only
  python generate_dataset.py --max-images 10 --visualize

  # Custom paths
  python generate_dataset.py --input-dir path/to/images --output-dir path/to/output
        """
    )
    
    parser.add_argument('--input-dir', type=str, 
                       default=str(Config.INPUT_DIR),
                       help='Input directory containing images')
    
    parser.add_argument('--output-dir', type=str,
                       default=str(Config.OUTPUT_DIR),
                       help='Output directory for cropped images')
    
    parser.add_argument('--model-path', type=str,
                       default=str(Config.MODEL_PATH),
                       help='Path to segmentation model checkpoint')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Save debug visualizations')
    
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process (for testing)')
    
    parser.add_argument('--padding', type=int, default=Config.PADDING,
                       help='Padding around bounding box (pixels)')
    
    args = parser.parse_args()
    
    # Update config from arguments
    config = Config()
    config.INPUT_DIR = Path(args.input_dir)
    config.OUTPUT_DIR = Path(args.output_dir)
    config.MODEL_PATH = Path(args.model_path)
    config.PADDING = args.padding
    
    # Validate paths
    if not config.INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {config.INPUT_DIR}")
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {config.MODEL_PATH}")
    
    # Run pipeline
    process_dataset(config, args.visualize, args.max_images)


if __name__ == "__main__":
    main()
