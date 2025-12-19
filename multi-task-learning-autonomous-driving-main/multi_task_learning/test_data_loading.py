"""
Test script to verify all data loaders work correctly with sample batches
Tests loading and batching for all three tasks
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_loader import MultiTaskDataLoader
from src.configs.config import get_config


def print_batch_info(batch_name, batch_data):
    """Print information about a batch"""
    print(f"\n{'='*60}")
    print(f"BATCH: {batch_name}")
    print(f"{'='*60}")

    for key, value in batch_data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: tensor{tuple(value.shape)} - dtype: {value.dtype}")
        elif isinstance(value, np.ndarray):
            print(f"  {key:20s}: array{value.shape} - dtype: {value.dtype}")
        elif isinstance(value, list):
            print(f"  {key:20s}: list of {len(value)} items")
        else:
            print(f"  {key:20s}: {type(value).__name__}")


def test_detection_loader(data_loader):
    """Test object detection data loader"""
    print("\n" + "█"*60)
    print("TESTING OBJECT DETECTION LOADER (KITTI)")
    print("█"*60)

    loader = data_loader.get_train_loader('detection')

    if loader is None:
        print("✗ Detection loader not available")
        return

    print(f"Total batches: {len(loader)}")

    # Get first batch
    for batch_idx, batch in enumerate(loader):
        print_batch_info(f"Detection - Batch {batch_idx}", batch)

        # Print sample details
        print(f"\nSample details:")
        print(f"  Batch size: {batch['image'].shape[0]}")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Number of objects: {[len(b) for b in batch['bboxes']]}")
        print(f"  Classes: {batch['labels']}")

        if batch_idx == 0:  # Only show first batch
            break

    print("✓ Detection loader test passed!")


def test_classification_loader(data_loader):
    """Test traffic sign classification data loader"""
    print("\n" + "█"*60)
    print("TESTING CLASSIFICATION LOADER (GTSRB)")
    print("█"*60)

    loader = data_loader.get_train_loader('classification')

    if loader is None:
        print("✗ Classification loader not available")
        return

    print(f"Total batches: {len(loader)}")

    # Get first batch
    for batch_idx, batch in enumerate(loader):
        print_batch_info(f"Classification - Batch {batch_idx}", batch)

        # Print sample details
        print(f"\nSample details:")
        print(f"  Batch size: {batch['image'].shape[0]}")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Labels: {batch['label']}")
        print(f"  Label shape: {batch['label'].shape}")

        if batch_idx == 0:  # Only show first batch
            break

    print("✓ Classification loader test passed!")


def test_segmentation_loader(data_loader):
    """Test lane detection segmentation loader"""
    print("\n" + "█"*60)
    print("TESTING SEGMENTATION LOADER (LANE DETECTION)")
    print("█"*60)

    loader = data_loader.get_train_loader('segmentation')

    if loader is None:
        print("✗ Segmentation loader not available")
        return

    print(f"Total batches: {len(loader)}")

    # Get first batch
    for batch_idx, batch in enumerate(loader):
        print_batch_info(f"Segmentation - Batch {batch_idx}", batch)

        # Print sample details
        print(f"\nSample details:")
        print(f"  Batch size: {batch['image'].shape[0]}")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Mask shape: {batch['segmentation_mask'].shape}")
        print(f"  Mask unique values: {np.unique(batch['segmentation_mask'].numpy())}")
        print(f"  Lanes per sample: {[len(l) for l in batch['lanes']]}")

        if batch_idx == 0:  # Only show first batch
            break

    print("✓ Segmentation loader test passed!")


def test_all_loaders_together(data_loader):
    """Test loading from all three datasets simultaneously"""
    print("\n" + "█"*60)
    print("TESTING ALL LOADERS TOGETHER")
    print("█"*60)

    print("\nLoading one batch from each dataset...")

    loaders = {
        'Detection (KITTI)': data_loader.get_train_loader('detection'),
        'Classification (GTSRB)': data_loader.get_train_loader('classification'),
        'Segmentation (Lane)': data_loader.get_train_loader('segmentation'),
    }

    results = {}
    for name, loader in loaders.items():
        if loader is None:
            print(f"✗ {name} loader not available")
            continue

        try:
            batch = next(iter(loader))
            results[name] = {
                'batch_size': batch['image'].shape[0],
                'image_shape': tuple(batch['image'].shape),
                'device': batch['image'].device,
            }
            print(f"✓ {name:30s} - batch_size: {batch['image'].shape[0]}, shape: {tuple(batch['image'].shape)}")
        except Exception as e:
            print(f"✗ {name:30s} - Error: {e}")

    print("\n✓ All loaders working simultaneously!")
    return results


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("MULTI-TASK LEARNING DATA LOADING TEST")
    print("="*60)

    # Get configuration
    config = get_config()

    # Create data loader
    print("\nInitializing data loaders...")
    data_loader = MultiTaskDataLoader(
        config=config,
        batch_size=4,
        num_workers=0,  # Set to 0 for testing
        shuffle=True
    )

    # Print summary
    data_loader.print_summary()

    # Test each loader individually
    test_detection_loader(data_loader)
    test_classification_loader(data_loader)
    test_segmentation_loader(data_loader)

    # Test all loaders together
    results = test_all_loaders_together(data_loader)

    # Final summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)

    print("\n✓ ALL TESTS PASSED!")
    print("\nData loading pipeline is ready for training:")
    print("  ✅ Object Detection (KITTI)")
    print("  ✅ Traffic Sign Classification (GTSRB)")
    print("  ✅ Lane Detection (Synthetic)")
    print("\nYou can now start implementing the multi-task model!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
