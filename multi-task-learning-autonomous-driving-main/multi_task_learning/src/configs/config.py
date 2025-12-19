"""
Configuration file for Multi-Task Learning project
Autonomous Driving: Object Detection, Lane Detection, Traffic Sign Classification
"""

from pathlib import Path
import torch

# ==================== PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = Path("/Users/shreyasreeburugadda/Desktop/Jaya/datasets")

# Dataset paths
KITTI_ROOT = DATA_ROOT / "kitti"
GTSRB_ROOT = DATA_ROOT / "gtsrb"
LANE_ROOT = DATA_ROOT / "tusimple/synthetic_lanes"

# Output paths
CHECKPOINT_DIR = PROJECT_ROOT / "results/checkpoints"
LOG_DIR = PROJECT_ROOT / "results/logs"

# Create directories if they don't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==================== DEVICE ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==================== DATASET CONFIGURATION ====================

# KITTI Configuration
class KITTIConfig:
    """KITTI Object Detection Dataset Configuration"""
    train_img_dir = KITTI_ROOT / "data_object_image_2/training/image_2"
    train_label_dir = KITTI_ROOT / "training/label_2"
    train_calib_dir = KITTI_ROOT / "data_object_calib/training/calib"

    test_img_dir = KITTI_ROOT / "data_object_image_2/testing/image_2"
    test_calib_dir = KITTI_ROOT / "data_object_calib/testing/calib"

    # Image properties
    img_height = 375
    img_width = 1242

    # Classes for object detection
    classes = ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Tram', 'Misc']
    num_classes = len(classes)

    # Data split
    train_split = 0.8  # 80% train, 20% validation

    # Minimum object size (filter very small objects)
    min_width = 20
    min_height = 20


# GTSRB Configuration
class GTSRBConfig:
    """GTSRB Traffic Sign Classification Dataset Configuration"""
    train_dir = GTSRB_ROOT / "GTSRB/Final_Training/Images"
    test_dir = GTSRB_ROOT / "GTSRB/Final_Test/Images"

    # Image properties (variable size, typically 32x32 to 64x64)
    resize_height = 64
    resize_width = 64

    # 43 traffic sign classes
    num_classes = 43

    # Data split
    train_split = 0.9  # 90% train, 10% validation


# Lane Detection Configuration
class LaneConfig:
    """Lane Detection Dataset Configuration"""
    train_img_dir = LANE_ROOT / "images"
    train_label_dir = LANE_ROOT / "labels"

    # Image properties
    img_height = 720
    img_width = 1280

    # Lane detection uses semantic segmentation
    num_lanes = 4  # Usually 4 lanes

    # Data split
    train_split = 0.8  # 80% train, 20% validation


# ==================== TRAINING CONFIGURATION ====================

class TrainingConfig:
    """Training hyperparameters"""
    # Basic settings
    batch_size = 4
    num_workers = 4
    epochs = 5

    # Learning rate
    learning_rate = 1e-4
    weight_decay = 1e-5

    # Optimizer
    optimizer = "adam"  # Options: adam, sgd

    # Scheduler
    use_scheduler = True
    scheduler_type = "cosine"  # Options: cosine, step, exponential

    # Mixed precision training
    use_amp = False  # Automatic Mixed Precision

    # Gradient clipping
    clip_grad_norm = 1.0

    # Validation frequency
    val_frequency = 5  # Validate every N epochs

    # Checkpoint frequency
    checkpoint_frequency = 10  # Save checkpoint every N epochs

    # Early stopping
    use_early_stopping = False
    patience = 15


# ==================== MODEL CONFIGURATION ====================

class ModelConfig:
    """Model architecture configuration"""

    # Backbone
    backbone = "resnet50"  # Options: resnet50, resnet101, vgg16, efficientnet
    pretrained = True
    freeze_backbone = False

    # Task-specific configurations

    # Object Detection Head
    detection_head = "faster_rcnn"  # Options: faster_rcnn, yolo
    num_detection_classes = KITTIConfig.num_classes

    # Lane Detection Head
    lane_head = "unet"  # Options: unet, deeplabv3, fcn
    num_lane_classes = 2  # Binary: lane or not lane

    # Traffic Sign Classification Head
    classification_head = "cnn"  # Options: cnn, densenet
    num_classification_classes = GTSRBConfig.num_classes

    # Multi-task learning weights
    # Start with equal weights, can be tuned based on performance
    detection_weight = 1.0
    lane_weight = 1.0
    classification_weight = 1.0


# ==================== DATA AUGMENTATION ====================

class AugmentationConfig:
    """Data augmentation configuration"""

    # Common augmentations
    use_horizontal_flip = True
    use_vertical_flip = False
    use_rotation = True
    use_brightness = True
    use_contrast = True
    use_gaussian_noise = True

    # Augmentation parameters
    horizontal_flip_prob = 0.5
    rotation_angle = 15  # degrees
    brightness_factor = 0.2  # 20% brightness variation
    contrast_factor = 0.2  # 20% contrast variation
    gaussian_noise_std = 0.01  # Standard deviation


# ==================== LOGGING ====================

class LoggingConfig:
    """Logging configuration"""
    log_level = "INFO"
    log_frequency = 100  # Log every N batches
    tensorboard = True
    wandb = False  # Weights & Biases


# ==================== INFERENCE ====================

class InferenceConfig:
    """Inference configuration"""
    confidence_threshold = 0.5
    nms_threshold = 0.5  # Non-maximum suppression threshold

    # Lane detection inference
    lane_threshold = 0.5

    # Classification inference
    use_ensemble = False


# ==================== UTILITY FUNCTIONS ====================

def get_config():
    """Get all configurations"""
    return {
        'project_root': PROJECT_ROOT,
        'data_root': DATA_ROOT,
        'kitti': KITTIConfig,
        'gtsrb': GTSRBConfig,
        'lane': LaneConfig,
        'training': TrainingConfig,
        'model': ModelConfig,
        'augmentation': AugmentationConfig,
        'logging': LoggingConfig,
        'inference': InferenceConfig,
        'device': DEVICE,
    }


def print_config():
    """Print all configuration values"""
    config = get_config()

    print("\n" + "="*80)
    print("PROJECT CONFIGURATION")
    print("="*80)

    print(f"\nPaths:")
    print(f"  Project Root: {config['project_root']}")
    print(f"  Data Root: {config['data_root']}")
    print(f"  Device: {config['device']}")

    print(f"\nDataset Paths:")
    print(f"  KITTI: {KITTI_ROOT}")
    print(f"  GTSRB: {GTSRB_ROOT}")
    print(f"  Lane Detection: {LANE_ROOT}")

    print(f"\nKITTI Configuration:")
    print(f"  Image size: {KITTIConfig.img_width}x{KITTIConfig.img_height}")
    print(f"  Classes: {KITTIConfig.num_classes}")
    print(f"  Train images: ~{len(list(KITTIConfig.train_img_dir.glob('*.png')))}")

    print(f"\nGTSRB Configuration:")
    print(f"  Image size: {GTSRBConfig.resize_width}x{GTSRBConfig.resize_height}")
    print(f"  Classes: {GTSRBConfig.num_classes}")

    print(f"\nLane Detection Configuration:")
    print(f"  Image size: {LaneConfig.img_width}x{LaneConfig.img_height}")
    print(f"  Train images: ~{len(list(LaneConfig.train_img_dir.glob('*.png')))}")

    print(f"\nTraining Configuration:")
    print(f"  Batch size: {TrainingConfig.batch_size}")
    print(f"  Learning rate: {TrainingConfig.learning_rate}")
    print(f"  Epochs: {TrainingConfig.epochs}")

    print("="*80 + "\n")


if __name__ == "__main__":
    print_config()
