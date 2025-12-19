# Multi-Task Learning Model Architecture

## Overview

This document describes the unified multi-task learning model for autonomous driving that simultaneously performs:
- **Object Detection** (KITTI dataset)
- **Lane Detection** (TuSimple synthetic lanes)
- **Traffic Sign Classification** (GTSRB dataset)

## Model Components

### 1. Shared Backbone: ResNet50

**File:** `src/models/backbone.py`

The shared feature extractor for all three tasks:

```
Input: (B, 3, H, W)
  ↓
ResNet50 (without FC layers)
  - Conv1: 7×7 kernel, stride 2
  - Layer1-4: Residual blocks
  - Output: (B, 2048, H/32, W/32)
```

**Key Features:**
- Pre-trained on ImageNet (can be frozen for faster training)
- Output: 2048-channel feature maps at 1/32 input resolution
- Reduced computational overhead by sharing backbone across tasks
- Flexible: Can use ResNet50 or ResNet101

**Parameters:** ~23.5M (ResNet50)

**Usage:**
```python
from src.models import get_backbone

backbone = get_backbone("resnet50", pretrained=True, freeze=False)
features = backbone(images)  # (B, 2048, H/32, W/32)
```

---

### 2. Object Detection Head

**File:** `src/models/detection_head.py`

Detects vehicles, pedestrians, and other objects with bounding boxes and class labels.

```
Features (B, 2048, H/32, W/32)
  ↓
Reduce Conv (3×3): 2048 → 512 channels
  ↓
RPN Branch                    Detection Branch
  ├─ RPN Conv 3×3             ├─ Avg Pool (7×7)
  ├─ RPN Class Head           ├─ FC 512 → 1024 (ReLU)
  │  Output: (B, num_anchors, H/32, W/32)  ├─ FC 1024 → 1024 (ReLU)
  │                           ├─ Class Pred: → (B, 8)
  └─ RPN Bbox Head            └─ Bbox Pred: → (B, 32)
     Output: (B, num_anchors*4, H/32, W/32)
```

**Architecture Details:**
- **Anchors:** 3 scales × 3 aspect ratios = 9 anchors per spatial location
- **Input:** Feature maps from backbone
- **Output:**
  - RPN Classification: Objectness scores for anchors
  - RPN Bbox Regression: Box coordinate adjustments
  - Class Predictions: Object class probabilities (7 classes + background)
  - Bbox Predictions: Final bounding box coordinates

**Classes:** 7
- Car, Pedestrian, Cyclist, Van, Truck, Tram, Miscellaneous

**Loss Function:**
- Cross-entropy loss for classification
- Smooth L1 loss for bounding box regression

**Parameters:** ~1.2M

---

### 3. Lane Detection Head

**File:** `src/models/lane_head.py`

Performs semantic segmentation to detect lane markings.

```
Features (B, 2048, H/32, W/32)
  ↓
Initial Conv: 2048 → 512
  ↓
Decoder Blocks (5× upsampling by 2)
  ├─ Decoder1: 512 → 256 (H/16, W/16)
  ├─ Decoder2: 256 → 128 (H/8, W/8)
  ├─ Decoder3: 128 → 64 (H/4, W/4)
  ├─ Decoder4: 64 → 32 (H/2, W/2)
  └─ Decoder5: 32 → 16 (H, W)
  ↓
Final Conv: 16 → 16
  ↓
Segmentation Head: 16 → 2 (binary: lane/not-lane)
  ↓
Output: (B, 2, H, W)
```

**Architecture Details:**
- **Type:** U-Net style decoder
- **Upsampling:** Bilinear interpolation with refinement convolutions
- **Each Decoder Block:**
  - Upsample by 2× (bilinear)
  - 3×3 convolution
  - Batch norm + ReLU
  - 3×3 convolution
  - Batch norm + ReLU

**Output:** Binary segmentation maps (lane vs. non-lane)

**Loss Function:**
- Cross-entropy loss with class weighting (2× weight for lane class)
- Optional Dice loss for handling class imbalance
- Formula: `(1 - dice_weight) * CE + dice_weight * Dice`

**Parameters:** ~850K

---

### 4. Traffic Sign Classification Head

**File:** `src/models/classification_head.py`

Classifies traffic signs into 43 categories.

**Standard Version:**
```
Features (B, 2048, H/32, W/32)
  ↓
Global Average Pooling → (B, 2048)
  ↓
FC Layer: 2048 → 1024 (BatchNorm + ReLU + Dropout)
  ↓
FC Layer: 1024 → 512 (BatchNorm + ReLU + Dropout)
  ↓
Classification Layer: 512 → 43
  ↓
Output: (B, 43)
```

**Advanced Version (with Channel Attention):**
```
Features (B, 2048, H/32, W/32)
  ├─ Global Avg Pool → (B, 2048)
  │  ↓
  │  Attention FC: 2048 → 128 → 2048
  │  ↓
  │  Sigmoid
  │
  └─ Global Max Pool → (B, 2048)
     ↓
     Attention FC: 2048 → 128 → 2048
     ↓
     Sigmoid
  ↓
Combined Attention × Pooled Features
  ↓
FC Layers (1024, 512)
  ↓
Classification: 512 → 43
```

**Classes:** 43 traffic sign categories
- Speed limits, stop signs, yield signs, warning signs, etc.

**Loss Function:**
- Cross-entropy loss with label smoothing (ε=0.1)
- Optional focal loss for hard examples

**Parameters:**
- Standard: ~2.0M
- Advanced: ~2.1M

---

## Unified Multi-Task Model

**File:** `src/models/multi_task_model.py`

Integrates all components into a single model.

```python
class MultiTaskModel(nn.Module):
    def __init__(self, ...):
        self.backbone = ResNet50Backbone(...)
        self.detection_head = ObjectDetectionHead(...)
        self.lane_head = LaneDetectionHead(...)
        self.classification_head = ClassificationHead(...)

    def forward(self, x, task=None):
        features = self.backbone(x)

        return {
            'detection': self.detection_head(features),
            'lane': self.lane_head(features),
            'classification': self.classification_head(features),
        }
```

**Key Features:**
- Shared backbone reduces computation compared to 3 separate models
- Each head can be trained/evaluated independently
- Task-specific weighting for multi-task loss
- Optional uncertainty-based loss weighting

**Model Size:**
- Total parameters: ~70.2M
- Trainable parameters: ~70.2M (if backbone not frozen)

### Forward Pass

#### Single Task Mode
```python
model = MultiTaskModel(...)
predictions = model(images, task='classification')
# Returns: {'classification': logits}
```

#### Multi-Task Mode (Default)
```python
predictions = model(images)
# Returns: {
#     'features': backbone_features,
#     'detection': detection_outputs,
#     'lane': lane_segmentation_maps,
#     'classification': class_logits,
# }
```

---

## Loss Functions

**File:** `src/losses/multi_task_loss.py`

### Multi-Task Loss

Combines losses from all three tasks with weighting:

```
L_total = w_det * L_det + w_lane * L_lane + w_cls * L_cls
```

**Modes:**
1. **Fixed Weights:** Manually specified weights
2. **Uncertainty Weighting:** Learnable task weights (learns task importance)

### Task-Specific Losses

1. **Detection Loss:**
   - Classification: Cross-entropy loss
   - Bbox Regression: Smooth L1 loss

2. **Lane Detection Loss:**
   - Primary: Cross-entropy loss (with class weighting)
   - Optional: Dice loss (for imbalance)

3. **Classification Loss:**
   - Primary: Cross-entropy loss (with label smoothing)
   - Optional: Focal loss (for hard examples)

---

## Input/Output Specifications

### Input
- **Shape:** (B, 3, H, W)
- **Format:** RGB image tensors normalized to [0, 1]
- **Supported Sizes:**
  - KITTI: (B, 3, 375, 1242)
  - GTSRB: (B, 3, 64, 64)
  - Lane: (B, 3, 720, 1280)

### Output

**Detection Head:**
```python
{
    'rpn_cls': (B, 9, H/32, W/32),          # RPN objectness
    'rpn_reg': (B, 36, H/32, W/32),         # RPN bbox deltas
    'cls_scores': (B, 8),                   # Class predictions
    'bbox_deltas': (B, 32),                 # Final bbox deltas
}
```

**Lane Head:**
```python
(B, 2, H, W)  # Binary segmentation map
```

**Classification Head:**
```python
(B, 43)  # Class logits
```

---

## Training Configuration

**File:** `src/configs/config.py`

### Default Hyperparameters

```python
batch_size = 4
learning_rate = 1e-4
weight_decay = 1e-5
epochs = 50
optimizer = "adam"
scheduler = "cosine"
gradient_clip = 1.0

# Task weights
detection_weight = 1.0
lane_weight = 1.0
classification_weight = 1.0
```

### Data Splits

| Task | Train | Val | Total |
|------|-------|-----|-------|
| Detection | 5,984 | 1,497 | 7,481 |
| Lane | 400 | 100 | 500 |
| Classification | 35,288 | 3,921 | 39,209 |

---

## Performance Metrics

### Classification
- Accuracy
- Top-5 Accuracy
- Precision / Recall / F1

### Detection
- mAP (mean Average Precision)
- Precision / Recall
- IoU (Intersection over Union)

### Lane Detection
- Pixel Accuracy
- IoU
- Dice Coefficient

---

## Memory Footprint

| Component | Parameters | Size (MB) |
|-----------|-----------|-----------|
| Backbone | 23.5M | 90 |
| Detection Head | 1.2M | 5 |
| Lane Head | 850K | 3 |
| Classification Head | 2.0M | 8 |
| **Total** | **27.5M** | **106** |

---

## Design Decisions

1. **Shared Backbone:** Reduces redundancy and computation
   - Alternative: 3 separate models (3× backbone parameters)

2. **ResNet50:** Good balance of accuracy and speed
   - Alternative: ResNet101 (more accurate but slower)

3. **U-Net Decoder:** Efficient for dense prediction tasks
   - Alternative: DeepLabv3, FCN (more parameters)

4. **Feature Sharing:** All heads use same backbone features
   - Alternative: Task-specific feature extraction (more parameters)

5. **Task Weighting:** Flexible, tunable loss weights
   - Allows balancing task importance during training

---

## Usage Examples

### Basic Training
```python
from src.models import MultiTaskModel
from src.losses import MultiTaskLoss

model = MultiTaskModel(backbone_name="resnet50", pretrained=True)
model = model.to(device)

# Forward pass
predictions = model(images)

# Compute losses
detection_loss = detection_loss_fn(predictions['detection'], targets)
lane_loss = lane_loss_fn(predictions['lane'], targets)
classification_loss = classification_loss_fn(predictions['classification'], targets)

# Combined loss
losses = {
    'detection': detection_loss,
    'lane': lane_loss,
    'classification': classification_loss,
}
total_loss = multi_task_loss(losses)
```

### Single Task Inference
```python
model.eval()
with torch.no_grad():
    # Classification only
    predictions = model(images, task='classification')
    logits = predictions['classification']

    # Get class predictions
    class_indices = torch.argmax(logits, dim=1)
```

### Model Analysis
```python
# Print model structure
print(model)

# Count parameters
total_params = model.get_total_params()
trainable_params = model.get_trainable_params()

# Freeze backbone for faster training
model.freeze_backbone()
```

---

## Future Improvements

1. **Attention Mechanisms:**
   - Cross-task attention
   - Spatial attention for detection

2. **Advanced Architectures:**
   - EfficientNet backbone
   - Transformer-based heads

3. **Multi-Scale Processing:**
   - Feature pyramid for detection
   - Multi-scale decoder for segmentation

4. **Auxiliary Heads:**
   - Depth estimation
   - Optical flow
   - Instance segmentation

5. **Loss Functions:**
   - Uncertainty weighting (learnable task weights)
   - Gradient normalization for balanced task learning

---

## References

- ResNet: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
- Faster R-CNN: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection" (NIPS 2015)
- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
- Multi-Task Learning: Ruder et al., "An Overview of Multi-Task Learning in Deep Neural Networks"
