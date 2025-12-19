"""
Unified Multi-Task Learning Model
Combines backbone and all three task-specific heads
"""

import torch
import torch.nn as nn
from .backbone import get_backbone
from .detection_head import ObjectDetectionHead, DetectionLoss
from .lane_head import LaneDetectionHead, LaneLoss
from .classification_head import ClassificationHead, ClassificationHeadAdvanced, ClassificationLoss


class MultiTaskModel(nn.Module):
    """
    Multi-Task Learning Model for Autonomous Driving
    Performs simultaneous:
    - Object Detection
    - Lane Detection (Segmentation)
    - Traffic Sign Classification
    """

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        freeze_backbone=False,
        num_detection_classes=7,
        num_lane_classes=2,
        num_classification_classes=43,
        use_advanced_classification=False,
    ):
        """
        Args:
            backbone_name: Name of backbone ("resnet50" or "resnet101")
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights
            num_detection_classes: Number of object detection classes
            num_lane_classes: Number of lane segmentation classes (usually 2)
            num_classification_classes: Number of traffic sign classes
            use_advanced_classification: Whether to use attention-based classification head
        """
        super(MultiTaskModel, self).__init__()

        # Get shared backbone
        self.backbone = get_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )

        in_channels = self.backbone.get_out_channels()

        # Task-specific heads
        self.detection_head = ObjectDetectionHead(
            in_channels=in_channels,
            num_classes=num_detection_classes,
        )

        self.lane_head = LaneDetectionHead(
            in_channels=in_channels,
            num_classes=num_lane_classes,
        )

        if use_advanced_classification:
            self.classification_head = ClassificationHeadAdvanced(
                in_channels=in_channels,
                num_classes=num_classification_classes,
            )
        else:
            self.classification_head = ClassificationHead(
                in_channels=in_channels,
                num_classes=num_classification_classes,
            )

        # Loss functions
        self.detection_loss_fn = DetectionLoss(num_classes=num_detection_classes)
        self.lane_loss_fn = LaneLoss(num_classes=num_lane_classes)
        self.classification_loss_fn = ClassificationLoss(
            num_classes=num_classification_classes,
            use_label_smoothing=True,
        )

        # Task weights for multi-task learning
        self.detection_weight = 1.0
        self.lane_weight = 1.0
        self.classification_weight = 1.0

    def forward(self, x, task=None):
        """
        Forward pass

        Args:
            x: Input image tensor (B, 3, H, W)
            task: Optional specific task to run ('detection', 'lane', 'classification')
                  If None, runs all tasks

        Returns:
            If task is None: dict with all predictions
            If task is specified: dict with that task's predictions
        """

        # Shared backbone
        features = self.backbone(x)

        if task is None:
            # Run all tasks
            return {
                'features': features,
                'detection': self.detection_head(features),
                'lane': self.lane_head(features),
                'classification': self.classification_head(features),
            }
        elif task == 'detection':
            return {'detection': self.detection_head(features)}
        elif task == 'lane':
            return {'lane': self.lane_head(features)}
        elif task == 'classification':
            return {'classification': self.classification_head(features)}
        else:
            raise ValueError(f"Unknown task: {task}")

    def compute_loss(self, predictions, targets):
        """
        Compute weighted multi-task loss

        Args:
            predictions: Output from forward pass
            targets: Dict with targets for each task

        Returns:
            dict with individual losses and total loss
        """

        losses = {}

        # Detection loss
        if 'detection' in predictions and 'detection' in targets:
            detection_loss = self.detection_loss_fn(predictions['detection'], targets['detection'])
            losses['detection'] = detection_loss
        else:
            losses['detection'] = torch.tensor(0.0)

        # Lane loss
        if 'lane' in predictions and 'lane' in targets:
            lane_loss = self.lane_loss_fn(predictions['lane'], targets['lane'])
            losses['lane'] = lane_loss
        else:
            losses['lane'] = torch.tensor(0.0)

        # Classification loss
        if 'classification' in predictions and 'classification' in targets:
            classification_loss = self.classification_loss_fn(
                predictions['classification'],
                targets['classification'],
            )
            losses['classification'] = classification_loss
        else:
            losses['classification'] = torch.tensor(0.0)

        # Weighted total loss
        total_loss = (
            self.detection_weight * losses['detection']
            + self.lane_weight * losses['lane']
            + self.classification_weight * losses['classification']
        )

        losses['total'] = total_loss

        return losses

    def set_task_weights(self, detection_weight=1.0, lane_weight=1.0, classification_weight=1.0):
        """
        Set weights for multi-task loss weighting

        Args:
            detection_weight: Weight for detection loss
            lane_weight: Weight for lane detection loss
            classification_weight: Weight for classification loss
        """
        self.detection_weight = detection_weight
        self.lane_weight = lane_weight
        self.classification_weight = classification_weight

    def freeze_backbone(self):
        """Freeze backbone weights"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone weights"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_multi_task_model(config):
    """
    Factory function to create model from config

    Args:
        config: Configuration object with model settings

    Returns:
        MultiTaskModel instance
    """

    model = MultiTaskModel(
        backbone_name=config.model.backbone,
        pretrained=config.model.pretrained,
        freeze_backbone=config.model.freeze_backbone,
        num_detection_classes=config.model.num_detection_classes,
        num_lane_classes=config.model.num_lane_classes,
        num_classification_classes=config.model.num_classification_classes,
        use_advanced_classification=False,
    )

    return model


if __name__ == "__main__":
    # Test multi-task model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskModel(
        backbone_name="resnet50",
        pretrained=False,
        freeze_backbone=False,
        num_detection_classes=7,
        num_lane_classes=2,
        num_classification_classes=43,
    ).to(device)

    print("Multi-Task Model Test")
    print(f"Device: {device}")
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}\n")

    # Test with different input sizes
    test_cases = [
        ("KITTI (detection)", torch.randn(2, 3, 375, 1242).to(device)),
        ("GTSRB (classification)", torch.randn(4, 3, 64, 64).to(device)),
        ("Lane Detection", torch.randn(2, 3, 720, 1280).to(device)),
    ]

    for name, x in test_cases:
        print(f"Testing {name}:")
        print(f"  Input shape: {x.shape}")

        predictions = model(x)

        print(f"  Outputs:")
        for key, val in predictions.items():
            if key == 'features':
                print(f"    {key}: {val.shape}")
            elif isinstance(val, dict):
                print(f"    {key}:")
                for sub_key, sub_val in val.items():
                    print(f"      {sub_key}: {sub_val.shape}")
            else:
                print(f"    {key}: {val.shape}")
        print()

    # Test single task
    print("Testing single task (classification only):")
    x = torch.randn(4, 3, 64, 64).to(device)
    predictions = model(x, task='classification')
    print(f"  Input: {x.shape}")
    print(f"  Output: {predictions['classification'].shape}")
    print()

    # Test loss computation
    print("Testing loss computation:")
    predictions = model(torch.randn(2, 3, 64, 64).to(device))

    targets = {
        'classification': torch.randint(0, 43, (2,)).to(device),
    }

    losses = model.compute_loss(predictions, targets)
    print(f"  Detection loss: {losses['detection']:.4f}")
    print(f"  Lane loss: {losses['lane']:.4f}")
    print(f"  Classification loss: {losses['classification']:.4f}")
    print(f"  Total loss: {losses['total']:.4f}")

    print("\nMulti-task model test passed!")
