"""
Object Detection Head for multi-task learning
Predicts bounding boxes and class labels for objects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectDetectionHead(nn.Module):
    """
    Object Detection Head
    Takes features from backbone and predicts bounding boxes and class labels
    Implements a simplified Faster R-CNN style detector
    """

    def __init__(self, in_channels=2048, num_classes=7, anchor_scales=None, anchor_ratios=None):
        """
        Args:
            in_channels: Number of input channels from backbone (default 2048 for ResNet50)
            num_classes: Number of object classes (7 for KITTI)
            anchor_scales: List of anchor scales (default: [0.5, 1.0, 2.0])
            anchor_ratios: List of anchor aspect ratios (default: [0.5, 1.0, 2.0])
        """
        super(ObjectDetectionHead, self).__init__()

        if anchor_scales is None:
            anchor_scales = [0.5, 1.0, 2.0]
        if anchor_ratios is None:
            anchor_ratios = [0.5, 1.0, 2.0]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.num_anchors = len(anchor_scales) * len(anchor_ratios)

        # Feature refinement - reduce channels
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Region Proposal Network (RPN)
        self.rpn_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Classification head: predict objectness score for each anchor
        self.rpn_cls = nn.Conv2d(512, self.num_anchors, kernel_size=1)

        # Regression head: predict bbox deltas for each anchor
        self.rpn_reg = nn.Conv2d(512, self.num_anchors * 4, kernel_size=1)

        # Detection head for class prediction (after RPN proposals)
        self.detection_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.detection_fc2 = nn.Linear(1024, 1024)

        # Class prediction
        self.class_pred = nn.Linear(1024, num_classes + 1)  # +1 for background

        # Bounding box refinement
        self.bbox_pred = nn.Linear(1024, (num_classes + 1) * 4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features):
        """
        Forward pass for object detection head

        Args:
            features: Feature maps from backbone (B, 2048, H/32, W/32)

        Returns:
            dict containing:
                - rpn_cls: RPN classification scores (B, num_anchors, H/32, W/32)
                - rpn_reg: RPN bbox regressions (B, num_anchors*4, H/32, W/32)
                - cls_scores: Class prediction scores (B, num_classes+1)
                - bbox_deltas: Bbox refinement deltas (B, num_classes*4)
        """

        # Reduce feature channels
        reduced = self.reduce_conv(features)  # (B, 512, H/32, W/32)

        # RPN forward pass
        rpn_feat = self.rpn_conv(reduced)  # (B, 512, H/32, W/32)
        rpn_cls = self.rpn_cls(rpn_feat)   # (B, num_anchors, H/32, W/32)
        rpn_reg = self.rpn_reg(rpn_feat)   # (B, num_anchors*4, H/32, W/32)

        # For detection head, use global average pooling on RPN features
        # This simulates proposal pooling in a simplified manner
        pool_feat = F.adaptive_avg_pool2d(rpn_feat, (7, 7))  # (B, 512, 7, 7)
        pool_feat = pool_feat.view(pool_feat.size(0), -1)     # (B, 512*7*7)

        # Dense layers for final classification
        det_feat = self.relu(self.detection_fc1(pool_feat))
        det_feat = self.dropout(det_feat)
        det_feat = self.relu(self.detection_fc2(det_feat))
        det_feat = self.dropout(det_feat)

        # Final predictions
        cls_scores = self.class_pred(det_feat)   # (B, num_classes+1)
        bbox_deltas = self.bbox_pred(det_feat)   # (B, (num_classes+1)*4)

        return {
            'rpn_cls': rpn_cls,
            'rpn_reg': rpn_reg,
            'cls_scores': cls_scores,
            'bbox_deltas': bbox_deltas,
        }


class DetectionLoss(nn.Module):
    """
    Loss function for object detection
    Combines classification and bounding box regression losses
    """

    def __init__(self, num_classes=7):
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        # Use weight to handle class imbalance (background vs objects)
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1] + [1.0] * num_classes))
        self.bbox_loss_fn = nn.SmoothL1Loss(reduction='mean')

    def forward(self, predictions, targets):
        """
        Calculate detection loss

        Args:
            predictions: Dict from detection head
            targets: Dict with 'bboxes' and 'labels'

        Returns:
            Total loss (scalar)
        """
        # Placeholder loss calculation
        # In practice, need to:
        # 1. Assign anchors to ground truth
        # 2. Calculate RPN losses
        # 3. Extract proposals from RPN
        # 4. Pool features for proposals
        # 5. Calculate detection losses

        cls_scores = predictions['cls_scores']
        bbox_deltas = predictions['bbox_deltas']

        # For simplified training, we use labels if available
        if 'class_labels' in targets:
            cls_loss = self.cls_loss_fn(cls_scores, targets['class_labels'])
        else:
            cls_loss = torch.tensor(0.0, device=cls_scores.device)

        # Bbox loss (only for positive samples)
        if 'bbox_targets' in targets:
            bbox_loss = self.bbox_loss_fn(bbox_deltas, targets['bbox_targets'])
        else:
            bbox_loss = torch.tensor(0.0, device=bbox_deltas.device)

        return cls_loss + bbox_loss


if __name__ == "__main__":
    # Test detection head
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head = ObjectDetectionHead(in_channels=2048, num_classes=7).to(device)

    # Test with feature maps from ResNet50
    features = torch.randn(2, 2048, 12, 39).to(device)  # Approx. KITTI size / 32

    outputs = head(features)

    print("Object Detection Head Test")
    print(f"Device: {device}")
    print(f"Input features: {features.shape}")
    print(f"\nOutputs:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")

    print("\nDetection head test passed!")
