"""
Multi-task loss functions for autonomous driving
Combines losses from detection, lane detection, and classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Weighted combination of all task losses for multi-task learning
    """

    def __init__(
        self,
        detection_weight=1.0,
        lane_weight=1.0,
        classification_weight=1.0,
        use_uncertainty=False,
    ):
        """
        Args:
            detection_weight: Weight for object detection loss
            lane_weight: Weight for lane detection loss
            classification_weight: Weight for traffic sign classification loss
            use_uncertainty: Use uncertainty weighting (learnable task weights)
        """
        super(MultiTaskLoss, self).__init__()

        self.detection_weight = detection_weight
        self.lane_weight = lane_weight
        self.classification_weight = classification_weight
        self.use_uncertainty = use_uncertainty

        if use_uncertainty:
            # Learnable task weights (log precision)
            self.log_det_var = nn.Parameter(torch.tensor(0.0))
            self.log_lane_var = nn.Parameter(torch.tensor(0.0))
            self.log_cls_var = nn.Parameter(torch.tensor(0.0))

    def forward(self, losses_dict):
        """
        Compute weighted multi-task loss

        Args:
            losses_dict: Dictionary with keys:
                - 'detection': detection loss (scalar tensor)
                - 'lane': lane detection loss (scalar tensor)
                - 'classification': classification loss (scalar tensor)

        Returns:
            Weighted total loss (scalar tensor)
        """

        if self.use_uncertainty:
            # Uncertainty-based weighting
            # Lower loss means higher precision (smaller variance)
            det_loss = losses_dict.get('detection', torch.tensor(0.0))
            lane_loss = losses_dict.get('lane', torch.tensor(0.0))
            cls_loss = losses_dict.get('classification', torch.tensor(0.0))

            # Weighted loss with learned precisions
            # Formula: sum((1 / (2 * sigma_i^2)) * L_i + 0.5 * log(sigma_i^2))
            det_weighted = (1 / (2 * torch.exp(self.log_det_var))) * det_loss + 0.5 * self.log_det_var
            lane_weighted = (1 / (2 * torch.exp(self.log_lane_var))) * lane_loss + 0.5 * self.log_lane_var
            cls_weighted = (1 / (2 * torch.exp(self.log_cls_var))) * cls_loss + 0.5 * self.log_cls_var

            total_loss = det_weighted + lane_weighted + cls_weighted

        else:
            # Fixed weight combination
            det_loss = losses_dict.get('detection', torch.tensor(0.0))
            lane_loss = losses_dict.get('lane', torch.tensor(0.0))
            cls_loss = losses_dict.get('classification', torch.tensor(0.0))

            total_loss = (
                self.detection_weight * det_loss
                + self.lane_weight * lane_loss
                + self.classification_weight * cls_loss
            )

        return total_loss

    def set_weights(self, detection_weight=None, lane_weight=None, classification_weight=None):
        """Update loss weights"""
        if detection_weight is not None:
            self.detection_weight = detection_weight
        if lane_weight is not None:
            self.lane_weight = lane_weight
        if classification_weight is not None:
            self.classification_weight = classification_weight


class DetectionLossFull(nn.Module):
    """
    Complete object detection loss
    Combines RPN loss and detection loss
    """

    def __init__(self, num_classes=7, positive_iou_threshold=0.5, negative_iou_threshold=0.3):
        super(DetectionLossFull, self).__init__()

        self.num_classes = num_classes
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold

        # RPN losses
        self.rpn_cls_loss = nn.CrossEntropyLoss()
        self.rpn_bbox_loss = nn.SmoothL1Loss(reduction='mean')

        # Detection losses
        self.det_cls_loss = nn.CrossEntropyLoss()
        self.det_bbox_loss = nn.SmoothL1Loss(reduction='mean')

    def compute_rpn_targets(self, anchors, gt_bboxes, gt_labels):
        """
        Compute RPN targets for anchor classification and regression

        Args:
            anchors: Generated anchors (N_anchors, 4)
            gt_bboxes: Ground truth bounding boxes (N_gt, 4)
            gt_labels: Ground truth labels (N_gt,)

        Returns:
            rpn_cls_targets: Binary labels for each anchor (N_anchors,)
            rpn_bbox_targets: Bbox regression targets (N_anchors, 4)
        """

        # Simplified: assign anchors to GT boxes based on IoU
        # Positive: IoU > positive_threshold
        # Negative: IoU < negative_threshold
        # Ignore: in between

        N_anchors = anchors.shape[0]
        N_gt = gt_bboxes.shape[0]

        # Compute IoU between all anchors and GT boxes
        ious = self._compute_iou_matrix(anchors, gt_bboxes)  # (N_anchors, N_gt)

        # Get max IoU for each anchor
        max_iou_per_anchor = ious.max(dim=1)[0]  # (N_anchors,)
        argmax_iou_per_anchor = ious.argmax(dim=1)  # (N_anchors,)

        # RPN classification targets (0: negative, 1: positive, -1: ignore)
        rpn_cls_targets = torch.full((N_anchors,), -1, dtype=torch.long)
        rpn_cls_targets[max_iou_per_anchor < self.negative_iou_threshold] = 0  # Negative
        rpn_cls_targets[max_iou_per_anchor > self.positive_iou_threshold] = 1  # Positive

        # RPN bbox targets (only for positive anchors)
        rpn_bbox_targets = torch.zeros((N_anchors, 4), dtype=torch.float32)

        positive_mask = rpn_cls_targets == 1
        if positive_mask.any():
            positive_anchors = anchors[positive_mask]
            positive_gt_idx = argmax_iou_per_anchor[positive_mask]
            positive_gt_boxes = gt_bboxes[positive_gt_idx]

            # Compute bbox deltas
            rpn_bbox_targets[positive_mask] = self._compute_bbox_deltas(
                positive_anchors, positive_gt_boxes
            )

        return rpn_cls_targets, rpn_bbox_targets

    def _compute_iou_matrix(self, boxes1, boxes2):
        """Compute IoU matrix between two sets of boxes"""
        # Simplified IoU computation
        # In practice, use torchvision.ops.box_iou
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        inter_x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(0)
        inter_h = (inter_y2 - inter_y1).clamp(0)
        inter_area = inter_w * inter_h

        union_area = area1[:, None] + area2 - inter_area

        iou = inter_area / (union_area + 1e-8)

        return iou

    def _compute_bbox_deltas(self, anchors, gt_boxes):
        """Compute bbox regression targets"""
        # Bbox deltas: (tx, ty, tw, th)
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        anchor_cx = anchors[:, 0] + anchor_w / 2
        anchor_cy = anchors[:, 1] + anchor_h / 2

        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_cx = gt_boxes[:, 0] + gt_w / 2
        gt_cy = gt_boxes[:, 1] + gt_h / 2

        dx = (gt_cx - anchor_cx) / (anchor_w + 1e-8)
        dy = (gt_cy - anchor_cy) / (anchor_h + 1e-8)
        dw = torch.log((gt_w + 1e-8) / (anchor_w + 1e-8))
        dh = torch.log((gt_h + 1e-8) / (anchor_h + 1e-8))

        deltas = torch.stack([dx, dy, dw, dh], dim=1)

        return deltas

    def forward(self, predictions, targets):
        """
        Compute detection loss

        Args:
            predictions: Dict with detection outputs
            targets: Dict with detection targets

        Returns:
            Total detection loss
        """

        # Simplified implementation
        # In full implementation, need to:
        # 1. Compute RPN targets from GT
        # 2. Filter out ignore anchors
        # 3. Compute RPN classification and bbox losses
        # 4. Generate proposals from RPN
        # 5. Assign proposals to GT
        # 6. Compute detection classification and bbox losses

        cls_scores = predictions['cls_scores']
        bbox_deltas = predictions['bbox_deltas']

        if 'class_labels' in targets:
            cls_loss = self.det_cls_loss(cls_scores, targets['class_labels'])
        else:
            cls_loss = torch.tensor(0.0, device=cls_scores.device)

        if 'bbox_targets' in targets:
            bbox_loss = self.det_bbox_loss(bbox_deltas, targets['bbox_targets'])
        else:
            bbox_loss = torch.tensor(0.0, device=bbox_deltas.device)

        return cls_loss + bbox_loss


class LaneSegmentationLoss(nn.Module):
    """
    Lane detection loss using dice loss + cross entropy
    Handles class imbalance (lanes are minority)
    """

    def __init__(self, num_classes=2, dice_weight=0.5, use_focal=False, focal_gamma=2.0):
        super(LaneSegmentationLoss, self).__init__()

        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma

        # Cross entropy with class weights
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 2.0]),  # More weight to lane class
            reduction='mean'
        )

    def dice_loss(self, predictions, targets):
        """
        Compute Dice loss (F1-score based)
        """
        predictions = torch.softmax(predictions, dim=1)  # (B, C, H, W)

        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        intersection = torch.sum(predictions * targets_one_hot, dim=(2, 3))
        union = torch.sum(predictions, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))

        dice = 1 - (2 * intersection) / (union + 1e-8)

        return dice.mean()

    def forward(self, predictions, targets):
        """
        Compute lane segmentation loss

        Args:
            predictions: Segmentation logits (B, num_classes, H, W)
            targets: Target masks (B, H, W)

        Returns:
            Combined loss
        """

        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)

        loss = (1 - self.dice_weight) * ce + self.dice_weight * dice

        return loss


class ClassificationLossFull(nn.Module):
    """
    Traffic sign classification loss with optional label smoothing and focal loss
    """

    def __init__(
        self,
        num_classes=43,
        use_label_smoothing=True,
        label_smoothing=0.1,
        use_focal=False,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        super(ClassificationLossFull, self).__init__()

        self.num_classes = num_classes
        self.use_label_smoothing = use_label_smoothing
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        if use_label_smoothing:
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def focal_loss(self, predictions, targets):
        """
        Focal loss: down-weight easy examples
        """
        ce_loss = self.ce_loss(predictions, targets)

        p = torch.softmax(predictions, dim=1)
        p_t = p.gather(1, targets.view(-1, 1))
        focal_weight = (1 - p_t) ** self.focal_gamma
        focal_loss = focal_weight.squeeze() * ce_loss

        return focal_loss.mean()

    def forward(self, predictions, targets):
        """
        Compute classification loss

        Args:
            predictions: Class logits (B, num_classes)
            targets: Target class indices (B,)

        Returns:
            Loss value
        """

        if self.use_focal:
            loss = self.focal_loss(predictions, targets)
        else:
            loss = self.ce_loss(predictions, targets)

        return loss


if __name__ == "__main__":
    # Test loss functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing Multi-Task Loss Functions")
    print(f"Device: {device}\n")

    # Test MultiTaskLoss
    print("1. Testing MultiTaskLoss:")
    mt_loss = MultiTaskLoss(detection_weight=1.0, lane_weight=1.0, classification_weight=1.0)

    losses_dict = {
        'detection': torch.tensor(0.5),
        'lane': torch.tensor(0.3),
        'classification': torch.tensor(0.8),
    }

    total_loss = mt_loss(losses_dict)
    print(f"   Total loss: {total_loss.item():.4f}\n")

    # Test with uncertainty weighting
    print("2. Testing MultiTaskLoss with uncertainty:")
    mt_loss_unc = MultiTaskLoss(use_uncertainty=True)
    total_loss_unc = mt_loss_unc(losses_dict)
    print(f"   Total loss (uncertainty): {total_loss_unc.item():.4f}\n")

    # Test ClassificationLoss
    print("3. Testing ClassificationLossFull:")
    cls_loss = ClassificationLossFull(num_classes=43, use_label_smoothing=True, use_focal=False)
    predictions = torch.randn(4, 43)
    targets = torch.randint(0, 43, (4,))
    loss = cls_loss(predictions, targets)
    print(f"   Classification loss: {loss.item():.4f}\n")

    # Test LaneSegmentationLoss
    print("4. Testing LaneSegmentationLoss:")
    lane_loss = LaneSegmentationLoss(num_classes=2, dice_weight=0.5)
    predictions = torch.randn(2, 2, 64, 64)
    targets = torch.randint(0, 2, (2, 64, 64))
    loss = lane_loss(predictions, targets)
    print(f"   Lane loss: {loss.item():.4f}\n")

    # Test DetectionLossFull
    print("5. Testing DetectionLossFull:")
    det_loss = DetectionLossFull(num_classes=7)
    predictions = {
        'cls_scores': torch.randn(4, 8),
        'bbox_deltas': torch.randn(4, 32),
    }
    targets = {
        'class_labels': torch.randint(0, 8, (4,)),
        'bbox_targets': torch.randn(4, 32),
    }
    loss = det_loss(predictions, targets)
    print(f"   Detection loss: {loss.item():.4f}\n")

    print("All loss function tests passed!")
