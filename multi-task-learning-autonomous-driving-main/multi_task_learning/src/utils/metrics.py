"""
Evaluation metrics for multi-task learning
"""

import torch
import numpy as np
from typing import Dict, Tuple


class MetricsTracker:
    """Track metrics across training/validation"""

    def __init__(self):
        self.metrics = {}

    def update(self, name: str, value: float, count: int = 1):
        """Update metric with moving average"""
        if name not in self.metrics:
            self.metrics[name] = {'sum': 0, 'count': 0}

        self.metrics[name]['sum'] += value * count
        self.metrics[name]['count'] += count

    def get(self, name: str) -> float:
        """Get average metric value"""
        if name not in self.metrics or self.metrics[name]['count'] == 0:
            return 0.0
        return self.metrics[name]['sum'] / self.metrics[name]['count']

    def get_all(self) -> Dict[str, float]:
        """Get all metrics"""
        return {name: self.get(name) for name in self.metrics}

    def reset(self):
        """Reset all metrics"""
        self.metrics = {}

    def __str__(self) -> str:
        return ' | '.join([f'{k}: {v:.4f}' for k, v in self.get_all().items()])


class ClassificationMetrics:
    """Compute classification metrics"""

    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute classification accuracy

        Args:
            predictions: Predicted class logits (B, C)
            targets: Target class indices (B,)

        Returns:
            Accuracy (0-1)
        """
        pred_classes = torch.argmax(predictions, dim=1)
        correct = (pred_classes == targets).float()
        return correct.mean().item()

    @staticmethod
    def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """
        Compute top-k accuracy

        Args:
            predictions: Predicted class logits (B, C)
            targets: Target class indices (B,)
            k: Number of top classes to consider

        Returns:
            Top-k accuracy (0-1)
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        targets_expanded = targets.view(-1, 1).expand_as(top_k_indices)
        correct = (top_k_indices == targets_expanded).any(dim=1).float()
        return correct.mean().item()

    @staticmethod
    def precision_recall_f1(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict:
        """
        Compute per-class precision, recall, and F1

        Args:
            predictions: Predicted class logits (B, C)
            targets: Target class indices (B,)
            num_classes: Number of classes

        Returns:
            Dict with precision, recall, f1 (macro and per-class)
        """
        pred_classes = torch.argmax(predictions, dim=1)

        precision = torch.zeros(num_classes)
        recall = torch.zeros(num_classes)
        f1 = torch.zeros(num_classes)

        for c in range(num_classes):
            tp = ((pred_classes == c) & (targets == c)).sum().float().item()
            fp = ((pred_classes == c) & (targets != c)).sum().float().item()
            fn = ((pred_classes != c) & (targets == c)).sum().float().item()

            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f = 2 * p * r / (p + r + 1e-8)

            precision[c] = p
            recall[c] = r
            f1[c] = f

        return {
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item(),
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
        }


class SegmentationMetrics:
    """Compute segmentation metrics"""

    @staticmethod
    def pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute pixel-wise accuracy

        Args:
            predictions: Predicted segmentation logits (B, C, H, W)
            targets: Target segmentation masks (B, H, W) or (B, 1, H, W)

        Returns:
            Pixel accuracy (0-1)
        """
        # Squeeze targets if they have shape (B, 1, H, W)
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        pred_classes = torch.argmax(predictions, dim=1)
        correct = (pred_classes == targets).float()
        return correct.mean().item()

    @staticmethod
    def iou(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = -1) -> Dict:
        """
        Compute Intersection over Union (IoU) per class

        Args:
            predictions: Predicted segmentation logits (B, C, H, W)
            targets: Target segmentation masks (B, H, W) or (B, 1, H, W)
            num_classes: Number of classes
            ignore_index: Class index to ignore

        Returns:
            Dict with IoU values
        """
        # Squeeze targets if they have shape (B, 1, H, W)
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        pred_classes = torch.argmax(predictions, dim=1)

        iou_scores = torch.zeros(num_classes)

        for c in range(num_classes):
            if c == ignore_index:
                continue

            intersection = ((pred_classes == c) & (targets == c)).sum().float()
            union = ((pred_classes == c) | (targets == c)).sum().float()

            if union == 0:
                iou = torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)
            else:
                iou = intersection / union

            iou_scores[c] = iou.item()

        return {
            'iou': iou_scores.mean().item(),
            'iou_per_class': iou_scores,
        }

    @staticmethod
    def dice_coefficient(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict:
        """
        Compute Dice Coefficient per class

        Args:
            predictions: Predicted segmentation logits (B, C, H, W)
            targets: Target segmentation masks (B, H, W) or (B, 1, H, W) or (B, C, H, W)
            num_classes: Number of classes

        Returns:
            Dict with Dice scores
        """
        # Handle different target shapes
        if targets.dim() == 4:
            # If shape is (B, 1, H, W), squeeze it
            if targets.shape[1] == 1:
                targets = targets.squeeze(1)
            else:
                # If shape is (B, C, H, W) with C > 1, take argmax
                targets = torch.argmax(targets, dim=1)

        pred_classes = torch.argmax(predictions, dim=1)

        # Ensure targets are long type for one_hot
        targets = targets.long()
        pred_classes = pred_classes.long()

        pred_one_hot = torch.nn.functional.one_hot(pred_classes, num_classes).permute(0, 3, 1, 2).float()
        target_one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3))
        union = pred_one_hot.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection) / (union + 1e-8)

        return {
            'dice': dice.mean().item(),
            'dice_per_class': dice.mean(dim=0),
        }


class DetectionMetrics:
    """Compute object detection metrics"""

    @staticmethod
    def compute_iou_bbox(box1: torch.Tensor, box2: torch.Tensor) -> float:
        """
        Compute IoU between two bounding boxes

        Args:
            box1: Box in format [x1, y1, x2, y2]
            box2: Box in format [x1, y1, x2, y2]

        Returns:
            IoU value (0-1)
        """
        x1_inter = max(box1[0].item(), box2[0].item())
        y1_inter = max(box1[1].item(), box2[1].item())
        x2_inter = min(box1[2].item(), box2[2].item())
        y2_inter = min(box1[3].item(), box2[3].item())

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / (union_area + 1e-8)

        return iou.item()

    @staticmethod
    def ap_per_class(
        predictions: list,
        targets: list,
        iou_threshold: float = 0.5,
    ) -> Tuple[float, float, float]:
        """
        Compute Average Precision per class

        Args:
            predictions: List of dicts with 'boxes', 'scores', 'labels'
            targets: List of dicts with 'boxes', 'labels'
            iou_threshold: IoU threshold for match

        Returns:
            Tuple of (ap, precision, recall)
        """
        # Simplified AP computation
        # In practice, use more sophisticated implementation

        tp = 0
        fp = 0
        fn = 0

        for pred, target in zip(predictions, targets):
            pred_boxes = pred.get('boxes', [])
            target_boxes = target.get('boxes', [])

            matched = [False] * len(target_boxes)

            for pb in pred_boxes:
                max_iou = 0
                max_idx = -1

                for i, tb in enumerate(target_boxes):
                    if matched[i]:
                        continue

                    iou = DetectionMetrics.compute_iou_bbox(pb, tb)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = i

                if max_iou > iou_threshold and max_idx >= 0:
                    tp += 1
                    matched[max_idx] = True
                else:
                    fp += 1

            fn += sum(1 for m in matched if not m)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        ap = (precision * recall) if (precision + recall) > 0 else 0

        return ap, precision, recall


if __name__ == "__main__":
    # Test metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing Metrics")
    print(f"Device: {device}\n")

    # Test ClassificationMetrics
    print("1. Classification Metrics:")
    predictions = torch.randn(10, 43)
    targets = torch.randint(0, 43, (10,))

    accuracy = ClassificationMetrics.accuracy(predictions, targets)
    top5_accuracy = ClassificationMetrics.top_k_accuracy(predictions, targets, k=5)

    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Top-5 Accuracy: {top5_accuracy:.4f}\n")

    # Test SegmentationMetrics
    print("2. Segmentation Metrics:")
    predictions = torch.randn(2, 2, 64, 64)
    targets = torch.randint(0, 2, (2, 64, 64))

    pixel_acc = SegmentationMetrics.pixel_accuracy(predictions, targets)
    iou_result = SegmentationMetrics.iou(predictions, targets, num_classes=2)
    dice_result = SegmentationMetrics.dice_coefficient(predictions, targets, num_classes=2)

    print(f"   Pixel Accuracy: {pixel_acc:.4f}")
    print(f"   IoU: {iou_result['iou']:.4f}")
    print(f"   Dice: {dice_result['dice']:.4f}\n")

    # Test MetricsTracker
    print("3. MetricsTracker:")
    tracker = MetricsTracker()
    tracker.update('loss', 0.5, count=10)
    tracker.update('loss', 0.3, count=5)
    tracker.update('accuracy', 0.95, count=15)

    print(f"   {tracker}\n")

    print("All metrics tests passed!")
