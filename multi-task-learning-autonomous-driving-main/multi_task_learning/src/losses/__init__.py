"""
Loss functions for multi-task learning
"""

from .multi_task_loss import (
    MultiTaskLoss,
    DetectionLossFull,
    LaneSegmentationLoss,
    ClassificationLossFull,
)

__all__ = [
    'MultiTaskLoss',
    'DetectionLossFull',
    'LaneSegmentationLoss',
    'ClassificationLossFull',
]
