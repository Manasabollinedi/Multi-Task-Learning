"""
Models module for multi-task learning
Backbone and task-specific heads
"""

from .backbone import ResNet50Backbone, ResNet101Backbone, get_backbone
from .detection_head import ObjectDetectionHead, DetectionLoss
from .lane_head import LaneDetectionHead, LaneLoss
from .classification_head import (
    ClassificationHead,
    ClassificationHeadAdvanced,
    ClassificationLoss,
)
from .multi_task_model import MultiTaskModel, create_multi_task_model

__all__ = [
    'ResNet50Backbone',
    'ResNet101Backbone',
    'get_backbone',
    'ObjectDetectionHead',
    'DetectionLoss',
    'LaneDetectionHead',
    'LaneLoss',
    'ClassificationHead',
    'ClassificationHeadAdvanced',
    'ClassificationLoss',
    'MultiTaskModel',
    'create_multi_task_model',
]
