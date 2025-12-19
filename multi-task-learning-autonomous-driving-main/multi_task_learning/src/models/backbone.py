"""
Backbone architectures for multi-task learning
Shared feature extractor for all tasks
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Backbone(nn.Module):
    """
    ResNet50 backbone for feature extraction
    Removes the last FC layer, returns feature maps
    """

    def __init__(self, pretrained=True, freeze=False):
        super(ResNet50Backbone, self).__init__()

        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)

        # Remove the average pooling and classification layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])

        # Feature map dimensions after ResNet50
        # Input: (3, H, W)
        # Output: (2048, H/32, W/32)
        self.out_channels = 2048

        if freeze:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze backbone weights during training"""
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            features: Feature maps of shape (B, 2048, H/32, W/32)
        """
        features = self.features(x)
        return features

    def get_out_channels(self):
        """Get output channels of backbone"""
        return self.out_channels


class ResNet101Backbone(nn.Module):
    """
    ResNet101 backbone for feature extraction
    Larger model than ResNet50, better accuracy at cost of more computation
    """

    def __init__(self, pretrained=True, freeze=False):
        super(ResNet101Backbone, self).__init__()

        # Load pretrained ResNet101
        self.backbone = models.resnet101(pretrained=pretrained)

        # Remove the average pooling and classification layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])

        # Feature map dimensions
        self.out_channels = 2048

        if freeze:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze backbone weights during training"""
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            features: Feature maps of shape (B, 2048, H/32, W/32)
        """
        features = self.features(x)
        return features

    def get_out_channels(self):
        """Get output channels of backbone"""
        return self.out_channels


def get_backbone(backbone_name="resnet50", pretrained=True, freeze=False):
    """
    Factory function to get backbone model

    Args:
        backbone_name: Name of backbone ("resnet50" or "resnet101")
        pretrained: Whether to use pretrained weights
        freeze: Whether to freeze backbone weights

    Returns:
        Backbone model instance
    """
    if backbone_name == "resnet50":
        return ResNet50Backbone(pretrained=pretrained, freeze=freeze)
    elif backbone_name == "resnet101":
        return ResNet101Backbone(pretrained=pretrained, freeze=freeze)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


if __name__ == "__main__":
    # Test backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = get_backbone("resnet50", pretrained=True).to(device)

    # Test with different input sizes
    test_inputs = [
        torch.randn(2, 3, 375, 1242).to(device),  # KITTI
        torch.randn(2, 3, 64, 64).to(device),      # GTSRB
        torch.randn(2, 3, 720, 1280).to(device),   # Lane detection
    ]

    print("Testing ResNet50 Backbone")
    print(f"Device: {device}")
    print(f"Backbone output channels: {backbone.get_out_channels()}\n")

    for i, x in enumerate(test_inputs):
        features = backbone(x)
        print(f"Input {i+1}: {x.shape} -> Features: {features.shape}")

    print("\nBackbone test passed!")
