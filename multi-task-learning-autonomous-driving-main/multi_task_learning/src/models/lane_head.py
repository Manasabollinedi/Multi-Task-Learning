"""
Lane Detection Head for multi-task learning
Performs semantic segmentation to detect lane markings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """
    Decoder block for upsampling features
    Used in U-Net style architecture
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (B, C_in, H, W)
        Returns:
            Output tensor (B, C_out, 2*H, 2*W)
        """
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LaneDetectionHead(nn.Module):
    """
    Lane Detection Head using U-Net style decoder
    Takes features from backbone and produces segmentation masks
    """

    def __init__(self, in_channels=2048, num_classes=2):
        """
        Args:
            in_channels: Number of input channels from backbone (2048 for ResNet50)
            num_classes: Number of segmentation classes (2 for binary: lane/not-lane)
        """
        super(LaneDetectionHead, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Initial feature reduction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder path: gradually upsample features
        # ResNet50 features are 1/32 of input resolution
        # We need to upsample back to original resolution (32x upsampling)

        # Upsample by 2x, 2x, 2x, 2x, 2x (total 32x)
        self.decoder1 = DecoderBlock(512, 256)    # 1/16
        self.decoder2 = DecoderBlock(256, 128)    # 1/8
        self.decoder3 = DecoderBlock(128, 64)     # 1/4
        self.decoder4 = DecoderBlock(64, 32)      # 1/2
        self.decoder5 = DecoderBlock(32, 16)      # Full resolution

        # Final segmentation head
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Output segmentation map
        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, features):
        """
        Forward pass for lane detection

        Args:
            features: Feature maps from backbone (B, 2048, H/32, W/32)

        Returns:
            segmentation_map: Segmentation mask (B, num_classes, H, W)
        """

        # Initial feature reduction
        x = self.initial_conv(features)  # (B, 512, H/32, W/32)

        # Decoder path - gradually upsample
        x = self.decoder1(x)  # (B, 256, H/16, W/16)
        x = self.decoder2(x)  # (B, 128, H/8, W/8)
        x = self.decoder3(x)  # (B, 64, H/4, W/4)
        x = self.decoder4(x)  # (B, 32, H/2, W/2)
        x = self.decoder5(x)  # (B, 16, H, W)

        # Final convolution
        x = self.final_conv(x)  # (B, 16, H, W)

        # Segmentation output
        segmentation_map = self.segmentation_head(x)  # (B, num_classes, H, W)

        return segmentation_map


class LaneLoss(nn.Module):
    """
    Loss function for lane detection (semantic segmentation)
    Uses cross entropy loss with optional focal loss for hard negatives
    """

    def __init__(self, num_classes=2, use_focal=False, focal_alpha=0.25, focal_gamma=2.0):
        super(LaneLoss, self).__init__()

        self.num_classes = num_classes
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Cross entropy loss with class weighting (more weight to lane pixels)
        # Typically lanes are minority class
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 2.0]),  # Give more weight to lane class
            reduction='mean'
        )

    def forward(self, predictions, targets):
        """
        Calculate lane detection loss

        Args:
            predictions: Predicted segmentation maps (B, num_classes, H, W)
            targets: Target segmentation masks (B, H, W) with class indices

        Returns:
            loss: Scalar loss value
        """

        # Cross entropy loss
        loss = self.ce_loss(predictions, targets)

        # Optional focal loss for hard negatives
        if self.use_focal:
            # Focal loss: downweight easy negatives
            p = torch.softmax(predictions, dim=1)
            p_t = p.gather(1, targets.unsqueeze(1))  # Probability of target class
            focal_weight = (1 - p_t) ** self.focal_gamma
            loss = loss * focal_weight.mean()

        return loss


if __name__ == "__main__":
    # Test lane detection head
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head = LaneDetectionHead(in_channels=2048, num_classes=2).to(device)

    # Test with feature maps from ResNet50
    # Lane images are 1280x720, so features are roughly 40x23 (1/32 resolution)
    features = torch.randn(2, 2048, 23, 40).to(device)

    segmentation_map = head(features)

    print("Lane Detection Head Test")
    print(f"Device: {device}")
    print(f"Input features: {features.shape}")
    print(f"Output segmentation map: {segmentation_map.shape}")

    # Test loss
    targets = torch.randint(0, 2, (2, 720, 1280)).to(device)
    loss_fn = LaneLoss(num_classes=2)

    # Resize segmentation map to match targets for loss calculation
    seg_map_resized = F.interpolate(segmentation_map, size=(720, 1280), mode='bilinear', align_corners=False)
    loss = loss_fn(seg_map_resized, targets)

    print(f"Loss: {loss.item():.4f}")
    print("\nLane detection head test passed!")
