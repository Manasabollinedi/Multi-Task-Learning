"""
Traffic Sign Classification Head for multi-task learning
Classifies traffic signs into 43 categories
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Traffic Sign Classification Head
    Takes features from backbone and predicts one of 43 traffic sign classes
    """

    def __init__(self, in_channels=2048, num_classes=43):
        """
        Args:
            in_channels: Number of input channels from backbone (2048 for ResNet50)
            num_classes: Number of traffic sign classes (43 for GTSRB)
        """
        super(ClassificationHead, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Global average pooling to get fixed-size feature vector
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature refinement through fully connected layers
        self.fc1 = nn.Linear(in_channels, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)

        # Classification head
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, features):
        """
        Forward pass for traffic sign classification

        Args:
            features: Feature maps from backbone (B, 2048, H/32, W/32)

        Returns:
            logits: Class logits (B, num_classes)
        """

        # Global average pooling to get (B, 2048, 1, 1)
        x = self.global_avg_pool(features)

        # Flatten to (B, 2048)
        x = x.view(x.size(0), -1)

        # First FC layer with batch norm and dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Second FC layer with batch norm and dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Classification layer
        logits = self.classifier(x)

        return logits


class ClassificationHeadAdvanced(nn.Module):
    """
    Advanced Classification Head with attention mechanism
    Uses channel attention to focus on important features
    """

    def __init__(self, in_channels=2048, num_classes=43, reduction=16):
        """
        Args:
            in_channels: Number of input channels from backbone (2048 for ResNet50)
            num_classes: Number of traffic sign classes (43 for GTSRB)
            reduction: Reduction ratio for attention mechanism
        """
        super(ClassificationHeadAdvanced, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Channel attention mechanism
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Attention FC layers
        self.fc_avg = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )

        self.fc_max = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )

        # Classification layers
        self.fc1 = nn.Linear(in_channels, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, features):
        """
        Forward pass with channel attention

        Args:
            features: Feature maps from backbone (B, 2048, H/32, W/32)

        Returns:
            logits: Class logits (B, num_classes)
        """

        # Channel attention
        avg_out = self.global_avg_pool(features)  # (B, 2048, 1, 1)
        max_out = self.global_max_pool(features)  # (B, 2048, 1, 1)

        avg_out = avg_out.view(avg_out.size(0), -1)  # (B, 2048)
        max_out = max_out.view(max_out.size(0), -1)  # (B, 2048)

        avg_att = self.fc_avg(avg_out)  # (B, 2048)
        max_att = self.fc_max(max_out)  # (B, 2048)

        # Combine attention
        att = torch.sigmoid(avg_att + max_att)  # (B, 2048)

        # Apply attention to average pooled features
        pooled = self.global_avg_pool(features)  # (B, 2048, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, 2048)
        x = pooled * att

        # Classification layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        logits = self.classifier(x)

        return logits


class ClassificationLoss(nn.Module):
    """
    Loss function for traffic sign classification
    Uses cross entropy loss with optional label smoothing
    """

    def __init__(self, num_classes=43, use_label_smoothing=False, label_smoothing=0.1):
        super(ClassificationLoss, self).__init__()

        self.num_classes = num_classes
        self.use_label_smoothing = use_label_smoothing

        if use_label_smoothing:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        """
        Calculate classification loss

        Args:
            predictions: Predicted logits (B, num_classes)
            targets: Target class indices (B,)

        Returns:
            loss: Scalar loss value
        """

        loss = self.loss_fn(predictions, targets)
        return loss


if __name__ == "__main__":
    # Test classification head
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Standard head
    head = ClassificationHead(in_channels=2048, num_classes=43).to(device)

    # Advanced head with attention
    head_advanced = ClassificationHeadAdvanced(in_channels=2048, num_classes=43).to(device)

    # Test with feature maps from ResNet50
    # GTSRB images are 64x64, so features are roughly 2x2 (1/32 resolution)
    features = torch.randn(4, 2048, 2, 2).to(device)

    logits = head(features)
    logits_advanced = head_advanced(features)

    print("Classification Head Test")
    print(f"Device: {device}")
    print(f"Input features: {features.shape}")
    print(f"Standard head output: {logits.shape}")
    print(f"Advanced head output: {logits_advanced.shape}")

    # Test loss
    targets = torch.randint(0, 43, (4,)).to(device)
    loss_fn = ClassificationLoss(num_classes=43, use_label_smoothing=True)

    loss = loss_fn(logits, targets)
    print(f"\nLoss: {loss.item():.4f}")

    # Test softmax probabilities
    probs = torch.softmax(logits, dim=1)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities sum to 1: {probs.sum(dim=1).mean().item():.4f}")

    print("\nClassification head test passed!")
