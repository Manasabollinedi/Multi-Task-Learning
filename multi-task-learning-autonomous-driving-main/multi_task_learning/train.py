"""
Main training script for multi-task learning
Trains the model on all three tasks simultaneously
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.configs.config import (
    DEVICE, PROJECT_ROOT, CHECKPOINT_DIR, LOG_DIR,
    TrainingConfig, ModelConfig
)
from src.models import MultiTaskModel
from src.losses import MultiTaskLoss, DetectionLossFull, LaneSegmentationLoss, ClassificationLossFull
from src.data import MultiTaskDataLoader
from src.utils import MetricsTracker, CheckpointManager, ClassificationMetrics, SegmentationMetrics


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'training.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class MultiTaskTrainer:
    """Trainer for multi-task learning model"""

    def __init__(self, model, train_loaders, val_loaders, config):
        """
        Args:
            model: MultiTaskModel instance
            train_loaders: Dict of train dataloaders
            val_loaders: Dict of validation dataloaders
            config: Configuration object
        """

        self.model = model.to(DEVICE)
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.config = config

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Learning rate scheduler
        if config.training.use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.epochs,
            )
        else:
            self.scheduler = None

        # Loss functions
        self.detection_loss = DetectionLossFull(num_classes=config.model.num_detection_classes)
        self.lane_loss = LaneSegmentationLoss(num_classes=config.model.num_lane_classes)
        self.classification_loss = ClassificationLossFull(
            num_classes=config.model.num_classification_classes,
            use_label_smoothing=True,
        )
        self.multi_task_loss = MultiTaskLoss(
            detection_weight=config.model.detection_weight,
            lane_weight=config.model.lane_weight,
            classification_weight=config.model.classification_weight,
        )

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(CHECKPOINT_DIR)

        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, epoch):
        """Train for one epoch"""

        self.model.train()
        metrics = MetricsTracker()

        # Create progress bar
        pbar = tqdm(total=len(self.train_loaders['classification']), desc=f"Epoch {epoch} Train")

        # Iterate through batches
        # For now, just use classification loader
        for batch_idx, batch in enumerate(self.train_loaders['classification']):
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            # Forward pass
            self.optimizer.zero_grad()

            predictions = self.model(images)

            # Compute losses for each task
            losses = {}

            # Classification loss (we have labels for this)
            cls_pred = predictions['classification']
            losses['classification'] = self.classification_loss(cls_pred, labels)

            # Lane and detection losses are not computed in this simple version
            # because we don't have proper targets from other dataloaders
            losses['detection'] = torch.tensor(0.0, device=DEVICE)
            losses['lane'] = torch.tensor(0.0, device=DEVICE)

            # Combined loss
            total_loss = self.multi_task_loss(losses)

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            if self.config.training.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clip_grad_norm)

            # Optimizer step
            self.optimizer.step()

            # Metrics
            metrics.update('total_loss', total_loss.item())
            metrics.update('classification_loss', losses['classification'].item())

            # Classification metrics
            accuracy = ClassificationMetrics.accuracy(cls_pred, labels)
            metrics.update('accuracy', accuracy)

            pbar.update(1)

            # Log progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Batch {batch_idx+1}/{len(self.train_loaders['classification'])}: {metrics}")

        pbar.close()

        return metrics.get_all()

    def validate(self, epoch):
        """Validate the model"""

        self.model.eval()
        metrics = MetricsTracker()

        with torch.no_grad():
            # Classification validation
            pbar = tqdm(total=len(self.val_loaders['classification']), desc=f"Epoch {epoch} Val")

            for batch in self.val_loaders['classification']:
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                predictions = self.model(images)

                # Classification loss
                cls_pred = predictions['classification']
                loss = self.classification_loss(cls_pred, labels)
                metrics.update('classification_loss', loss.item())

                # Classification metrics
                accuracy = ClassificationMetrics.accuracy(cls_pred, labels)
                metrics.update('accuracy', accuracy)

                pbar.update(1)

            pbar.close()

        return metrics.get_all()

    def train(self):
        """Main training loop"""

        logger.info("="*80)
        logger.info("Starting Multi-Task Learning Training")
        logger.info("="*80)

        logger.info(f"Model parameters: {self.model.get_total_params():,}")
        logger.info(f"Trainable parameters: {self.model.get_trainable_params():,}")

        for epoch in range(1, self.config.training.epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            if epoch % self.config.training.val_frequency == 0:
                val_metrics = self.validate(epoch)

                logger.info(f"\nEpoch {epoch}/{self.config.training.epochs}")
                logger.info(f"  Train: {train_metrics}")
                logger.info(f"  Val:   {val_metrics}")

                # Check if best
                val_loss = val_metrics.get('classification_loss', float('inf'))
                is_best = val_loss < self.best_val_loss

                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    scheduler=self.scheduler,
                    is_best=is_best,
                )

                # Early stopping
                if (self.config.training.use_early_stopping and
                    self.patience_counter >= self.config.training.patience):
                    logger.info(f"\nEarly stopping at epoch {epoch}")
                    break

            # Learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        logger.info("="*80)
        logger.info("Training Complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("="*80)


def main():
    """Main entry point"""

    logger.info("Loading configuration...")

    # Load data
    logger.info("Loading data...")

    # Create a simple config object for the data loader
    class DataConfig:
        pass

    data_config = DataConfig()

    data_loader = MultiTaskDataLoader(
        config=data_config,
        batch_size=TrainingConfig.batch_size,
        num_workers=TrainingConfig.num_workers,
    )

    train_loaders = {
        'detection': data_loader.get_train_loader('detection'),
        'lane': data_loader.get_train_loader('lane'),
        'classification': data_loader.get_train_loader('classification'),
    }

    val_loaders = {
        'detection': data_loader.get_val_loader('detection'),
        'lane': data_loader.get_val_loader('lane'),
        'classification': data_loader.get_val_loader('classification'),
    }

    logger.info("Data loaded successfully")

    # Create model
    logger.info("Creating model...")
    model = MultiTaskModel(
        backbone_name=ModelConfig.backbone,
        pretrained=ModelConfig.pretrained,
        freeze_backbone=ModelConfig.freeze_backbone,
        num_detection_classes=ModelConfig.num_detection_classes,
        num_lane_classes=ModelConfig.num_lane_classes,
        num_classification_classes=ModelConfig.num_classification_classes,
    )

    logger.info(f"Model created on {DEVICE}")

    # Create config object for trainer
    class TrainerConfig:
        def __init__(self):
            self.training = TrainingConfig
            self.model = ModelConfig
            self.logging = type('LoggingConfig', (), {'log_frequency': 100})

    trainer_config = TrainerConfig()

    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        config=trainer_config,
    )

    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
