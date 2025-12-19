"""
Checkpoint management utilities
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime


class CheckpointManager:
    """Manage model checkpointing and restoration"""

    def __init__(self, checkpoint_dir: Path):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_metrics = {}

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        is_best: bool = False,
        model_name: str = "model",
    ):
        """
        Save model checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            scheduler: Learning rate scheduler (optional)
            is_best: Whether this is the best checkpoint
            model_name: Name prefix for checkpoint
        """

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"{model_name}_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f"{model_name}_best.pt"
            torch.save(checkpoint, best_path)

            # Update best metrics
            self.best_metrics = metrics.copy()

        print(f"Checkpoint saved: {checkpoint_path}")

        if is_best:
            print(f"  -> New best checkpoint! Metrics: {metrics}")

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Path,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict:
        """
        Load model checkpoint

        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            checkpoint_path: Path to checkpoint file
            scheduler: Learning rate scheduler (optional)

        Returns:
            Dict with checkpoint info (epoch, metrics)
        """

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Epoch: {epoch}")
        print(f"  Metrics: {metrics}")

        return {'epoch': epoch, 'metrics': metrics}

    def load_best_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        model_name: str = "model",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict:
        """Load best checkpoint"""
        best_path = self.checkpoint_dir / f"{model_name}_best.pt"

        if not best_path.exists():
            print(f"No best checkpoint found: {best_path}")
            return {'epoch': 0, 'metrics': {}}

        return self.load_checkpoint(model, optimizer, best_path, scheduler)

    def get_checkpoint_info(self, model_name: str = "model") -> Optional[Dict]:
        """Get info about best checkpoint"""
        if not self.best_metrics:
            return None

        return {
            'model_name': model_name,
            'best_metrics': self.best_metrics,
            'checkpoint_dir': str(self.checkpoint_dir),
        }


class TrainingState:
    """Track and save training state"""

    def __init__(self, state_file: Path):
        """
        Args:
            state_file: File to save state to
        """
        self.state_file = Path(state_file)

    def save_state(
        self,
        epoch: int,
        best_loss: float,
        patience_counter: int,
        learning_rate: float,
        metrics: Dict[str, float],
    ):
        """Save training state"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'best_loss': best_loss,
            'patience_counter': patience_counter,
            'learning_rate': learning_rate,
            'metrics': metrics,
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> Optional[Dict]:
        """Load training state"""
        if not self.state_file.exists():
            return None

        with open(self.state_file, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    import tempfile

    print("Testing Checkpoint Manager\n")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        # Create simple model and optimizer
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Test CheckpointManager
        manager = CheckpointManager(checkpoint_dir)

        # Save checkpoint
        metrics = {'loss': 0.5, 'accuracy': 0.95}
        manager.save_checkpoint(
            model, optimizer, epoch=1, metrics=metrics, is_best=True, model_name="test_model"
        )

        # Modify model
        for p in model.parameters():
            p.data.fill_(0)

        # Load checkpoint
        loaded_info = manager.load_checkpoint(
            model, optimizer, checkpoint_dir / "test_model_best.pt"
        )

        print(f"Loaded epoch: {loaded_info['epoch']}")
        print(f"Loaded metrics: {loaded_info['metrics']}")

        # Test TrainingState
        print("\nTesting TrainingState:")
        state_file = checkpoint_dir / "training_state.json"
        state_tracker = TrainingState(state_file)

        state_tracker.save_state(
            epoch=5,
            best_loss=0.3,
            patience_counter=2,
            learning_rate=1e-4,
            metrics={'train_loss': 0.4, 'val_loss': 0.5},
        )

        loaded_state = state_tracker.load_state()
        print(f"Loaded state: {loaded_state}")

    print("\nCheckpoint manager tests passed!")
