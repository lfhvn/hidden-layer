"""Service for SAE training and management."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..models.sae import SparseAutoencoder, SAETrainingConfig
from ..storage import Experiment, get_session
from ..storage.schemas import ExperimentStatus

logger = logging.getLogger(__name__)


class SAEService:
    """Service for training and managing Sparse Autoencoders."""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """Initialize SAE service.

        Args:
            checkpoint_dir: Directory to save model checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        name: str,
        model_name: str,
        layer_name: str,
        layer_index: int,
        config: SAETrainingConfig,
        description: Optional[str] = None,
    ) -> Experiment:
        """Create new experiment record in database.

        Args:
            name: Experiment name
            model_name: Name of model being analyzed
            layer_name: Layer being hooked
            layer_index: Index of layer
            config: SAE training configuration
            description: Optional description

        Returns:
            Experiment database record
        """
        experiment = Experiment(
            name=name,
            description=description,
            model_name=model_name,
            layer_name=layer_name,
            layer_index=layer_index,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            sparsity_coef=config.sparsity_coef,
            learning_rate=config.learning_rate,
            num_epochs=config.num_epochs,
            status=ExperimentStatus.PENDING,
        )

        with get_session() as session:
            session.add(experiment)
            session.commit()
            session.refresh(experiment)

        logger.info(f"Created experiment {experiment.id}: {name}")

        return experiment

    def train(
        self,
        sae: SparseAutoencoder,
        train_data: torch.Tensor,
        experiment_id: Optional[int] = None,
        val_data: Optional[torch.Tensor] = None,
        callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Train SAE on activation data.

        Args:
            sae: SAE model to train
            train_data: Training activations (N, input_dim)
            experiment_id: Database experiment ID for tracking
            val_data: Optional validation data
            callback: Optional callback function called after each epoch

        Returns:
            Dictionary with training metrics
        """
        # Update experiment status
        if experiment_id is not None:
            with get_session() as session:
                experiment = session.get(Experiment, experiment_id)
                if experiment:
                    experiment.status = ExperimentStatus.RUNNING
                    experiment.started_at = datetime.utcnow()
                    session.add(experiment)
                    session.commit()

        config = sae.config
        device = config.device

        # Create data loader
        dataset = TensorDataset(train_data)
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        )

        # Setup optimizer
        optimizer = optim.Adam(sae.parameters(), lr=config.learning_rate, weight_decay=config.l2_coef)

        # Training loop
        sae.train()
        history = {
            "train_loss": [],
            "reconstruction_loss": [],
            "sparsity_loss": [],
            "val_loss": [],
        }

        logger.info(f"Starting training for {config.num_epochs} epochs")

        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_sparsity_loss = 0.0
            num_batches = 0

            for (batch,) in dataloader:
                batch = batch.to(device)

                # Forward pass
                optimizer.zero_grad()
                output = sae(batch, return_loss=True)

                # Backward pass
                output.loss.backward()
                optimizer.step()

                # Track metrics
                epoch_loss += output.loss.item()
                epoch_recon_loss += output.reconstruction_loss.item()
                epoch_sparsity_loss += output.sparsity_loss.item()
                num_batches += 1

            # Average losses
            avg_loss = epoch_loss / num_batches
            avg_recon = epoch_recon_loss / num_batches
            avg_sparsity = epoch_sparsity_loss / num_batches

            history["train_loss"].append(avg_loss)
            history["reconstruction_loss"].append(avg_recon)
            history["sparsity_loss"].append(avg_sparsity)

            # Validation
            if val_data is not None:
                val_loss = self._validate(sae, val_data)
                history["val_loss"].append(val_loss)
            else:
                history["val_loss"].append(None)

            logger.info(
                f"Epoch {epoch + 1}/{config.num_epochs} - "
                f"Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, Sparsity: {avg_sparsity:.4f})"
            )

            # Callback
            if callback is not None:
                callback(epoch, history)

        # Save checkpoint
        if experiment_id is not None:
            checkpoint_path = self.checkpoint_dir / f"sae_exp_{experiment_id}.pt"
            sae.save(str(checkpoint_path))

            # Update experiment
            with get_session() as session:
                experiment = session.get(Experiment, experiment_id)
                if experiment:
                    experiment.status = ExperimentStatus.COMPLETED
                    experiment.completed_at = datetime.utcnow()
                    experiment.num_samples = len(train_data)
                    experiment.train_loss = history["train_loss"][-1]
                    experiment.reconstruction_loss = history["reconstruction_loss"][-1]
                    experiment.sparsity_loss = history["sparsity_loss"][-1]
                    session.add(experiment)
                    session.commit()

            logger.info(f"Saved checkpoint to {checkpoint_path}")

        sae.eval()

        return history

    def _validate(self, sae: SparseAutoencoder, val_data: torch.Tensor) -> float:
        """Run validation.

        Args:
            sae: SAE model
            val_data: Validation data

        Returns:
            Average validation loss
        """
        sae.eval()
        device = sae.config.device

        with torch.no_grad():
            val_data = val_data.to(device)
            output = sae(val_data, return_loss=True)
            val_loss = output.loss.item()

        sae.train()

        return val_loss


def train_sae(
    activations: torch.Tensor,
    config: SAETrainingConfig,
    experiment_name: Optional[str] = None,
) -> SparseAutoencoder:
    """Convenience function to train an SAE.

    Args:
        activations: Training activations
        config: Training configuration
        experiment_name: Optional experiment name for tracking

    Returns:
        Trained SAE model
    """
    sae = SparseAutoencoder(config).to(config.device)

    service = SAEService()
    service.train(sae, activations)

    logger.info("SAE training completed")

    return sae
