"""Feature extraction pipeline for processing activations through SAE."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from ..models.activation_capture import ActivationCapture
from ..models.sae import SparseAutoencoder
from ..models.feature_extractor import FeatureExtractor, ExtractedFeature

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Results from feature extraction pipeline."""

    experiment_id: Optional[int] = None
    layer_name: str = ""
    num_samples: int = 0
    features: Dict[int, ExtractedFeature] = field(default_factory=dict)
    avg_sparsity: float = 0.0
    avg_reconstruction_loss: float = 0.0


class FeatureExtractionPipeline:
    """Pipeline for extracting and analyzing features from model activations.

    This pipeline:
    1. Loads a language model
    2. Captures activations from specified layers
    3. Runs activations through SAE
    4. Extracts and analyzes discovered features
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        layer_idx: int = -1,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """Initialize extraction pipeline.

        Args:
            model_name: HuggingFace model name or path
            layer_idx: Which layer to extract from (-1 for last)
            device: Device to run on
            cache_dir: Cache directory for models
        """
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.device = device
        self.cache_dir = cache_dir

        # Lazy-loaded components
        self.model = None
        self.tokenizer = None
        self.activation_capture = None
        self.sae = None
        self.feature_extractor = None

        logger.info(
            f"FeatureExtractionPipeline initialized for {model_name} layer {layer_idx}"
        )

    def load_model(self):
        """Load language model and tokenizer."""
        if self.model is None:
            logger.info(f"Loading model {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            self.model = AutoModel.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            self.model.to(self.device)
            self.model.eval()

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Model loaded successfully")

    def setup_activation_capture(self, layer_name: str):
        """Setup activation capture for specified layer.

        Args:
            layer_name: Name of layer to hook
        """
        if self.model is None:
            self.load_model()

        self.activation_capture = ActivationCapture(
            self.model, layer_names=[layer_name]
        )

        logger.info(f"Activation capture setup for {layer_name}")

    def load_sae(self, sae_path: Optional[str] = None, sae: Optional[SparseAutoencoder] = None):
        """Load or set SAE model.

        Args:
            sae_path: Path to saved SAE checkpoint
            sae: Pre-initialized SAE instance
        """
        if sae is not None:
            self.sae = sae
        elif sae_path is not None:
            self.sae = SparseAutoencoder.load(sae_path, device=self.device)
        else:
            raise ValueError("Must provide either sae_path or sae instance")

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            self.sae, tokenizer=self.tokenizer, device=self.device
        )

        logger.info("SAE and feature extractor loaded")

    def extract_activations(
        self, texts: List[str], max_length: int = 512
    ) -> torch.Tensor:
        """Extract activations from model for given texts.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length

        Returns:
            Activations tensor of shape (batch, seq_len, hidden_dim)
        """
        if self.model is None:
            self.load_model()

        if self.activation_capture is None:
            raise ValueError("Activation capture not setup. Call setup_activation_capture first.")

        # Tokenize
        inputs = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Capture activations
        with torch.no_grad(), self.activation_capture:
            _ = self.model(**inputs)
            activations_dict = self.activation_capture.get_activations()

        # Get first layer's activations
        layer_name = list(activations_dict.keys())[0]
        activations = activations_dict[layer_name]

        logger.info(f"Extracted activations: {activations.shape}")

        return activations

    def run(
        self,
        texts: List[str],
        layer_name: str,
        sae: SparseAutoencoder,
        compute_statistics: bool = True,
    ) -> ExtractionResult:
        """Run full extraction pipeline.

        Args:
            texts: List of text strings to analyze
            layer_name: Layer to extract from
            sae: Trained SAE model
            compute_statistics: Whether to compute feature statistics

        Returns:
            ExtractionResult with extracted features
        """
        # Setup
        self.setup_activation_capture(layer_name)
        self.load_sae(sae=sae)

        # Extract activations
        activations = self.extract_activations(texts)

        # Get tokens for analysis
        tokens = self.tokenizer(
            texts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        # Compute features
        features = {}
        if compute_statistics:
            features = self.feature_extractor.compute_feature_statistics(
                activations, tokens=tokens
            )

        # Compute metrics
        with torch.no_grad():
            activations_flat = activations.reshape(-1, activations.shape[-1]).to(self.device)
            sae_output = self.sae(activations_flat, return_loss=True)

            avg_reconstruction_loss = sae_output.reconstruction_loss.item()
            sparsity_metrics = self.sae.compute_sparsity_metrics(sae_output.hidden)
            avg_sparsity = sparsity_metrics["l0_sparsity"]

        result = ExtractionResult(
            layer_name=layer_name,
            num_samples=len(texts),
            features=features,
            avg_sparsity=avg_sparsity,
            avg_reconstruction_loss=avg_reconstruction_loss,
        )

        logger.info(
            f"Extraction complete: {len(features)} features, "
            f"sparsity={avg_sparsity:.2f}, loss={avg_reconstruction_loss:.4f}"
        )

        return result
