"""Dataset loading utilities for SAE training."""

import logging
from typing import Iterator, List, Optional, Dict, Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads and preprocesses datasets for SAE training.

    Supports streaming from HuggingFace datasets and custom text corpora.
    """

    def __init__(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "train",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        """Initialize dataset loader.

        Args:
            dataset_name: Name of HuggingFace dataset
            dataset_config: Dataset configuration/subset
            split: Dataset split to use
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
            batch_size: Batch size for loading
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

        self.dataset = None
        logger.info(f"DatasetLoader initialized for {dataset_name}/{dataset_config}")

    def load(self, streaming: bool = False, max_samples: Optional[int] = None):
        """Load dataset.

        Args:
            streaming: Whether to stream dataset (for large datasets)
            max_samples: Maximum number of samples to load

        Returns:
            Dataset instance
        """
        try:
            self.dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
                streaming=streaming,
            )

            if max_samples is not None and not streaming:
                self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

            logger.info(f"Loaded dataset with {len(self.dataset) if not streaming else 'streaming'} samples")

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        return self.dataset

    def get_batches(
        self, max_samples: Optional[int] = None
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Get batches of tokenized text.

        Args:
            max_samples: Maximum number of samples to process

        Yields:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        if self.dataset is None:
            self.load()

        if self.tokenizer is None:
            raise ValueError("Tokenizer required for batching")

        batch_texts = []
        num_samples = 0

        for example in self.dataset:
            # Extract text field
            text = example.get("text", example.get("content", ""))

            if not text or len(text.strip()) == 0:
                continue

            batch_texts.append(text)
            num_samples += 1

            # Yield batch when full
            if len(batch_texts) >= self.batch_size:
                yield self._tokenize_batch(batch_texts)
                batch_texts = []

            # Stop if max_samples reached
            if max_samples is not None and num_samples >= max_samples:
                break

        # Yield remaining samples
        if batch_texts:
            yield self._tokenize_batch(batch_texts)

    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            Dictionary with tokenized outputs
        """
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def get_text_samples(self, num_samples: int = 100) -> List[str]:
        """Get raw text samples.

        Args:
            num_samples: Number of samples to retrieve

        Returns:
            List of text strings
        """
        if self.dataset is None:
            self.load()

        texts = []

        for i, example in enumerate(self.dataset):
            if i >= num_samples:
                break

            text = example.get("text", example.get("content", ""))
            if text and len(text.strip()) > 0:
                texts.append(text)

        return texts


def load_wikitext(
    tokenizer: Optional[PreTrainedTokenizer] = None,
    split: str = "train",
    max_samples: int = 1000,
    batch_size: int = 32,
) -> DatasetLoader:
    """Convenience function to load WikiText dataset.

    Args:
        tokenizer: Tokenizer for encoding
        split: Dataset split
        max_samples: Maximum samples to load
        batch_size: Batch size

    Returns:
        DatasetLoader instance
    """
    loader = DatasetLoader(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split=split,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    loader.load(max_samples=max_samples)

    return loader
