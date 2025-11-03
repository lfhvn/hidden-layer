"""
Concept Vector Library

Store, manage, and search concept representations extracted from models.
"""

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConceptVector:
    """
    A concept representation extracted from a model.

    Attributes:
        name: Human-readable name (e.g., "happiness", "anger")
        vector: The activation vector
        layer: Layer index where extracted
        extraction_prompt: Prompt used to extract this concept
        contrastive_prompt: Optional negative prompt for contrastive extraction
        model_name: Model this was extracted from
        timestamp: When it was created
        metadata: Additional info (e.g., position, method)
    """

    name: str
    vector: np.ndarray
    layer: int
    extraction_prompt: str
    contrastive_prompt: Optional[str] = None
    model_name: str = "unknown"
    timestamp: str = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        d = asdict(self)
        d["vector"] = self.vector.tolist()  # Convert numpy array
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ConceptVector":
        """Load from dictionary"""
        d["vector"] = np.array(d["vector"])  # Convert back to numpy
        return cls(**d)

    def cosine_similarity(self, other: "ConceptVector") -> float:
        """Compute cosine similarity with another concept"""
        dot = np.dot(self.vector, other.vector)
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        return dot / (norm_self * norm_other) if (norm_self * norm_other) > 0 else 0.0

    def euclidean_distance(self, other: "ConceptVector") -> float:
        """Compute Euclidean distance to another concept"""
        return np.linalg.norm(self.vector - other.vector)


class ConceptLibrary:
    """
    Manage a collection of concept vectors.

    Usage:
        library = ConceptLibrary()

        # Add concepts
        library.add_concept(
            name="happiness",
            vector=happy_vec,
            layer=15,
            extraction_prompt="I feel very happy",
            model_name="llama-3.2-3b"
        )

        # Search
        similar = library.find_similar("happiness", top_k=5)

        # Save/load
        library.save("concepts/my_library.pkl")
        library = ConceptLibrary.load("concepts/my_library.pkl")
    """

    def __init__(self):
        self.concepts: Dict[str, ConceptVector] = {}
        self.metadata = {"created": datetime.now().isoformat(), "version": "1.0"}

    def add_concept(
        self,
        name: str,
        vector: np.ndarray,
        layer: int,
        extraction_prompt: str,
        contrastive_prompt: Optional[str] = None,
        model_name: str = "unknown",
        metadata: Optional[Dict] = None,
        overwrite: bool = False,
    ) -> ConceptVector:
        """
        Add a concept to the library.

        Args:
            name: Unique identifier for the concept
            vector: Activation vector
            layer: Layer index
            extraction_prompt: Prompt used for extraction
            contrastive_prompt: Optional negative prompt
            model_name: Source model
            metadata: Additional information
            overwrite: If True, replace existing concept with same name

        Returns:
            The created ConceptVector
        """
        if name in self.concepts and not overwrite:
            raise ValueError(f"Concept '{name}' already exists. Use overwrite=True to replace.")

        concept = ConceptVector(
            name=name,
            vector=vector,
            layer=layer,
            extraction_prompt=extraction_prompt,
            contrastive_prompt=contrastive_prompt,
            model_name=model_name,
            metadata=metadata or {},
        )

        self.concepts[name] = concept
        return concept

    def get(self, name: str) -> Optional[ConceptVector]:
        """Get a concept by name"""
        return self.concepts.get(name)

    def remove(self, name: str) -> bool:
        """Remove a concept by name"""
        if name in self.concepts:
            del self.concepts[name]
            return True
        return False

    def list_concepts(self) -> List[str]:
        """Get list of all concept names"""
        return list(self.concepts.keys())

    def find_similar(self, query: str, top_k: int = 5, metric: str = "cosine") -> List[Tuple[str, float]]:
        """
        Find concepts similar to a query concept.

        Args:
            query: Name of query concept
            top_k: Number of results to return
            metric: "cosine" or "euclidean"

        Returns:
            List of (concept_name, similarity_score) tuples
        """
        if query not in self.concepts:
            raise ValueError(f"Query concept '{query}' not found in library")

        query_concept = self.concepts[query]
        similarities = []

        for name, concept in self.concepts.items():
            if name == query:
                continue

            if metric == "cosine":
                score = query_concept.cosine_similarity(concept)
            elif metric == "euclidean":
                score = -query_concept.euclidean_distance(concept)  # Negative for sorting
            else:
                raise ValueError(f"Unknown metric: {metric}")

            similarities.append((name, score))

        # Sort by score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def find_similar_to_vector(
        self, vector: np.ndarray, top_k: int = 5, metric: str = "cosine"
    ) -> List[Tuple[str, float]]:
        """
        Find concepts similar to an arbitrary vector.

        Args:
            vector: Query vector
            top_k: Number of results
            metric: "cosine" or "euclidean"

        Returns:
            List of (concept_name, similarity_score) tuples
        """
        similarities = []

        for name, concept in self.concepts.items():
            if metric == "cosine":
                dot = np.dot(vector, concept.vector)
                norm_v = np.linalg.norm(vector)
                norm_c = np.linalg.norm(concept.vector)
                score = dot / (norm_v * norm_c) if (norm_v * norm_c) > 0 else 0.0
            elif metric == "euclidean":
                score = -np.linalg.norm(vector - concept.vector)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            similarities.append((name, score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def filter_by_layer(self, layer: int) -> List[str]:
        """Get all concepts extracted from a specific layer"""
        return [name for name, concept in self.concepts.items() if concept.layer == layer]

    def filter_by_model(self, model_name: str) -> List[str]:
        """Get all concepts extracted from a specific model"""
        return [name for name, concept in self.concepts.items() if concept.model_name == model_name]

    def save(self, path: str):
        """
        Save library to disk.

        Format: pickle file containing:
        - concepts: Dict[str, ConceptVector]
        - metadata: Dict
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "concepts": {name: concept.to_dict() for name, concept in self.concepts.items()},
            "metadata": self.metadata,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved {len(self.concepts)} concepts to {path}")

    @classmethod
    def load(cls, path: str) -> "ConceptLibrary":
        """Load library from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        library = cls()
        library.metadata = data["metadata"]

        for name, concept_dict in data["concepts"].items():
            library.concepts[name] = ConceptVector.from_dict(concept_dict)

        print(f"Loaded {len(library.concepts)} concepts from {path}")
        return library

    def export_json(self, path: str):
        """Export library to JSON (without vectors, for inspection)"""
        data = {
            "metadata": self.metadata,
            "concepts": {
                name: {
                    "name": c.name,
                    "layer": c.layer,
                    "extraction_prompt": c.extraction_prompt,
                    "contrastive_prompt": c.contrastive_prompt,
                    "model_name": c.model_name,
                    "timestamp": c.timestamp,
                    "vector_shape": c.vector.shape,
                    "vector_norm": float(np.linalg.norm(c.vector)),
                    "metadata": c.metadata,
                }
                for name, c in self.concepts.items()
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Exported library metadata to {path}")

    def merge(self, other: "ConceptLibrary", prefix: Optional[str] = None):
        """
        Merge another library into this one.

        Args:
            other: Library to merge
            prefix: Optional prefix for concept names to avoid collisions
        """
        for name, concept in other.concepts.items():
            new_name = f"{prefix}_{name}" if prefix else name

            if new_name in self.concepts:
                print(f"Warning: Skipping duplicate concept '{new_name}'")
                continue

            self.concepts[new_name] = concept

        print(f"Merged {len(other.concepts)} concepts")

    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute pairwise similarity matrix for all concepts.

        Returns:
            NxN matrix where entry [i,j] is cosine similarity between concepts i and j
        """
        names = list(self.concepts.keys())
        n = len(names)
        matrix = np.zeros((n, n))

        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = self.concepts[name1].cosine_similarity(self.concepts[name2])

        return matrix, names

    def __len__(self):
        return len(self.concepts)

    def __repr__(self):
        return f"ConceptLibrary(concepts={len(self.concepts)})"


def build_emotion_library(steerer, layer: int = 15, model_name: str = "unknown") -> ConceptLibrary:
    """
    Build a library of basic emotion concepts.

    Args:
        steerer: ActivationSteerer instance
        layer: Layer to extract from
        model_name: Model identifier

    Returns:
        ConceptLibrary with emotion concepts
    """
    library = ConceptLibrary()

    emotions = {
        "happiness": ("I feel extremely happy, joyful, and delighted!", "I feel neutral, neither happy nor sad."),
        "sadness": ("I feel very sad, depressed, and sorrowful.", "I feel neutral, neither happy nor sad."),
        "anger": ("I feel extremely angry, furious, and enraged!", "I feel calm and neutral."),
        "fear": ("I feel very scared, frightened, and terrified!", "I feel safe and calm."),
        "surprise": ("I feel shocked, astonished, and amazed!", "I feel unsurprised and calm."),
        "disgust": ("I feel disgusted, repulsed, and revolted!", "I feel neutral and calm."),
    }

    print(f"Extracting {len(emotions)} emotion concepts from layer {layer}...")

    for emotion, (positive, negative) in emotions.items():
        print(f"  - {emotion}")
        vector = steerer.extract_contrastive_concept(
            positive_prompt=positive, negative_prompt=negative, layer_idx=layer, position="last"
        )

        library.add_concept(
            name=emotion,
            vector=vector,
            layer=layer,
            extraction_prompt=positive,
            contrastive_prompt=negative,
            model_name=model_name,
            metadata={"category": "emotion"},
        )

    return library


def build_topic_library(steerer, layer: int = 15, model_name: str = "unknown") -> ConceptLibrary:
    """
    Build a library of topic/domain concepts.
    """
    library = ConceptLibrary()

    topics = {
        "science": (
            "Let's discuss scientific research, experiments, and discoveries.",
            "Let's have a general conversation.",
        ),
        "politics": ("Let's discuss politics, government, and policy.", "Let's have a general conversation."),
        "sports": ("Let's talk about sports, athletics, and competition.", "Let's have a general conversation."),
        "art": ("Let's discuss art, creativity, and artistic expression.", "Let's have a general conversation."),
        "technology": ("Let's talk about technology, computers, and innovation.", "Let's have a general conversation."),
    }

    print(f"Extracting {len(topics)} topic concepts from layer {layer}...")

    for topic, (positive, negative) in topics.items():
        print(f"  - {topic}")
        vector = steerer.extract_contrastive_concept(
            positive_prompt=positive, negative_prompt=negative, layer_idx=layer, position="last"
        )

        library.add_concept(
            name=topic,
            vector=vector,
            layer=layer,
            extraction_prompt=positive,
            contrastive_prompt=negative,
            model_name=model_name,
            metadata={"category": "topic"},
        )

    return library


if __name__ == "__main__":
    # Example usage
    print("Concept Library Demo\n")

    # Create a dummy library
    library = ConceptLibrary()

    # Add some dummy concepts
    library.add_concept(
        name="happiness",
        vector=np.random.randn(512),
        layer=15,
        extraction_prompt="I feel happy",
        model_name="test-model",
    )

    library.add_concept(
        name="sadness", vector=np.random.randn(512), layer=15, extraction_prompt="I feel sad", model_name="test-model"
    )

    print(f"Library: {library}")
    print(f"Concepts: {library.list_concepts()}")

    # Test save/load
    library.save("/tmp/test_library.pkl")
    loaded = ConceptLibrary.load("/tmp/test_library.pkl")
    print(f"Loaded: {loaded}")
