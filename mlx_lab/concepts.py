"""
Concept vector management for MLX Lab

Browse and manage concept vectors used in activation steering.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from mlx_lab.utils import get_repo_root, format_bytes


@dataclass
class ConceptInfo:
    """Information about a concept vector"""

    name: str
    path: Path
    size_bytes: int
    dimensions: Optional[int] = None
    layer: Optional[int] = None
    source_model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConceptBrowser:
    """Browse and manage concept vectors"""

    def __init__(self):
        self.concepts_dir = get_repo_root() / "shared" / "concepts"
        self.concepts_dir.mkdir(parents=True, exist_ok=True)

    def list_concepts(self) -> List[ConceptInfo]:
        """List all available concept vectors"""
        concepts = []

        for concept_file in self.concepts_dir.glob("*.pkl"):
            # Get file size
            size = concept_file.stat().st_size
            name = concept_file.stem

            # Try to load metadata
            try:
                with open(concept_file, "rb") as f:
                    data = pickle.load(f)

                # Handle different formats
                dimensions = None
                layer = None
                source_model = None
                metadata = None

                # If it's a numpy array directly
                if hasattr(data, "shape"):
                    dimensions = data.shape[0] if len(data.shape) > 0 else None

                # If it's a dictionary with metadata
                elif isinstance(data, dict):
                    if "vector" in data and hasattr(data["vector"], "shape"):
                        dimensions = data["vector"].shape[0]
                    dimensions = data.get("dimensions", dimensions)
                    layer = data.get("layer")
                    source_model = data.get("source_model")
                    metadata = data.get("metadata")

                # If it's a ConceptLibrary object (from introspection code)
                elif hasattr(data, "concepts"):
                    # It's a library, not a single concept
                    # We'll skip detailed info for libraries
                    dimensions = len(data.concepts) if hasattr(data, "concepts") else None
                    metadata = {"type": "concept_library", "num_concepts": dimensions}

                concepts.append(
                    ConceptInfo(
                        name=name,
                        path=concept_file,
                        size_bytes=size,
                        dimensions=dimensions,
                        layer=layer,
                        source_model=source_model,
                        metadata=metadata,
                    )
                )

            except Exception as e:
                # If we can't load it, still list it but without details
                concepts.append(
                    ConceptInfo(
                        name=name,
                        path=concept_file,
                        size_bytes=size,
                        metadata={"error": str(e)},
                    )
                )

        return sorted(concepts, key=lambda c: c.name)

    def get_concept_info(self, name: str) -> Optional[ConceptInfo]:
        """Get info about a specific concept"""
        concepts = self.list_concepts()

        for concept in concepts:
            if concept.name == name or concept.name.startswith(name):
                return concept

        return None

    def check_compatibility(
        self, concept_name: str, model_name: str
    ) -> Dict[str, Any]:
        """
        Check if a concept is compatible with a model

        Args:
            concept_name: Name of the concept
            model_name: Name of the model

        Returns:
            Dictionary with compatibility info
        """
        concept = self.get_concept_info(concept_name)
        if not concept:
            return {
                "compatible": False,
                "reason": f"Concept '{concept_name}' not found",
            }

        # Basic compatibility check
        # In practice, concepts should work across models with same architecture
        # but this is a placeholder for more sophisticated checking

        result = {"compatible": True, "warnings": []}

        if concept.source_model and concept.source_model not in model_name:
            result["warnings"].append(
                f"Concept extracted from {concept.source_model}, "
                f"may not transfer perfectly to {model_name}"
            )

        if concept.metadata and "error" in concept.metadata:
            result["compatible"] = False
            result["reason"] = f"Concept file error: {concept.metadata['error']}"

        return result

    def format_concept_list(self, concepts: List[ConceptInfo]) -> str:
        """Format concept list for display"""
        if not concepts:
            return f"No concept vectors found in {self.concepts_dir}"

        lines = []
        lines.append(f"Concept Vectors (in {self.concepts_dir}):")
        lines.append("=" * 70)

        for concept in concepts:
            lines.append(f"  • {concept.name}")

            if concept.metadata and concept.metadata.get("type") == "concept_library":
                num = concept.metadata.get("num_concepts", "unknown")
                lines.append(f"    Type: Concept Library ({num} concepts)")
            elif concept.dimensions:
                lines.append(f"    Dimensions: {concept.dimensions}")

            if concept.layer is not None:
                lines.append(f"    Layer: {concept.layer}")

            if concept.source_model:
                lines.append(f"    Source: {concept.source_model}")

            lines.append(f"    Size: {format_bytes(concept.size_bytes)}")

            if concept.metadata and "error" in concept.metadata:
                lines.append(f"    ⚠️  Warning: {concept.metadata['error']}")

            lines.append("")

        lines.append("=" * 70)
        lines.append(f"Total: {len(concepts)} concept vector files")

        return "\n".join(lines)

    def format_concept_info(self, concept: ConceptInfo) -> str:
        """Format detailed concept info for display"""
        lines = []
        lines.append(f"Concept: {concept.name}")
        lines.append("=" * 70)
        lines.append(f"Path: {concept.path}")
        lines.append(f"Size: {format_bytes(concept.size_bytes)}")

        if concept.metadata and concept.metadata.get("type") == "concept_library":
            lines.append(f"Type: Concept Library")
            num = concept.metadata.get("num_concepts", "unknown")
            lines.append(f"Contains: {num} concepts")
        else:
            if concept.dimensions:
                lines.append(f"Dimensions: {concept.dimensions}")

            if concept.layer is not None:
                lines.append(f"Layer: {concept.layer}")

            if concept.source_model:
                lines.append(f"Source Model: {concept.source_model}")

        if concept.metadata:
            if "error" in concept.metadata:
                lines.append("")
                lines.append(f"⚠️  Error: {concept.metadata['error']}")
            elif concept.metadata.get("type") != "concept_library":
                lines.append("")
                lines.append("Metadata:")
                for key, value in concept.metadata.items():
                    lines.append(f"  {key}: {value}")

        lines.append("")
        lines.append("Usage:")
        lines.append("  Use this concept with activation steering in notebooks")
        lines.append(f"  See: theory-of-mind/introspection/notebooks/")

        return "\n".join(lines)
