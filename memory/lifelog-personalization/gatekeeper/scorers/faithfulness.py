"""RAG faithfulness metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    from datasets import Dataset  # type: ignore
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
except ImportError:  # pragma: no cover - ragas is optional
    Dataset = None  # type: ignore
    evaluate = None  # type: ignore
    answer_relevancy = None  # type: ignore
    context_precision = None  # type: ignore
    context_recall = None  # type: ignore
    faithfulness = None  # type: ignore


@dataclass
class RAGFaithfulnessScores:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def evaluate_with_ragas(records: List[Dict[str, str]]) -> Optional[RAGFaithfulnessScores]:
    """Run ragas evaluation if the dependency is available."""

    if evaluate is None or Dataset is None:
        return None

    dataset = Dataset.from_list(records)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return RAGFaithfulnessScores(
        faithfulness=float(result["faithfulness"]),
        answer_relevancy=float(result["answer_relevancy"]),
        context_precision=float(result["context_precision"]),
        context_recall=float(result["context_recall"]),
    )


def lexical_faithfulness(records: Iterable[Dict[str, str]]) -> RAGFaithfulnessScores:
    """Fallback heuristic faithfulness metrics.

    The heuristic penalises hallucinations by measuring overlap between answers
    and retrieved contexts.
    """

    overlaps: List[float] = []
    relevancies: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    for record in records:
        answer_tokens = set(record.get("answer", "").lower().split())
        prediction_tokens = set(record.get("prediction", "").lower().split())
        context_tokens = set(" ".join(record.get("contexts", [])).lower().split())
        overlaps.append(len(prediction_tokens & context_tokens) / max(1, len(prediction_tokens)))
        relevancies.append(len(prediction_tokens & answer_tokens) / max(1, len(answer_tokens)))
        precisions.append(len(context_tokens & answer_tokens) / max(1, len(context_tokens)))
        recalls.append(len(context_tokens & answer_tokens) / max(1, len(answer_tokens)))

    return RAGFaithfulnessScores(
        faithfulness=float(np.mean(overlaps)) if overlaps else 0.0,
        answer_relevancy=float(np.mean(relevancies)) if relevancies else 0.0,
        context_precision=float(np.mean(precisions)) if precisions else 0.0,
        context_recall=float(np.mean(recalls)) if recalls else 0.0,
    )


def score_rag_truth(records: List[Dict[str, str]]) -> RAGFaithfulnessScores:
    """Evaluate RAG outputs with ragas when possible, otherwise fallback."""

    ragas_scores = evaluate_with_ragas(records)
    if ragas_scores is not None:
        return ragas_scores
    return lexical_faithfulness(records)
