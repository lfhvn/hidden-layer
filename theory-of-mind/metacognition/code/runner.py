"""
Calibration experiment runner.

Poses every question in a dataset to a subject, collects (confidence, correctness)
pairs, and computes a full calibration report. Optionally logs each item and the
summary through the harness ``ExperimentTracker`` for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from harness import ExperimentConfig, ExperimentResult, get_tracker

from .dataset import Dataset, load_dataset
from .metrics import CalibrationReport, calibration_report
from .subjects import AnswerRecord

Subject = object  # any object exposing .answer(question) -> AnswerRecord


@dataclass
class CalibrationRun:
    """Result of running a subject over a dataset."""

    dataset_name: str
    n_bins: int
    records: List[AnswerRecord]
    report: CalibrationReport

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "n_bins": self.n_bins,
            "report": self.report.to_dict(),
            "records": [
                {
                    "id": r.question_id,
                    "question": r.question,
                    "answer": r.answer,
                    "confidence": r.confidence,
                    "correct": r.correct,
                }
                for r in self.records
            ],
        }


def run_calibration(
    subject: Subject,
    dataset: Optional[Dataset] = None,
    n_bins: int = 10,
    track: bool = False,
    experiment_name: str = "calibration",
    provider_label: str = "subject",
    model_label: str = "subject",
) -> CalibrationRun:
    """
    Run ``subject`` over ``dataset`` (defaults to the bundled factual_qa set) and
    return a CalibrationRun.

    Args:
        subject: anything with ``.answer(question) -> AnswerRecord``.
        dataset: questions to ask; defaults to the bundled dataset.
        n_bins: number of confidence bins for ECE/MCE/reliability.
        track: if True, log per-item results and summary via the harness tracker.
        experiment_name / provider_label / model_label: tracker metadata.

    Subjects that omit a confidence on some item are skipped for calibration
    (those metrics are undefined without a stated confidence) but still answered.
    """
    if dataset is None:
        dataset = load_dataset()

    tracker = None
    config = None
    if track:
        config = ExperimentConfig(
            experiment_name=experiment_name,
            task_type="calibration",
            strategy="answer+confidence",
            provider=provider_label,
            model=model_label,
        )
        tracker = get_tracker()
        tracker.start_experiment(config)

    records: List[AnswerRecord] = []
    for q in dataset:
        rec = subject.answer(q)
        records.append(rec)
        if tracker is not None:
            tracker.log_result(
                ExperimentResult(
                    config=config,
                    task_input=q.question,
                    output=rec.raw_response,
                    latency_s=0.0,
                    eval_scores={
                        "correct": float(rec.correct),
                        "confidence": float(rec.confidence) if rec.confidence is not None else -1.0,
                    },
                    eval_metadata={"answer": rec.answer, "question_id": rec.question_id},
                )
            )

    confidences = [r.confidence for r in records if r.confidence is not None]
    correct = [r.correct for r in records if r.confidence is not None]
    report = calibration_report(confidences, correct, n_bins=n_bins)

    if tracker is not None:
        tracker.finish_experiment()

    return CalibrationRun(
        dataset_name=dataset.name,
        n_bins=n_bins,
        records=records,
        report=report,
    )
