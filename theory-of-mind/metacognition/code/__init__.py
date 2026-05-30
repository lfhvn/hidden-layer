"""
Metacognition Research Project

Measuring whether models *know what they know*: metacognitive calibration
(does stated confidence track correctness?) and introspective self-prediction
(can a model forecast its own answer?).

Core pieces:
- metrics:        Brier, ECE, MCE, AUROC, reliability diagrams (pure stdlib)
- dataset:        short-answer questions with deterministic scoring
- subjects:       ModelSubject (real LLM) and SimulatedSubject (deterministic)
- runner:         run_calibration -> CalibrationRun
- selfprediction: self-prediction / introspection task
- report:         JSON + optional matplotlib reliability plots

Uses the harness for the LLM provider abstraction (including the offline ``sim``
provider) and experiment tracking.
"""

from .dataset import Dataset, Question, load_dataset, make_arithmetic_dataset, normalize
from .metrics import (
    CalibrationReport,
    ReliabilityBin,
    accuracy,
    ascii_reliability_diagram,
    brier_score,
    calibration_report,
    confidence_auroc,
    expected_calibration_error,
    maximum_calibration_error,
    mean_confidence,
    reliability_bins,
)
from .report import plot_reliability, save_json
from .runner import CalibrationRun, run_calibration
from .selfprediction import (
    ModelIntrospector,
    SelfPredictionRecord,
    SelfPredictionRun,
    SimulatedIntrospector,
    answers_match,
    build_prediction_prompt,
    parse_prediction,
    run_self_prediction,
)
from .subjects import (
    CALIBRATION_PROFILES,
    AnswerRecord,
    ModelSubject,
    SimulatedSubject,
    build_prompt,
    parse_answer_and_confidence,
)

__version__ = "0.1.0"

__all__ = [
    # dataset
    "Dataset",
    "Question",
    "load_dataset",
    "make_arithmetic_dataset",
    "normalize",
    # metrics
    "CalibrationReport",
    "ReliabilityBin",
    "accuracy",
    "mean_confidence",
    "brier_score",
    "reliability_bins",
    "expected_calibration_error",
    "maximum_calibration_error",
    "confidence_auroc",
    "calibration_report",
    "ascii_reliability_diagram",
    # subjects
    "AnswerRecord",
    "ModelSubject",
    "SimulatedSubject",
    "CALIBRATION_PROFILES",
    "build_prompt",
    "parse_answer_and_confidence",
    # runner
    "CalibrationRun",
    "run_calibration",
    # self-prediction
    "SelfPredictionRecord",
    "SelfPredictionRun",
    "ModelIntrospector",
    "SimulatedIntrospector",
    "run_self_prediction",
    "answers_match",
    "build_prediction_prompt",
    "parse_prediction",
    # report
    "save_json",
    "plot_reliability",
]
