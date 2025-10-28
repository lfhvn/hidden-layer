"""
Benchmark Dataset Loaders for SELPHI

This module provides loaders for established Theory of Mind benchmark datasets:
- ToMBench: 2,860 samples across 8 ToM tasks
- OpenToM: 696 narratives with 16,008 questions
- SocialIQA: 38,000 QA pairs on social reasoning

Usage:
    >>> from selphi.benchmarks import load_tombench, load_opentom, load_socialiqa
    >>> scenarios = load_tombench()
    >>> # Use with existing SELPHI evaluation pipeline
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from selphi.scenarios import ToMScenario, ToMType


@dataclass
class BenchmarkDataset:
    """A benchmark dataset with metadata"""
    name: str
    source: str
    scenarios: List[ToMScenario]
    total_count: int
    metadata: Dict[str, Any]


def load_tombench(
    data_path: Optional[str] = None,
    tasks: Optional[List[str]] = None
) -> BenchmarkDataset:
    """
    Load ToMBench dataset.

    ToMBench (ACL 2024) contains 2,860 testing samples across 8 ToM tasks
    and 31 abilities in social cognition.

    Installation:
        git clone https://github.com/zhchen18/ToMBench.git
        # Data will be in ./ToMBench/data/

    Args:
        data_path: Path to ToMBench data directory (default: ./ToMBench/data/)
        tasks: Optional list of specific tasks to load

    Returns:
        BenchmarkDataset with ToMScenario objects

    Note:
        ToMBench should be used for evaluation only, not training.
    """
    if data_path is None:
        data_path = "./ToMBench/data/"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"ToMBench data not found at {data_path}. "
            "Please clone the repository:\n"
            "git clone https://github.com/zhchen18/ToMBench.git"
        )

    scenarios = []

    # ToMBench has JSONL files for each task
    jsonl_files = [f for f in os.listdir(data_path) if f.endswith('.jsonl')]

    for jsonl_file in jsonl_files:
        task_name = jsonl_file.replace('.jsonl', '')

        # Skip if specific tasks requested and this isn't one
        if tasks and task_name not in tasks:
            continue

        file_path = os.path.join(data_path, jsonl_file)

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())

                # Convert ToMBench item to ToMScenario
                scenario = _tombench_to_scenario(item, task_name)
                scenarios.append(scenario)

    return BenchmarkDataset(
        name="ToMBench",
        source="https://github.com/zhchen18/ToMBench",
        scenarios=scenarios,
        total_count=len(scenarios),
        metadata={
            "citation": "Chen et al., ACL 2024",
            "tasks_loaded": tasks or "all",
            "note": "Use for evaluation only, not training"
        }
    )


def load_opentom(
    data_path: Optional[str] = None,
    include_long: bool = True,
    question_types: Optional[List[str]] = None
) -> BenchmarkDataset:
    """
    Load OpenToM dataset.

    OpenToM contains 696 narratives with 16,008 questions testing
    theory-of-mind reasoning with explicit character personalities.

    Installation:
        git clone https://github.com/seacowx/OpenToM.git
        # Data will be in ./OpenToM/

    Or use HuggingFace:
        from datasets import load_dataset
        dataset = load_dataset("SeacowX/OpenToM")

    Args:
        data_path: Path to OpenToM directory (default: ./OpenToM/)
        include_long: Include long narratives (default: True)
        question_types: Optional filter for question types
                       (location, multihop, attitude)

    Returns:
        BenchmarkDataset with ToMScenario objects

    Note:
        OpenToM should NOT be used for training or fine-tuning.
    """
    if data_path is None:
        data_path = "./OpenToM/"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"OpenToM data not found at {data_path}. "
            "Please clone the repository:\n"
            "git clone https://github.com/seacowx/OpenToM.git"
        )

    scenarios = []

    # Load main dataset
    main_file = os.path.join(data_path, "opentom.json")
    if os.path.exists(main_file):
        with open(main_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                scenario = _opentom_to_scenario(item)
                if question_types is None or scenario.metadata.get('question_type') in question_types:
                    scenarios.append(scenario)

    # Load long narratives if requested
    if include_long:
        long_file = os.path.join(data_path, "opentom_long.json")
        if os.path.exists(long_file):
            with open(long_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    scenario = _opentom_to_scenario(item)
                    scenario.metadata['narrative_length'] = 'long'
                    if question_types is None or scenario.metadata.get('question_type') in question_types:
                        scenarios.append(scenario)

    return BenchmarkDataset(
        name="OpenToM",
        source="https://github.com/seacowx/OpenToM",
        scenarios=scenarios,
        total_count=len(scenarios),
        metadata={
            "citation": "2024",
            "include_long": include_long,
            "question_types": question_types or "all",
            "note": "Do NOT use for training or fine-tuning"
        }
    )


def load_socialiqa(
    split: str = "validation",
    max_samples: Optional[int] = None,
    use_huggingface: bool = True
) -> BenchmarkDataset:
    """
    Load SocialIQA dataset.

    SocialIQA contains 38,000 multiple choice questions about social
    situations, testing commonsense reasoning and theory of mind.

    Installation (HuggingFace):
        pip install datasets
        from datasets import load_dataset
        dataset = load_dataset("allenai/social_i_qa")

    Args:
        split: Dataset split to load ("train", "validation")
        max_samples: Optional limit on number of samples
        use_huggingface: Use HuggingFace datasets library (recommended)

    Returns:
        BenchmarkDataset with ToMScenario objects

    Note:
        Requires `datasets` package: pip install datasets
    """
    if use_huggingface:
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library required. "
                "Install with: pip install datasets"
            )

        dataset = hf_load_dataset("allenai/social_i_qa", split=split)

        scenarios = []
        count = 0

        for item in dataset:
            if max_samples and count >= max_samples:
                break

            scenario = _socialiqa_to_scenario(item)
            scenarios.append(scenario)
            count += 1

        return BenchmarkDataset(
            name="SocialIQA",
            source="https://huggingface.co/datasets/allenai/social_i_qa",
            scenarios=scenarios,
            total_count=len(scenarios),
            metadata={
                "citation": "Sap et al., 2019",
                "split": split,
                "full_dataset_size": 38000
            }
        )
    else:
        raise NotImplementedError(
            "Non-HuggingFace loading not yet implemented. "
            "Please use use_huggingface=True"
        )


# ============================================================================
# Conversion Helpers
# ============================================================================

def _tombench_to_scenario(item: Dict[str, Any], task_name: str) -> ToMScenario:
    """Convert ToMBench item to ToMScenario"""

    # Map ToMBench tasks to our ToMType
    task_mapping = {
        "false_belief": ToMType.FALSE_BELIEF,
        "knowledge_attribution": ToMType.KNOWLEDGE_ATTRIBUTION,
        "perspective": ToMType.PERSPECTIVE_TAKING,
        "belief_update": ToMType.BELIEF_UPDATING,
        "second_order": ToMType.SECOND_ORDER_BELIEF,
    }

    tom_type = task_mapping.get(task_name.lower(), ToMType.KNOWLEDGE_ATTRIBUTION)

    # Extract fields (adjust based on actual ToMBench structure)
    context = item.get('context', '')
    question = item.get('question', '')
    choices = item.get('choices', [])
    answer_key = item.get('answer_key', 'A')

    # Find correct answer
    answer_idx = ord(answer_key) - ord('A')
    correct_answer = choices[answer_idx] if answer_idx < len(choices) else ""

    return ToMScenario(
        name=f"tombench_{task_name}_{item.get('id', '')}",
        tom_type=tom_type,
        setup=context,
        events=[],  # ToMBench may not have explicit events
        test_questions=[question],
        correct_answers=[correct_answer],
        reasoning=f"ToMBench {task_name} task",
        difficulty="medium",
        metadata={
            "source": "ToMBench",
            "task": task_name,
            "id": item.get('id'),
            "choices": choices,
            "answer_key": answer_key
        }
    )


def _opentom_to_scenario(item: Dict[str, Any]) -> ToMScenario:
    """Convert OpenToM item to ToMScenario"""

    # Extract narrative and question
    narrative = item.get('narrative', '')
    question = item.get('question', '')
    choices = item.get('choices', [])
    answer = item.get('answer', '')
    question_type = item.get('question_type', '')

    # Map question types to ToMType
    type_mapping = {
        "location": ToMType.KNOWLEDGE_ATTRIBUTION,
        "multihop": ToMType.SECOND_ORDER_BELIEF,
        "attitude": ToMType.PERSPECTIVE_TAKING,
    }

    tom_type = type_mapping.get(question_type.lower(), ToMType.KNOWLEDGE_ATTRIBUTION)

    return ToMScenario(
        name=f"opentom_{item.get('id', '')}",
        tom_type=tom_type,
        setup=narrative,
        events=[],  # Narrative contains integrated events
        test_questions=[question],
        correct_answers=[answer],
        reasoning=f"OpenToM {question_type} question",
        difficulty="medium",
        metadata={
            "source": "OpenToM",
            "id": item.get('id'),
            "question_type": question_type,
            "choices": choices,
            "characters": item.get('characters', [])
        }
    )


def _socialiqa_to_scenario(item: Dict[str, Any]) -> ToMScenario:
    """Convert SocialIQA item to ToMScenario"""

    context = item.get('context', '')
    question = item.get('question', '')

    # SocialIQA has 3 answer choices
    choices = [
        item.get('answerA', ''),
        item.get('answerB', ''),
        item.get('answerC', '')
    ]

    # Correct answer label (1, 2, or 3)
    answer_label = item.get('label', '1')
    correct_answer = choices[int(answer_label) - 1]

    return ToMScenario(
        name=f"socialiqa_{item.get('id', '')}",
        tom_type=ToMType.PRAGMATIC_REASONING,  # Social commonsense
        setup=context,
        events=[],
        test_questions=[question],
        correct_answers=[correct_answer],
        reasoning="SocialIQA commonsense reasoning task",
        difficulty="medium",
        metadata={
            "source": "SocialIQA",
            "choices": choices,
            "label": answer_label
        }
    )


# ============================================================================
# Utility Functions
# ============================================================================

def list_available_benchmarks() -> Dict[str, Dict[str, Any]]:
    """List all available benchmarks with information"""
    return {
        "ToMBench": {
            "size": 2860,
            "source": "https://github.com/zhchen18/ToMBench",
            "citation": "Chen et al., ACL 2024",
            "installation": "git clone https://github.com/zhchen18/ToMBench.git",
            "usage": "load_tombench()",
            "note": "Use for evaluation only, not training"
        },
        "OpenToM": {
            "size": 16008,
            "narratives": 696,
            "source": "https://github.com/seacowx/OpenToM",
            "huggingface": "SeacowX/OpenToM",
            "citation": "2024",
            "installation": "git clone https://github.com/seacowx/OpenToM.git",
            "usage": "load_opentom()",
            "note": "Do NOT use for training or fine-tuning"
        },
        "SocialIQA": {
            "size": 38000,
            "source": "https://huggingface.co/datasets/allenai/social_i_qa",
            "citation": "Sap et al., 2019",
            "installation": "pip install datasets",
            "usage": "load_socialiqa()",
            "note": "Available on HuggingFace"
        }
    }


def print_benchmark_info():
    """Print information about all available benchmarks"""
    benchmarks = list_available_benchmarks()

    print("Available Theory of Mind Benchmarks for SELPHI:\n")
    print("=" * 70)

    for name, info in benchmarks.items():
        print(f"\n{name}")
        print("-" * 70)
        for key, value in info.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
