"""
Agent benchmarks for ACE evaluation.

Includes multi-step reasoning, tool use, and planning tasks.
"""

from typing import List, Dict, Any
import json
import re

from .base import Benchmark, Task


class SimpleAgentBenchmark(Benchmark):
    """
    Simple agent benchmark with multi-step reasoning tasks.

    Inspired by AppWorld but simplified for initial testing.
    """

    def __init__(self):
        super().__init__("SimpleAgent")

    def load_tasks(self) -> List[Task]:
        """Load simple agent tasks."""
        tasks = []

        # Task 1: Email management
        tasks.append(Task(
            id="email_1",
            description=(
                "You have 3 unread emails. The first is from your boss asking for a report by EOD. "
                "The second is from marketing about a meeting at 3pm. The third is spam. "
                "What should you prioritize?"
            ),
            expected_output="boss report",
            metadata={"split": "test", "category": "prioritization"},
            difficulty="easy"
        ))

        # Task 2: Calendar scheduling
        tasks.append(Task(
            id="calendar_1",
            description=(
                "You have a meeting at 2pm that will last 1 hour. Someone wants to schedule "
                "a call at 2:30pm. Can you accept this request?"
            ),
            expected_output="no",
            metadata={"split": "test", "category": "scheduling"},
            difficulty="easy"
        ))

        # Task 3: File organization
        tasks.append(Task(
            id="file_1",
            description=(
                "You have files named: report_draft.pdf, report_final.pdf, notes.txt, "
                "and data.csv. Which file should you send to your manager for review?"
            ),
            expected_output="report_final.pdf",
            metadata={"split": "test", "category": "file_management"},
            difficulty="easy"
        ))

        # Task 4: Multi-step planning
        tasks.append(Task(
            id="planning_1",
            description=(
                "You need to: (1) review a document, (2) send feedback to Alice, "
                "(3) schedule a follow-up meeting. The document review takes 30 minutes, "
                "Alice is only available until noon, and it's currently 11am. "
                "What order should you do these tasks?"
            ),
            expected_output="send feedback, schedule meeting, review document",
            metadata={"split": "test", "category": "planning"},
            difficulty="medium"
        ))

        # Task 5: Tool selection
        tasks.append(Task(
            id="tool_1",
            description=(
                "You need to analyze sales data from the past quarter. "
                "You have access to: Excel, Email, Calendar, and Calculator. "
                "Which tool should you use?"
            ),
            expected_output="excel",
            metadata={"split": "test", "category": "tool_selection"},
            difficulty="easy"
        ))

        # Task 6: Information retrieval
        tasks.append(Task(
            id="retrieval_1",
            description=(
                "You need to find the deadline for the Q3 budget report. "
                "Where should you look: (a) Email, (b) Calendar, (c) Recent files, (d) All of the above?"
            ),
            expected_output="all of the above",
            metadata={"split": "test", "category": "retrieval"},
            difficulty="medium"
        ))

        # Task 7: Complex multi-step
        tasks.append(Task(
            id="complex_1",
            description=(
                "A client emails asking for a project update. You need to: "
                "1) Check project status in the tracking system, "
                "2) Review the latest milestone completion, "
                "3) Draft a response email, "
                "4) Get approval from your manager, "
                "5) Send the email. "
                "The client email says 'urgent'. What should you do first?"
            ),
            expected_output="check project status",
            metadata={"split": "test", "category": "workflow"},
            difficulty="hard"
        ))

        # Task 8: Constraint satisfaction
        tasks.append(Task(
            id="constraint_1",
            description=(
                "You need to schedule a 2-hour meeting with 4 people. "
                "Person A is available 9-11am and 2-4pm. "
                "Person B is available 10am-12pm and 3-5pm. "
                "Person C is available 9am-1pm. "
                "Person D is available 1-5pm. "
                "When can you schedule the meeting?"
            ),
            expected_output="3-5pm",
            metadata={"split": "test", "category": "constraint_satisfaction"},
            difficulty="hard"
        ))

        # Add training tasks
        tasks.extend(self._create_training_tasks())

        return tasks

    def _create_training_tasks(self) -> List[Task]:
        """Create training tasks."""
        training_tasks = []

        # Similar structure but different content
        training_tasks.append(Task(
            id="train_email_1",
            description=(
                "You have 2 emails: one from HR about benefits enrollment (due tomorrow) "
                "and one from a colleague about lunch plans. Which is more important?"
            ),
            expected_output="hr benefits",
            metadata={"split": "train", "category": "prioritization"},
            difficulty="easy"
        ))

        training_tasks.append(Task(
            id="train_calendar_1",
            description=(
                "Your calendar shows a dentist appointment at 10am lasting 1 hour. "
                "Can you join a team standup at 10:30am?"
            ),
            expected_output="no",
            metadata={"split": "train", "category": "scheduling"},
            difficulty="easy"
        ))

        training_tasks.append(Task(
            id="train_file_1",
            description=(
                "You have: budget_2024_v1.xlsx, budget_2024_v2.xlsx, budget_2024_final.xlsx. "
                "Which should you submit to finance?"
            ),
            expected_output="budget_2024_final.xlsx",
            metadata={"split": "train", "category": "file_management"},
            difficulty="easy"
        ))

        return training_tasks

    def evaluate(self, task: Task, output: str) -> Dict[str, Any]:
        """
        Evaluate agent task output.

        Uses fuzzy matching since exact matches are unrealistic.
        """
        output_lower = output.lower().strip()
        expected_lower = str(task.expected_output).lower().strip()

        # Check if expected answer is in output
        correct = expected_lower in output_lower

        # Additional fuzzy matching for common variations
        if not correct:
            # Handle variations like "no" vs "cannot" vs "can't"
            if expected_lower == "no":
                correct = any(word in output_lower for word in ["no", "cannot", "can't", "unable"])
            elif expected_lower == "yes":
                correct = any(word in output_lower for word in ["yes", "can", "accept"])

        feedback = "Correct" if correct else f"Expected: {task.expected_output}, Got: {output[:100]}"

        return {
            "correct": correct,
            "feedback": feedback
        }


class ToolUseBenchmark(Benchmark):
    """
    Benchmark for tool use and function calling.
    """

    def __init__(self):
        super().__init__("ToolUse")

    def load_tasks(self) -> List[Task]:
        """Load tool use tasks."""
        tasks = []

        # Task 1: Calculator
        tasks.append(Task(
            id="calc_1",
            description="What is 234 * 567? Use the calculator tool.",
            expected_output="132678",
            metadata={"split": "test", "tool": "calculator"},
            difficulty="easy"
        ))

        # Task 2: Search
        tasks.append(Task(
            id="search_1",
            description="Find the capital of France using the search tool.",
            expected_output="Paris",
            metadata={"split": "test", "tool": "search"},
            difficulty="easy"
        ))

        # Task 3: File read
        tasks.append(Task(
            id="file_read_1",
            description="Read the contents of 'data.txt' which contains: 'Hello World'",
            expected_output="Hello World",
            metadata={"split": "test", "tool": "file_read"},
            difficulty="easy"
        ))

        # Task 4: Multi-tool
        tasks.append(Task(
            id="multi_tool_1",
            description=(
                "Calculate 15 * 23, then search for that number of days in hours. "
                "How many hours is it?"
            ),
            expected_output="8280",  # 345 days * 24 hours
            metadata={"split": "test", "tool": "multi"},
            difficulty="hard"
        ))

        return tasks

    def evaluate(self, task: Task, output: str) -> Dict[str, Any]:
        """Evaluate tool use output."""
        output_clean = output.lower().strip()
        expected = str(task.expected_output).lower().strip()

        # Extract numbers from output if expected is numeric
        if expected.isdigit():
            numbers = re.findall(r'\d+', output)
            correct = expected in numbers
        else:
            correct = expected in output_clean

        feedback = "Correct" if correct else f"Expected: {task.expected_output}"

        return {
            "correct": correct,
            "feedback": feedback
        }


class ReasoningBenchmark(Benchmark):
    """
    Multi-step reasoning benchmark.
    """

    def __init__(self):
        super().__init__("Reasoning")

    def load_tasks(self) -> List[Task]:
        """Load reasoning tasks."""
        tasks = []

        # Logical reasoning
        tasks.append(Task(
            id="logic_1",
            description=(
                "All roses are flowers. Some flowers are red. "
                "Can we conclude that some roses are red?"
            ),
            expected_output="no",
            metadata={"split": "test", "type": "logic"},
            difficulty="medium"
        ))

        # Mathematical reasoning
        tasks.append(Task(
            id="math_reason_1",
            description=(
                "A train travels 60 km/h for 2 hours, then 80 km/h for 1.5 hours. "
                "What is the total distance traveled?"
            ),
            expected_output="240",  # 120 + 120
            metadata={"split": "test", "type": "math"},
            difficulty="medium"
        ))

        # Causal reasoning
        tasks.append(Task(
            id="causal_1",
            description=(
                "The ground is wet. It could be because: "
                "(a) it rained, (b) a sprinkler was on, (c) someone washed their car. "
                "Which is MOST likely if the sky is cloudy?"
            ),
            expected_output="it rained",
            metadata={"split": "test", "type": "causal"},
            difficulty="easy"
        ))

        # Planning reasoning
        tasks.append(Task(
            id="planning_reason_1",
            description=(
                "You need to bake a cake. The steps are: "
                "1) Preheat oven (10 min), 2) Mix ingredients (15 min), "
                "3) Bake (30 min), 4) Cool (20 min). "
                "If you start at 2:00 PM, when will the cake be ready?"
            ),
            expected_output="3:15",  # 75 minutes total
            metadata={"split": "test", "type": "planning"},
            difficulty="medium"
        ))

        return tasks

    def evaluate(self, task: Task, output: str) -> Dict[str, Any]:
        """Evaluate reasoning output."""
        output_clean = output.lower().strip()
        expected = str(task.expected_output).lower().strip()

        correct = expected in output_clean

        feedback = "Correct" if correct else f"Expected: {task.expected_output}"

        return {
            "correct": correct,
            "feedback": feedback
        }
