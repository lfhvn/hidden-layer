"""
Domain-specific benchmarks for ACE evaluation.

Includes math, finance, and code benchmarks.
"""

from typing import List, Dict, Any
import re

from .base import Benchmark, Task


class MathBenchmark(Benchmark):
    """
    Mathematical reasoning benchmark.

    Includes arithmetic, algebra, and word problems.
    """

    def __init__(self):
        super().__init__("Math")

    def load_tasks(self) -> List[Task]:
        """Load math tasks."""
        tasks = []

        # Arithmetic
        tasks.append(Task(
            id="arith_1",
            description="What is 456 + 789?",
            expected_output="1245",
            metadata={"split": "test", "type": "arithmetic"},
            difficulty="easy"
        ))

        tasks.append(Task(
            id="arith_2",
            description="Calculate 15 * 24.",
            expected_output="360",
            metadata={"split": "test", "type": "arithmetic"},
            difficulty="easy"
        ))

        # Word problems
        tasks.append(Task(
            id="word_1",
            description=(
                "Sarah has 12 apples. She gives 5 to John and buys 8 more. "
                "How many apples does she have now?"
            ),
            expected_output="15",
            metadata={"split": "test", "type": "word_problem"},
            difficulty="easy"
        ))

        tasks.append(Task(
            id="word_2",
            description=(
                "A store sells notebooks for $3 each and pens for $2 each. "
                "If you buy 4 notebooks and 6 pens, how much do you spend?"
            ),
            expected_output="24",
            metadata={"split": "test", "type": "word_problem"},
            difficulty="medium"
        ))

        # Fractions
        tasks.append(Task(
            id="frac_1",
            description="What is 1/2 + 1/4?",
            expected_output="3/4",
            metadata={"split": "test", "type": "fractions"},
            difficulty="medium"
        ))

        # Percentages
        tasks.append(Task(
            id="percent_1",
            description="What is 25% of 80?",
            expected_output="20",
            metadata={"split": "test", "type": "percentages"},
            difficulty="easy"
        ))

        # Multi-step
        tasks.append(Task(
            id="multi_1",
            description=(
                "A rectangle has length 8 cm and width 5 cm. "
                "What is its perimeter and area?"
            ),
            expected_output="perimeter: 26, area: 40",
            metadata={"split": "test", "type": "multi_step"},
            difficulty="medium"
        ))

        # Algebra
        tasks.append(Task(
            id="algebra_1",
            description="Solve for x: 2x + 5 = 13",
            expected_output="4",
            metadata={"split": "test", "type": "algebra"},
            difficulty="medium"
        ))

        # Training tasks
        tasks.extend(self._create_training_tasks())

        return tasks

    def _create_training_tasks(self) -> List[Task]:
        """Create training tasks."""
        return [
            Task(
                id="train_arith_1",
                description="What is 234 + 567?",
                expected_output="801",
                metadata={"split": "train", "type": "arithmetic"},
                difficulty="easy"
            ),
            Task(
                id="train_word_1",
                description=(
                    "A classroom has 25 students. 8 are absent today. "
                    "How many students are present?"
                ),
                expected_output="17",
                metadata={"split": "train", "type": "word_problem"},
                difficulty="easy"
            ),
            Task(
                id="train_percent_1",
                description="What is 10% of 200?",
                expected_output="20",
                metadata={"split": "train", "type": "percentages"},
                difficulty="easy"
            ),
        ]

    def evaluate(self, task: Task, output: str) -> Dict[str, Any]:
        """Evaluate math output."""
        # Extract numbers from output
        output_numbers = re.findall(r'\d+(?:\.\d+)?', output)
        expected = str(task.expected_output)

        # Check if expected answer is in output
        correct = expected in output or expected in output_numbers

        # For fraction answers
        if '/' in expected:
            correct = expected in output

        # For multi-part answers
        if ',' in expected:
            parts = expected.split(',')
            correct = all(part.strip() in output for part in parts)

        feedback = "Correct" if correct else f"Expected: {task.expected_output}"

        return {
            "correct": correct,
            "feedback": feedback
        }


class FinanceBenchmark(Benchmark):
    """
    Finance domain benchmark.

    Inspired by FiNER but focused on reasoning tasks rather than NER.
    """

    def __init__(self):
        super().__init__("Finance")

    def load_tasks(self) -> List[Task]:
        """Load finance tasks."""
        tasks = []

        # Financial calculations
        tasks.append(Task(
            id="fin_calc_1",
            description=(
                "A stock is priced at $50. It increases by 20% in one day. "
                "What is the new price?"
            ),
            expected_output="60",
            metadata={"split": "test", "type": "calculation"},
            difficulty="easy"
        ))

        # ROI calculation
        tasks.append(Task(
            id="roi_1",
            description=(
                "You invest $1000 and receive $1200 back. "
                "What is your return on investment (ROI) as a percentage?"
            ),
            expected_output="20",
            metadata={"split": "test", "type": "roi"},
            difficulty="medium"
        ))

        # Interest calculation
        tasks.append(Task(
            id="interest_1",
            description=(
                "You deposit $5000 in a bank account with 3% annual interest. "
                "How much interest do you earn in one year?"
            ),
            expected_output="150",
            metadata={"split": "test", "type": "interest"},
            difficulty="easy"
        ))

        # Budget analysis
        tasks.append(Task(
            id="budget_1",
            description=(
                "Monthly income: $4000. Expenses: rent $1200, food $600, "
                "utilities $200, entertainment $300. What is the savings amount?"
            ),
            expected_output="1700",
            metadata={"split": "test", "type": "budget"},
            difficulty="medium"
        ))

        # Financial terminology
        tasks.append(Task(
            id="term_1",
            description=(
                "A company's revenue is $10M and costs are $7M. "
                "What is the profit margin as a percentage?"
            ),
            expected_output="30",
            metadata={"split": "test", "type": "terminology"},
            difficulty="medium"
        ))

        # Investment comparison
        tasks.append(Task(
            id="invest_1",
            description=(
                "Investment A: $100 upfront, returns $120 after 1 year. "
                "Investment B: $100 upfront, returns $115 after 6 months. "
                "Which has better annualized return?"
            ),
            expected_output="B",
            metadata={"split": "test", "type": "comparison"},
            difficulty="hard"
        ))

        # Entity recognition (inspired by FiNER)
        tasks.append(Task(
            id="entity_1",
            description=(
                "Extract the organizations from: "
                "'Apple Inc. reported earnings that beat Goldman Sachs estimates.'"
            ),
            expected_output="Apple Inc., Goldman Sachs",
            metadata={"split": "test", "type": "entity"},
            difficulty="medium"
        ))

        return tasks

    def evaluate(self, task: Task, output: str) -> Dict[str, Any]:
        """Evaluate finance output."""
        output_clean = output.lower().strip()
        expected = str(task.expected_output).lower().strip()

        # Extract numbers if expected is numeric
        if expected.replace('.', '').isdigit():
            output_numbers = re.findall(r'\d+(?:\.\d+)?', output)
            correct = expected in output_numbers or expected in output_clean
        else:
            # For entity extraction, check if all expected entities are present
            if ',' in expected:
                entities = [e.strip() for e in expected.split(',')]
                correct = all(entity.lower() in output_clean for entity in entities)
            else:
                correct = expected in output_clean

        feedback = "Correct" if correct else f"Expected: {task.expected_output}"

        return {
            "correct": correct,
            "feedback": feedback
        }


class CodeBenchmark(Benchmark):
    """
    Code understanding and generation benchmark.
    """

    def __init__(self):
        super().__init__("Code")

    def load_tasks(self) -> List[Task]:
        """Load code tasks."""
        tasks = []

        # Code understanding
        tasks.append(Task(
            id="code_understand_1",
            description=(
                "What does this function do?\n"
                "```python\n"
                "def f(x):\n"
                "    return x * 2\n"
                "```"
            ),
            expected_output="doubles",
            metadata={"split": "test", "type": "understanding"},
            difficulty="easy"
        ))

        # Bug identification
        tasks.append(Task(
            id="bug_1",
            description=(
                "Find the bug:\n"
                "```python\n"
                "def sum_list(nums):\n"
                "    total = 0\n"
                "    for i in range(len(nums)):\n"
                "        total += i\n"
                "    return total\n"
                "```"
            ),
            expected_output="total += nums[i]",
            metadata={"split": "test", "type": "bug_finding"},
            difficulty="medium"
        ))

        # Code completion
        tasks.append(Task(
            id="complete_1",
            description=(
                "Complete this function to check if a number is even:\n"
                "```python\n"
                "def is_even(n):\n"
                "    # TODO\n"
                "```"
            ),
            expected_output="return n % 2 == 0",
            metadata={"split": "test", "type": "completion"},
            difficulty="easy"
        ))

        # Algorithm selection
        tasks.append(Task(
            id="algo_1",
            description=(
                "You need to find if an element exists in a sorted list. "
                "Which algorithm is most efficient: linear search or binary search?"
            ),
            expected_output="binary search",
            metadata={"split": "test", "type": "algorithm"},
            difficulty="easy"
        ))

        # Complexity analysis
        tasks.append(Task(
            id="complexity_1",
            description=(
                "What is the time complexity of this code?\n"
                "```python\n"
                "for i in range(n):\n"
                "    for j in range(n):\n"
                "        print(i, j)\n"
                "```"
            ),
            expected_output="O(n^2)",
            metadata={"split": "test", "type": "complexity"},
            difficulty="medium"
        ))

        return tasks

    def evaluate(self, task: Task, output: str) -> Dict[str, Any]:
        """Evaluate code output."""
        output_clean = output.lower().strip()
        expected = str(task.expected_output).lower().strip()

        # Fuzzy matching for code-related answers
        correct = expected in output_clean

        # Special handling for code snippets
        if "return" in expected or "%" in expected:
            # Remove whitespace for comparison
            output_no_space = re.sub(r'\s+', '', output_clean)
            expected_no_space = re.sub(r'\s+', '', expected)
            correct = expected_no_space in output_no_space

        feedback = "Correct" if correct else f"Expected: {task.expected_output}"

        return {
            "correct": correct,
            "feedback": feedback
        }
