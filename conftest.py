"""Root pytest configuration.

Ensures the repository root is on ``sys.path`` so test suites can ``import harness``
(and the research packages) regardless of how pytest is invoked — ``pytest``,
``python -m pytest``, an IDE runner, or collecting several suites at once. Without
this, bare ``pytest`` does not add the CWD to ``sys.path`` and imports fail.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
