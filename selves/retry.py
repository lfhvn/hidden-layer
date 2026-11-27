"""llm_call that rides out a model server restart."""

import time

from harness import llm_call as _call


def llm_call(*args, **kwargs):
    for attempt in range(10):
        try:
            return _call(*args, **kwargs)
        except RuntimeError:
            if attempt == 9:
                raise
            time.sleep(min(10 * (attempt + 1), 60))
