#!/usr/bin/env python3
"""
Update notebook configuration cells to include MLX provider option.

This script adds or updates configuration cells in Jupyter notebooks to ensure
MLX is documented as a provider option.
"""

import json
import sys
from pathlib import Path

# Configuration cell template
CONFIG_CELL_TEMPLATE = """# ═══════════════════════════════════════════════════════════════════════
# ⚙️  CONFIGURATION - EDIT THIS CELL TO CHANGE SETTINGS
# ═══════════════════════════════════════════════════════════════════════

PROVIDER = 'ollama'  # Options: 'mlx', 'ollama', 'anthropic', 'openai'
MODEL = None         # Examples:
                     # MLX: 'mlx-community/Llama-3.2-3B-Instruct-4bit'
                     # Ollama: 'llama3.2:latest', 'qwen2.5:latest'
                     # Anthropic: 'claude-3-5-sonnet-20241022'
                     # OpenAI: 'gpt-4o'

# Use defaults if not specified
if MODEL is None:
    MODEL = DEFAULT_MODEL

# ═══════════════════════════════════════════════════════════════════════

print("Current Configuration:")
print("="*70)
print(f"📍 Provider: {PROVIDER}")
print(f"🤖 Model: {MODEL or '(default for provider)'}")
print("="*70)"""


def has_config_cell(notebook):
    """Check if notebook already has a configuration cell."""
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if '⚙️  CONFIGURATION' in source or 'EDIT THIS CELL TO CHANGE SETTINGS' in source:
                return True
    return False


def has_default_imports(notebook):
    """Check if notebook imports DEFAULT_MODEL and DEFAULT_PROVIDER."""
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'from harness.defaults import' in source and 'DEFAULT_' in source:
                return True
    return False


def update_notebook(notebook_path):
    """Update a single notebook with proper configuration cell."""
    print(f"\nProcessing: {notebook_path}")

    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    # Skip if already has config cell
    if has_config_cell(notebook):
        print(f"  ✓ Already has configuration cell")
        return False

    # Skip if doesn't use harness defaults
    if not has_default_imports(notebook):
        print(f"  ⊘ Doesn't use DEFAULT_MODEL/DEFAULT_PROVIDER")
        return False

    # Find the imports cell (first code cell that imports from harness)
    import_cell_index = None
    for i, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'from harness' in source and 'import' in source:
                import_cell_index = i
                break

    if import_cell_index is None:
        print(f"  ✗ Could not find imports cell")
        return False

    # Clean up the imports cell (remove configuration instructions)
    import_cell = notebook['cells'][import_cell_index]
    source_lines = import_cell.get('source', [])

    # Remove lines that show configuration instructions
    cleaned_lines = []
    skip_section = False
    for line in source_lines:
        line_str = line if isinstance(line, str) else ''

        # Start skipping at configuration instruction section
        if 'NOTEBOOK CONFIGURATION' in line_str or 'TO CHANGE:' in line_str:
            skip_section = True

        # Stop skipping after the separator line
        if skip_section and line_str.strip().startswith('print("="'):
            if 'NOTEBOOK CONFIGURATION' not in line_str:
                skip_section = False
            continue

        if not skip_section:
            cleaned_lines.append(line)

    # Add simple success message if not present
    if cleaned_lines and 'print("✓' not in ''.join(cleaned_lines):
        if cleaned_lines[-1].strip() and not cleaned_lines[-1].endswith('\n'):
            cleaned_lines.append('\n')
        cleaned_lines.append('\nprint("✓ Imports successful")\n')

    import_cell['source'] = cleaned_lines

    # Create configuration cell
    config_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': CONFIG_CELL_TEMPLATE.split('\n')
    }

    # Insert configuration cell after imports
    notebook['cells'].insert(import_cell_index + 1, config_cell)

    # Save updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)

    print(f"  ✓ Added configuration cell")
    return True


def main():
    """Update all notebooks that need configuration cells."""
    repo_root = Path(__file__).parent.parent

    # Notebooks to update
    notebooks = [
        # Multi-agent (skip 00_quickstart.ipynb - already has it)
        'communication/multi-agent/notebooks/01_baseline_experiments.ipynb',
        'communication/multi-agent/notebooks/02_debate_experiments.ipynb',
        'communication/multi-agent/notebooks/02_multi_agent_comparison.ipynb',
        'communication/multi-agent/notebooks/03_introspection_experiments.ipynb',
        'communication/multi-agent/notebooks/04_api_introspection.ipynb',
        'communication/multi-agent/notebooks/09_reasoning_and_rationale.ipynb',
        # CRIT
        'communication/multi-agent/notebooks/crit/01_basic_critique_experiments.ipynb',
        'communication/multi-agent/notebooks/crit/02_uicrit_benchmark.ipynb',
        # SELPHI
        'theory-of-mind/selphi/notebooks/01_basic_tom_tests.ipynb',
        'theory-of-mind/selphi/notebooks/02_benchmark_evaluation.ipynb',
        # Steerability
        'alignment/steerability/notebooks/01_dashboard_testing.ipynb',
    ]

    updated_count = 0
    for notebook_rel_path in notebooks:
        notebook_path = repo_root / notebook_rel_path
        if notebook_path.exists():
            if update_notebook(notebook_path):
                updated_count += 1
        else:
            print(f"\n✗ Not found: {notebook_path}")

    print(f"\n{'='*70}")
    print(f"Updated {updated_count} notebooks")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
