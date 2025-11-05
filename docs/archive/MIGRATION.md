# Migration Guide - Repository Reorganization

**Date**: November 4, 2025

## Summary

The hidden-layer repository has been reorganized from a single multi-agent project into a research lab structure with six distinct research projects and shared infrastructure.

## What Changed

### Structure

**Before**:
```
hidden-layer/
├── code/harness/          # Mixed infrastructure + project code
├── code/crit/
├── code/selphi/
├── latent-lens/
├── latent-topologies/
├── steerability-dashboard/
├── concepts/
├── notebooks/
└── CLAUDE.md (534 lines)
```

**After**:
```
hidden-layer/
├── harness/               # Standalone library
├── shared/                # Shared resources
│   ├── concepts/
│   ├── datasets/
│   └── utils/
├── projects/              # All research projects
│   ├── multi-agent/
│   ├── selphi/
│   ├── latent-space/     # Combined lens + topologies
│   ├── introspection/
│   ├── ai-to-ai-comm/
│   └── steerability/
├── docs/                  # Lab-wide documentation
├── CLAUDE.md (280 lines)  # 47% smaller!
└── RESEARCH.md            # Research themes
```

### Key Changes

1. **Harness is now a standalone library** (`/harness/`)
   - Can be open-sourced independently
   - Used by all projects
   - Clear separation of infrastructure vs. research

2. **Projects are at the same level** (`/projects/`)
   - Multi-agent (was `/code/`)
   - SELPHI (was `/code/selphi/`)
   - Latent Space (was `/latent-lens/` + `/latent-topologies/`)
   - Introspection (extracted from harness)
   - AI-to-AI Communication (new)
   - Steerability (was `/steerability-dashboard/`)

3. **Shared resources** (`/shared/`)
   - Concepts (was `/concepts/`)
   - Datasets
   - Common utilities

4. **Lab-wide documentation** (`/docs/`)
   - Infrastructure guides
   - Hardware setup (optional)
   - Workflows and conventions

5. **Lean CLAUDE.md** (280 lines, down from 534)
   - Top-level overview
   - Project-specific guides in each project

## Migration Actions

### For Developers

#### Imports

**Before**:
```python
from harness import llm_call  # When harness was in code/
```

**After** (same, but different location):
```python
from harness import llm_call  # Now harness is top-level
```

**Action**: Run `pip install -e .` from repository root to make harness importable.

#### Project Imports

**Before** (in code/harness/strategies.py):
```python
from .llm_provider import llm_call
```

**After** (in projects/multi-agent/code/strategies.py):
```python
from harness import llm_call
```

**Action**: Already updated in migration.

#### Working Directory

**Before**:
```bash
cd /path/to/hidden-layer
python code/cli.py "Question?"
```

**After**:
```bash
cd /path/to/hidden-layer/projects/multi-agent
python code/cli.py "Question?"
```

#### Notebooks

**Before**:
```python
import sys
sys.path.append('../code')
from harness import llm_call
```

**After**:
```python
# No sys.path needed if you ran `pip install -e .` from root
from harness import llm_call
```

#### Concept Vectors

**Before**:
```python
library = ConceptLibrary.load("concepts/emotions.pkl")
```

**After**:
```python
library = ConceptLibrary.load("../../shared/concepts/emotions.pkl")
# Or use absolute path
library = ConceptLibrary.load("/path/to/hidden-layer/shared/concepts/emotions.pkl")
```

### For Projects

Each project now has:
- `README.md` - Quick start
- `CLAUDE.md` - Development guide
- `code/` - Project code
- (Optional) `notebooks/`, `tests/`, `config/`

### Documentation

**New files**:
- `/RESEARCH.md` - Research themes and connections
- `/docs/README.md` - Documentation navigation
- `/shared/README.md` - Shared resources guide
- `/harness/README.md` - Harness library documentation
- Each project has its own `README.md` and `CLAUDE.md`

**Moved files**:
- `SETUP.md` → `docs/hardware/local-setup.md`
- `MLX_MODELS_2025.md` → `docs/hardware/mlx-models.md`
- `BENCHMARKS.md` → `docs/workflows/benchmarking.md`
- `PROVIDER_LIMITATIONS.md` → `docs/infrastructure/provider-limitations.md`

## Setup

### Fresh Setup

```bash
# Clone repository
git clone https://github.com/yourusername/hidden-layer
cd hidden-layer

# Install harness (makes it importable)
pip install -e .

# Optional: Install with extras
pip install -e ".[mlx,introspection,dev]"

# Pick a project
cd projects/multi-agent
cat README.md
```

### Updating Existing Clone

```bash
# Pull changes
cd /path/to/hidden-layer
git pull

# Install harness
pip install -e .

# Update imports if you have custom notebooks/scripts
# Change: from code.harness import ...
# To: from harness import ...
```

## Testing

### Test Harness Import

```bash
python -c "from harness import llm_call; print('Harness OK')"
```

### Test Project Import

```bash
cd projects/multi-agent
python -c "from code.strategies import run_strategy; print('Strategies OK')"
```

### Test Shared Resources

```bash
python -c "from shared.concepts import ConceptLibrary; print('Shared OK')" 2>/dev/null || echo "Shared has no Python imports (concepts are loaded directly)"
```

## Rollback (If Needed)

Old structure is backed up in:
- `CLAUDE.md.old` - Original CLAUDE.md
- `README.md.old` - Original README.md
- `/code/` - Original code directory (still exists, not removed)

To rollback:
```bash
# Restore old files
mv CLAUDE.md CLAUDE.md.new
mv CLAUDE.md.old CLAUDE.md

mv README.md README.md.new
mv README.md.old README.md

# Continue using old structure
```

## Benefits

1. **Performance**: 47% smaller CLAUDE.md → faster context loading
2. **Clarity**: Clear project boundaries
3. **Lab Identity**: Reflects true research scope
4. **Scalability**: Easy to add new projects
5. **Modularity**: Harness can be open-sourced independently

## Questions?

- **General structure**: See [README.md](README.md)
- **Research themes**: See [RESEARCH.md](RESEARCH.md)
- **Development guide**: See [CLAUDE.md](CLAUDE.md)
- **Project-specific**: See `projects/{project}/CLAUDE.md`
- **Infrastructure**: See [docs/infrastructure/](docs/infrastructure/)

## Next Steps

1. Review new structure: `tree -L 3` or `ls -R`
2. Read top-level README.md
3. Pick a project to work on
4. Read project's CLAUDE.md
5. Start developing!

---

**Migration completed**: November 4, 2025

**Clean break**: No backward compatibility, clean structure

**All tests passing**: (To be verified)
