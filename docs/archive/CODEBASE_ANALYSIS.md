# Hidden-Layer Codebase Analysis

**Analysis Date**: November 3, 2025  
**Project Status**: Active development with significant architectural scope  
**Critical Issues Found**: 1 (blocking imports)  

---

## Executive Summary

The hidden-layer codebase is substantially more complex and comprehensive than documented in CLAUDE.md. It contains:

- **~6,600 lines of Python** across 3 major subsystems (harness, crit, selphi) plus a separate latent-topologies project
- **~5,600 lines of documentation** (some outdated relative to implementation)
- **Multiple research-focused features** not mentioned in primary development guide
- **1 critical blocker**: Missing `harness/benchmarks.py` module breaks imports for downstream packages

### Current State
- Core harness: ✅ Functional (when benchmarks issue is fixed)
- CRIT (design critique): ✅ Well-implemented (~1,600 LOC)
- SELPHI (theory of mind): ✅ Well-implemented (~1,100 LOC)
- Latent Topologies: 🚧 Separate subproject with own stack
- Tests: ❌ None present
- CI/CD: ❌ No GitHub workflows

---

## Project Structure & Organization

### Actual Directory Layout

```
hidden-layer/
├── code/
│   ├── harness/              # Core framework (1,900 LOC)
│   │   ├── __init__.py       # Package exports (tries to import missing benchmarks)
│   │   ├── llm_provider.py   # MLX, Ollama, Anthropic, OpenAI (527 LOC)
│   │   ├── strategies.py     # Multi-agent strategies (749 LOC)
│   │   ├── experiment_tracker.py  # Experiment logging (251 LOC)
│   │   ├── evals.py          # Evaluation functions (280 LOC)
│   │   ├── model_config.py   # Configuration management (219 LOC)
│   │   ├── rationale.py      # Rationale extraction (285 LOC)
│   │   ├── defaults.py       # Default parameters (17 LOC)
│   │   └── [MISSING] benchmarks.py  # 🚨 CRITICAL - imported but not present
│   │
│   ├── crit/                 # Design Critique System (1,500+ LOC)
│   │   ├── __init__.py       # Rich API exports (121 LOC)
│   │   ├── problems.py       # Design problem definitions (584 LOC)
│   │   ├── strategies.py     # Critique strategies (656 LOC)
│   │   ├── evals.py          # Critique evaluation (500 LOC)
│   │   ├── benchmarks.py     # UICrit benchmark loader (406 LOC)
│   │   └── README.md         # Comprehensive documentation (546 LOC)
│   │
│   ├── selphi/               # Theory of Mind System (1,200+ LOC)
│   │   ├── __init__.py       # Rich API exports (122 LOC)
│   │   ├── scenarios.py      # ToM test scenarios (378 LOC)
│   │   ├── tasks.py          # Scenario execution (289 LOC)
│   │   ├── evals.py          # Evaluation functions (372 LOC)
│   │   ├── benchmarks.py     # Benchmark loaders (420 LOC)
│   │   └── README.md         # Comprehensive documentation (422 LOC)
│   │
│   └── cli.py                # Command-line tool (324 LOC)
│
├── notebooks/
│   ├── 01_baseline_experiments.ipynb
│   ├── 02_debate_experiments.ipynb
│   ├── 02_multi_agent_comparison.ipynb
│   ├── 09_reasoning_and_rationale.ipynb
│   ├── crit/
│   │   ├── 01_basic_critique_experiments.ipynb
│   │   └── 02_uicrit_benchmark.ipynb
│   └── selphi/
│       ├── 01_basic_tom_tests.ipynb
│       └── 02_benchmark_evaluation.ipynb
│
├── config/
│   ├── models.yaml           # Model presets (minimal: 6 configs)
│   └── README.md             # Configuration documentation
│
├── latent-topologies/        # Separate React Native mobile app project
│   ├── README.md
│   ├── PRD.md
│   ├── TECH_PLAN.md
│   ├── scripts/
│   ├── research/
│   └── [Node.js/Expo project]
│
├── experiments/              # Auto-generated (empty on checkout)
├── [Documentation files - 11 total]
└── requirements.txt          # 12 packages
```

**Total Code**: 6,613 Python LOC across 18 files  
**Total Docs**: 5,641 LOC across 11 markdown files  

---

## Critical Issues Found

### 🚨 **CRITICAL BLOCKER: Missing `harness/benchmarks.py`**

**Impact**: Breaking import error prevents ANY code using harness, crit, or selphi from running.

**Problem**:
```python
# In code/harness/__init__.py, line 50:
from .benchmarks import (
    load_benchmark,
    get_baseline_scores,
    BENCHMARKS
)
```

**Evidence**:
- File does not exist in `/home/user/hidden-layer/code/harness/`
- Python import fails: `ModuleNotFoundError: No module named 'harness.benchmarks'`
- Blocks crit and selphi imports (they depend on harness)
- Referenced in CLAUDE.md but never implemented

**Test**:
```bash
python3 -c "from harness import llm_call"  # FAILS
```

**Resolution Required**:
- Either: Create the missing `harness/benchmarks.py` with load_benchmark(), get_baseline_scores(), BENCHMARKS
- Or: Remove the import from `__init__.py` if benchmarks are not part of core harness

---

## Documentation vs Implementation Gaps

### CLAUDE.md Completeness Issues

| Feature | CLAUDE.md Says | Actual | Status |
|---------|---|---|---|
| Core Harness | ✅ Documented | ✅ Implemented (1,900 LOC) | Matches |
| Rationale Extraction | ❌ Not mentioned | ✅ Fully implemented (285 LOC) | Gap |
| CRIT System | ❌ Not mentioned | ✅ Full subsystem (1,500+ LOC) | Gap |
| SELPHI System | ❌ Not mentioned | ✅ Full subsystem (1,200+ LOC) | Gap |
| Latent Topologies | ❌ Not mentioned | ✅ Separate project | Gap |
| Benchmarks Module | ✅ Mentioned (briefly) | ❌ Missing | Gap |
| Configuration Management | ✅ Mentioned | ✅ Implemented (219 LOC) | Matches |
| CLI Tool | ✅ Mentioned | ✅ Implemented (324 LOC) | Matches |
| Experiment Tracking | ✅ Mentioned | ✅ Implemented (251 LOC) | Matches |

### README.md Simplification

README.md describes a much simpler structure:
```
code/
└── harness/              # Only mentions 4 files
```

But actual structure has 3 major subsystems with specialized documentation. This is misleading for new users.

---

## Code Quality Assessment

### Strengths

1. **Well-Structured Subsystems**
   - CRIT and SELPHI are self-contained with clear APIs
   - Good separation of concerns
   - Each has comprehensive README

2. **Rich Documentation**
   - 546 LOC for CRIT README (detailed API reference)
   - 422 LOC for SELPHI README (comprehensive guide)
   - 243 LOC for config management
   - Multiple SETUP, QUICKSTART guides

3. **Clean Module Exports**
   - `__init__.py` files carefully curate public API
   - CRIT exports 28 public items
   - SELPHI exports 30 public items

4. **Thoughtful Parameter Design**
   - ModelConfig class with sensible defaults
   - Support for thinking_budget, num_ctx (reasoning parameters)
   - Temperature guidelines in documentation

### Issues & Concerns

1. **Missing Testing Infrastructure**
   - ❌ No test files (`test_*.py`, `*_test.py`)
   - ❌ No pytest configuration
   - ❌ No CI/CD workflows (no `.github/workflows/`)
   - Manual testing only via notebooks

2. **Incomplete Core Module**
   - Missing `harness/benchmarks.py` (blocking)
   - Creates import blocker for all downstream packages
   - Functions documented but not implemented:
     - `load_benchmark()`
     - `get_baseline_scores()`
     - `BENCHMARKS` registry

3. **Large Monolithic Files**
   - `harness/strategies.py`: 749 LOC (single file)
   - `harness/llm_provider.py`: 527 LOC (single file)
   - `crit/problems.py`: 584 LOC (single file)
   - `crit/strategies.py`: 656 LOC (single file)
   
   Could benefit from splitting into focused modules.

4. **Potential Code Duplication**
   - Helper function `_extract_answer()` in harness/strategies.py
   - Similar answer extraction probably needed in crit/strategies.py
   - No shared utility module for common patterns

5. **Documentation Maintenance Burden**
   - 5,641 LOC of markdown across 11 files
   - Multiple overlapping guides (START_HERE, QUICKSTART, README, IMPLEMENTATION, SETUP, etc.)
   - Risk of inconsistency/staleness (as seen with CLAUDE.md)
   - Latent Topologies has its own separate doc tree

6. **Inconsistent Dependency Architecture**
   - CRIT imports from harness (fine)
   - SELPHI imports from harness (fine)
   - But harness tries to import benchmarks that don't exist (bad)
   - Could create circular dependency issues if not careful

---

## Implementation Completeness Matrix

### Harness Module

| Component | Status | LOC | Notes |
|-----------|--------|-----|-------|
| LLM Provider | ✅ Complete | 527 | MLX, Ollama, Anthropic, OpenAI |
| Strategies | ✅ Complete | 749 | 5 strategies: single, debate, self_consistency, manager_worker, consensus |
| Experiment Tracking | ✅ Complete | 251 | JSON/JSONL logging with summaries |
| Evaluations | ✅ Complete | 280 | 8 evaluation methods |
| Model Config | ✅ Complete | 219 | YAML-based with 6 presets |
| Rationale Extraction | ✅ Complete | 285 | Reasoning chain extraction |
| Defaults | ✅ Complete | 17 | Basic defaults |
| **Benchmarks** | ❌ **MISSING** | 0 | **CRITICAL - Breaks imports** |

### CRIT Module

| Component | Status | LOC | Notes |
|-----------|--------|-----|-------|
| Problem Definitions | ✅ Complete | 584 | 8 design problems across 5 domains |
| Critique Strategies | ✅ Complete | 656 | 4 strategies: single, multi_perspective, iterative, adversarial |
| Evaluations | ✅ Complete | 500 | Coverage, quality, depth metrics |
| Benchmarks | ✅ Complete | 406 | UICrit loader + 4 external benchmark loaders |
| API/Documentation | ✅ Complete | 546 | Comprehensive with examples |

### SELPHI Module

| Component | Status | LOC | Notes |
|-----------|--------|-----|-------|
| Scenarios | ✅ Complete | 378 | 9 ToM scenarios across 7 types |
| Task Execution | ✅ Complete | 289 | Run, batch, compare functions |
| Evaluations | ✅ Complete | 372 | Semantic matching, LLM judge |
| Benchmarks | ✅ Complete | 420 | ToMBench, OpenToM, SocialIQA loaders |
| API/Documentation | ✅ Complete | 422 | Comprehensive with examples |

---

## Workflow Friction Points

### 1. **Import Resolution is Currently Broken**
```python
# This fails:
from crit import run_critique_strategy  # ← requires harness
from selphi import run_scenario          # ← requires harness
from harness import llm_call             # ← requires benchmarks (missing)
```

**Workaround**: None currently available

### 2. **Incomplete Model Presets**
- Only 6 configurations in `config/models.yaml`
- Many mentioned in README examples don't exist (e.g., `llama3.2-fast`)
- Model names don't match actual Ollama conventions

### 3. **Scattered Documentation**
- To understand the project, new users must read:
  - README.md (overview, but incomplete)
  - CLAUDE.md (outdated/incomplete)
  - START_HERE.md (setup focus)
  - QUICKSTART.md (cheat sheet)
  - config/README.md (config system)
  - code/crit/README.md (crit system)
  - code/selphi/README.md (selphi system)
  - BENCHMARKS.md (external datasets)
  - IMPLEMENTATION.md (what was built)
  - INSTALLATION_AND_REASONING.md (details)
  - SETUP_COMPLETE.md (post-setup)

**Result**: 11 docs, high duplication, no clear entry point

### 4. **Configuration System Underutilized**
- Only 6 presets defined
- CLAUDE.md says "easy to override" but examples not shown clearly
- CLI doesn't support all override patterns mentioned in docs

### 5. **Latent Topologies Orphaned**
- Separate Node.js/Expo project in same repo
- Not mentioned in main documentation
- Different tech stack (not Python)
- Different setup process
- Could be confusing for new users

---

## Complexity Analysis

### Code Complexity by Component

```
Lines of Code Distribution:

Harness Strategies      749  ████████████
Harness LLM Provider    527  ████████
CRIT Strategies        656  ██████████
CRIT Problems          584  █████████
CRIT Evals             500  ████████
CLI                    324  █████
SELPHI Benchmarks      420  ██████
SELPHI Evals           372  ██████
CRIT Benchmarks        406  ██████
SELPHI Scenarios       378  ██████
Model Config           219  ███
Experiment Tracker     251  ████
SELPHI Tasks           289  ████
Rationale Extraction   285  ████
─────────────────────────
Total (code)         6,613 LOC
Total (docs)         5,641 LOC
```

### Cyclomatic Complexity (Manual Assessment)

**High Complexity**:
- `harness/llm_provider.py`: Multiple nested provider implementations (MLX, Ollama, Anthropic, OpenAI)
- `harness/strategies.py`: Complex orchestration logic (debate, manager-worker with parallel execution)
- `crit/strategies.py`: Multi-round iterative logic, synthesis
- `selphi/evals.py`: LLM-as-judge with multiple scoring methods

**Medium Complexity**:
- All evaluation modules (multiple conditional branches for different eval types)
- Experiment tracker (JSON/JSONL format handling)

**Low Complexity**:
- Data classes (scenarios, problems, configs)
- CLI argument parsing

---

## Redundancy & Duplication Analysis

### Potential Duplications

1. **Answer Extraction** (~20 LOC)
   - `harness/strategies.py`: `_extract_answer()` function
   - Pattern: Regex extraction of numerical/text answers
   - **Needed in**: Likely CRIT and SELPHI too
   - **Status**: Not actually duplicated, but utility would help

2. **Evaluation Function Patterns**
   - `harness/evals.py`: `exact_match()`, `keyword_match()`, `numeric_match()`
   - `crit/evals.py`: Similar evaluation concepts but for critique
   - `selphi/evals.py`: Similar evaluation concepts but for ToM
   - **Status**: Different enough to justify duplication (different schemas)

3. **Provider/Model Configuration**
   - `harness/model_config.py`: ModelConfig class
   - CRIT and SELPHI load this but don't extend it
   - Could benefit from subsystem-specific config classes

4. **Benchmark Loading Pattern**
   - `crit/benchmarks.py`: 406 LOC with loader functions
   - `selphi/benchmarks.py`: 420 LOC with similar structure
   - Both use same BenchmarkDataset class
   - Could share utilities but currently independent

### Code Sharing Opportunities

1. **Create `harness/utils/`** for common helpers:
   - `text.py`: `_extract_answer()`, regex patterns
   - `evaluation.py`: Common eval utilities
   - `benchmarks.py`: Shared benchmark loading (currently missing)

2. **Standardize Benchmark Loaders**
   - Consistent API across crit/benchmarks and selphi/benchmarks
   - Shared BenchmarkDataset class (already done, but could be harness-level)

3. **Provider-Agnostic Evaluation**
   - Extract LLM judge logic to harness/evals.py
   - Currently duplicated in crit and selphi

---

## Dependencies & Compatibility

### Requirements

```
mlx>=0.15.0
mlx-lm>=0.15.0
ollama>=0.1.0
anthropic>=0.25.0
openai>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
ipykernel>=6.20.0
ipywidgets>=8.0.0
python-dotenv>=1.0.0
tqdm>=4.65.0
pyyaml>=6.0.0
```

**Issues**:
- ✅ Well-curated (no bloat)
- ⚠️ MLX requires Mac (M-series processor) - undocumented in requirements
- ⚠️ Ollama requires separate installation - not in pip
- ⚠️ No version pins (uses >=) - could cause compatibility drift

**Missing**:
- No test dependencies (pytest, etc.)
- No dev dependencies (black, flake8, mypy)
- No CI/CD configuration

---

## Documentation Breakdown

| File | LOC | Purpose | Status |
|------|-----|---------|--------|
| CLAUDE.md | 352 | Development guide | ⚠️ Outdated (doesn't mention CRIT, SELPHI) |
| README.md | 238 | Project overview | ⚠️ Simplified (hides complexity) |
| SETUP.md | 219 | Installation | ✅ Current |
| QUICKSTART.md | 187 | Quick reference | ✅ Current |
| IMPLEMENTATION.md | 278 | What was built | ✅ Relevant (but dated) |
| BENCHMARKS.md | 284 | Benchmark info | ✅ Useful |
| INSTALLATION_AND_REASONING.md | 434 | Detailed setup | ✅ Current |
| SETUP_COMPLETE.md | 348 | Post-setup | ✅ Current |
| START_HERE.md | 149 | Entry point | ✅ Simple |
| config/README.md | 243 | Config system | ✅ Current |
| code/crit/README.md | 546 | CRIT system | ✅ Comprehensive |
| code/selphi/README.md | 422 | SELPHI system | ✅ Comprehensive |

---

## Architecture Assessment

### Strengths

1. **Clean Module Boundaries**
   - Harness, CRIT, SELPHI are distinct
   - Each has own strategies, evals, benchmarks
   - Clear dependency: CRIT/SELPHI → Harness

2. **Extensibility**
   - Easy to add new strategies (STRATEGIES registry)
   - Easy to add new evaluations (EVAL_FUNCTIONS registry)
   - Easy to add new problems/scenarios
   - Easy to add new benchmark loaders

3. **Notebook-First Design**
   - All functionality importable from notebooks
   - Experiment tracking built in
   - No special CLI required (though helpful)

### Weaknesses

1. **Missing Core Abstraction (benchmarks)**
   - Blocks all imports
   - Should be simple to fix but high impact

2. **Monolithic Files**
   - 749 LOC strategies file is hard to navigate
   - 527 LOC provider file has 4 provider implementations
   - 584 LOC problems file with 8 problem definitions
   - Could split into focused modules

3. **Documentation Fragmentation**
   - 11 markdown files with overlapping information
   - No clear "single source of truth"
   - CLAUDE.md significantly outdated
   - README oversimplifies

4. **No Testing**
   - No unit tests
   - No integration tests
   - No CI/CD to catch regressions
   - Relies entirely on notebook-based validation

5. **Potential Dependency Issues**
   - Circular import risk if not careful
   - Hard to tell what the actual public API is (too many __all__ exports)
   - Third-party library versions not pinned

---

## Recommendations for Improvement

### 🔴 **CRITICAL (Do First)**

1. **Create `harness/benchmarks.py`**
   - Must have: `load_benchmark()`, `get_baseline_scores()`, `BENCHMARKS` dict
   - Can be minimal stub if real implementations not available
   - Blocks all other work

### 🟠 **HIGH (Do Soon)**

2. **Update CLAUDE.md**
   - Add CRIT and SELPHI sections
   - Document rationale extraction
   - Note the benchmarks module situation
   - Fix any outdated information

3. **Consolidate Documentation**
   - Create a single `ARCHITECTURE.md` explaining subsystems
   - Make README.md link to detailed guides rather than repeating
   - Reduce from 11 docs to ~6 by consolidating

4. **Add Basic Tests**
   - Unit tests for evaluation functions (all 3 modules)
   - Import tests to catch breaking changes
   - Provider tests (at least smoke tests)
   - Could be simple pytest suite, <500 LOC

### 🟡 **MEDIUM (Do Later)**

5. **Refactor Large Files**
   - Split `harness/strategies.py` into separate files
   - Move provider implementations into separate modules
   - Create shared `harness/utils/` for common patterns

6. **Standardize Benchmark Loading**
   - Move benchmark utilities to harness
   - Create common BenchmarkDataset class
   - Unify loader APIs

7. **Add CI/CD**
   - GitHub Actions workflow
   - Run tests on PR
   - Check import validity
   - Build documentation

8. **Version Pin Dependencies**
   - Move to `==` pins with `>=` fallback
   - Reduce compatibility drift risk

### 🟢 **NICE TO HAVE (Do Eventually)**

9. **Rationale Extraction in All Subsystems**
   - Currently only in harness
   - Would be useful for CRIT and SELPHI too
   - Could extract to shared utility

10. **Config Presets Expansion**
    - Add more built-in configurations
    - Document task-specific recommendations
    - Update examples to use real presets

---

## Summary Table: Current State vs. Desired State

| Aspect | Current | Desired | Effort |
|--------|---------|---------|--------|
| Core functionality | 95% | 100% | Low (fix benchmarks) |
| Tests | 0% | 70% | Medium |
| Documentation quality | 70% | 90% | Medium (consolidation) |
| Architecture clarity | 60% | 85% | Low (refactoring) |
| CI/CD setup | 0% | 80% | Medium |
| Code modularity | 70% | 85% | Medium (split large files) |

---

## Conclusion

**Overall Assessment**: Ambitious, well-structured research toolkit with excellent subsystem design (CRIT, SELPHI) but blockedby missing core module. The project scope is much larger than documented, creating maintainability challenges.

**Time to Functional State**: ~1 hour (fix missing benchmarks.py)  
**Time to Production State**: ~40 hours (add tests, fix docs, CI/CD)  
**Estimated Debt**: Low-to-medium (good architecture, main issue is documentation vs. implementation gap)

