# Hidden Layer - Independent AI Research Lab

## Mission

Hidden Layer investigates fundamental questions about AI systems:
- How do agents communicate and coordinate?
- What is AI theory of mind and self-knowledge?
- What are the internal representations models use?
- How can we build aligned, honest AI systems?

## Research Themes

### 1. Agent Communication & Coordination
Understanding how multiple AI agents can work together effectively.

**Projects**:
- **[Multi-Agent Architecture](communication/multi-agent/)** - Debate, CRIT, XFN teams, coordination strategies
- **[AI-to-AI Communication](communication/ai-to-ai-comm/)** - Non-linguistic communication via latent representations

### 2. Theory of Mind & Self-Knowledge
Exploring how AI systems understand mental states (their own and others').

**Projects**:
- **[SELPHI](theory-of-mind/selphi/)** - Theory of mind evaluation and benchmarking
- **[Introspection](theory-of-mind/introspection/)** - Model introspection experiments

### 3. Internal Representations & Interpretability
Making sense of high-dimensional latent spaces and learned features.

**Projects**:
- **[Latent Space](representations/latent-space/)** - SAE interpretability (Lens) + Mobile exploration (Topologies)
- **[Introspection](theory-of-mind/introspection/)** - Concept vectors and activation steering

### 4. Alignment, Steerability & Deception
Building controllable, honest, aligned AI systems.

**Projects**:
- **[Steerability](alignment/steerability/)** - Steering vectors and adherence metrics
- **[SELPHI](theory-of-mind/selphi/)** - Deception detection
- **[Introspection](theory-of-mind/introspection/)** - Honest self-reporting

---

## Projects

### [Multi-Agent Architecture](communication/multi-agent/)
Research platform for multi-agent coordination strategies.

**Status**: Active | **Stack**: Python, Ollama/Claude/GPT

**Features**: Debate, CRIT, self-consistency, manager-worker, consensus strategies

### [SELPHI](theory-of-mind/selphi/)
Theory of Mind evaluation and benchmarking.

**Status**: Active | **Stack**: Python, ToMBench/OpenToM/SocialIQA

**Features**: 9+ ToM scenarios, 7 ToM types, benchmark integration

### [Latent Space](representations/latent-space/)

#### Lens (SAE Interpretability)
Interactive web app for training Sparse Autoencoders and discovering features.

**Status**: Active | **Stack**: FastAPI, Next.js, PyTorch

**Features**: SAE training, feature gallery, activation lens, labeling

#### Topologies (Mobile Exploration)
Mobile app for experiencing latent spaces through vision, sound, and haptics.

**Status**: Concept | **Stack**: React Native, Expo

**Features**: Visual constellation navigation, audio mapping, haptic feedback

### [Introspection](theory-of-mind/introspection/)
Model introspection experiments (Anthropic-style).

**Status**: Active | **Stack**: Python, MLX

**Features**: Activation steering, concept vectors, introspection tasks, API introspection

### [AI-to-AI Communication](communication/ai-to-ai-comm/)
Non-linguistic communication between LLMs via latent representations.

**Status**: Early | **Stack**: Python

**Focus**: Efficient agent communication through internal states

### [Steerability](alignment/steerability/)
Real-time steering with adherence metrics.

**Status**: Active | **Stack**: FastAPI, Next.js

**Features**: Steering vectors, adherence tracking, A/B comparison

---

## Infrastructure

### The Harness (Standalone Library)

**Location**: `/harness/`

Core infrastructure providing:
- **Unified LLM Provider**: Seamlessly switch between Ollama, MLX, Claude, GPT
- **Experiment Tracking**: Automatic logging, reproducibility
- **Evaluation Utilities**: Exact match, LLM-as-judge, benchmarks
- **Model Configuration**: Named presets, system prompts

**Philosophy**: Flexible - use local models (Ollama, MLX) for rapid iteration OR frontier models (Claude, GPT) via API.

```python
from harness import llm_call

# Local
response = llm_call("Question?", provider="ollama", model="llama3.2:latest")

# API
response = llm_call("Question?", provider="anthropic", model="claude-3-5-sonnet-20241022")
```

**Can be open-sourced independently** - useful to other researchers.

### Shared Resources

**Location**: `/shared/`

- `concepts/` - Concept vectors (emotions, topics, custom)
- `datasets/` - Benchmark datasets
- `utils/` - Common utilities

---

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/hidden-layer
cd hidden-layer

# Setup (optional - only if using local models)
./setup.sh

# Pick a project
cd communication/multi-agent    # Or theory-of-mind/selphi, representations/latent-space, etc.

# See project README for specific instructions
cat README.md
```

### Example: Multi-Agent Debate

```python
from communication.multi_agent import run_strategy
from harness import get_tracker

# Run 3-agent debate
result = run_strategy(
    "debate",
    task_input="Should we invest in renewable energy?",
    n_debaters=3,
    provider="ollama",
    model="llama3.2:latest"
)

print(result.output)
```

### Example: Theory of Mind Evaluation

```python
from theory_of_mind.selphi.code import run_scenario, SALLY_ANNE
from harness import llm_call

# Run classic false belief test
response = llm_call(
    SALLY_ANNE.get_prompt(),
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)

print(response.text)
```

### Example: Introspection

```python
from theory_of_mind.introspection.code import ConceptLibrary, ActivationSteerer
from mlx_lm import load

# Load concept library
library = ConceptLibrary.load("shared/concepts/emotions_layer15.pkl")

# Use with local model
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
steerer = ActivationSteerer(model, tokenizer)

# Steer toward happiness
steered_output = steerer.generate_with_steering(
    prompt="Write a story",
    concept_vector=library.get("happiness"),
    strength=2.0
)
```

---

## Documentation

### Lab-Wide
- **[RESEARCH.md](RESEARCH.md)** - Research themes and cross-project connections
- **[CLAUDE.md](CLAUDE.md)** - Development guide for Claude
- **[docs/](docs/)** - Shared documentation (infrastructure, workflows, conventions)

### Project-Specific
Each project has:
- `README.md` - Overview and quick start
- `CLAUDE.md` - Development guide
- Additional docs as needed

### Topics
- **Infrastructure**: [docs/infrastructure/](docs/infrastructure/)
- **Hardware**: [docs/hardware/](docs/hardware/) (optional - local models)
- **Workflows**: [docs/workflows/](docs/workflows/)

---

## Research Connections

Projects are deeply interconnected. For example:

- **Multi-agent + AI-to-AI comm** → Can agents coordinate via latent messaging?
- **SELPHI + Introspection** → Understanding others vs. understanding self
- **Latent Space + Introspection** → What features enable self-knowledge?
- **SELPHI + Steerability** → Detecting and preventing deception

See [RESEARCH.md](RESEARCH.md) for detailed connections.

---

## File Organization

```
hidden-layer/
├── harness/                    # Core infrastructure (standalone library)
├── shared/                     # Shared resources (concepts, datasets, utils)
├── web-tools/                  # Deployment versions of web applications
│
├── communication/              # Research Area: Agent Communication
│   ├── multi-agent/           # Multi-agent coordination
│   └── ai-to-ai-comm/         # Non-linguistic communication
│
├── theory-of-mind/             # Research Area: Theory of Mind & Self-Knowledge
│   ├── selphi/                # Theory of mind evaluation
│   └── introspection/         # Model introspection
│
├── representations/            # Research Area: Internal Representations
│   └── latent-space/
│       ├── lens/              # SAE interpretability
│       └── topologies/        # Mobile latent exploration
│
├── alignment/                  # Research Area: Alignment & Steerability
│   └── steerability/          # Steering vectors & metrics
│
└── docs/                       # Lab-wide documentation
    ├── infrastructure/
    ├── hardware/
    ├── workflows/
    └── archive/
```

---

## Hardware

**Not local-only!** Infrastructure supports:
- **Local** (optional): Ollama, MLX on Apple Silicon
- **API**: Claude, GPT, etc. for frontier capabilities

**If using local models** (M4 Max 128GB RAM):
- Run 70B models (~35GB, 4-bit quantized)
- Run 3-4 7B models in parallel (~12GB total)
- Fine-tune 13B models with LoRA (~20GB)

See [docs/hardware/](docs/hardware/) for setup guides.

---

## Publications & Research

See [RESEARCH.md](RESEARCH.md) for:
- Research questions
- Findings & publications
- Ongoing experiments
- Cross-project connections

---

## Contributing

This is an independent research lab. Interested in collaborating? Open an issue or reach out.

**For developers**: See [CLAUDE.md](CLAUDE.md) for development guidelines.

---

## License

- **Code**: MIT
- **Documentation**: CC-BY
- **Research outputs**: TBD per project

---

## Acknowledgments

- **Apple MLX Team** - Apple Silicon optimization
- **Ollama** - Local model serving
- **Anthropic, OpenAI** - API access for baseline comparisons
- **Research Community** - ToMBench, OpenToM, UICrit datasets
- **Open Source** - Meta (Llama), Mistral AI, HuggingFace

---

**Ready to dive in?**
- New to the lab? Start with [RESEARCH.md](RESEARCH.md)
- Want to develop? See [CLAUDE.md](CLAUDE.md)
- Pick a project: explore [communication/](communication/), [theory-of-mind/](theory-of-mind/), [representations/](representations/), or [alignment/](alignment/)
