# Hidden Layer Project Guide

> **Quick Navigation**: Jump to any project and start experimenting in minutes

This guide provides a complete overview of all research projects in Hidden Layer, organized by research area, with direct links to get started quickly.

---

## 🚀 First Time Here?

**Start here**: [QUICKSTART.md](QUICKSTART.md) - Set up your environment (5 minutes)

**Then choose** a research area below based on your interests:

---

## 📚 Research Areas & Projects

### 1. 🗣️ Communication

*How do AI agents communicate and coordinate?*

#### [Multi-Agent Architecture](communication/multi-agent/)
- **What**: Multi-agent coordination strategies (debate, consensus, manager-worker, CRIT)
- **Quick Start**: `jupyter lab communication/multi-agent/notebooks/00_quickstart.ipynb`
- **Use When**: Comparing single vs. multi-agent performance, exploring coordination strategies
- **Documentation**: [README](communication/multi-agent/README.md) | [CLAUDE.md](communication/multi-agent/CLAUDE.md)

#### [AI-to-AI Communication](communication/ai-to-ai-comm/)
- **What**: Non-linguistic communication via latent representations (Cache-to-Cache)
- **Quick Start**: `jupyter lab communication/ai-to-ai-comm/notebooks/01_c2c_quickstart.ipynb`
- **Use When**: Exploring efficient communication between models, latent space messaging
- **Documentation**: [README](communication/ai-to-ai-comm/README.md) | [C2C_README](communication/ai-to-ai-comm/C2C_README.md)

---

### 2. 🧠 Theory of Mind

*How do AI systems understand mental states (self and others)?*

#### [SELPHI - Theory of Mind Evaluation](theory-of-mind/selphi/)
- **What**: ToM scenarios (Sally-Anne, false beliefs) and benchmarks (ToMBench, OpenToM, SocialIQA)
- **Quick Start**: `jupyter lab theory-of-mind/selphi/notebooks/01_basic_tom_tests.ipynb`
- **Use When**: Evaluating perspective-taking, false belief reasoning, mental state understanding
- **Documentation**: [README](theory-of-mind/selphi/README.md) | [CLAUDE.md](theory-of-mind/selphi/CLAUDE.md)

#### [Introspection - Self-Knowledge](theory-of-mind/introspection/)
- **What**: Activation steering, concept vectors, model self-knowledge experiments
- **Quick Start**: `jupyter lab theory-of-mind/introspection/notebooks/01_concept_vectors.ipynb`
- **Use When**: Steering model behavior, testing introspection accuracy, exploring activations
- **Documentation**: [README](theory-of-mind/introspection/README.md) | [CLAUDE.md](theory-of-mind/introspection/CLAUDE.md)

---

### 3. 🔍 Representations

*What are internal representations and how can we interpret them?*

#### [Latent Lens - SAE Interpretability](representations/latent-space/lens/)
- **What**: Web app for training Sparse Autoencoders and discovering interpretable features
- **Quick Start**: `cd representations/latent-space/lens && make dev`
- **Use When**: Discovering features in activations, training SAEs, analyzing model internals
- **Documentation**: [README](representations/latent-space/lens/README.md) | [CLAUDE.md](representations/latent-space/CLAUDE.md)

#### [Latent Topologies - Mobile Exploration](representations/latent-space/topologies/)
- **What**: Mobile app for exploring latent space with visual, audio, and haptic feedback
- **Quick Start**: See [README](representations/latent-space/topologies/README.md) for setup
- **Use When**: Experiencing latent space navigation, phenomenological research
- **Documentation**: [README](representations/latent-space/topologies/README.md) | [PRD](representations/latent-space/topologies/PRD.md)

#### [CALM - Continuous Language Models](representations/latent-space/calm/)
- **What**: Vector-by-vector generation (replaces token-by-token)
- **Quick Start**: See [README](representations/latent-space/calm/README.md) for training
- **Use When**: Exploring efficient generation, continuous representations
- **Documentation**: [README](representations/latent-space/calm/README.md) | [CLAUDE.md](representations/latent-space/calm/CLAUDE.md)

#### [LLM State Explorer](representations/state-explorer/)
- **What**: Real-time visualization of LLM internal activations
- **Quick Start**: `cd representations/state-explorer/backend && uvicorn app.main:app --reload`
- **Use When**: Visualizing activations, exploring layer dynamics, debugging model behavior
- **Documentation**: [README](representations/state-explorer/README.md)

---

### 4. 🎯 Alignment

*How can we steer AI systems and verify alignment?*

#### [Steerability Dashboard](alignment/steerability/)
- **What**: Live steering controls with adherence metrics and constraint enforcement
- **Quick Start**: `cd alignment/steerability && make dev`
- **Use When**: Steering model behavior, tracking adherence, enforcing constraints
- **Documentation**: [README](alignment/steerability/README.md) | [CLAUDE.md](alignment/steerability/CLAUDE.md)

---

### 5. 💾 Memory & Personalization

*How can AI systems use personal memory effectively?*

#### [Lifelog Personalization Gatekeeper](memory/lifelog-personalization/)
- **What**: Lifelog retrieval and personalization evaluation harness
- **Quick Start**: `jupyter lab memory/lifelog-personalization/notebooks/00_quickstart.ipynb`
- **Use When**: Evaluating personalization, testing memory retrieval strategies
- **Documentation**: [README](memory/lifelog-personalization/README.md)

---

## 🔧 Infrastructure & Tools

### [Harness - Core Infrastructure](harness/)
- **What**: Provider-agnostic LLM infrastructure, experiment tracking, evaluation utilities
- **Use**: Imported by all research projects
- **Documentation**: [README](harness/README.md)

### [MLX Lab - Model Management CLI](mlx_lab/)
- **What**: CLI tool for managing MLX models, benchmarking, concept browsing
- **Quick Start**: `mlx-lab setup` then `mlx-lab models download qwen3-8b-4bit`
- **Use When**: Managing local models on Apple Silicon
- **Documentation**: [README](mlx_lab/README.md)

### [AgentMesh - Product Platform](agentmesh/)
- **What**: Production multi-agent workflow orchestration (wraps research)
- **Quick Start**: See [README](agentmesh/README.md) for setup
- **Use When**: Deploying multi-agent systems in production
- **Documentation**: [README](agentmesh/README.md)

---

## 🌐 Web Tools (Public Demos)

Deployment-ready versions of research projects:

- **[Latent Lens](web-tools/latent-lens/)** - SAE interpretability web demo
- **[Multi-Agent Arena](web-tools/multi-agent-arena/)** - Multi-agent strategy playground
- **[Steerability](web-tools/steerability/)** - Live steering interface

All web tools:
- Share infrastructure with research projects (don't duplicate code)
- Include Docker Compose for easy deployment
- Have separate backend (FastAPI) and frontend (Next.js)

---

## 📖 Documentation Structure

Each project follows this pattern:

```
project-name/
├── README.md              # Overview, prerequisites, quick start
├── CLAUDE.md              # Development guide (for research projects)
├── notebooks/             # Jupyter notebooks (research projects)
├── code/ or src/          # Python implementation
└── docs/                  # Additional documentation
```

### Lab-Wide Documentation

- **[README.md](README.md)** - Lab overview and philosophy
- **[QUICKSTART.md](QUICKSTART.md)** - Zero-to-notebook setup (start here!)
- **[CLAUDE.md](CLAUDE.md)** - Development guide for contributors
- **[RESEARCH.md](RESEARCH.md)** - Research themes and cross-connections
- **[SETUP.md](SETUP.md)** - Detailed setup instructions
- **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** - This file

---

## 🎯 Quick Decision Guide

**Choose your path based on your goal:**

### I want to explore multi-agent strategies
→ Start with [Multi-Agent](communication/multi-agent/) or [AgentMesh](agentmesh/)

### I want to understand model internals
→ Try [Latent Lens](representations/latent-space/lens/) or [State Explorer](representations/state-explorer/)

### I want to steer model behavior
→ Use [Steerability](alignment/steerability/) or [Introspection](theory-of-mind/introspection/)

### I want to test theory of mind
→ Explore [SELPHI](theory-of-mind/selphi/)

### I want to improve agent communication
→ Try [AI-to-AI Comm](communication/ai-to-ai-comm/) or [CALM](representations/latent-space/calm/)

### I want to manage local models
→ Use [MLX Lab](mlx_lab/)

---

## 🔬 Research Workflows

### Typical Research Flow

1. **Setup Environment**: Follow [QUICKSTART.md](QUICKSTART.md)
2. **Pick a Project**: Use this guide to choose based on interests
3. **Run Quickstart Notebook**: Most projects have `00_quickstart.ipynb` or `01_*.ipynb`
4. **Experiment**: Modify parameters, try different models, log experiments
5. **Analyze**: Use harness experiment tracker to compare results
6. **Iterate**: Refine hypotheses based on findings

### Cross-Project Research

Many research questions span multiple projects:

- **Multi-agent + AI-to-AI Comm**: Can agents coordinate via latent messages?
- **SELPHI + Introspection**: Is introspection related to theory of mind?
- **Latent Lens + Steerability**: What features control behavior?
- **Introspection + Alignment**: Can we detect deception via introspection?

See [RESEARCH.md](RESEARCH.md) for detailed cross-project connections.

---

## 🆘 Getting Help

### Documentation
- Project-specific: Check project's `README.md` and `CLAUDE.md`
- Infrastructure: See `harness/README.md` and `docs/`
- Setup issues: See `SETUP.md` and run `python check_setup.py`

### Common Issues

**Import errors**: Make sure you're running from repository root and harness is installed
**Model not found**: Check `config/models.yaml` and provider setup (Ollama/MLX/API)
**Notebook errors**: Verify environment setup with `python check_setup.py`

---

## 📊 Project Status Summary

| Project | Status | Notebooks | Web UI | API Required |
|---------|--------|-----------|--------|--------------|
| Multi-Agent | ✅ Active | 7 | ❌ | Optional |
| AI-to-AI Comm | ✅ Active | 2 | ❌ | No |
| SELPHI | ✅ Active | 2 | ❌ | Optional |
| Introspection | ✅ Active | 2 | ❌ | Optional |
| Latent Lens | ✅ Active | 1 | ✅ | No |
| Latent Topologies | 🔄 Early Dev | 1 | 📱 Mobile | No |
| CALM | ✅ Active | 0 | ❌ | No |
| State Explorer | 🔄 MVP | 0 | ✅ | No |
| Steerability | ✅ Active | 1 | ✅ | No |
| Lifelog | ✅ Active | 1 | ❌ | Optional |
| AgentMesh | ✅ Active | 0 | ✅ | Optional |
| MLX Lab | ✅ Active | 0 | CLI | No |

**Legend**:
- ✅ Active - Fully functional
- 🔄 MVP/Early Dev - Working but evolving
- ❌ - Not applicable
- 📱 - Mobile app
- CLI - Command-line interface

---

## 🎓 Learning Path

**Beginner**:
1. Complete [QUICKSTART.md](QUICKSTART.md)
2. Run [Multi-Agent](communication/multi-agent/) `00_quickstart.ipynb`
3. Explore [SELPHI](theory-of-mind/selphi/) ToM scenarios
4. Try [State Explorer](representations/state-explorer/) visualization

**Intermediate**:
1. Train SAE with [Latent Lens](representations/latent-space/lens/)
2. Steer models with [Steerability](alignment/steerability/)
3. Extract concept vectors in [Introspection](theory-of-mind/introspection/)
4. Compare strategies in [Multi-Agent](communication/multi-agent/)

**Advanced**:
1. Implement C2C communication in [AI-to-AI Comm](communication/ai-to-ai-comm/)
2. Train CALM models in [CALM](representations/latent-space/calm/)
3. Design cross-project experiments (see [RESEARCH.md](RESEARCH.md))
4. Deploy with [AgentMesh](agentmesh/)

---

## 🔄 Recent Updates

- **2025-11**: Enhanced project READMEs with prerequisites and installation sections
- **2025-11**: Added this PROJECT_GUIDE.md for easier navigation
- **2025-11**: Improved cross-references between projects and main QUICKSTART

---

## 📝 Contributing

Each project welcomes contributions! See individual project `CLAUDE.md` files for development guidelines.

**General principles**:
- Maintain backward compatibility with harness
- Log experiments for reproducibility
- Document design decisions
- Test with local models first

---

**Need help choosing?** Ask yourself:
- What aspect of AI behavior am I most curious about?
- Do I prefer notebooks, web UIs, or CLI tools?
- Am I exploring or building production systems?

Then find your project above and dive in! 🚀
