# Hidden Layer - Video Walkthrough Scripts

> Production-ready scripts for creating video tutorials

This document provides detailed scripts for recording video walkthroughs of Hidden Layer projects. Each script includes timing, commands, voiceover text, and key points to demonstrate.

---

## Table of Contents

1. [Quick Start (5 minutes)](#1-quick-start-5-minutes)
2. [Multi-Agent Debate (8 minutes)](#2-multi-agent-debate-8-minutes)
3. [SELPHI Theory of Mind (10 minutes)](#3-selphi-theory-of-mind-10-minutes)
4. [Latent Lens SAE Training (12 minutes)](#4-latent-lens-sae-training-12-minutes)
5. [Setting Up Ollama (4 minutes)](#5-setting-up-ollama-4-minutes)
6. [Comparing Strategies (7 minutes)](#6-comparing-strategies-7-minutes)

---

## General Recording Guidelines

**Screen Setup**:
- Terminal: Bottom half of screen
- Browser/Editor: Top half
- Use large fonts (14-16pt minimum)
- Dark theme recommended

**Audio**:
- Clear microphone (no background noise)
- Speak slowly and clearly
- Pause 1-2 seconds between major steps

**Editing**:
- Speed up slow processes (model downloads, training)
- Add captions for key commands
- Include timestamps in video description

---

## 1. Quick Start (5 minutes)

**Goal**: Get from zero to first experiment in 5 minutes

**Target Audience**: Complete beginners

**Prerequisites**: macOS or Linux, Python 3.10+

### Script

**[0:00 - 0:15] Introduction**

*Voiceover*: "In this video, we'll go from zero to running your first AI experiment with Hidden Layer in just 5 minutes. Let's get started."

*Screen*: Show Hidden Layer repository homepage

---

**[0:15 - 0:45] Clone Repository**

*Voiceover*: "First, clone the repository and navigate into it."

*Commands*:
```bash
git clone https://github.com/yourusername/hidden-layer
cd hidden-layer
ls  # Show structure
```

*Screen*: Show terminal with colorized output, highlight key directories (harness/, communication/, etc.)

---

**[0:45 - 1:30] Setup Environment**

*Voiceover*: "Hidden Layer provides a one-command setup. Run make setup to create a virtual environment and install all dependencies."

*Commands*:
```bash
make setup
source venv/bin/activate
```

*Screen*: Show installation progress (speed up 2x), highlight when complete

---

**[1:30 - 2:00] Verify Setup**

*Voiceover*: "Let's verify everything is working with the check setup script."

*Commands*:
```bash
python check_setup.py
```

*Screen*: Show green checkmarks appearing, highlight key components (Python version, harness, providers)

---

**[2:00 - 2:30] Install Ollama**

*Voiceover*: "We'll use Ollama for free local models. Install it with brew, start the server, and pull a small model."

*Commands*:
```bash
brew install ollama
ollama serve  # In separate terminal
ollama pull llama3.2:latest
```

*Screen*: Split screen - left: installation, right: server logs

---

**[2:30 - 3:15] First Experiment**

*Voiceover*: "Now the fun part - let's run our first multi-agent debate in Python."

*Screen*: Open Python REPL or Jupyter notebook

*Commands*:
```python
from communication.multi_agent import run_strategy

result = run_strategy(
    "debate",
    task_input="Should we invest in renewable energy?",
    n_debaters=3,
    provider="ollama",
    model="llama3.2:latest"
)

print(result.output)
```

*Screen*: Show code typing (can be sped up), then show model thinking, then final output appearing

---

**[3:15 - 4:00] View Results**

*Voiceover*: "The debate strategy had three AI agents discuss the question, and a judge synthesized their arguments. Here's the final answer."

*Screen*: Highlight the output, scroll through to show structure

*Key Points to Show*:
- Arguments from multiple perspectives
- Synthesis of ideas
- Well-structured final answer

---

**[4:00 - 4:45] Open Notebook**

*Voiceover*: "For a more interactive experience, Hidden Layer includes Jupyter notebooks with pre-built experiments. Let's open the multi-agent quickstart notebook."

*Commands*:
```bash
jupyter lab communication/multi-agent/notebooks/00_quickstart.ipynb
```

*Screen*: Show Jupyter opening, navigate to notebook, run first cell

---

**[4:45 - 5:00] Wrap Up**

*Voiceover*: "That's it! You've just run your first multi-agent AI experiment. Check out PROJECT_GUIDE.md to explore 15+ more research projects. Happy experimenting!"

*Screen*: Show PROJECT_GUIDE.md in browser, scroll through project list

**END SCREEN**: Links to documentation, GitHub repo, next videos

---

## 2. Multi-Agent Debate (8 minutes)

**Goal**: Deep dive into multi-agent strategies

**Target Audience**: Users who completed Quick Start

**Prerequisites**: Hidden Layer set up, Ollama running

### Script

**[0:00 - 0:20] Introduction**

*Voiceover*: "In this video, we'll explore multi-agent strategies in depth: how they work, when to use them, and how to compare their performance."

*Screen*: Show multi-agent README.md

---

**[0:20 - 1:30] Available Strategies**

*Voiceover*: "Hidden Layer includes six multi-agent strategies, each with different strengths."

*Screen*: Show code editor with strategies.py open

*Visual*: Create animated diagram showing each strategy:

1. **Single** (Baseline)
   - One model → one answer
   - Fast, no coordination

2. **Debate**
   - Multiple agents argue → judge synthesizes
   - Good for controversial questions

3. **Self-Consistency**
   - Multiple independent answers → majority vote
   - Good for factual questions

4. **Manager-Worker**
   - Manager breaks down task → workers execute → manager synthesizes
   - Good for complex tasks

5. **Consensus**
   - Agents discuss until they agree
   - Good for collaborative decisions

6. **CRIT**
   - Multiple perspectives critique a design
   - Good for creative work

---

**[1:30 - 3:00] Running a Debate**

*Voiceover*: "Let's run a debate with three agents on a controversial topic."

*Screen*: Jupyter notebook

*Code*:
```python
from communication.multi_agent import run_strategy

question = "Is artificial general intelligence achievable within 10 years?"

result = run_strategy(
    "debate",
    task_input=question,
    n_debaters=3,
    rounds=2,
    provider="ollama",
    model="llama3.2:latest"
)
```

*Screen*: Show execution, then display result.metadata to show structure:
- Number of debaters
- Number of rounds
- Individual arguments
- Judge's synthesis

---

**[3:00 - 4:30] Comparing Strategies**

*Voiceover*: "Let's compare how different strategies answer the same question."

*Code*:
```python
from communication.multi_agent import run_strategy
from harness import get_tracker

tracker = get_tracker()
question = "What are the pros and cons of remote work?"

strategies = ["single", "debate", "self_consistency", "consensus"]

for strategy in strategies:
    result = run_strategy(
        strategy,
        task_input=question,
        provider="ollama"
    )
    print(f"\n{strategy.upper()}:")
    print(result.output)
    print("-" * 50)
    tracker.log_result(f"{strategy}_remote_work", result)
```

*Screen*: Show each strategy's output appearing, highlight differences

---

**[4:30 - 5:30] Analyzing Results**

*Voiceover*: "The experiment tracker lets us compare results systematically."

*Code*:
```python
# View summary
tracker.summarize()

# Export to CSV for analysis
tracker.export("strategy_comparison.csv")

# Load in pandas for visualization
import pandas as pd
df = pd.read_csv("strategy_comparison.csv")
df.head()
```

*Screen*: Show DataFrame, maybe create a quick bar chart of response lengths or quality scores

---

**[5:30 - 6:30] Performance Considerations**

*Voiceover*: "Multi-agent strategies trade latency for quality. Here's what affects performance."

*Screen*: Show comparison table (pre-prepared)

| Factor | Impact | Recommendation |
|--------|--------|----------------|
| Number of agents | Linear increase in calls | Use 2-3 for development, 3-5 for production |
| Model size | Affects per-call latency | Start with small models (3B), scale up |
| Number of rounds | Multiplies calls | Single round often sufficient |
| Provider | Local vs API tradeoff | Local: faster iteration, API: better quality |

---

**[6:30 - 7:30] Advanced: CRIT Strategy**

*Voiceover*: "The CRIT strategy is specialized for design critique. Let's see it in action."

*Code*:
```python
from communication.multi_agent.crit import run_crit

result = run_crit(
    task_input="Design a mobile app for mental health tracking",
    perspectives=[
        "UX Designer - focus on usability",
        "Clinical Psychologist - focus on therapeutic value",
        "Privacy Expert - focus on data security"
    ],
    provider="ollama"
)

print(result.output)
```

*Screen*: Show output with each perspective's critique, then synthesis

---

**[7:30 - 8:00] Wrap Up**

*Voiceover*: "You've now seen all major multi-agent strategies. Experiment with different questions, models, and parameters to find what works best for your use case. Check out the notebooks for more examples."

*Screen*: Show notebooks/ directory, highlight available notebooks

**END SCREEN**: Next video (SELPHI Theory of Mind), GitHub repo, documentation

---

## 3. SELPHI Theory of Mind (10 minutes)

**Goal**: Understand and evaluate AI theory of mind

**Target Audience**: Researchers, AI safety enthusiasts

**Prerequisites**: Hidden Layer set up

### Script

**[0:00 - 0:30] Introduction**

*Voiceover*: "Theory of mind - the ability to understand that others have different beliefs, knowledge, and perspectives - is a crucial aspect of intelligence. In this video, we'll test how well AI models perform on theory of mind tasks."

*Screen*: Show Sally-Anne test diagram (pre-prepared illustration)

---

**[0:30 - 2:00] The Sally-Anne Test**

*Voiceover*: "Let's start with the classic Sally-Anne test, which tests false belief understanding."

*Screen*: Animated diagram showing:
1. Sally puts marble in basket
2. Sally leaves
3. Anne moves marble to box
4. Sally returns

*Question*: "Where will Sally look for the marble?"
*Correct answer*: "In the basket" (Sally's false belief)

*Voiceover*: "Humans develop this ability around age 4. Let's see how AI does."

---

**[2:00 - 3:30] Running the Test**

*Screen*: Jupyter notebook

*Code*:
```python
from harness import llm_call
from theory_of_mind.selphi import SALLY_ANNE, evaluate_scenario

# Get the prompt
prompt = SALLY_ANNE.get_prompt()
print(prompt)

# Run with Claude
response = llm_call(
    prompt,
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)

print("\nResponse:")
print(response.text)

# Evaluate
result = evaluate_scenario(SALLY_ANNE, response.text)
print(f"\nCorrect: {result['correct']}")
print(f"Score: {result['average_score']:.2f}")
```

*Screen*: Show prompt, model response, evaluation

*Voiceover*: "The model correctly identifies Sally's false belief. Let's try more scenarios."

---

**[3:30 - 5:00] Multiple Scenarios**

*Voiceover*: "SELPHI includes 9+ scenarios testing different aspects of theory of mind."

*Code*:
```python
from theory_of_mind.selphi.code.scenarios import (
    SALLY_ANNE,
    CHOCOLATE_BAR,
    BIRTHDAY_PUPPY,
    SMARTIES_TEST
)

scenarios = [SALLY_ANNE, CHOCOLATE_BAR, BIRTHDAY_PUPPY, SMARTIES_TEST]

results = {}
for scenario in scenarios:
    response = llm_call(scenario.get_prompt(), provider="anthropic")
    results[scenario.name] = evaluate_scenario(scenario, response.text)

# Display results
for name, result in results.items():
    print(f"{name}: {result['average_score']:.2f}")
```

*Screen*: Show table of results, maybe visualize with a bar chart

---

**[5:00 - 7:00] Benchmark Evaluation**

*Voiceover*: "For systematic evaluation, SELPHI includes three major benchmarks: ToMBench, OpenToM, and SocialIQA."

*Code*:
```python
from theory_of_mind.selphi import run_benchmark_evaluation

# Run on ToMBench subset
results = run_benchmark_evaluation(
    benchmark="tombench",
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    sample_size=100
)

# View results breakdown
results.print_summary()
```

*Screen*: Show output with ASCII table:
```
ToMBench Evaluation Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: claude-3-5-sonnet-20241022
Samples: 100/388
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

By ToM Type:
  False Belief:        87.5% (35/40)
  Second-Order:        82.1% (23/28)
  Epistemic States:    91.2% (31/34)
  Perspective-Taking:  88.9% (16/18)

Overall Accuracy: 86.0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

**[7:00 - 8:30] Comparing Models**

*Voiceover*: "Let's compare different models on the same scenarios."

*Code*:
```python
models = [
    ("anthropic", "claude-3-5-sonnet-20241022"),
    ("openai", "gpt-4"),
    ("ollama", "llama3.2:latest")
]

comparison = {}
for provider, model in models:
    result = llm_call(
        SALLY_ANNE.get_prompt(),
        provider=provider,
        model=model
    )
    score = evaluate_scenario(SALLY_ANNE, result.text)
    comparison[model] = score

# Visualize
import pandas as pd
df = pd.DataFrame(comparison).T
df.plot(kind='bar', title='ToM Performance by Model')
```

*Screen*: Show bar chart comparing models

---

**[8:30 - 9:30] Research Insights**

*Voiceover*: "Here's what we've learned from running these evaluations."

*Screen*: Show slides with key findings:

1. **Frontier models (Claude, GPT-4) perform well** (~85-90% accuracy)
2. **Smaller local models struggle** (~60-70% accuracy)
3. **Second-order beliefs are hardest** (requires understanding A's belief about B's belief)
4. **Models improve with chain-of-thought** reasoning
5. **Connection to alignment**: ToM relates to deception detection

---

**[9:30 - 10:00] Wrap Up**

*Voiceover*: "Theory of mind is fundamental to AI safety and alignment. SELPHI makes it easy to evaluate any model. Try it with your own scenarios or benchmarks."

*Screen*: Show SELPHI README.md, highlight research questions section

**END SCREEN**: Next video (Introspection), GitHub repo, research papers

---

## 4. Latent Lens SAE Training (12 minutes)

**Goal**: Train a Sparse Autoencoder and discover interpretable features

**Target Audience**: Interpretability researchers, advanced users

**Prerequisites**: Hidden Layer set up, Docker installed

### Script

**[0:00 - 0:45] Introduction**

*Voiceover*: "Sparse Autoencoders are a powerful tool for understanding what's happening inside language models. In this video, we'll train an SAE, discover features, and analyze which features activate for different inputs."

*Screen*: Show diagram of SAE architecture (encoder → sparse representation → decoder)

---

**[0:45 - 2:00] Starting Latent Lens**

*Voiceover*: "Latent Lens is a web app that makes SAE training interactive. Let's start it with Docker Compose."

*Commands*:
```bash
cd representations/latent-space/lens
make dev
```

*Screen*: Show terminal output, highlight when services are ready

*Voiceover*: "This starts both the backend (FastAPI) and frontend (Next.js)."

---

**[2:00 - 3:30] Creating an Experiment**

*Voiceover*: "Navigate to localhost:3000 and let's create our first experiment."

*Screen*: Browser showing Latent Lens UI

*Steps*:
1. Click "Layer Explorer"
2. Select model: `gpt2`
3. Select layer: `6` (middle layer)
4. Configure SAE:
   - Hidden dimension: 4096
   - Sparsity coefficient: 0.01
5. Click "Create Experiment"

*Screen*: Show form filling, then training starting

---

**[3:30 - 5:00] Watching Training**

*Voiceover*: "The SAE is now training on WikiText data. We can watch the loss decrease in real-time."

*Screen*: Show training dashboard with:
- Loss curve decreasing
- Reconstruction accuracy increasing
- Sparsity metrics
- Estimated time remaining

*Speed up 4x* - training takes ~5 minutes but show sped up

---

**[5:00 - 7:00] Exploring Features**

*Voiceover*: "Once training completes, we can browse the discovered features."

*Screen*: Navigate to Feature Gallery

*Show*:
1. Grid of features with activation patterns
2. Click on a feature (e.g., "sports words")
3. See top activating tokens: "basketball", "football", "soccer", etc.
4. Click on another feature (e.g., "pronouns")
5. See: "he", "she", "they", "it"

*Voiceover*: "Each feature represents a meaningful pattern the model learned."

---

**[7:00 - 8:30] Activation Lens**

*Voiceover*: "The Activation Lens shows which features activate for specific text."

*Screen*: Navigate to Activation Lens

*Steps*:
1. Enter text: "The cat sat on the mat and purred softly."
2. Select experiment
3. Click "Analyze"

*Screen*: Show heatmap:
- X-axis: tokens
- Y-axis: features
- Color: activation strength
- Highlight that "cat" activates animal feature
- "purred" activates sound feature
- "softly" activates adverb feature

---

**[8:30 - 9:30] Labeling Features**

*Voiceover*: "We can add human-readable labels to features for future reference."

*Screen*: Navigate to Labeling view

*Steps*:
1. Select a feature
2. Add label: "animal words"
3. Add tags: "semantic", "nouns"
4. Save

*Show labeling a few more features quickly*

---

**[9:30 - 10:30] Analyzing Results**

*Voiceover*: "Let's analyze what we've learned about GPT-2's layer 6."

*Screen*: Show summary statistics:
- Total features: 4096
- Labeled features: 15 (user labeled)
- Sparsity: 92% (most features inactive for any token)
- Reconstruction accuracy: 99.2%

*Voiceover*: "The SAE successfully compressed layer activations into interpretable features."

---

**[10:30 - 11:30] API Usage**

*Voiceover*: "You can also use Latent Lens programmatically via the API."

*Screen*: Show Python code:

```python
import requests

# Create experiment
response = requests.post("http://localhost:8000/api/experiments", json={
    "model_name": "gpt2",
    "layer_index": 6,
    "hidden_dim": 4096,
    "sparsity_coef": 0.01
})

experiment_id = response.json()["id"]

# Analyze text
response = requests.post("http://localhost:8000/api/activations/analyze", json={
    "experiment_id": experiment_id,
    "text": "The cat sat on the mat"
})

activations = response.json()
print(activations)
```

---

**[11:30 - 12:00] Wrap Up**

*Voiceover*: "You've just trained your first Sparse Autoencoder and discovered interpretable features. Experiment with different models, layers, and texts to understand what models learn."

*Screen*: Show Latent Lens README.md, highlight roadmap section

**END SCREEN**: Next video (State Explorer), GitHub repo, SAE research papers

---

## 5. Setting Up Ollama (4 minutes)

**Goal**: Get Ollama running for local model inference

**Target Audience**: Complete beginners

**Prerequisites**: macOS or Linux

### Script

**[0:00 - 0:15] Introduction**

*Voiceover*: "Ollama lets you run large language models locally on your computer for free. Let's get it set up."

---

**[0:15 - 0:45] Installation**

*Voiceover*: "On macOS, install with Homebrew. On Linux, use the official script."

*macOS*:
```bash
brew install ollama
```

*Linux*:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

*Screen*: Show installation progress

---

**[0:45 - 1:15] Starting Server**

*Voiceover*: "Start the Ollama server. Keep this terminal running."

*Commands*:
```bash
ollama serve
```

*Screen*: Show server logs appearing:
```
Ollama server started
Listening on 127.0.0.1:11434
```

---

**[1:15 - 2:00] Pulling Models**

*Voiceover*: "Now pull some models. Start with Llama 3.2 - it's small and fast."

*Commands*:
```bash
# Open new terminal (keep server running)
ollama pull llama3.2:latest

# Optional: pull more models
ollama pull mistral-small:latest
```

*Screen*: Show download progress (speed up 2-3x)

---

**[2:00 - 2:30] Testing**

*Voiceover*: "Let's test it works."

*Commands*:
```bash
ollama run llama3.2:latest "Explain quantum computing in one sentence"
```

*Screen*: Show response appearing

---

**[2:30 - 3:00] Model Management**

*Voiceover*: "Here are some useful Ollama commands."

*Commands*:
```bash
ollama list  # See installed models
ollama ps    # See running models
ollama rm mistral-small:latest  # Remove a model
```

*Screen*: Show output for each command

---

**[3:00 - 3:45] Using with Hidden Layer**

*Voiceover*: "Now use it with Hidden Layer."

*Code*:
```python
from harness import llm_call

response = llm_call(
    "What is the capital of France?",
    provider="ollama",
    model="llama3.2:latest"
)

print(response.text)
```

*Screen*: Show execution and response

---

**[3:45 - 4:00] Wrap Up**

*Voiceover*: "That's it! You now have unlimited local inference. See the FAQ for troubleshooting."

**END SCREEN**: Back to Quick Start video, FAQ, Ollama docs

---

## 6. Comparing Strategies (7 minutes)

**Goal**: Systematically compare multi-agent strategies

**Target Audience**: Researchers, users optimizing for specific tasks

**Prerequisites**: Hidden Layer set up, Ollama or API configured

### Script

**[0:00 - 0:30] Introduction**

*Voiceover*: "Not all multi-agent strategies work well for all tasks. Let's systematically compare strategies on different question types."

---

**[0:30 - 1:30] Setup**

*Screen*: Jupyter notebook

*Code*:
```python
from communication.multi_agent import run_strategy
from harness import get_tracker
import pandas as pd

tracker = get_tracker()

# Define test questions
questions = {
    "factual": "What is the capital of France?",
    "analytical": "What are the pros and cons of remote work?",
    "creative": "Design a logo for an AI safety organization",
    "controversial": "Should AI research be regulated?"
}

strategies = ["single", "debate", "self_consistency", "consensus"]
```

---

**[1:30 - 4:00] Running Comparisons**

*Voiceover*: "Let's run each strategy on each question type."

*Code*:
```python
results = []

for q_type, question in questions.items():
    print(f"\n{'='*50}")
    print(f"Question Type: {q_type}")
    print(f"Question: {question}")
    print('='*50)

    for strategy in strategies:
        print(f"\nRunning {strategy}...")

        result = run_strategy(
            strategy,
            task_input=question,
            provider="ollama",
            model="llama3.2:latest"
        )

        # Log result
        tracker.log_result(
            f"{q_type}_{strategy}",
            result
        )

        # Store for analysis
        results.append({
            "question_type": q_type,
            "strategy": strategy,
            "response_length": len(result.output),
            "output": result.output[:100] + "..."  # Truncate for display
        })

        print(f"Response: {result.output[:200]}...")
```

*Screen*: Show execution, speed up 2x, show key responses

---

**[4:00 - 5:30] Analysis**

*Voiceover*: "Now let's analyze the results."

*Code*:
```python
# Convert to DataFrame
df = pd.DataFrame(results)

# Group by question type and strategy
summary = df.groupby(['question_type', 'strategy'])['response_length'].mean()
print(summary)

# Visualize
import matplotlib.pyplot as plt

df_pivot = df.pivot(
    index='strategy',
    columns='question_type',
    values='response_length'
)

df_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Response Length by Strategy and Question Type')
plt.ylabel('Characters')
plt.xlabel('Strategy')
plt.legend(title='Question Type')
plt.show()
```

*Screen*: Show bar chart, discuss patterns

---

**[5:30 - 6:30] Insights**

*Voiceover*: "Here's what we learned."

*Screen*: Show findings slide:

**Findings**:
1. **Factual questions**: Self-consistency wins (fastest, correct)
2. **Analytical questions**: Debate wins (multiple perspectives)
3. **Creative questions**: Consensus wins (collaborative)
4. **Controversial questions**: Debate wins (argues both sides)

**Performance**:
- Single: Fastest but limited perspective
- Debate: 3-4x slower, highest quality
- Self-consistency: 3x slower, good for factual
- Consensus: 4-5x slower, best for collaboration

---

**[6:30 - 7:00] Wrap Up**

*Voiceover*: "Choose your strategy based on your task type and time budget. See the notebooks for more comparisons."

**END SCREEN**: Multi-agent README, notebooks, next video

---

## Production Notes

### Recording Tools

**Screen Recording**:
- macOS: QuickTime, ScreenFlow, or OBS Studio
- Linux: OBS Studio, SimpleScreenRecorder
- Windows: OBS Studio, Camtasia

**Audio**:
- Audacity (free, open source)
- Adobe Audition (professional)
- Built-in screen recording audio (adequate)

**Editing**:
- DaVinci Resolve (free, professional)
- iMovie (macOS, simple)
- Adobe Premiere Pro (professional)
- Kdenlive (Linux, open source)

### Video Format

- **Resolution**: 1920x1080 minimum (1080p)
- **Frame rate**: 30 fps
- **Format**: MP4 (H.264 codec)
- **Audio**: AAC, 128kbps minimum

### Publishing

**YouTube**:
- Create playlist: "Hidden Layer Tutorials"
- Add timestamps in description
- Add links to documentation
- Enable subtitles/captions

**Repository**:
- Add video links to README.md
- Create `docs/videos.md` with index
- Embed in documentation where relevant

### Thumbnail Design

- **Size**: 1280x720 pixels
- **Text**: Large, readable font (60pt+)
- **Colors**: High contrast
- **Include**:  project logo, video title, key visual

### Example Thumbnails

1. **Quick Start**: Terminal with green checkmarks, "5 Minutes to First Experiment"
2. **Multi-Agent**: Diagram of 3 agents debating, "AI Debate in Action"
3. **SELPHI**: Sally-Anne diagram, "Testing AI Theory of Mind"
4. **Latent Lens**: Feature heatmap, "See Inside Language Models"

---

## Next Steps

1. **Record videos** following these scripts
2. **Edit** for clarity and pacing
3. **Publish** to YouTube/Vimeo
4. **Link** from documentation
5. **Gather feedback** and iterate

---

**Last updated**: 2025-11-19

**Questions?** See [FAQ.md](FAQ.md) or open an issue.
