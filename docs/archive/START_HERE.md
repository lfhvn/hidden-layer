# Research Lab Setup - Downloaded Files

## What This Is

This is the complete MLX-based experimentation harness I built for your M4 Max. All the code is here and ready to use.

## How to Use These Files

### Option 1: Quick Start (Recommended)

1. **Download this folder** to your Mac (wherever you want your project)

2. **Run the setup script:**
   ```bash
   cd research-lab-setup
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Install Ollama** (if you haven't):
   ```bash
   brew install ollama
   ollama serve &
   ollama pull llama3.2:latest
   ```

4. **Test it works:**
   ```bash
   source venv/bin/activate
   cd code
   python cli.py "What is 2+2?" --strategy single --provider ollama
   ```

5. **Start experimenting:**
   ```bash
   cd ../notebooks
   jupyter notebook
   ```
   Open `01_baseline_experiments.ipynb`

### Option 2: Add to Existing Git Repo

If you already have a GitHub repo for your research lab:

```bash
# From your repo root
cp -r /path/to/downloaded/research-lab-setup/* .
git add .
git commit -m "Add MLX experimentation harness"
git push
```

### Option 3: Create New Git Repo

```bash
cd research-lab-setup
git init
git add .
git commit -m "Initial commit: MLX harness setup"

# Create repo on GitHub, then:
git remote add origin https://github.com/yourusername/research-lab.git
git push -u origin main
```

## What's Included

```
research-lab-setup/
├── code/
│   ├── harness/              # Core library
│   │   ├── llm_provider.py   # MLX, Ollama, APIs
│   │   ├── strategies.py     # Multi-agent strategies
│   │   ├── experiment_tracker.py
│   │   └── evals.py
│   └── cli.py                # Command-line tool
│
├── notebooks/
│   ├── 01_baseline_experiments.ipynb
│   └── 02_multi_agent_comparison.ipynb
│
├── README.md                 # Project overview
├── SETUP.md                  # Detailed setup guide
├── QUICKSTART.md            # Cheat sheet
├── IMPLEMENTATION.md        # What was built
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── setup.sh                # Quick setup script
```

## Quick Test

After setup, try this:

```python
from harness import llm_call, run_strategy

# Test Ollama
response = llm_call("Hi!", provider="ollama", model="llama3.2:latest")
print(response.text)

# Test debate strategy
result = run_strategy("debate", "Should we invest in solar?", 
                     n_debaters=3, provider="ollama")
print(result.output)
```

## Integration with Your Existing Project Files

You mentioned you have project files (project_plan.md, etc.). You can:

1. **Keep them separate** - This harness is just the code/tooling
2. **Merge them** - Copy your planning docs into this folder
3. **Create structure** - Make a `docs/` folder for planning, `code/` for harness

Suggested structure:
```
your-research-lab/
├── code/              # This harness
├── notebooks/         # Your experiments
├── docs/              # Your planning docs (project_plan.md, etc.)
├── experiments/       # Auto-generated logs
└── README.md
```

## Next Steps

1. ✅ Run `setup.sh`
2. ✅ Test with CLI: `python code/cli.py "test"`
3. ✅ Open notebook: `notebooks/01_baseline_experiments.ipynb`
4. ✅ Read `QUICKSTART.md` for common patterns

## Need Help?

- Installation issues? Check `SETUP.md`
- How to use? Check `QUICKSTART.md`
- How it works? Check `IMPLEMENTATION.md`
- Code details? Everything is commented

## Ready?

```bash
cd research-lab-setup
./setup.sh
source venv/bin/activate
jupyter notebook notebooks/01_baseline_experiments.ipynb
```

Let's build! 🚀
