# Hidden Layer Integrations

This directory contains guides for enhancing the Hidden Layer platform with powerful integrations.

## Overview

Hidden Layer supports several integrations to accelerate your research workflow:

1. **MCP Servers** - Give Claude direct access to external data and services
2. **Claude Skills** - Reusable automation modules for common workflows
3. **Weights & Biases** - Professional experiment tracking and visualization

## Available Guides

### MCP Servers

**📁 [MCP_SETUP_GUIDE.md](MCP_SETUP_GUIDE.md)**

Configure Model Context Protocol servers for:
- **File System MCP**: Direct access to experiment results, notebooks, and code
- **Brave Search MCP**: Automatic literature review while analyzing results

**Why use MCPs?**
- Zero-friction access to external data
- Automatic context retrieval
- No manual copy/paste of experiment data

### Claude Skills

**⚡ [SKILLS_GUIDE.md](SKILLS_GUIDE.md)**

Documentation for included skills:
- **analyze-experiment**: Automatically analyze experiment results with research context
- **run-benchmark**: Systematically evaluate models across benchmarks
- **introspection-sweep**: Test activation steering across parameter spaces
- **write-paper**: Generate LaTeX paper sections from experiments

**Why use skills?**
- Automate repetitive workflows
- Consistent analysis methodology
- Reduce cognitive load

### Experiment Tracking

**📊 [WANDB_GUIDE.md](WANDB_GUIDE.md)**

Integrate Weights & Biases for:
- Professional dashboards
- Real-time monitoring
- Easy comparison across experiments
- Automatic hyperparameter tracking

**Why use W&B?**
- Industry-standard experiment tracking
- Beautiful visualizations
- Share results with collaborators
- Track experiments across days/weeks

## Quick Start

### 1. Install W&B (Optional)

```bash
pip install wandb
wandb login
```

### 2. Configure MCP Servers

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/hidden-layer"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

### 3. Use Claude Skills

Skills are already included in `.claude/skills/`. Just invoke them:

```
User: "Use analyze-experiment skill with directory=experiments/my_run_20241103/"
```

## Integration Workflow

### Typical Research Session

1. **Design experiment** in notebook
2. **Run experiment** with W&B tracking enabled:
   ```python
   from harness import get_tracker, ExperimentConfig

   config = ExperimentConfig(
       experiment_name="debate_vs_single",
       strategy="debate",
       provider="ollama",
       model="llama3.1:8b"
   )

   tracker = get_tracker(use_wandb=True)
   tracker.start_experiment(config)
   # ... run your experiment ...
   tracker.finish_experiment()
   ```

3. **Monitor progress** in W&B dashboard (real-time)
4. **Analyze results** with Claude using:
   - File System MCP (reads experiment files automatically)
   - Brave Search MCP (finds related research)
   - analyze-experiment skill (generates insights)

5. **Write paper sections** using write-paper skill

### Example Commands

**Analyze an experiment:**
```
User: "Use analyze-experiment skill with directory=experiments/debate_llama3.1_20241103_a1b2c3d4"
```

**Run systematic benchmark:**
```
User: "Use run-benchmark skill with benchmark=tombench models=llama3.2:3b,llama3.1:8b strategies=single,debate"
```

**Parameter sweep:**
```
User: "Use introspection-sweep skill with layers=15,20,25 concepts=honesty,creativity strengths=1.0,2.0,3.0"
```

## Implementation Status

✅ **Completed**:
- W&B integration in experiment tracker
- File System MCP guide
- Brave Search MCP guide
- 4 essential Claude Skills
- Comprehensive documentation

📋 **Recommended Next Steps**:
1. Install W&B: `pip install wandb && wandb login`
2. Configure MCP servers in Claude Desktop
3. Test with small experiment using W&B
4. Try analyze-experiment skill on results

## Additional Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [MCP Documentation](https://modelcontextprotocol.io/)
- [Claude Skills Documentation](https://docs.anthropic.com/claude/docs/skills)

## Troubleshooting

### W&B Issues

**Problem**: "wandb module not found"
```bash
pip install wandb==0.18.7
```

**Problem**: "wandb.errors.UsageError: api_key not configured"
```bash
wandb login
```

### MCP Issues

**Problem**: MCP servers not appearing in Claude
- Check `claude_desktop_config.json` syntax (valid JSON)
- Restart Claude Desktop
- Check server logs in Claude → Settings → Developer

**Problem**: File System MCP can't access files
- Verify path in config matches your installation
- Use absolute paths, not relative

### Skills Issues

**Problem**: Skills not found
- Verify `.claude/skills/` directory exists
- Check skill file names match documentation
- Ensure skill files have `.md` extension

## Questions?

See individual guides for detailed troubleshooting and examples:
- [MCP_SETUP_GUIDE.md](MCP_SETUP_GUIDE.md)
- [WANDB_GUIDE.md](WANDB_GUIDE.md)
- [SKILLS_GUIDE.md](SKILLS_GUIDE.md)
