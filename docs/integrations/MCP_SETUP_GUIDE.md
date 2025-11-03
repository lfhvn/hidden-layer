# MCP Server Setup Guide

**Last Updated**: 2025-11-03
**Purpose**: Configure Model Context Protocol (MCP) servers to give Claude direct access to your research data and external tools

---

## What Are MCP Servers?

MCP (Model Context Protocol) servers give Claude **direct access** to external data and services. Instead of you manually reading files or searching the web, Claude can do it automatically while helping you.

**Think of it like this**:
- **Without MCP**: "Claude, let me read this experiment file... ok here's the data... now analyze it"
- **With MCP**: "Claude, analyze my latest experiment" → Claude reads the file directly

---

## Quick Start (30 Minutes)

We're setting up 2 MCP servers:
1. **File System** - Access your experiment results, notebooks, code
2. **Brave Search** - Search for papers, benchmarks, research while analyzing

---

## Part 1: File System MCP (15 minutes)

### Why You Need This

**Without File System MCP**:
```
You: "Analyze my debate experiment from yesterday"
Claude: "Please copy the contents of the results file"
You: *Opens file, copies 500 lines*
Claude: "Thanks, analyzing..."
```

**With File System MCP**:
```
You: "Analyze experiments/debate_energy_20251103_143022_a3f9/"
Claude: *Reads config.json, results.jsonl, summary.json automatically*
Claude: "I found 47 results. The debate strategy achieved 82% accuracy
        with avg latency of 3.2s. Here are the key insights..."
```

### What Claude Can Do With It

✅ Read experiment results directly
✅ Analyze multiple experiments and compare
✅ Search through logs
✅ Read and modify notebooks
✅ Access any file in your project
✅ Generate reports from data

### Setup Instructions

#### Step 1: Find Your Claude Config File

**macOS**:
```bash
# Location of Claude Desktop config
~/.claude/claude_desktop_config.json

# If it doesn't exist, create it:
mkdir -p ~/.claude
touch ~/.claude/claude_desktop_config.json
```

**Linux**:
```bash
~/.config/Claude/claude_desktop_config.json
```

**Windows**:
```
%APPDATA%\Claude\claude_desktop_config.json
```

#### Step 2: Add File System MCP Configuration

Open the config file and add:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/YOUR_USERNAME/hidden-layer"
      ]
    }
  }
}
```

**⚠️ IMPORTANT**: Replace `/Users/YOUR_USERNAME/hidden-layer` with your **actual path** to the hidden-layer directory.

**To find your path**:
```bash
cd ~/hidden-layer  # or wherever your project is
pwd
# Copy this output and use it in the config
```

#### Step 3: Restart Claude Desktop

Close and reopen Claude Desktop app for changes to take effect.

#### Step 4: Test It Works

In Claude Desktop, try:
```
"List the files in my experiments directory"
```

If it works, Claude will show you the files! If not, see Troubleshooting below.

### Usage Examples

Once configured, you can ask Claude:

**Analyze Experiments**:
```
"Read experiments/debate_energy_20251103/summary.json and analyze the results"

"Compare the last 3 experiments in my experiments/ directory"

"What was the average latency across all debate experiments?"
```

**Work With Code**:
```
"Read code/harness/strategies.py and explain the debate strategy"

"Show me all TODOs in the codebase"

"Find all files that import 'llm_call'"
```

**Analyze Notebooks**:
```
"Summarize what notebooks/03_introspection_experiments.ipynb does"

"Are there any errors in my notebook outputs?"
```

### Security & Permissions

**What Claude Can Access**:
- Only the directory you specify (e.g., `/Users/you/hidden-layer`)
- Subdirectories within that path
- Can read AND write files

**What Claude CANNOT Access**:
- Files outside the specified directory
- System files
- Other projects

**Best Practices**:
- Only grant access to your project directory
- Don't use your home directory (`~`) unless necessary
- Be specific about the path

---

## Part 2: Brave Search MCP (15 minutes)

### Why You Need This

**The Problem**: Staying current with research while analyzing results

**Without Brave Search**:
```
You: "My ToM accuracy is 78%. What's state-of-the-art?"
Claude: "Based on my training data from Jan 2025, typical accuracy is 70-85%"
You: *Opens browser, searches papers, reads for 30 minutes*
You: "Ok I found this paper that says..."
```

**With Brave Search**:
```
You: "My ToM accuracy is 78%. What's state-of-the-art?"
Claude: *Searches automatically*
Claude: "I found 3 recent papers from 2025. The current SOTA on ToMBench
        is 89% using chain-of-thought prompting [link]. Here's a paper
        from last month with a new benchmark FANToM-G [link] that might
        be relevant. Should we try their approach?"
```

### What Claude Can Do With It

✅ Search for research papers while analyzing
✅ Find new benchmarks and datasets
✅ Discover related work automatically
✅ Get proper citations
✅ Stay current with latest methods
✅ Verify claims against current literature

### Setup Instructions

#### Step 1: Get Brave Search API Key

1. Go to https://brave.com/search/api/
2. Click "Get Started" or "Sign Up"
3. Choose the **Free tier** (2,000 queries/month - plenty for research)
4. Verify your email
5. Go to your dashboard
6. Copy your API key (looks like: `BSA...`)

**💡 Tip**: The free tier is sufficient for research use. You get 2,000 queries/month.

#### Step 2: Add Brave Search to Config

Open your `~/.claude/claude_desktop_config.json` and update it:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/YOUR_USERNAME/hidden-layer"
      ]
    },
    "brave-search": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-brave-search"
      ],
      "env": {
        "BRAVE_API_KEY": "YOUR_API_KEY_HERE"
      }
    }
  }
}
```

**⚠️ IMPORTANT**: Replace `YOUR_API_KEY_HERE` with your actual Brave API key.

**Security Note**: This config file is local to your machine. Your API key stays on your computer.

#### Step 3: Restart Claude Desktop

Close and reopen Claude Desktop.

#### Step 4: Test It Works

Try:
```
"Search for recent papers on multi-agent LLM strategies"
```

Claude should return relevant search results with links!

### Usage Examples

**Find Related Papers**:
```
"Search for papers on theory of mind in LLMs published in 2024-2025"

"Find the UICrit paper and give me the citation"

"What are the latest multi-agent strategies for LLMs?"
```

**Discover Benchmarks**:
```
"Search for theory of mind benchmarks besides ToMBench"

"Are there design critique datasets similar to UICrit?"

"Find benchmarks for testing LLM introspection"
```

**Research Specific Topics**:
```
"Search for papers on activation steering in transformers"

"Find implementations of debate strategies in LLMs"

"What's the current SOTA on ToMBench?"
```

**Get Citations**:
```
"Search for 'Anthropic introspection paper' and give me BibTeX"

"Find citation for OpenToM benchmark"
```

### How It Works With Your Research

Claude will **automatically search** when it's helpful:

**Example 1**: Analyzing Results
```
You: "Analyze my SELPHI results and suggest improvements"

Claude: *Reads your results*
Claude: *Automatically searches for recent ToM papers*
Claude: "Your false belief accuracy is 78%, which is above the 2024
        baseline of 72% but below recent SOTA of 89%. I found this
        paper [link] that uses chain-of-thought prompting and achieves
        +12% improvement. Should we try that approach?"
```

**Example 2**: Comparing Benchmarks
```
You: "Should I add more ToM benchmarks?"

Claude: *Searches automatically*
Claude: "I found 3 additional ToM benchmarks:
        1. FANToM-G (2024) - Focuses on second-order beliefs
        2. SocialToM (2025) - Real-world social scenarios
        3. ProToM (2024) - Pragmatic reasoning

        Based on your research questions, FANToM-G would complement
        ToMBench well. Here's the paper [link] and dataset [link]."
```

### Rate Limits

**Free Tier**: 2,000 queries/month
- That's ~65 searches/day
- Plenty for research use
- Claude only searches when actually helpful

**Monitor Usage**: Check your Brave dashboard periodically

---

## Troubleshooting

### File System MCP Not Working

**Issue**: Claude says "I don't have access to the file system"

**Solutions**:

1. **Check config path**:
   ```bash
   cat ~/.claude/claude_desktop_config.json
   # Verify the path is correct
   ```

2. **Verify directory exists**:
   ```bash
   ls -la /path/you/specified
   ```

3. **Check permissions**:
   ```bash
   # Make sure you can read the directory
   ls /path/to/hidden-layer/experiments
   ```

4. **Restart Claude Desktop**:
   - Fully quit the app (not just close window)
   - Reopen

5. **Check for JSON errors**:
   - Use a JSON validator: https://jsonlint.com
   - Common issue: Missing comma between servers

**Still not working?** Try this minimal config:
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/you/Desktop"]
    }
  }
}
```

Test with Desktop first, then switch to your project path.

### Brave Search MCP Not Working

**Issue**: Claude says searches aren't working

**Solutions**:

1. **Verify API key**:
   - Go to https://brave.com/search/api/
   - Check your key is active
   - Copy it again (might have copied wrong)

2. **Check environment variable**:
   ```json
   "env": {
     "BRAVE_API_KEY": "BSA..."  // Should start with BSA
   }
   ```

3. **Test API key manually**:
   ```bash
   curl -H "X-Subscription-Token: YOUR_API_KEY" \
     "https://api.search.brave.com/res/v1/web/search?q=test"
   ```

4. **Check rate limits**:
   - Log into Brave dashboard
   - See if you've hit your limit

### Both Servers Configured But Not Working

**Issue**: Config looks right but neither works

**Common causes**:

1. **npx not installed**:
   ```bash
   # Check if Node.js is installed
   node --version
   npm --version

   # If not, install from https://nodejs.org
   ```

2. **JSON syntax error**:
   ```bash
   # Validate your config
   cat ~/.claude/claude_desktop_config.json | python -m json.tool
   # Should output formatted JSON with no errors
   ```

3. **Wrong config file location**:
   ```bash
   # Make sure you're editing the right file
   # On macOS it's:
   ls -la ~/.claude/claude_desktop_config.json
   ```

### Getting Help

If you're still stuck:

1. Check your config matches the examples exactly
2. Try the minimal config first (just filesystem, just Desktop)
3. Check Claude Desktop → Settings → Advanced → Developer Tools → Console for errors
4. Restart your computer (seriously, sometimes this helps)

---

## Complete Working Example

Here's a complete, tested config with both servers:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/yourname/hidden-layer"
      ]
    },
    "brave-search": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-brave-search"
      ],
      "env": {
        "BRAVE_API_KEY": "BSAxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
      }
    }
  }
}
```

**Replace**:
- `/Users/yourname/hidden-layer` → Your actual project path
- `BSAxxxxxxxxxxxxxxxxxxxxxxxxxxxx` → Your actual Brave API key

---

## Testing Your Setup

### Full Integration Test

Once both servers are configured, try this comprehensive test:

```
"Do a complete research workflow:
1. List my experiment directories
2. Analyze the most recent experiment
3. Search for recent papers related to the strategy used
4. Compare my results to published benchmarks
5. Suggest improvements based on current research"
```

If this works, you're all set! Claude will:
- ✅ Read your files directly
- ✅ Analyze results
- ✅ Search for papers
- ✅ Give you research-backed recommendations

---

## What's Next?

Now that MCP is configured:

1. **See it in action**: Ask Claude to analyze experiments
2. **Add W&B integration**: Track experiments professionally (see `docs/integrations/WANDB_GUIDE.md`)
3. **Create skills**: Build custom automation (see `docs/integrations/SKILLS_GUIDE.md`)

---

## Quick Reference Card

Save this for easy access:

### File System Commands
```
"List experiments/"
"Read experiments/{name}/summary.json"
"Compare last 3 experiments"
"Search codebase for {term}"
"Analyze notebook {name}"
```

### Brave Search Commands
```
"Search for {topic} papers"
"Find benchmarks for {task}"
"What's SOTA on {benchmark}?"
"Get citation for {paper}"
"Find similar datasets to {name}"
```

### Combined Workflow
```
"Analyze my experiment and search for relevant papers"
"Compare my results to published benchmarks"
"Suggest improvements based on recent research"
```

---

**Questions?** See troubleshooting section above or check the main integration guide: `INTEGRATION_OPPORTUNITIES.md`
