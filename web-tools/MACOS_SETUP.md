# macOS Setup Guide

Quick guide for running web tools on macOS.

## TL;DR - Create Python Alias (Recommended)

If you hate typing `python3`, create an alias:

```bash
# For zsh (default on modern macOS)
echo "alias python='python3'" >> ~/.zshrc
echo "alias pip='pip3'" >> ~/.zshrc
source ~/.zshrc

# For bash (older macOS)
echo "alias python='python3'" >> ~/.bash_profile
echo "alias pip='pip3'" >> ~/.bash_profile
source ~/.bash_profile

# Verify
python --version  # Should show Python 3.x
```

**Good news**: The Makefiles now auto-detect `python` or `python3`, so they work either way!

## Prerequisites

### 1. Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python 3
```bash
brew install python3
# Verify installation
python3 --version  # Should show Python 3.11+
```

### 3. Install Node.js
```bash
brew install node
# Verify installation
node --version   # Should show v18+
npm --version    # Should show v9+
```

## Setup Any Tool

All tools follow the same pattern:

```bash
cd web-tools/[tool-name]
make setup
```

This will:
- Create Python virtual environment (using `python3`)
- Install Python dependencies
- Install Node.js dependencies
- Create `.env` files from templates

## Running Tools

### Multi-Agent Arena

```bash
cd web-tools/multi-agent-arena

# Setup (one time)
make setup

# Add your API key to backend/.env
echo "ANTHROPIC_API_KEY=sk-ant-your-key" >> backend/.env

# Run (in separate terminals)
# Terminal 1:
make dev-backend

# Terminal 2:
make dev-frontend

# Visit http://localhost:3001
```

### Steerability Dashboard

```bash
cd web-tools/steerability

make setup
echo "ANTHROPIC_API_KEY=sk-ant-your-key" >> backend/.env

# Terminal 1:
make dev-backend

# Terminal 2:
make dev-frontend

# Visit http://localhost:3000
```

### Latent Lens

```bash
cd web-tools/latent-lens

make setup

# Terminal 1:
make dev-backend

# Terminal 2:
make dev-frontend

# Visit http://localhost:3002
```

## Common Issues

### Issue: `python: command not found`

**Solution**: The Makefiles now use `python3` (fixed). If you still see this, ensure Python 3 is installed:
```bash
brew install python3
python3 --version
```

### Issue: `ModuleNotFoundError` when starting backend

**Solution**: Activate the virtual environment first:
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Issue: `npm: command not found`

**Solution**: Install Node.js:
```bash
brew install node
node --version
```

### Issue: Port already in use

**Solution**: Kill the process using the port:
```bash
# Find process on port 8000 (backend)
lsof -ti:8000 | xargs kill -9

# Find process on port 3000/3001/3002 (frontend)
lsof -ti:3001 | xargs kill -9
```

### Issue: `ImportError: No module named 'fastapi'`

**Solution**: Make sure you're in the virtual environment:
```bash
cd backend
source venv/bin/activate
# You should see (venv) in your prompt
pip install -r requirements.txt
```

### Issue: Frontend won't start - `EACCES` permission error

**Solution**: Fix npm permissions:
```bash
sudo chown -R $(whoami) ~/.npm
sudo chown -R $(whoami) /usr/local/lib/node_modules
```

## Tips for macOS

### Use iTerm2 for Better Terminal Experience
```bash
brew install --cask iterm2
```

### Split Terminals
- **iTerm2**: `Cmd+D` (split vertically), `Cmd+Shift+D` (split horizontally)
- **Terminal**: Open new tabs with `Cmd+T`

### Use tmux for Multiple Panes (Optional)
```bash
brew install tmux

# Start tmux
tmux

# Split panes
Ctrl+b then "   # Split horizontally
Ctrl+b then %   # Split vertically

# Switch panes
Ctrl+b then arrow keys
```

### Environment Variables

Add to `~/.zshrc` (or `~/.bash_profile` for bash):
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Then reload:
```bash
source ~/.zshrc
```

## Quick Start Script (Optional)

Create `~/start-multi-agent.sh`:
```bash
#!/bin/bash

# Start Multi-Agent Arena
cd ~/path/to/hidden-layer/web-tools/multi-agent-arena

# Start backend in background
osascript -e 'tell application "Terminal" to do script "cd '"$(pwd)"'/backend && source venv/bin/activate && uvicorn app.main:app --reload --port 8000"'

# Start frontend in new window
osascript -e 'tell application "Terminal" to do script "cd '"$(pwd)"'/frontend && npm run dev"'

echo "✅ Multi-Agent Arena starting..."
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3001"
```

Make executable:
```bash
chmod +x ~/start-multi-agent.sh
```

Run:
```bash
~/start-multi-agent.sh
```

## Debugging

### Check if ports are open
```bash
lsof -i :8000  # Backend
lsof -i :3001  # Frontend
```

### Check Python installation
```bash
which python3
python3 --version
python3 -m pip --version
```

### Check Node installation
```bash
which node
node --version
npm --version
```

### View backend logs
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000 --log-level debug
```

### View frontend logs
```bash
cd frontend
npm run dev
# Logs will show in terminal
```

## Performance Tips

### Use Haiku Model (Faster & Cheaper)
The default is already Haiku (`claude-3-haiku-20240307`), which is:
- **10x faster** than Sonnet
- **20x cheaper** than Sonnet
- Still very capable

### Clear Node Cache (If Slow)
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Clear Python Cache
```bash
cd backend
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

## Getting Help

If you encounter issues:
1. Check this guide first
2. Read `web-tools/QUICKSTART.md`
3. Check tool-specific READMEs
4. Open an issue on GitHub

## Useful Commands

```bash
# Check what's running
ps aux | grep uvicorn
ps aux | grep node

# Kill all Python processes
pkill -9 python3

# Kill all Node processes
pkill -9 node

# Check disk space
df -h

# Check memory usage
top

# Monitor network
netstat -an | grep LISTEN
```

---

**Happy developing on macOS! 🍎**
