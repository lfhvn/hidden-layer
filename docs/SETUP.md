# Advanced / Manual Setup Guide
 
 This guide is for users who want to manually configure their environment or need specific setups (e.g., custom MLX builds, non-standard paths).
 
 **For most users, just run:**
 ```bash
 ./setup.sh
 ```
 
 ---
 
 ## Manual Installation
 
 ### 1. Python Environment
 Requires Python 3.10+.
 ```bash
 python3 -m venv venv
 source venv/bin/activate
 pip install --upgrade pip
 ```
 
 ### 2. Dependencies
 ```bash
 pip install -r requirements.txt
 ```
 *Note: On Apple Silicon, this includes MLX. On other platforms, MLX is skipped.*
 
 ### 3. Ollama (Local Models)
 1. Install [Ollama](https://ollama.ai).
 2. Start the server: `ollama serve`
 3. Pull the default model: `ollama pull llama3.2:latest`
 
 ### 4. API Keys (Optional)
 If you want to use Anthropic or OpenAI:
 ```bash
 cp .env.example .env
 # Edit .env with your keys
 ```
 
 ## Troubleshooting
 
 - **MLX Issues**: Ensure you are on macOS 13.3+ with an Apple Silicon chip. Python 3.13 is not yet supported for MLX.
 - **Ollama Connection**: Ensure `ollama serve` is running in a separate terminal or background.
