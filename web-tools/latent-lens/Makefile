.PHONY: dev setup test clean

# Auto-detect Python command (python3 or python)
PYTHON := $(shell command -v python3 2> /dev/null || command -v python 2> /dev/null)
PIP := $(shell command -v pip3 2> /dev/null || command -v pip 2> /dev/null)

# Development
dev:
	@echo "🔬 Starting Latent Lens (local mode)..."
	@echo ""
	@echo "Run these commands in separate terminals:"
	@echo ""
	@echo "Terminal 1 (Backend):"
	@echo "  cd backend && source venv/bin/activate && uvicorn app.main:app --reload --port 8002"
	@echo ""
	@echo "Terminal 2 (Frontend):"
	@echo "  cd frontend && npm run dev"
	@echo ""
	@echo "Or use: make dev-backend (in terminal 1) and make dev-frontend (in terminal 2)"
	@echo ""
	@echo "Frontend: http://localhost:3002"
	@echo "Backend:  http://localhost:8002"

dev-backend:
	@echo "Starting backend on http://localhost:8002..."
	@cd backend && source venv/bin/activate && uvicorn app.main:app --reload --port 8002

dev-frontend:
	@echo "Starting frontend on http://localhost:3002..."
	@cd frontend && npm run dev

setup:
	@echo "🔧 Setting up Latent Lens..."
	@echo ""
	@echo "Creating environment files..."
	@cp backend/.env.example backend/.env || true
	@cp frontend/.env.example frontend/.env.local || true
	@echo ""
	@echo "Setting up backend..."
	@cd backend && $(PYTHON) -m venv venv && . venv/bin/activate && $(PIP) install -r requirements.txt
	@echo ""
	@echo "Setting up frontend..."
	@cd frontend && npm install
	@echo ""
	@echo "✅ Setup complete!"
	@echo ""
	@echo "📝 Next steps:"
	@echo "  1. Run 'make dev' to see how to start the servers"
	@echo "  2. Or run 'make dev-backend' and 'make dev-frontend' in separate terminals"

clean:
	rm -rf backend/venv
	rm -rf backend/__pycache__
	rm -rf frontend/.next
	rm -rf frontend/node_modules
