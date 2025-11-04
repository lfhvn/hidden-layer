.PHONY: help dev setup test clean

help:
	@echo "🎛️  Steerability Dashboard"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup       - Initial setup (create venv, install deps)"
	@echo "  make dev         - Show how to start development servers"
	@echo "  make dev-backend - Start backend server"
	@echo "  make dev-frontend- Start frontend server"
	@echo "  make dev-docker  - Start with Docker"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Clean up"

setup:
	@echo "🔧 Setting up Steerability Dashboard..."
	@echo ""
	@echo "Creating environment files..."
	@cp backend/.env.example backend/.env || true
	@cp frontend/.env.example frontend/.env.local || true
	@echo ""
	@echo "Setting up backend..."
	@cd backend && python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt
	@echo ""
	@echo "Setting up frontend..."
	@cd frontend && npm install
	@echo ""
	@echo "✅ Setup complete!"
	@echo ""
	@echo "📝 Next steps:"
	@echo "  1. Edit backend/.env and add your ANTHROPIC_API_KEY"
	@echo "  2. Run 'make dev' to see how to start the servers"

dev:
	@echo "🚀 Starting Steerability Dashboard (local mode)..."
	@echo ""
	@echo "Run these commands in separate terminals:"
	@echo ""
	@echo "Terminal 1 (Backend):"
	@echo "  cd backend && source venv/bin/activate && uvicorn app.main:app --reload --port 8000"
	@echo ""
	@echo "Terminal 2 (Frontend):"
	@echo "  cd frontend && npm run dev"
	@echo ""
	@echo "Or use: make dev-backend (in terminal 1) and make dev-frontend (in terminal 2)"

dev-docker:
	@echo "Starting Steerability Dashboard with Docker..."
	@docker-compose up --build

dev-backend:
	@echo "Starting backend on http://localhost:8000..."
	@cd backend && source venv/bin/activate && uvicorn app.main:app --reload --port 8000

dev-frontend:
	@echo "Starting frontend on http://localhost:3000..."
	@cd frontend && npm run dev

test:
	cd backend && pytest -v --cov=app

clean:
	rm -rf backend/venv
	rm -rf backend/__pycache__
	rm -rf frontend/.next
	rm -rf frontend/node_modules
