.PHONY: dev setup test clean deploy

# Development (without Docker)
dev:
	@echo "🚀 Starting Multi-Agent Arena (local mode)..."
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
	@echo "Starting Multi-Agent Arena with Docker..."
	@docker-compose up

dev-frontend:
	@echo "Starting frontend on http://localhost:3001..."
	@cd frontend && npm run dev

dev-backend:
	@echo "Starting backend on http://localhost:8000..."
	@cd backend && source venv/bin/activate && uvicorn app.main:app --reload --port 8000

# Setup
setup:
	@echo "🔧 Setting up Multi-Agent Arena..."
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
	@echo "  3. Or run 'make dev-backend' and 'make dev-frontend' in separate terminals"

# Testing
test:
	cd backend && pytest
	cd frontend && npm test

# Clean
clean:
	rm -rf backend/venv
	rm -rf backend/__pycache__
	rm -rf frontend/.next
	rm -rf frontend/node_modules

# Deployment
deploy-backend:
	@echo "Deploying backend to Railway..."
	cd backend && railway up

deploy-frontend:
	@echo "Deploying frontend to Vercel..."
	cd frontend && vercel deploy --prod

deploy: deploy-backend deploy-frontend
	@echo "✅ Deployment complete!"
