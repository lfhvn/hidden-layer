.PHONY: dev setup test clean deploy

# Development
dev:
	@echo "Starting Multi-Agent Arena..."
	@docker-compose up

dev-frontend:
	cd frontend && npm install && npm run dev

dev-backend:
	cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && uvicorn app.main:app --reload --port 8000

# Setup
setup:
	@echo "Setting up Multi-Agent Arena..."
	@cp backend/.env.example backend/.env
	@cp frontend/.env.example frontend/.env.local
	@cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
	@cd frontend && npm install
	@echo "✅ Setup complete! Edit backend/.env and frontend/.env.local, then run 'make dev'"

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
