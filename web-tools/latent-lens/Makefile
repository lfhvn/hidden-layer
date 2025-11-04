.PHONY: dev setup test clean

# Development
dev:
	@echo "Starting Latent Lens..."
	@echo "Frontend will be at http://localhost:3002"
	@echo "Backend will be at http://localhost:8002"
	@echo ""
	@echo "Run these in separate terminals:"
	@echo "  cd backend && uvicorn app.main:app --reload --port 8002"
	@echo "  cd frontend && npm run dev"

setup:
	@echo "Setting up Latent Lens..."
	@cp backend/.env.example backend/.env
	@cp frontend/.env.example frontend/.env.local
	@cd backend && pip install -r requirements.txt
	@cd frontend && npm install
	@echo "✅ Setup complete! Run 'make dev' to start"

clean:
	rm -rf backend/__pycache__
	rm -rf frontend/.next
	rm -rf frontend/node_modules
