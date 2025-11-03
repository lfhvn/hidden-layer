.PHONY: help dev build test clean install backend frontend

# Default target
help:
	@echo "Latent Lens - Interactive LLM Interpretability"
	@echo ""
	@echo "Available targets:"
	@echo "  make dev        - Start development environment"
	@echo "  make build      - Build Docker images"
	@echo "  make test       - Run tests"
	@echo "  make install    - Install dependencies"
	@echo "  make backend    - Run backend only"
	@echo "  make frontend   - Run frontend only"
	@echo "  make clean      - Clean up containers and volumes"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"

# Development environment
dev:
	docker-compose up --build

# Build images
build:
	docker-compose build

# Install dependencies locally
install:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements-dev.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

# Run backend only
backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run frontend only
frontend:
	cd frontend && npm run dev

# Run tests
test:
	@echo "Running backend tests..."
	cd backend && pytest -v --cov=app --cov-report=term-missing
	@echo "Running frontend type check..."
	cd frontend && npm run type-check

# Clean up
clean:
	docker-compose down -v
	rm -rf backend/__pycache__ backend/app/__pycache__
	rm -rf backend/.pytest_cache
	rm -rf frontend/.next frontend/node_modules
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Linting
lint:
	@echo "Linting backend..."
	cd backend && ruff check app/
	cd backend && mypy app/
	@echo "Linting frontend..."
	cd frontend && npm run lint

# Format code
format:
	@echo "Formatting backend..."
	cd backend && black app/
	cd backend && ruff check --fix app/
	@echo "Formatting frontend..."
	cd frontend && npm run lint -- --fix

# Database migrations (for future use)
migrate:
	@echo "Running database migrations..."
	cd backend && alembic upgrade head

# Start services in background
up:
	docker-compose up -d

# Stop services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Restart services
restart:
	docker-compose restart
