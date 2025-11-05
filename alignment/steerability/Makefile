.PHONY: help dev build test clean

help:
	@echo "Steerability Dashboard"
	@echo "Available targets:"
	@echo "  make dev        - Start development environment"
	@echo "  make build      - Build Docker images"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean up"

dev:
	docker-compose up --build

build:
	docker-compose build

test:
	cd backend && pytest -v --cov=app

clean:
	docker-compose down -v
	rm -rf backend/__pycache__ frontend/.next
