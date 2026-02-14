.PHONY: help install install-dev test lint format clean build upload

help:
	@echo "Ollama Agent CLI - Development Commands"
	@echo ""
	@echo "  make install        Install package"
	@echo "  make install-dev    Install in development mode with dev dependencies"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linter (ruff)"
	@echo "  make format        Format code (black)"
	@echo "  make clean         Clean build artifacts"
	@echo "  make build         Build distribution packages"
	@echo "  make upload        Upload to PyPI"
	@echo ""

install:
	uv tool install . --reinstall

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check ollama_agent/

format:
	black ollama_agent/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf ollama_agent/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	twine upload dist/*

# Development shortcuts
dev: install-dev
	@echo "Development environment ready!"

check: lint test
	@echo "All checks passed!"
