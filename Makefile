# GraphGuard Fraud Detection Makefile
# Make targets for common development tasks

.PHONY: help install train serve-api serve-dashboard test clean

# Default target
help:
	@echo "GraphGuard Fraud Detection - Available Commands:"
	@echo ""
	@echo "  install          - Install Python dependencies"
	@echo "  train            - Train the fraud detection model"
	@echo "  serve-api        - Start the FastAPI service"
	@echo "  serve-dashboard  - Start the Streamlit dashboard"
	@echo "  test             - Run all tests"
	@echo "  clean            - Clean up generated files"
	@echo "  help             - Show this help message"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Train the model
train:
	@echo "Training fraud detection model..."
	@echo "TODO: Implement training pipeline"
	@echo "This will:"
	@echo "  1. Load and preprocess data"
	@echo "  2. Build graph features with Node2Vec"
	@echo "  3. Train baseline XGBoost"
	@echo "  4. Train hybrid model (tabular + graph)"
	@echo "  5. Save artifacts to app/artifacts/"
	@echo ""
	@echo "Run: python -m src.train"

# Start the FastAPI service
serve-api:
	@echo "Starting FastAPI service..."
	@echo "API will be available at: http://localhost:8000"
	@echo "API docs at: http://localhost:8000/docs"
	@echo ""
	@echo "Run: uvicorn app.api:app --reload --port 8000"

# Start the Streamlit dashboard
serve-dashboard:
	@echo "Starting Streamlit dashboard..."
	@echo "Dashboard will be available at: http://localhost:8501"
	@echo ""
	@echo "Run: streamlit run dashboard/app.py"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v
	@echo "Tests completed!"

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf */*/__pycache__/
	rm -rf .pytest_cache/
	rm -rf app/artifacts/*.pkl
	rm -rf app/artifacts/*.joblib
	rm -rf app/artifacts/*.npy
	@echo "Cleanup completed!"

# Development setup
dev-setup: install
	@echo "Setting up development environment..."
	@echo "Creating necessary directories..."
	mkdir -p app/artifacts
	mkdir -p data/raw
	mkdir -p data/processed
	@echo "Development environment ready!"

# Quick start (install + dev-setup)
quick-start: dev-setup
	@echo "GraphGuard is ready for development!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Add your dataset to data/raw/"
	@echo "  2. Run 'make train' to train the model"
	@echo "  3. Run 'make serve-api' to start the API"
	@echo "  4. Run 'make serve-dashboard' to start the dashboard"
	@echo "  5. Run 'make test' to verify everything works"
