"""
GraphGuard FastAPI Service
Real-time fraud detection API endpoint
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from typing import Dict, Any
import os

from ..src.schema import TransactionRequest, TransactionResponse, HealthResponse
from ..src.explain import FraudExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GraphGuard Fraud Detection API",
    description="Real-time fraud detection using graph-enhanced ML",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
model = None
preprocessor = None
graph_features = None
explainer = None
model_version = "20250209"  # TODO: Update with actual training date


@app.on_event("startup")
async def load_models():
    """Load trained models and artifacts on startup"""
    global model, preprocessor, graph_features, explainer
    
    try:
        artifacts_dir = "app/artifacts"
        
        # Load XGBoost model
        model_path = os.path.join(artifacts_dir, "model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found, using dummy model")
            model = None
            
        # Load preprocessor
        preprocessor_path = os.path.join(artifacts_dir, "encoders.joblib")
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logger.info("Preprocessor loaded successfully")
        else:
            logger.warning("Preprocessor file not found")
            preprocessor = None
            
        # Load graph features
        graph_path = os.path.join(artifacts_dir, "vecs.npy")
        if os.path.exists(graph_path):
            graph_features = np.load(graph_path, allow_pickle=True).item()
            logger.info("Graph features loaded successfully")
        else:
            logger.warning("Graph features file not found")
            graph_features = None
            
        # Initialize explainer
        if model is not None:
            feature_names = []  # TODO: Get actual feature names
            explainer = FraudExplainer(model, feature_names)
            logger.info("Explainer initialized")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        model_version=model_version,
        timestamp=datetime.now().isoformat()
    )


@app.post("/score", response_model=TransactionResponse)
async def score_transaction(request: TransactionRequest):
    """Score a transaction for fraud probability"""
    try:
        # TODO: Implement actual scoring logic:
        # 1. Preprocess transaction data
        # 2. Extract graph features
        # 3. Make prediction
        # 4. Generate explanation
        
        # Placeholder implementation
        probability = 0.15  # Random probability for demo
        threshold = 0.80
        label = "flag" if probability >= threshold else "safe"
        
        # Placeholder feature contributions
        top_features = [
            {"name": "device_id_freq", "contrib": 0.21},
            {"name": "node2vec_src_07", "contrib": 0.15},
            {"name": "amount_z", "contrib": 0.12}
        ]
        
        # Placeholder reason
        reason = "High device reuse and risky neighbors for src_account_id; large amount vs. history."
        
        # Placeholder graph context
        graph_context = {
            "src_degree": 27,
            "dst_degree": 11,
            "src_pagerank": 0.023,
            "dst_pagerank": 0.008
        }
        
        return TransactionResponse(
            transaction_id=request.transaction_id,
            probability=probability,
            label=label,
            threshold=threshold,
            top_features=top_features,
            reason=reason,
            graph_context=graph_context
        )
        
    except Exception as e:
        logger.error(f"Error scoring transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "GraphGuard Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "score": "/score"
        },
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
