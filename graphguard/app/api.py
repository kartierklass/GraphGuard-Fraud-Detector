"""
GraphGuard FastAPI Application
Real-time fraud detection API using hybrid XGBoost + Graph features
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import uvicorn
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GraphGuard Fraud Detection API",
    description="Real-time fraud detection using graph-enhanced machine learning",
    version="1.0.0"
)

# Global variables for loaded artifacts
model = None
feature_names = None
baseline_model = None

# Path to artifacts
ARTIFACTS_DIR = Path("app/artifacts")

# Pydantic Models for Request/Response
class TransactionRequest(BaseModel):
    """Transaction data for fraud scoring"""
    # Core transaction features
    TransactionAmt: float = Field(..., description="Transaction amount")
    ProductCD: str = Field(..., description="Product code")
    
    # Card features
    card1: Optional[int] = Field(None, description="Card feature 1")
    card2: Optional[float] = Field(None, description="Card feature 2") 
    card3: Optional[float] = Field(None, description="Card feature 3")
    card4: Optional[str] = Field(None, description="Card feature 4")
    card5: Optional[float] = Field(None, description="Card feature 5")
    card6: Optional[str] = Field(None, description="Card feature 6")
    
    # Address features
    addr1: Optional[float] = Field(None, description="Address 1")
    addr2: Optional[float] = Field(None, description="Address 2")
    
    # Distance features
    dist1: Optional[float] = Field(None, description="Distance 1")
    dist2: Optional[float] = Field(None, description="Distance 2")
    
    # Email domain features
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    
    # Device features
    DeviceType: Optional[str] = Field(None, description="Device type")
    DeviceInfo: Optional[str] = Field(None, description="Device info")
    
    # C features (anonymized)
    C1: Optional[float] = Field(None, description="C feature 1")
    C2: Optional[float] = Field(None, description="C feature 2")
    C3: Optional[float] = Field(None, description="C feature 3")
    C4: Optional[float] = Field(None, description="C feature 4")
    C5: Optional[float] = Field(None, description="C feature 5")
    C6: Optional[float] = Field(None, description="C feature 6")
    C7: Optional[float] = Field(None, description="C feature 7")
    C8: Optional[float] = Field(None, description="C feature 8")
    C9: Optional[float] = Field(None, description="C feature 9")
    C10: Optional[float] = Field(None, description="C feature 10")
    C11: Optional[float] = Field(None, description="C feature 11")
    C12: Optional[float] = Field(None, description="C feature 12")
    C13: Optional[float] = Field(None, description="C feature 13")
    C14: Optional[float] = Field(None, description="C feature 14")
    
    # D features (time deltas)
    D1: Optional[float] = Field(None, description="D feature 1")
    D2: Optional[float] = Field(None, description="D feature 2")
    D3: Optional[float] = Field(None, description="D feature 3")
    D4: Optional[float] = Field(None, description="D feature 4")
    D5: Optional[float] = Field(None, description="D feature 5")
    D6: Optional[float] = Field(None, description="D feature 6")
    D7: Optional[float] = Field(None, description="D feature 7")
    D8: Optional[float] = Field(None, description="D feature 8")
    D9: Optional[float] = Field(None, description="D feature 9")
    D10: Optional[float] = Field(None, description="D feature 10")
    D11: Optional[float] = Field(None, description="D feature 11")
    D12: Optional[float] = Field(None, description="D feature 12")
    D13: Optional[float] = Field(None, description="D feature 13")
    D14: Optional[float] = Field(None, description="D feature 14")
    D15: Optional[float] = Field(None, description="D feature 15")
    
    # M features (match features)
    M1: Optional[str] = Field(None, description="M feature 1")
    M2: Optional[str] = Field(None, description="M feature 2")
    M3: Optional[str] = Field(None, description="M feature 3")
    M4: Optional[str] = Field(None, description="M feature 4")
    M5: Optional[str] = Field(None, description="M feature 5")
    M6: Optional[str] = Field(None, description="M feature 6")
    M7: Optional[str] = Field(None, description="M feature 7")
    M8: Optional[str] = Field(None, description="M feature 8")
    M9: Optional[str] = Field(None, description="M feature 9")

class FeatureContribution(BaseModel):
    """Individual feature contribution to prediction"""
    feature_name: str = Field(..., description="Name of the feature")
    value: float = Field(..., description="Feature value")
    importance: float = Field(..., description="Feature importance score")

class GraphContext(BaseModel):
    """Graph-based context for the prediction"""
    connected_entities: int = Field(default=0, description="Number of connected entities")
    risk_score: float = Field(default=0.0, description="Graph-based risk score")
    network_flags: List[str] = Field(default_factory=list, description="Network-based risk flags")

class TransactionResponse(BaseModel):
    """Fraud prediction response"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    prediction: str = Field(..., description="Fraud prediction (FRAUD/LEGITIMATE)")
    confidence: str = Field(..., description="Prediction confidence (HIGH/MEDIUM/LOW)")
    risk_score: int = Field(..., description="Risk score (0-100)")
    reason: str = Field(..., description="Human-readable explanation")
    top_features: List[FeatureContribution] = Field(..., description="Top contributing features")
    graph_context: GraphContext = Field(..., description="Graph-based context")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Health check timestamp")

def load_artifacts():
    """Load model and preprocessing artifacts at startup"""
    global model, feature_names, baseline_model
    
    try:
        # Load hybrid model
        model_path = ARTIFACTS_DIR / "hybrid_model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info("✅ Hybrid model loaded successfully")
        else:
            # Fallback to baseline model
            baseline_path = ARTIFACTS_DIR / "baseline_model.pkl"
            if baseline_path.exists():
                model = joblib.load(baseline_path)
                logger.info("✅ Baseline model loaded as fallback")
            else:
                raise FileNotFoundError("No model files found")
        
        # Load feature names
        feature_names_path = ARTIFACTS_DIR / "hybrid_feature_names.pkl"
        if feature_names_path.exists():
            feature_names = joblib.load(feature_names_path)
            logger.info(f"✅ Feature names loaded: {len(feature_names)} features")
        else:
            # Fallback to baseline feature names
            baseline_features_path = ARTIFACTS_DIR / "baseline_feature_names.pkl"
            if baseline_features_path.exists():
                feature_names = joblib.load(baseline_features_path)
                logger.info(f"✅ Baseline feature names loaded: {len(feature_names)} features")
            else:
                logger.warning("⚠️ No feature names found, using model features")
                feature_names = None
                
        logger.info("🚀 All artifacts loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading artifacts: {str(e)}")
        return False

def preprocess_transaction(transaction_data: dict) -> pd.DataFrame:
    """Preprocess transaction data to match training format"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Add all missing numeric features with default values
        numeric_defaults = {}
        
        # V features (V1-V339)
        for i in range(1, 340):
            numeric_defaults[f'V{i}'] = 0.0
            
        # Add any missing core features with proper defaults
        core_features = {
            'card2': 0.0, 'card3': 0.0, 'card5': 0.0,
            'addr2': 0.0, 'dist1': 0.0, 'dist2': 0.0,
            'C3': 0.0, 'C4': 0.0, 'C5': 0.0, 'C6': 0.0, 'C7': 0.0, 'C8': 0.0, 
            'C9': 0.0, 'C10': 0.0, 'C11': 0.0, 'C12': 0.0, 'C13': 0.0, 'C14': 0.0,
            'D1': 0.0, 'D2': 0.0, 'D3': 0.0, 'D4': 0.0, 'D5': 0.0, 'D6': 0.0,
            'D7': 0.0, 'D8': 0.0, 'D9': 0.0, 'D10': 0.0, 'D11': 0.0, 'D12': 0.0,
            'D13': 0.0, 'D14': 0.0, 'D15': 0.0
        }
        
        # Combine all defaults
        numeric_defaults.update(core_features)
        
        # Add missing features to DataFrame
        for feature, default_value in numeric_defaults.items():
            if feature not in df.columns:
                df[feature] = default_value
            elif df[feature].isnull().all():
                df[feature] = default_value
        
        # Handle categorical features - frequency encoding simulation
        categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 
                          'DeviceType', 'DeviceInfo', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
        
        # Simple frequency encoding (in production, use saved encodings from training)
        categorical_encodings = {
            'ProductCD': {'W': 73698, 'C': 31506, 'H': 5000, 'S': 4000, 'R': 3000},
            'card4': {'visa': 326400, 'mastercard': 94700, 'american express': 13500, 'discover': 8200},
            'card6': {'debit': 317900, 'credit': 80200, 'charge card': 600},
            'DeviceType': {'desktop': 146600, 'mobile': 105900},
            'P_emaildomain': {'gmail.com': 76500, 'yahoo.com': 24600, 'hotmail.com': 14800, 'anonymous.com': 11100},
            'R_emaildomain': {'gmail.com': 33200, 'yahoo.com': 8400, 'hotmail.com': 5100}
        }
        
        for col in categorical_cols:
            if col in df.columns:
                # Fill missing with 'MISSING'
                df[col] = df[col].fillna('MISSING')
                # Apply frequency encoding
                if col in categorical_encodings:
                    df[col] = df[col].map(categorical_encodings[col]).fillna(1)  # Unknown = 1
                else:
                    df[col] = 1  # Default encoding for unknown categories
        
        # Handle numerical features - fill missing with -999 and ensure proper types
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numerical_cols] = df[numerical_cols].fillna(-999.0)
        
        # Ensure all non-categorical columns are numeric
        for col in df.columns:
            if col not in categorical_cols:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-999.0)
                else:
                    df[col] = df[col].astype(float)
        
        # Add placeholder graph features (all zeros for single transaction scoring)
        graph_feature_names = [
            'degree_centrality_card1', 'pagerank_card1', 'clustering_coefficient_card1', 'betweenness_centrality_card1',
            'degree_centrality_addr1', 'pagerank_addr1', 'clustering_coefficient_addr1', 'betweenness_centrality_addr1',
            'degree_centrality_p_emaildomain', 'pagerank_p_emaildomain', 'clustering_coefficient_p_emaildomain', 'betweenness_centrality_p_emaildomain',
            'degree_centrality_productcd', 'pagerank_productcd', 'clustering_coefficient_productcd', 'betweenness_centrality_productcd',
            'degree_centrality_devicetype', 'pagerank_devicetype', 'clustering_coefficient_devicetype', 'betweenness_centrality_devicetype',
            'degree_centrality_card4', 'pagerank_card4', 'clustering_coefficient_card4', 'betweenness_centrality_card4'
        ]
        
        for feature in graph_feature_names:
            df[feature] = 0.0
        
        # Ensure all expected features are present
        if feature_names:
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            # Reorder columns to match training order
            df = df.reindex(columns=feature_names, fill_value=0.0)
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")

def generate_explanation(prediction_proba: float, feature_contributions: List[dict]) -> str:
    """Generate human-readable explanation for the prediction"""
    if prediction_proba >= 0.7:
        risk_level = "HIGH RISK"
        explanation = f"This transaction shows strong fraud indicators ({prediction_proba:.1%} probability). "
    elif prediction_proba >= 0.3:
        risk_level = "MEDIUM RISK"
        explanation = f"This transaction shows moderate fraud indicators ({prediction_proba:.1%} probability). "
    else:
        risk_level = "LOW RISK"
        explanation = f"This transaction appears legitimate ({prediction_proba:.1%} fraud probability). "
    
    # Add top contributing factors
    if feature_contributions:
        top_feature = feature_contributions[0]
        explanation += f"Key factor: {top_feature['feature_name']} "
        
        if len(feature_contributions) >= 2:
            second_feature = feature_contributions[1]
            explanation += f"and {second_feature['feature_name']}. "
    
    explanation += f"Risk assessment: {risk_level}."
    return explanation

@app.on_event("startup")
async def startup_event():
    """Load artifacts when the application starts"""
    logger.info("🚀 Starting GraphGuard API...")
    success = load_artifacts()
    if not success:
        logger.error("❌ Failed to load artifacts. API may not function correctly.")
    else:
        logger.info("✅ GraphGuard API ready for fraud detection!")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/score", response_model=TransactionResponse)
async def score_transaction(transaction: TransactionRequest):
    """Score a transaction for fraud probability"""
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Generate transaction ID
        transaction_id = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Preprocess the transaction
        df = preprocess_transaction(transaction.dict())
        
        # Make prediction
        # Ensure all columns are numeric for XGBoost
        df_numeric = df.copy()
        
        # Debug: Log column types
        logger.info(f"DataFrame shape: {df_numeric.shape}")
        object_cols = df_numeric.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            logger.error(f"Object columns found: {object_cols[:10]}...")  # Show first 10
        
        # Force all columns to numeric
        for col in df_numeric.columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce').fillna(0.0).astype(float)
        
        fraud_proba = model.predict_proba(df_numeric)[0, 1]  # Probability of fraud (class 1)
        prediction = "FRAUD" if fraud_proba >= 0.5 else "LEGITIMATE"
        
        # Determine confidence
        if fraud_proba >= 0.8 or fraud_proba <= 0.2:
            confidence = "HIGH"
        elif fraud_proba >= 0.6 or fraud_proba <= 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Calculate risk score (0-100)
        risk_score = int(fraud_proba * 100)
        
        # Get feature importance (top 5 features)
        if hasattr(model, 'feature_importances_') and feature_names:
            feature_importance = list(zip(feature_names, model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            top_features = []
            for i, (feature_name, importance) in enumerate(feature_importance[:5]):
                if i < len(df.columns):
                    feature_value = df.iloc[0][feature_name] if feature_name in df.columns else 0.0
                    top_features.append(FeatureContribution(
                        feature_name=feature_name,
                        value=feature_value,
                        importance=importance
                    ))
        else:
            # Fallback top features
            top_features = [
                FeatureContribution(feature_name="TransactionAmt", value=transaction.TransactionAmt, importance=0.15),
                FeatureContribution(feature_name="ProductCD", value=1.0, importance=0.12),
                FeatureContribution(feature_name="card1", value=float(transaction.card1 or 0), importance=0.10),
                FeatureContribution(feature_name="addr1", value=float(transaction.addr1 or 0), importance=0.08),
                FeatureContribution(feature_name="C1", value=float(transaction.C1 or 0), importance=0.07)
            ]
        
        # Generate explanation
        reason = generate_explanation(fraud_proba, [f.dict() for f in top_features])
        
        # Create graph context (placeholder for single transaction)
        graph_context = GraphContext(
            connected_entities=0,
            risk_score=fraud_proba * 0.3,  # Reduced since no graph features for single transaction
            network_flags=["single_transaction_mode"] if fraud_proba > 0.5 else []
        )
        
        # Return response
        return TransactionResponse(
            transaction_id=transaction_id,
            fraud_probability=fraud_proba,
            prediction=prediction,
            confidence=confidence,
            risk_score=risk_score,
            reason=reason,
            top_features=top_features,
            graph_context=graph_context,
            model_version="hybrid_v1.0" if "hybrid" in str(ARTIFACTS_DIR / "hybrid_model.pkl") else "baseline_v1.0",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Scoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "GraphGuard Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "score": "/score",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)