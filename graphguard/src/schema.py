"""
GraphGuard API Schema Definitions
Pydantic models for request/response validation
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Input schema for transaction scoring"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., description="Transaction amount")
    src_account_id: str = Field(..., description="Source account identifier")
    dst_account_id: str = Field(..., description="Destination account identifier")
    device_id: Optional[str] = Field(None, description="Device identifier")
    ip_address: Optional[str] = Field(None, description="IP address")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    timestamp: Optional[str] = Field(None, description="Transaction timestamp")


class FeatureContribution(BaseModel):
    """Feature contribution for explainability"""
    name: str = Field(..., description="Feature name")
    contrib: float = Field(..., description="SHAP contribution value")


class GraphContext(BaseModel):
    """Graph-based context information"""
    src_degree: Optional[int] = Field(None, description="Source account degree in graph")
    dst_degree: Optional[int] = Field(None, description="Destination account degree in graph")
    src_pagerank: Optional[float] = Field(None, description="Source account PageRank")
    dst_pagerank: Optional[float] = Field(None, description="Destination account PageRank")


class TransactionResponse(BaseModel):
    """Output schema for transaction scoring"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    probability: float = Field(..., description="Fraud probability score (0-1)")
    label: str = Field(..., description="Predicted label: 'flag' or 'safe'")
    threshold: float = Field(..., description="Threshold used for classification")
    top_features: List[FeatureContribution] = Field(..., description="Top contributing features")
    reason: str = Field(..., description="Human-readable explanation")
    graph_context: Optional[GraphContext] = Field(None, description="Graph-based context")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_version: str = Field(..., description="Model version timestamp")
    timestamp: str = Field(..., description="Current timestamp")
