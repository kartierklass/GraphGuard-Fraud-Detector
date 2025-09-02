"""
GraphGuard Streamlit Dashboard
Analyst console for reviewing fraud alerts and inspecting transactions
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import networkx as nx
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="GraphGuard Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000"
THRESHOLD_DEFAULT = 0.80


def main():
    """Main dashboard application"""
    st.title("🛡️ GraphGuard Fraud Detection Dashboard")
    st.markdown("Real-time fraud detection with graph-enhanced ML")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        threshold = st.slider("Fraud Threshold", 0.0, 1.0, THRESHOLD_DEFAULT, 0.05)
        api_url = st.text_input("API URL", API_BASE_URL)
        
        st.header("Actions")
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Live Alerts", "🔍 Transaction Inspector", "📈 Analytics"])
    
    with tab1:
        show_live_alerts(api_url, threshold)
    
    with tab2:
        show_transaction_inspector(api_url)
    
    with tab3:
        show_analytics(api_url)


def show_live_alerts(api_url: str, threshold: float):
    """Display live fraud alerts table"""
    st.header("Live Fraud Alerts")
    
    # Simulate live data (replace with actual API calls)
    alerts_data = generate_sample_alerts(threshold)
    
    if alerts_data:
        df = pd.DataFrame(alerts_data)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_device = st.selectbox("Filter by Device", ["All"] + list(df['device_id'].unique()))
        with col2:
            selected_account = st.selectbox("Filter by Account", ["All"] + list(df['src_account_id'].unique()))
        with col3:
            selected_merchant = st.selectbox("Filter by Merchant", ["All"] + list(df['merchant_id'].unique()))
        
        # Apply filters
        filtered_df = df.copy()
        if selected_device != "All":
            filtered_df = filtered_df[filtered_df['device_id'] == selected_device]
        if selected_account != "All":
            filtered_df = filtered_df[filtered_df['src_account_id'] == selected_account]
        if selected_merchant != "All":
            filtered_df = filtered_df[filtered_df['merchant_id'] == selected_merchant]
        
        # Display alerts
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export button
        if st.button("📥 Export Alerts"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"fraud_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No alerts found above the threshold.")


def show_transaction_inspector(api_url: str):
    """Allow analysts to inspect individual transactions"""
    st.header("Transaction Inspector")
    
    # Input form
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_id = st.text_input("Transaction ID", "TXN_001")
            amount = st.number_input("Amount", min_value=0.0, value=100.0)
            src_account = st.text_input("Source Account", "ACC_001")
            dst_account = st.text_input("Destination Account", "ACC_002")
        
        with col2:
            device_id = st.text_input("Device ID", "DEV_001")
            ip_address = st.text_input("IP Address", "192.168.1.1")
            merchant_id = st.text_input("Merchant ID", "MERCH_001")
            timestamp = st.text_input("Timestamp", "2025-02-09T10:00:00")
        
        submitted = st.form_submit_button("🔍 Score Transaction")
    
    if submitted:
        # Prepare request
        transaction_data = {
            "transaction_id": transaction_id,
            "amount": amount,
            "src_account_id": src_account,
            "dst_account_id": dst_account,
            "device_id": device_id,
            "ip_address": ip_address,
            "merchant_id": merchant_id,
            "timestamp": timestamp
        }
        
        # Call API (placeholder for now)
        result = simulate_api_call(transaction_data)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Fraud Analysis")
            st.metric("Fraud Probability", f"{result['probability']:.3f}")
            st.metric("Prediction", result['label'].upper(), 
                     delta="🚨 HIGH RISK" if result['label'] == "flag" else "✅ SAFE")
            
            st.subheader("Top Contributing Features")
            for feature in result['top_features']:
                st.write(f"• **{feature['name']}**: {feature['contrib']:.3f}")
            
            st.subheader("Explanation")
            st.info(result['reason'])
        
        with col2:
            st.subheader("Graph Context")
            if result.get('graph_context'):
                gc = result['graph_context']
                st.metric("Source Degree", gc.get('src_degree', 'N/A'))
                st.metric("Dest Degree", gc.get('dst_degree', 'N/A'))
                st.metric("Source PageRank", f"{gc.get('src_pagerank', 0):.4f}")
                st.metric("Dest PageRank", f"{gc.get('dst_pagerank', 0):.4f}")


def show_analytics(api_url: str):
    """Show analytics and model performance"""
    st.header("Analytics & Model Performance")
    
    # Placeholder for analytics
    st.info("Analytics dashboard will show model performance metrics, feature importance, and trends.")
    
    # Sample metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", "1,234")
    with col2:
        st.metric("Fraud Rate", "2.3%")
    with col3:
        st.metric("Model AUC", "0.92")
    with col4:
        st.metric("Avg Response Time", "25ms")


def generate_sample_alerts(threshold: float) -> List[Dict[str, Any]]:
    """Generate sample alert data for demonstration"""
    sample_data = [
        {
            "transaction_id": "TXN_001",
            "timestamp": "2025-02-09T10:00:00",
            "amount": 1500.0,
            "src_account_id": "ACC_001",
            "dst_account_id": "ACC_002",
            "device_id": "DEV_001",
            "ip_address": "192.168.1.1",
            "merchant_id": "MERCH_001",
            "probability": 0.85,
            "label": "flag"
        },
        {
            "transaction_id": "TXN_002",
            "timestamp": "2025-02-09T10:05:00",
            "amount": 2500.0,
            "src_account_id": "ACC_003",
            "dst_account_id": "ACC_004",
            "device_id": "DEV_002",
            "ip_address": "192.168.1.2",
            "merchant_id": "MERCH_002",
            "probability": 0.92,
            "label": "flag"
        }
    ]
    
    # Filter by threshold
    return [alert for alert in sample_data if alert['probability'] >= threshold]


def simulate_api_call(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate API call for demonstration"""
    # In production, this would call the actual FastAPI endpoint
    return {
        "transaction_id": transaction_data["transaction_id"],
        "probability": 0.75,
        "label": "flag",
        "threshold": 0.80,
        "top_features": [
            {"name": "device_id_freq", "contrib": 0.21},
            {"name": "node2vec_src_07", "contrib": 0.15},
            {"name": "amount_z", "contrib": 0.12}
        ],
        "reason": "High device reuse and risky neighbors for src_account_id; large amount vs. history.",
        "graph_context": {
            "src_degree": 27,
            "dst_degree": 11,
            "src_pagerank": 0.023,
            "dst_pagerank": 0.008
        }
    }


if __name__ == "__main__":
    main()
