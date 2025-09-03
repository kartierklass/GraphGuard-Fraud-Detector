#!/usr/bin/env python3
"""
Test script for GraphGuard graph features
Tests the new NetworkX-based graph metrics implementation
"""

import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'graphguard', 'src'))

from graph_features import GraphFeatureBuilder

def test_graph_features():
    """Test the graph features implementation"""
    print("🧪 Testing GraphGuard Graph Features...")
    
    # Create sample transaction data
    sample_data = pd.DataFrame({
        'transaction_id': ['TXN_001', 'TXN_002', 'TXN_003', 'TXN_004'],
        'src_account_id': ['ACC_001', 'ACC_002', 'ACC_001', 'ACC_003'],
        'dst_account_id': ['ACC_002', 'ACC_003', 'ACC_004', 'ACC_001'],
        'device_id': ['DEV_001', 'DEV_002', 'DEV_001', 'DEV_003'],
        'ip_address': ['IP_001', 'IP_002', 'IP_001', 'IP_003'],
        'merchant_id': ['MERCH_001', 'MERCH_002', 'MERCH_001', 'MERCH_003'],
        'amount': [100.0, 200.0, 150.0, 300.0]
    })
    
    print(f"📊 Sample data created with {len(sample_data)} transactions")
    
    # Initialize the graph feature builder
    builder = GraphFeatureBuilder()
    
    try:
        # Build the complete pipeline
        print("🔨 Building graph and computing features...")
        result_df = builder.build_all_features(sample_data)
        
        print(f"✅ Success! Added {len([col for col in result_df.columns if col.startswith(('src_', 'dst_', 'device_', 'ip_'))])} graph features")
        
        # Show the new features
        graph_columns = [col for col in result_df.columns if col.startswith(('src_', 'dst_', 'device_', 'ip_'))]
        print(f"📈 Graph features added: {graph_columns}")
        
        # Show sample results
        print("\n📋 Sample results:")
        print(result_df[['transaction_id', 'src_account_id', 'src_pagerank', 'src_degree', 'src_clustering_coefficient']].head())
        
        # Test ego-graph functionality
        print("\n🔍 Testing ego-graph functionality...")
        ego_graph = builder.get_ego_graph('ACC_001', radius=1)
        print(f"✅ Ego-graph created with {ego_graph.number_of_nodes()} nodes and {ego_graph.number_of_edges()} edges")
        
        # Test saving/loading
        print("\n💾 Testing save/load functionality...")
        builder.save_graph_stats('test_graph_stats.joblib')
        print("✅ Graph statistics saved successfully")
        
        # Clean up test file
        if os.path.exists('test_graph_stats.joblib'):
            os.remove('test_graph_stats.joblib')
            print("✅ Test file cleaned up")
        
        print("\n🎉 All tests passed! GraphGuard graph features are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_graph_features()
    sys.exit(0 if success else 1)
