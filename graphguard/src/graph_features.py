"""
GraphGuard Graph Features Module
Builds heterogeneous graphs and computes graph metrics (PageRank, Degree, Clustering)
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging

# Note: Using NetworkX-based graph metrics instead of complex embeddings
# This provides stable, interpretable graph features for fraud detection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphFeatureBuilder:
    """Builds graph features for fraud detection using NetworkX metrics"""
    
    def __init__(self):
        self.graph = None
        self.graph_stats = {}
        self.node_features = {}
        
    def build_transaction_graph(self, df: pd.DataFrame) -> nx.Graph:
        """Build heterogeneous graph from transaction data"""
        logger.info("Building transaction graph from transaction data")
        
        # Create a new graph
        G = nx.Graph()
        
        # Add nodes for each unique entity
        unique_accounts = set(df['src_account_id'].unique()) | set(df['dst_account_id'].unique())
        unique_devices = set(df['device_id'].dropna().unique())
        unique_ips = set(df['ip_address'].dropna().unique())
        unique_merchants = set(df['merchant_id'].dropna().unique())
        
        # Add nodes with type attribute
        for account in unique_accounts:
            G.add_node(account, type='account')
        for device in unique_devices:
            G.add_node(device, type='device')
        for ip in unique_ips:
            G.add_node(ip, type='ip')
        for merchant in unique_merchants:
            G.add_node(merchant, type='merchant')
            
        # Add edges based on transactions
        for _, row in df.iterrows():
            # Account to account edge (transaction)
            if pd.notna(row['src_account_id']) and pd.notna(row['dst_account_id']):
                G.add_edge(row['src_account_id'], row['dst_account_id'], 
                          type='transaction', amount=row['amount'])
            
            # Account to device edge
            if pd.notna(row['src_account_id']) and pd.notna(row['device_id']):
                G.add_edge(row['src_account_id'], row['device_id'], type='uses_device')
            if pd.notna(row['dst_account_id']) and pd.notna(row['device_id']):
                G.add_edge(row['dst_account_id'], row['device_id'], type='uses_device')
                
            # Account to IP edge
            if pd.notna(row['src_account_id']) and pd.notna(row['ip_address']):
                G.add_edge(row['src_account_id'], row['ip_address'], type='uses_ip')
            if pd.notna(row['dst_account_id']) and pd.notna(row['ip_address']):
                G.add_edge(row['dst_account_id'], row['ip_address'], type='uses_ip')
                
            # Account to merchant edge
            if pd.notna(row['src_account_id']) and pd.notna(row['merchant_id']):
                G.add_edge(row['src_account_id'], row['merchant_id'], type='transacts_with')
            if pd.notna(row['dst_account_id']) and pd.notna(row['merchant_id']):
                G.add_edge(row['dst_account_id'], row['merchant_id'], type='transacts_with')
        
        self.graph = G
        logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def compute_graph_statistics(self, graph: nx.Graph) -> Dict[str, Dict]:
        """Compute graph statistics for each node"""
        logger.info("Computing graph statistics (PageRank, Degree, Clustering)")
        
        if graph is None:
            raise ValueError("Graph must be built before computing statistics")
        
        # Compute PageRank
        try:
            pagerank = nx.pagerank(graph, alpha=0.85)
        except:
            logger.warning("PageRank computation failed, using degree centrality instead")
            pagerank = nx.degree_centrality(graph)
        
        # Compute degree centrality
        degree_centrality = nx.degree_centrality(graph)
        
        # Compute clustering coefficient
        clustering_coeff = nx.clustering(graph)
        
        # Compute betweenness centrality (optional, can be slow for large graphs)
        try:
            betweenness_centrality = nx.betweenness_centrality(graph, k=min(100, len(graph.nodes())))
        except:
            logger.warning("Betweenness centrality computation failed, using degree instead")
            betweenness_centrality = {node: graph.degree(node) for node in graph.nodes()}
        
        # Store statistics for each node
        for node in graph.nodes():
            self.graph_stats[node] = {
                'pagerank': pagerank.get(node, 0.0),
                'degree_centrality': degree_centrality.get(node, 0.0),
                'clustering_coefficient': clustering_coeff.get(node, 0.0),
                'betweenness_centrality': betweenness_centrality.get(node, 0.0),
                'degree': graph.degree(node),
                'node_type': graph.nodes[node].get('type', 'unknown')
            }
        
        logger.info(f"Computed graph statistics for {len(self.graph_stats)} nodes")
        return self.graph_stats
    
    def extract_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract graph features for each transaction"""
        logger.info("Extracting transaction graph features")
        
        if not self.graph_stats:
            raise ValueError("Graph statistics must be computed before extracting features")
        
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Add graph-based features for source and destination accounts
        result_df['src_pagerank'] = result_df['src_account_id'].map(
            lambda x: self.graph_stats.get(x, {}).get('pagerank', 0.0)
        )
        result_df['src_degree_centrality'] = result_df['src_account_id'].map(
            lambda x: self.graph_stats.get(x, {}).get('degree_centrality', 0.0)
        )
        result_df['src_clustering_coefficient'] = result_df['src_account_id'].map(
            lambda x: self.graph_stats.get(x, {}).get('clustering_coefficient', 0.0)
        )
        result_df['src_betweenness_centrality'] = result_df['src_account_id'].map(
            lambda x: self.graph_stats.get(x, {}).get('betweenness_centrality', 0.0)
        )
        result_df['src_degree'] = result_df['src_account_id'].map(
            lambda x: self.graph_stats.get(x, {}).get('degree', 0)
        )
        
        result_df['dst_pagerank'] = result_df['dst_account_id'].map(
            lambda x: self.graph_stats.get(x, {}).get('pagerank', 0.0)
        )
        result_df['dst_degree_centrality'] = result_df['dst_account_id'].map(
            lambda x: self.graph_stats.get(x, {}).get('degree_centrality', 0.0)
        )
        result_df['dst_clustering_coefficient'] = result_df['dst_account_id'].map(
            lambda x: self.graph_stats.get(x, {}).get('clustering_coefficient', 0.0)
        )
        result_df['dst_betweenness_centrality'] = result_df['dst_account_id'].map(
            lambda x: self.graph_stats.get(x, {}).get('betweenness_centrality', 0.0)
        )
        result_df['dst_degree'] = result_df['dst_account_id'].map(
            lambda x: self.graph_stats.get(x, {}).get('degree', 0)
        )
        
        # Add device and IP graph features
        result_df['device_degree'] = result_df['device_id'].map(
            lambda x: self.graph_stats.get(x, {}).get('degree', 0) if pd.notna(x) else 0
        )
        result_df['ip_degree'] = result_df['ip_address'].map(
            lambda x: self.graph_stats.get(x, {}).get('degree', 0) if pd.notna(x) else 0
        )
        
        # Add aggregate features
        result_df['src_dst_pagerank_diff'] = result_df['src_pagerank'] - result_df['dst_pagerank']
        result_df['src_dst_degree_diff'] = result_df['src_degree'] - result_df['dst_degree']
        result_df['src_dst_clustering_diff'] = result_df['src_clustering_coefficient'] - result_df['dst_clustering_coefficient']
        
        # Add risk indicators
        result_df['high_risk_src'] = (result_df['src_pagerank'] > 0.01) | (result_df['src_degree'] > 10)
        result_df['high_risk_dst'] = (result_df['dst_pagerank'] > 0.01) | (result_df['dst_degree'] > 10)
        result_df['suspicious_device'] = result_df['device_degree'] > 5
        result_df['suspicious_ip'] = result_df['ip_degree'] > 3
        
        logger.info(f"Added {len([col for col in result_df.columns if col.startswith(('src_', 'dst_', 'device_', 'ip_'))])} graph features")
        return result_df
    
    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete pipeline: build graph, compute statistics, extract features"""
        logger.info("Building complete graph feature pipeline")
        
        # Step 1: Build graph
        self.build_transaction_graph(df)
        
        # Step 2: Compute graph statistics
        self.compute_graph_statistics(self.graph)
        
        # Step 3: Extract features
        result_df = self.extract_transaction_features(df)
        
        logger.info("Graph feature pipeline completed successfully")
        return result_df
    
    def save_graph_stats(self, file_path: str):
        """Save graph statistics to disk"""
        logger.info(f"Saving graph statistics to {file_path}")
        import joblib
        joblib.dump(self.graph_stats, file_path)
    
    def load_graph_stats(self, file_path: str):
        """Load graph statistics from disk"""
        logger.info(f"Loading graph statistics from {file_path}")
        import joblib
        self.graph_stats = joblib.load(file_path)
    
    def get_node_neighbors(self, node_id: str, max_neighbors: int = 10) -> List[str]:
        """Get neighbors of a specific node for ego-graph visualization"""
        if self.graph is None:
            return []
        
        if node_id not in self.graph:
            return []
        
        neighbors = list(self.graph.neighbors(node_id))
        return neighbors[:max_neighbors]
    
    def get_ego_graph(self, node_id: str, radius: int = 1) -> nx.Graph:
        """Get ego-graph around a specific node"""
        if self.graph is None:
            return nx.Graph()
        
        try:
            ego_graph = nx.ego_graph(self.graph, node_id, radius=radius)
            return ego_graph
        except:
            logger.warning(f"Could not create ego-graph for node {node_id}")
            return nx.Graph()


def main():
    """Main graph feature building pipeline"""
    builder = GraphFeatureBuilder()
    
    # Example usage
    logger.info("Graph feature pipeline initialized")
    logger.info("Use build_all_features() to process your transaction data")
    
    # Sample data structure for testing
    sample_data = pd.DataFrame({
        'transaction_id': ['TXN_001', 'TXN_002'],
        'src_account_id': ['ACC_001', 'ACC_002'],
        'dst_account_id': ['ACC_002', 'ACC_003'],
        'device_id': ['DEV_001', 'DEV_002'],
        'ip_address': ['IP_001', 'IP_002'],
        'merchant_id': ['MERCH_001', 'MERCH_002'],
        'amount': [100.0, 200.0]
    })
    
    logger.info("Sample data structure created for testing")


if __name__ == "__main__":
    main()
