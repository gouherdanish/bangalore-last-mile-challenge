import pandas as pd
import numpy as np
import logging
from .feature_engineering import FeatureExtractor

# Setup logging
logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    End-to-end feature engineering pipeline for inference.
    
    This class orchestrates the complete feature engineering process
    for real-time bus arrival prediction, including feature extraction,
    data preparation, and model-ready data formatting.
    
    Attributes:
        extractor (FeatureExtractor): Feature extraction engine
        feature_columns (List[str]): List of feature column names
        logger (logging.Logger): Logger instance for the class
    """
    
    def __init__(self):
        """
        Initialize the FeaturePipeline.
        """
        self.extractor = FeatureExtractor()
        self.feature_columns = []
        logger.info("Initializing FeaturePipeline for inference")
    
    def fit_transform(self, live_data: pd.DataFrame, route_id: str = None) -> pd.DataFrame:
        """
        Extract all features and prepare data for modeling.
        
        Orchestrates the complete feature extraction process and
        prepares the data for model inference in real-time scenarios.
        
        Args:
            live_data (pd.DataFrame): Input data for feature extraction
            route_id (str, optional): Route identifier for route-specific features
            
        Returns:
            pd.DataFrame: Data with all extracted features ready for modeling
        """
        logger.info(f"Starting fit_transform for route {route_id}")
        result = self.extractor.extract_all_features(live_data, route_id)
        logger.info(f"Feature extraction completed for route {route_id}")
        return result
    
    def get_feature_importance_ready_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get data ready for feature importance analysis and model inference.
        
        Prepares numeric data by selecting relevant features, handling
        missing values, and removing infinite values for model compatibility.
        
        Args:
            data (pd.DataFrame): Input data with extracted features
            
        Returns:
            pd.DataFrame: Cleaned numeric data ready for model inference
        """
        logger.info("Preparing data for feature importance analysis")
        
        # Store feature columns for later use
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        # Explicitly drop latitude/longitude from model features per requirement
        self.feature_columns = [
            col for col in numeric_cols 
            if col not in ['latitude', 'longitude', 'next_lat', 'next_lon', 'geometry', 'vehicle_timestamp', 'ts', 'target']
        ]
        
        logger.info(f"Selected {len(self.feature_columns)} feature columns for modeling")
        
        # Fill missing values
        numeric_data = data[self.feature_columns].fillna(0)
        
        # Remove infinite values
        numeric_data = numeric_data.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Data preparation completed: {numeric_data.shape}")
        return numeric_data
