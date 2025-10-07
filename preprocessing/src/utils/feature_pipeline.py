import pandas as pd
import numpy as np
import logging
from .feature_engineering import FeatureExtractor

# Setup logging
logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    End-to-end feature engineering pipeline for bus arrival prediction.
    
    This class orchestrates the complete feature engineering process,
    including temporal, spatial, and route-specific feature extraction
    and data preparation for machine learning models.
    
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
        logger.info("Initializing FeaturePipeline")
    
    def fit_transform(self, live_data: pd.DataFrame, route_id: str = None) -> pd.DataFrame:
        """
        Extract all features and prepare for modeling.
        
        Args:
            live_data (pd.DataFrame): Input data for feature extraction
            route_id (str, optional): Route ID for route-specific features
            
        Returns:
            pd.DataFrame: Data with all extracted features
        """
        logger.info(f"Starting feature extraction for route {route_id if route_id else 'all'}")
        return self.extractor.extract_all_features(live_data, route_id)
    
    def get_feature_importance_ready_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get data ready for feature importance analysis.
        
        Prepares numeric data by selecting relevant features,
        handling missing values, and removing infinite values.
        
        Args:
            data (pd.DataFrame): Input data with features
            
        Returns:
            pd.DataFrame: Cleaned numeric data ready for analysis
        """
        logger.info("Preparing data for feature importance analysis")
        
        # Store feature columns for later use
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        # Explicitly drop latitude/longitude from model features per requirement
        self.feature_columns = [
            col for col in numeric_cols 
            if col not in ['latitude', 'longitude', 'next_lat', 'next_lon', 'geometry', 'vehicle_timestamp', 'ts', 'target']
        ]
        
        logger.info(f"Selected {len(self.feature_columns)} feature columns")
        
        # Fill missing values
        numeric_data = data[self.feature_columns].fillna(0)
        
        # Remove infinite values
        numeric_data = numeric_data.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Data shape after cleaning: {numeric_data.shape}")
        return numeric_data
