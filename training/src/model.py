import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
import pickle
import logging

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

class HeuristicModel:
    """
    Main prediction engine for bus arrival times using heuristic methods.
    
    This class implements a heuristic-based model for predicting bus arrival
    times using historical patterns and statistical analysis of travel times
    between stops under different conditions.
    
    Attributes:
        historical_patterns (Dict): Cached historical patterns for different routes
        logger (logging.Logger): Logger instance for the class
    """
    
    def __init__(self):
        """
        Initialize the HeuristicModel.
        """
        self.historical_patterns = {}
        logger.info("Initialized HeuristicModel")
    
    def _train_default_model(self, trip_data: pd.DataFrame):
        """
        Train arrival prediction model for all routes using heuristic model.
        
        Args:
            trip_data (pd.DataFrame): Training data for all routes
        """
        if trip_data.empty:
            logger.warning("No training data provided for default model")
            return
        
        logger.info("Training default model for all routes")
        # Calculate historical arrival patterns
        patterns = self._calculate_arrival_patterns(trip_data)
        self.historical_patterns['default'] = patterns
        
        logger.info(f"Trained default model with {len(trip_data)} records")

    def _train_route_model(self, trip_data: pd.DataFrame, route_id: str):
        """
        Train arrival prediction model for a specific route using heuristic model.
        
        Args:
            trip_data (pd.DataFrame): Training data for the specific route
            route_id (str): Route identifier
        """
        if trip_data.empty:
            logger.warning(f"No training data provided for route {route_id}")
            return
        
        logger.info(f"Training model for route {route_id}")
        # Calculate historical arrival patterns
        patterns = self._calculate_arrival_patterns(trip_data)
        self.historical_patterns[route_id] = patterns
        
        logger.info(f"Trained model for route {route_id} with {len(trip_data)} records")
    
    def _calculate_arrival_patterns(self, trip_data: pd.DataFrame) -> Dict:
        """
        Calculate statistical patterns for arrival predictions.
        
        Analyzes historical trip data to extract patterns in travel times
        between stops under different temporal conditions.
        
        Args:
            trip_data (pd.DataFrame): Historical trip data with features
            
        Returns:
            Dict: Dictionary containing calculated patterns
        """
        logger.info("Calculating arrival patterns from trip data")
        patterns = {}
        
        # Group by stop pairs to calculate average travel times
        stop_pairs = trip_data.groupby(['current_stop_id', 'hour', 'day_of_week']).agg({
            'segment_duration': ['mean', 'std', 'count'],
            'speed': ['mean', 'std'],
            'segment_distance': 'mean'
        }).reset_index()
        
        patterns['stop_travel_times'] = stop_pairs
        
        logger.info(f"Calculated patterns for {len(stop_pairs)} stop-time combinations")
        return patterns

    def train(self, trip_data: pd.DataFrame, route_id: str = 'default'):
        """
        Train the model for a specific route or default.
        
        Args:
            trip_data (pd.DataFrame): Training data
            route_id (str, optional): Route identifier. Defaults to 'default'.
        """
        logger.info(f"Training model for route_id: {route_id}")
        if route_id == 'default':
            self._train_default_model(trip_data)
        else:
            self._train_route_model(trip_data, route_id)

    def save(self, model_path: str):
        """
        Save the model to a file.
        
        Args:
            model_path (str): Path where to save the model
        """
        logger.info(f"Saving model to {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(self.historical_patterns, f)
        logger.info("Model saved successfully")

    def load(self, model_path: str):
        """
        Load the model from a file.
        
        Args:
            model_path (str): Path to the saved model file
        """
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            self.historical_patterns = pickle.load(f)
        logger.info("Model loaded successfully")
            
    