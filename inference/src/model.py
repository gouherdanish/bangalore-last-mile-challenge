import pandas as pd
import warnings
import pickle
import logging
from datetime import timedelta

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

class HeuristicModel:
    """
    Main prediction engine for bus arrival times using heuristic methods.
    
    This class implements a heuristic-based model for predicting bus arrival
    times using historical patterns and statistical analysis of travel times
    between stops under different conditions in real-time inference scenarios.
    
    Attributes:
        historical_patterns (Dict): Cached historical patterns for different routes
        logger (logging.Logger): Logger instance for the class
    """
    
    def __init__(self):
        """
        Initialize the HeuristicModel.
        """
        self.historical_patterns = {}
        logger.info("Initializing HeuristicModel for inference")

    def load(self, model_path: str):
        """
        Load the trained model from a file.
        
        Loads pre-trained historical patterns and route characteristics
        from a pickle file for real-time inference.
        
        Args:
            model_path (str): Path to the model file
        """
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            self.historical_patterns = pickle.load(f)
        logger.info(f"Model loaded successfully with {len(self.historical_patterns)} route patterns")

    def predict(self, test_data: pd.DataFrame) -> tuple[list, list]:
        """
        Predict arrival times for future stops using heuristic methods.
        
        Uses historical patterns and statistical analysis to predict
        arrival times for remaining stops on the route based on current
        position, time, and route characteristics.
        
        Args:
            test_data (pd.DataFrame): Input data with current trip state
            
        Returns:
            tuple[list, list]: Predicted arrival times and durations for future stops
        """
        logger.info("Starting prediction for bus arrival times")
        
        lat = test_data['latitude'].iloc[-1]
        lon = test_data['longitude'].iloc[-1]
        ts = test_data['ts'].iloc[-1]
        route_id = test_data['route_id'].iloc[-1]
        current_hour = test_data['hour'].iloc[-1]
        current_dow = test_data['day_of_week'].iloc[-1]
        cumulative_time = test_data['ts'].iloc[-1]
        current_stop_id = test_data['current_stop_id'].iloc[-1]
        remaining_stops = test_data['future_stop_ids'].iloc[-1]
        distance_to_next_route_stop = test_data['distance_to_next_route_stop'].iloc[-1]
        future_segment_distances = test_data['future_segment_distances'].iloc[-1]
        predicted_arrivals, predicted_durations = {}, {}
        
        logger.info(f"Predicting for route {route_id}, current stop {current_stop_id}, hour {current_hour}")
        logger.debug(f"Current data: {test_data.iloc[-1,:].to_dict()}")
        # print(route_id, lat, lon, ts, current_hour, current_dow, current_stop_id, remaining_stops, distance_to_next_route_stop, future_segment_distances)
        # patterns = self.historical_patterns[route_id] if route_id in self.historical_patterns else self.historical_patterns['default']
        patterns_all = self.historical_patterns['default']
        travel_times = patterns_all.get('stop_travel_times', pd.DataFrame())
        # travel_times_all = patterns_all.get('stop_travel_times', pd.DataFrame())
        for i, stop_id in enumerate(remaining_stops):
            # Use historical patterns
            if i == 0:
                mask = travel_times.loc[
                    (travel_times['current_stop_id'] == current_stop_id) &
                    (travel_times['hour'] == current_hour) &
                    (travel_times['day_of_week'] == current_dow)
                ]
                if mask.empty:
                    mask = travel_times.loc[
                        (travel_times['current_stop_id'] == current_stop_id) &
                        (travel_times['hour'] == current_hour)
                    ]
                    if mask.empty:
                        mask = travel_times.loc[
                            (travel_times['current_stop_id'] == current_stop_id) &
                            (travel_times['hour'].isin([current_hour-1, current_hour+1]))
                        ]
                        # print(mask,current_hour-1,current_hour,current_hour+1,current_stop_id)
                        if mask.empty:
                            mask = travel_times.loc[
                                (travel_times['hour'] == current_hour) &
                                (travel_times['day_of_week'] == current_dow)
                            ]
                            if mask.empty:
                                mask = travel_times.loc[
                                    (travel_times['current_stop_id'] == current_stop_id)
                                ]
                # print(mask)
                mean_speed = mask['speed']['mean'].mean()
                if pd.isna(mean_speed) or mean_speed == 0:
                    # Fallback: use average speed from all data
                    mean_speed = travel_times['speed']['mean'].mean()
                    if pd.isna(mean_speed) or mean_speed == 0:
                        mean_speed = 20.0 / 3.6  # Default fallback speed in m/s
                estimated_time = distance_to_next_route_stop / mean_speed
            else:
                prev_stop = remaining_stops[i-1]
                mask = travel_times.loc[
                    (travel_times['current_stop_id'] == prev_stop) &
                    (travel_times['hour'] == current_hour) &
                    (travel_times['day_of_week'] == current_dow)
                ]
                if mask.empty:
                    mask = travel_times.loc[
                        (travel_times['current_stop_id'] == prev_stop) &
                        (travel_times['hour'] == current_hour)
                    ]
                    if mask.empty:
                        mask = travel_times.loc[
                            (travel_times['current_stop_id'] == prev_stop) &
                            (travel_times['hour'].isin([current_hour-1, current_hour+1]))
                        ]
                        # print(mask,current_hour-1,current_hour,current_hour+1,current_stop_id)
                        if mask.empty:
                            mask = travel_times.loc[
                                (travel_times['hour'] == current_hour) &
                                (travel_times['day_of_week'] == current_dow)
                            ]
                            if mask.empty:
                                mask = travel_times.loc[
                                    (travel_times['current_stop_id'] == prev_stop)
                                ]
                # print(mask)
                mean_speed = mask['speed']['mean'].mean()
                if pd.isna(mean_speed) or mean_speed == 0:
                    # Fallback: use average speed from all data
                    mean_speed = travel_times['speed']['mean'].mean()
                    if pd.isna(mean_speed) or mean_speed == 0:
                        mean_speed = 20.0 / 3.6  # Default fallback speed in m/s
                segment_distance = future_segment_distances[i-1]
                estimated_time = segment_distance / mean_speed
            # Final safety check for NaN
            if pd.isna(estimated_time):
                estimated_time = 60.0  # Default 1 minute fallback
                logger.warning(f"Using fallback time for stop {stop_id}")
            
            logger.debug(f"Stop {stop_id}: speed={mean_speed:.2f}, time={estimated_time:.2f}s")
            predicted_durations[stop_id] = estimated_time
            cumulative_time += timedelta(seconds=estimated_time)
            predicted_arrivals[stop_id] = cumulative_time.strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Prediction completed for {len(predicted_arrivals)} stops")
        return predicted_arrivals, predicted_durations
            