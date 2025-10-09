from .geoutils import GeoUtils
from pathlib import Path
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
from typing import List
import numpy as np
import logging
from shapely.geometry import Point

# Setup logging
logger = logging.getLogger(__name__)

class TestDataProcessor:
    """
    Data processor for real-time test data in inference pipeline.
    
    This class handles the processing of live trip data for prediction,
    including data cleaning, route position detection, and feature
    preparation for the prediction model.
    
    Attributes:
        data_path (Path): Path to the data directory containing static data
        stops_df (gpd.GeoDataFrame): Processed stops data with geometries
        route_sequences_df (pd.DataFrame): Route sequences with distances
        logger (logging.Logger): Logger instance for the class
    """
    
    def __init__(self, data_path: Path):
        """
        Initialize the TestDataProcessor.
        """
        self.data_path = Path(data_path)
        self.stops_df = None
        self.route_sequences_df = None
        
        logger.info("Initializing TestDataProcessor")
        self._load_static_data()

    def _load_static_data(self):
        """
        Load static reference data (stops, sequences).
        
        Loads processed static data including stops with geometries
        and route sequences with pre-calculated distances.
        """
        logger.info("Loading static reference data")
        
        # Load stops
        stops_path = self.data_path / "stops_0.shp"
        logger.info(f"Loading stops from {stops_path}")
        self.stops_df = gpd.read_file(stops_path)
        
        # Load route sequences
        from ast import literal_eval
        seq_path = self.data_path / "route_to_stop_sequence.csv"
        logger.info(f"Loading route sequences from {seq_path}")
        self.route_sequences_df = pd.read_csv(seq_path)
        self.route_sequences_df['stop_id_list'] = self.route_sequences_df['stop_id_list'].apply(lambda x: x.split(', '))
        self.route_sequences_df['next_stop_distances'] = self.route_sequences_df['next_stop_distances'].apply(lambda x: [float(i) for i in x.split(', ')])
        
        logger.info(f"Loaded {len(self.stops_df)} stops, {len(self.route_sequences_df)} route sequences")

    def clean_trip_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize trip data.
        
        Performs data cleaning including string normalization,
        timestamp conversion, coordinate validation, and filtering
        for scheduled trips only.
        
        Args:
            df (pd.DataFrame): Raw trip data
            
        Returns:
            pd.DataFrame: Cleaned trip data
        """
        logger.info(f"Cleaning trip data: {len(df)} records")
        
        # Clean string columns by removing quotes
        string_cols = ['route_id', 'trip_id', 'vehicle_id', 'stop_id']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('"', '', regex=False)
        
        # Convert timestamps
        if 'vehicle_timestamp' in df.columns:
            df['vehicle_timestamp'] = pd.to_numeric(df['vehicle_timestamp'].astype(str).str.replace('"', ''), errors='coerce')
            df['ts'] = pd.to_datetime(df['vehicle_timestamp'], unit='s', errors='coerce')
        
        # Filter valid GPS coordinates
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
        
        # Filter scheduled trips only
        if 'schedule_relationship' in df.columns:
            df = df[df['schedule_relationship'].isin(['"ADDED"', '"SCHEDULED"', 'ADDED', 'SCHEDULED'])]
        
        cleaned_df = df.dropna(subset=['ts', 'latitude', 'longitude'])
        logger.info(f"Cleaned data: {len(cleaned_df)} valid records")
        return cleaned_df
    
    def get_current_data(self, df: pd.DataFrame) -> tuple[dict, str]:
        """
        Get current data from dataframe.
        
        Extracts the most recent data point from the trip data
        and returns the current state and route ID.
        
        Args:
            df (pd.DataFrame): Trip data sorted by timestamp
            
        Returns:
            tuple[dict, str]: Current data dictionary and route ID
        """
        trip_data = df.sort_values('ts')
        current_data = trip_data.iloc[-1].to_dict()
        route_id = current_data['route_id']
        
        logger.info(f"Current route ID: {route_id}")
        logger.info(f"Trip data contains {trip_data['trip_id'].nunique()} unique trips")
        return current_data, route_id
    
    def get_route_sequence_with_distances(self, route_id: str) -> tuple[list, list]:
        """
        Get route sequence and distances for a specific route.
        
        Retrieves the stop sequence and pre-calculated distances
        between consecutive stops for the specified route.
        
        Args:
            route_id (str): Route identifier
            
        Returns:
            tuple[list, list]: Stop sequence and distance list
            
        Raises:
            AssertionError: If route sequence is not found
        """
        logger.info(f"Getting route sequence for route {route_id}")
        
        # Get route sequence
        route_sequence = self.route_sequences_df[self.route_sequences_df['route_id'] == int(route_id)]
        assert not route_sequence.empty, f"No sequence found for route_id: {route_id}"
        
        stop_sequence = route_sequence['stop_id_list'].iloc[0]
        distances = route_sequence['next_stop_distances'].iloc[0]
        
        logger.info(f"Found route sequence with {len(stop_sequence)} stops")
        return stop_sequence, distances

    def find_best_route_position(self, lat: float, lon: float, stop_sequence: list) -> int:
        """
        Find the best position in the route sequence to match the current location.
        
        Uses geometric projection to determine the most likely position
        of the bus on the route based on GPS coordinates and stop sequence.
        
        Args:
            lat (float): Current latitude
            lon (float): Current longitude
            stop_sequence (list): List of stop IDs in route order
            
        Returns:
            int: Index of the best matching stop position
        """
        logger.info(f"Finding best route position for coordinates ({lat}, {lon})")
        
        local_crs = GeoUtils.get_proj_crs(lon=lon, lat=lat)
        gps_point = gpd.points_from_xy([lon], [lat], crs=4326).to_crs(local_crs)[0]
        
        logger.debug(f"GPS point projected: {gps_point}")
        
        # Start from min_position and look at pairs of stops
        for i in range(len(stop_sequence)-1):
            current_stop_id = stop_sequence[i]
            next_stop_id = stop_sequence[i+1]
            
            # Get current stop coordinates
            current_stop = self.stops_df[self.stops_df['stop_id'] == current_stop_id].iloc[0]
            current_point = gpd.points_from_xy([current_stop['stop_lon']], [current_stop['stop_lat']],crs=4326).to_crs(local_crs)[0]
            # print(f"Current point: {current_point}")

            # Get next stop coordinates
            next_stop = self.stops_df[self.stops_df['stop_id'] == next_stop_id].iloc[0]
            next_point = gpd.points_from_xy([next_stop['stop_lon']], [next_stop['stop_lat']],crs=4326).to_crs(local_crs)[0]
            # print(f"Next point: {next_point}")

            # Calculate vector from current to next stop
            segment_vector = np.array([next_point.x - current_point.x, 
                                     next_point.y - current_point.y])
            segment_length = np.sqrt(np.sum(segment_vector**2))
            # print(f"Segment Vector: {segment_vector}")
            # print(f"Segment Length: {segment_length}")
            
            if segment_length == 0:
                continue  # Skip if stops are at same location
                
            # Calculate vector from current stop to GPS point
            gps_vector = np.array([gps_point.x - current_point.x,
                                 gps_point.y - current_point.y])
            gps_vector_length = np.linalg.norm(gps_vector)
            # print(f"GPS Vector: {gps_vector}")
            
            # Project GPS vector onto segment vector
            # projection = (a·b/|b|²)b where b is segment vector
            projection_scalar = np.dot(gps_vector, segment_vector) / (segment_length**2)
            # print(f"Projection Scalar: {projection_scalar}")
            
            # If projection is > 1, we're past the next stop
            # If projection is < 0, we haven't reached current stop
            # If 0 <= projection <= 1, we're between stops
            if projection_scalar > 1:
                continue  # We've passed this segment, keep looking
            elif projection_scalar < 0:
                continue  # we haven't reached current stop, keep looking
            else:
                # Additional check: Calculate perpendicular distance from GPS to segment
                # Perpendicular distance = |gps_vector - projection_vector|
                projection_vector = projection_scalar * segment_vector
                perpendicular_vector = gps_vector - projection_vector
                perpendicular_distance = np.linalg.norm(perpendicular_vector)
                
                # Define a maximum allowed perpendicular distance threshold
                # If GPS point is too far from the segment, it might not be on this route
                max_perpendicular_distance = max(100.0, segment_length * 0.5)  # 100m or 50% of segment length
                
                # Also check if GPS vector is too long compared to segment
                # This catches the edge case of acute angle but large distance
                max_gps_vector_length = segment_length * 1.5  # Allow 1.5x segment length
                
                logger.debug(f"Perpendicular distance: {perpendicular_distance:.2f}m")
                logger.debug(f"GPS vector length: {gps_vector_length:.2f}m vs max: {max_gps_vector_length:.2f}m")
                
                if perpendicular_distance > max_perpendicular_distance:
                    logger.debug(f"Too far from segment {i}, continuing search")
                    continue  # Too far from segment, keep looking
                
                if gps_vector_length > max_gps_vector_length:
                    logger.debug(f"GPS vector too long for segment {i}, continuing search")
                    continue  # GPS vector too long, likely not on this segment
                
                logger.info(f"Found best position at stop index {i} (stop {current_stop_id})")
                return i  # This is our previous stop
        
        # If we're past the last stop or very close to it
        logger.info(f"Position beyond last stop, using index {len(stop_sequence) - 1}")
        return len(stop_sequence) - 1

    def process_data(self, current_data: dict, route_id: str) -> pd.DataFrame:
        """
        Process current trip data and extract route position information.
        
        Analyzes current GPS position to determine bus location on route,
        calculates distances to upcoming stops, and prepares data for
        feature extraction and prediction.
        
        Args:
            current_data (dict): Current trip data with GPS coordinates and timestamp
            route_id (str): Route identifier
            
        Returns:
            pd.DataFrame: Processed data with route position and distance information
        """
        logger.info(f"Processing data for route {route_id}")
        
        current_lat = current_data['latitude']
        current_lon = current_data['longitude']
        current_ts = current_data['ts']
        current_point = Point(current_lon, current_lat)
        
        stop_sequence, next_route_stop_distances = self.get_route_sequence_with_distances(route_id)
        
        # Find best matching stop from current_furthest onwards (sequential constraint)
        best_position = self.find_best_route_position(current_lat, current_lon, stop_sequence)
        logger.info(f"Best position determined: {best_position}")
        
        # Get relevant data
        logger.info(f"Current point: {current_point}")
        logger.info(f"Current timestamp: {current_ts}")
        logger.info(f"Current latitude: {current_lat}")
        logger.info(f"Current longitude: {current_lon}")
        logger.info(f"Route ID: {route_id}")
        logger.info(f"Best position: {best_position}")
        logger.info(f"Stop sequence: {stop_sequence}")
        current_stop_id = stop_sequence[best_position]
        logger.info(f"Current Stop ID: {current_stop_id}")
        next_stop_id = stop_sequence[best_position + 1]
        logger.info(f"Next Stop ID: {next_stop_id}")
        future_stop_ids = stop_sequence[best_position + 1:]
        logger.info(f"Future Stop IDs: {future_stop_ids}")
        next_stop_geom = self.stops_df[self.stops_df['stop_id'] == next_stop_id].iloc[0]['geometry']
        
        # Calculate distance
        distance_to_next_route_stop = GeoUtils.calculate_dist(current_point, next_stop_geom)
        future_segment_distances = next_route_stop_distances[best_position + 1:]
        
        logger.info(f"Current stop: {current_stop_id}, Next stop: {next_stop_id}, Distance: {distance_to_next_route_stop:.2f}m")
        
        current_df = pd.DataFrame({
            'route_id': [route_id],
            'ts': [current_ts],
            'latitude': [current_lat],
            'longitude': [current_lon],
            'route_position': [best_position],
            'total_route_stops': [len(stop_sequence)],
            'current_stop_id': [current_stop_id],
            'next_stop_id': [next_stop_id],
            'future_stop_ids': [future_stop_ids],
            'distance_to_next_route_stop': [distance_to_next_route_stop],
            'future_segment_distances': [future_segment_distances]
        })
        
        logger.info(f"Data processing completed for route {route_id}")
        return current_df

