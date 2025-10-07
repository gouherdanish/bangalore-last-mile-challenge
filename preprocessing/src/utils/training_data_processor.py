from .geoutils import GeoUtils
from pathlib import Path
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
from typing import List
import numpy as np
import logging

# Setup logging
logger = logging.getLogger(__name__)

class TrainingDataProcessor:
    """
    Comprehensive data processing pipeline for bus arrival prediction.
    
    This class handles the processing of historical trip data, including
    data loading, cleaning, route processing, and feature preparation
    for machine learning model training.
    
    Attributes:
        raw_data_path (Path): Path to raw trip data directory
        processed_data_path (Path): Path to processed data directory
        stops_df (gpd.GeoDataFrame): Processed stops data with geometries
        route_sequences_df (pd.DataFrame): Processed route sequences with distances
        logger (logging.Logger): Logger instance for the class
    """
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        Initialize the TrainingDataProcessor.
        
        Args:
            raw_data_path (str): Path to the raw data directory
            processed_data_path (str): Path to the processed data directory
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.stops_df = None
        self.route_sequences_df = None
        
        logger.info("Initializing TrainingDataProcessor")
        self._load_static_data()
    
    def _load_static_data(self):
        """
        Load static reference data (stops, sequences).
        
        Loads processed static data including stops with geometries
        and route sequences with pre-calculated distances.
        """
        logger.info("Loading static reference data")
        
        # Load stops
        stops_path = self.processed_data_path / "static_data/stops_0.shp"
        logger.info(f"Loading stops from {stops_path}")
        self.stops_df = gpd.read_file(stops_path)
        
        # Load route sequences
        from ast import literal_eval
        seq_path = self.processed_data_path / "static_data/route_to_stop_sequence.csv"
        logger.info(f"Loading route sequences from {seq_path}")
        self.route_sequences_df = pd.read_csv(seq_path)
        self.route_sequences_df['stop_id_list'] = self.route_sequences_df['stop_id_list'].apply(lambda x: x.split(', '))
        self.route_sequences_df['next_stop_distances'] = self.route_sequences_df['next_stop_distances'].apply(lambda x: [float(i) for i in x.split(', ')])
        
        logger.info(f"Loaded {len(self.stops_df)} stops, {len(self.route_sequences_df)} route sequences")
    
    def load_trip_data_batch(self, date_folders: List[str]) -> pd.DataFrame:
        """
        Load and process trip data from multiple date folders.
        
        Args:
            date_folders (List[str]): List of date folder names to process
            
        Returns:
            pd.DataFrame: Combined and cleaned trip data from all folders
        """
        logger.info(f"Loading trip data for {len(date_folders)} date folders")
        all_trips = []
        
        for date_folder in date_folders:
            date_path = self.raw_data_path / "home/cistup-videoserver/nas_mount/Brij_work/BTS_project/s3_flatening/s3_bucket_aws_cli_processed" / date_folder
            if not date_path.exists():
                logger.warning(f"Date path does not exist: {date_path}")
                continue
                
            parquet_files = list(date_path.glob("*.parquet"))
            logger.info(f"Processing {len(parquet_files)} files from {date_folder}")
            
            for file_path in parquet_files:
                try:
                    df = pq.read_table(file_path).to_pandas()
                    df = self._clean_trip_data(df)
                    all_trips.append(df)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        if all_trips:
            combined_df = pd.concat(all_trips, ignore_index=True)
            logger.info(f"Loaded {len(combined_df)} trip records")
            return combined_df
        else:
            logger.warning("No trip data loaded")
            return pd.DataFrame()
    
    def _clean_trip_data(self, df: pd.DataFrame) -> pd.DataFrame:
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
        
        return df.dropna(subset=['ts', 'latitude', 'longitude'])
    
    def process_trips_by_route(self, trip_df: pd.DataFrame, route_id: str) -> pd.DataFrame:
        """Process all trips for a specific route with feature engineering"""
        route_trips = trip_df[trip_df['route_id'] == route_id].reset_index(drop=True).copy()
        
        if route_trips.empty:
            return pd.DataFrame()
        
        # Get route sequence
        route_sequence = self.route_sequences_df[self.route_sequences_df['route_id'] == int(route_id)]
        if route_sequence.empty:
            return pd.DataFrame()
        
        stop_sequence = route_sequence.iloc[0]['stop_id_list']
        next_route_stop_distances = route_sequence.iloc[0]['next_stop_distances']
        
        processed_trips = []
        
        for trip_id in route_trips['trip_id'].unique():
            trip_data = route_trips[route_trips['trip_id'] == trip_id].reset_index(drop=True).copy()
            trip_data = trip_data.sort_values('ts')
            
            # Add only basic sequential data (raw data for feature engineering)
            trip_data = self._add_basic_sequential_data(trip_data)
            
            # Add basic route context (not features)
            trip_data = self._add_route_context(trip_data, stop_sequence, next_route_stop_distances)
            
            processed_trips.append(trip_data)
        
        return pd.concat(processed_trips, ignore_index=True) if processed_trips else pd.DataFrame()
    
    def _add_basic_sequential_data(self, trip_data: pd.DataFrame) -> pd.DataFrame:
        """Add basic sequential data (next point info) - NOT features"""

        trip_data['geometry'] = gpd.points_from_xy(trip_data['longitude'], trip_data['latitude'])
        # Add the raw sequential data that feature engineering needs
        trip_data['next_lat'] = trip_data['latitude'].shift(-1)
        trip_data['next_lon'] = trip_data['longitude'].shift(-1)
        trip_data['next_timestamp'] = trip_data['ts'].shift(-1)
        
        # Fill nulls at end of trips with same values (cleaner than null checking)
        # This makes segment calculations naturally result in 0 distance/duration for trip end
        trip_data['next_lat'] = trip_data['next_lat'].fillna(trip_data['latitude'])
        trip_data['next_lon'] = trip_data['next_lon'].fillna(trip_data['longitude'])
        trip_data['next_timestamp'] = trip_data['next_timestamp'].fillna(trip_data['ts'])
        return trip_data
    
    def _add_route_context(self, trip_data: pd.DataFrame, route_stop_ids: List[str], next_route_stop_distances: List[float]) -> pd.DataFrame:
        """Add route context using lat/lon mapping to route sequence (stop_id unreliable)"""
        
        # Map each GPS point to route sequence using coordinates
        route_positions = self._map_gps_to_route_sequence(trip_data, route_stop_ids)
        
        trip_data['route_position'] = route_positions
        trip_data['total_route_stops'] = len(route_stop_ids)
        
        # Keep track of current stop
        trip_data['current_stop_id'] = [route_stop_ids[pos] if pos < len(route_stop_ids) else None 
                                       for pos in route_positions]
        # Add next stop ID from the sequence
        trip_data['next_stop_id'] = [route_stop_ids[pos + 1] if pos + 1 < len(route_stop_ids) else route_stop_ids[pos] 
                                    for pos in route_positions]
        
        trip_data['future_stop_ids'] = [route_stop_ids[pos + 1:] if pos + 1 < len(route_stop_ids) else route_stop_ids[pos:] 
                                    for pos in route_positions]
        
        # Add next stop distance from the sequence
        # Get next stop geometries
        next_stop_geoms = trip_data['next_stop_id'].map(self.stops_df.set_index('stop_id')['geometry'])
        # Calculate distances using vectorized operation
        trip_data['distance_to_next_route_stop'] = GeoUtils.calculate_dist_vectorized(
            trip_data['geometry'],
            next_stop_geoms
        )
        trip_data['future_segment_distances'] = [next_route_stop_distances[pos + 1:] if pos + 1 < len(next_route_stop_distances) else next_route_stop_distances[pos:] 
                                    for pos in route_positions]
        trip_data = self._calculate_duration_to_future_stops(trip_data)

        trip_data['distance_remaining_after_next_stop'] = [sum(next_route_stop_distances[(pos+1):]) if pos+1 < len(next_route_stop_distances) else 0 
                                    for pos in route_positions]

        trip_data['distance_remaining'] = trip_data['distance_to_next_route_stop'] + trip_data['distance_remaining_after_next_stop']
        return trip_data
    
    def _calculate_arrival_time_to_future_stops(self, trip_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate arrival time to future stops"""
        # Ensure the column exists with object dtype to hold lists
        if 'arrival_time_to_future_stops' not in trip_data.columns:
            trip_data['arrival_time_to_future_stops'] = None

        for i, row in trip_data.iterrows():
            future_stop_ids = row['future_stop_ids']
            ts = row['ts']
            arrival_times = []
            for future_stop_id in future_stop_ids:
                matches = trip_data.loc[trip_data['current_stop_id'] == future_stop_id, 'ts']
                if matches.empty:
                    continue
                future_ts = matches.iloc[0]
                arrival_times.append(future_ts)
            # Use .at for scalar cell assignment to avoid iterable shape issues
            trip_data.at[i, 'arrival_time_to_future_stops'] = durations
        return trip_data

    def _calculate_duration_to_future_stops(self, trip_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate duration to future stops"""
        # Ensure the column exists with object dtype to hold lists
        if 'duration_to_future_stops' not in trip_data.columns:
            trip_data['duration_to_future_stops'] = None

        for i, row in trip_data.iterrows():
            future_stop_ids = row['future_stop_ids']
            ts = row['ts']
            durations = []
            for future_stop_id in future_stop_ids:
                matches = trip_data.loc[trip_data['current_stop_id'] == future_stop_id, 'ts']
                if matches.empty:
                    continue
                future_ts = matches.iloc[0]
                durations.append((future_ts - ts).total_seconds())
            # Use .at for scalar cell assignment to avoid iterable shape issues
            trip_data.at[i, 'duration_to_future_stops'] = durations
        return trip_data
    
    def _calculate_distance_to_future_stops(self, trip_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate distance to future stops"""
        # Ensure the column exists with object dtype to hold lists
        if 'distance_to_future_stops' not in trip_data.columns:
            trip_data['distance_to_future_stops'] = None

        for i, row in trip_data.iterrows():
            future_stop_ids = row['future_stop_ids']

    def _map_gps_to_route_sequence(self, trip_data: pd.DataFrame, stop_sequence: List[str]) -> List[int]:
        """
        Map GPS coordinates to route sequence positions
        Uses sequential progress - can't go backwards on route
        """
        
        route_positions = []
        current_furthest = 0  # Track furthest position reached
        trip_id = trip_data['trip_id'].iloc[0]
        
        # print(f"Trip ID: {trip_id} Mapping {len(trip_data)} GPS points to route sequence of {len(stop_sequence)} stops...")
        
        for i, row in trip_data.iterrows():
            lat, lon = row['latitude'], row['longitude']
            
            # Find best matching stop from current_furthest onwards (sequential constraint)
            best_position = self._find_best_route_position(
                lat, lon, stop_sequence, current_furthest
            )
            
            # Update furthest position (sequential progress)
            if best_position >= current_furthest:
                current_furthest = best_position
            else:
                # If we get a position behind current furthest, use current furthest
                # (could be GPS noise or bus waiting at stop)
                best_position = current_furthest
            
            route_positions.append(best_position)
        
        # Log progress tracking
        if route_positions:
            final_position = route_positions[-1]
            progress_pct = (final_position + 1) / len(stop_sequence) * 100
            # print(f"Trip progress: {final_position + 1}/{len(stop_sequence)} stops ({progress_pct:.1f}%)")
        
        return route_positions
    
    def _find_best_route_position(self, lat: float, lon: float, 
                                 stop_sequence: List[str], min_position: int) -> int:
        """
        Find the last stop that the bus has passed
        Returns position of the previous stop in the sequence
        Uses vector projection to determine if we've passed a stop
        """
        local_crs = GeoUtils.get_proj_crs(lon=lon,lat=lat)
        gps_point = gpd.points_from_xy([lon], [lat],crs=4326).to_crs(local_crs)[0]
        
        # Start from min_position and look at pairs of stops
        for i in range(min_position, len(stop_sequence)-1):
            current_stop_id = stop_sequence[i]
            next_stop_id = stop_sequence[i+1]
            
            # Get current stop coordinates
            current_stop = self.stops_df[self.stops_df['stop_id'] == current_stop_id].iloc[0]
            current_point = gpd.points_from_xy([current_stop['stop_lon']], [current_stop['stop_lat']],crs=4326).to_crs(local_crs)[0]
            
            # Get next stop coordinates
            next_stop = self.stops_df[self.stops_df['stop_id'] == next_stop_id].iloc[0]
            next_point = gpd.points_from_xy([next_stop['stop_lon']], [next_stop['stop_lat']],crs=4326).to_crs(local_crs)[0]
            
            # Calculate vector from current to next stop
            segment_vector = np.array([next_point.x - current_point.x, 
                                     next_point.y - current_point.y])
            segment_length = np.sqrt(np.sum(segment_vector**2))
            
            if segment_length == 0:
                continue  # Skip if stops are at same location
                
            # Calculate vector from current stop to GPS point
            gps_vector = np.array([gps_point.x - current_point.x,
                                 gps_point.y - current_point.y])
            
            # Project GPS vector onto segment vector
            # projection = (a·b/|b|²)b where b is segment vector
            projection_scalar = np.dot(gps_vector, segment_vector) / (segment_length**2)
            
            # If projection is > 1, we're past the next stop
            # If projection is < 0, we haven't reached current stop
            # If 0 <= projection <= 1, we're between stops
            if projection_scalar > 1:
                continue  # We've passed this segment, keep looking
            else:
                return i  # This is our previous stop
        
        # If we're past the last stop or very close to it
        return len(stop_sequence) - 1