from utils.geoutils import GeoUtils
from utils.utils import Utils
from pathlib import Path
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
from typing import List
import numpy as np
import argparse
import shapely
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/static_data_processor.log'
)
logger = logging.getLogger(__name__)

class StaticDataProcessor:
    """
    Static data processing pipeline for bus arrival prediction.
    
    This class handles the preprocessing of static reference data including
    bus stops, route sequences, and route information for the Bangalore
    Last Mile Challenge.
    
    Attributes:
        data_path (Path): Path to the input data directory
        output_path (Path): Path to the output data directory
        stops_df (pd.DataFrame): Processed stops data with geometries
        route_sequences_df (pd.DataFrame): Processed route sequences with distances
        logger (logging.Logger): Logger instance for the class
    """
    
    def __init__(self, data_path: str, output_path: str):
        """
        Initialize the StaticDataProcessor.
        
        Args:
            data_path (str): Path to the input data directory
            output_path (str): Path to the output data directory
            
        Raises:
            AssertionError: If data_path or output_path does not exist
        """
        
        self.data_path = Path(data_path)
        assert self.data_path.exists(), f"Error: Data path does not exist {self.data_path}"
        self.output_path = Path(output_path)
        assert self.output_path.exists(), f"Error: Output path does not exist {self.output_path}"
        self.stops_df = None
        self.route_sequences_df = None
        
        logger.info("Initializing StaticDataProcessor")
        self._load_static_data()
    
    def _load_static_data(self):
        """
        Load and process static reference data (stops, sequences).
        
        This method loads stops data and route sequences, processes them
        to create geometric representations and calculate distances between
        consecutive stops in each route.
        
        Raises:
            AssertionError: If required data files do not exist
        """
        logger.info("Loading static reference data")
        
        # Load stops
        stops_path = self.data_path / "base_data/stops_0.csv"
        assert stops_path.exists(), f"Error: Stops Data does not exist {stops_path}"
        
        logger.info(f"Loading stops from {stops_path}")
        self.stops_df = pd.read_csv(stops_path)
        self.stops_df['stop_id'] = self.stops_df['stop_id'].astype(str)
        self.stops_df['geometry'] = gpd.points_from_xy(self.stops_df['stop_lon'], self.stops_df['stop_lat'], crs=4326)
        logger.info(f"Loaded {len(self.stops_df)} stops")
        
        # Load route sequences
        from ast import literal_eval
        seq_path = self.data_path / "base_data/route_to_stop_sequence_v2.csv"
        assert seq_path.exists(), f"Error: Route to Stop Sequence Data does not exist {seq_path}"
        
        logger.info(f"Loading route sequences from {seq_path}")
        self.route_sequences_df = pd.read_csv(seq_path)
        
        logger.info("Processing route sequences and calculating distances")
        self.route_sequences_df['stop_id_list'] = self.route_sequences_df['stop_id_list'].apply(literal_eval)
        self.route_sequences_df['route_stop_geoms'] = self.route_sequences_df['stop_id_list'].apply(self._prepare_stops_geometry)
        self.route_sequences_df['next_stop_distances'] = self.route_sequences_df['route_stop_geoms'].apply(self._calculate_route_stop_distances)
        self.route_sequences_df['stop_id_list'] = self.route_sequences_df['stop_id_list'].apply(lambda stop_id: ', '.join(map(str, stop_id)))
        self.route_sequences_df['next_stop_distances'] = self.route_sequences_df['next_stop_distances'].apply(lambda dist: ', '.join(map(str, dist)))
        self.route_sequences_df.drop(columns=['route_stop_geoms'], inplace=True)
        
        logger.info(f"Loaded {len(self.stops_df)} stops, {len(self.route_sequences_df)} route sequences")
    
    def _prepare_stops_geometry(self, stop_ids: List[str]) -> List[shapely.geometry.Point]:
        """
        Prepare geometry objects for a list of stop IDs.
        
        Args:
            stop_ids (List[str]): List of stop IDs to get geometries for
            
        Returns:
            List[shapely.geometry.Point]: List of Point geometries for the stops
            
        Raises:
            AssertionError: If stop_ids list is empty
        """
        stop_geoms = []
        assert len(stop_ids) > 0, "Error: Route must have at least one stop"
        for stop_id in stop_ids:
            stop_geom = self.stops_df.loc[self.stops_df.stop_id == stop_id, 'geometry'].values[0]
            stop_geoms.append(stop_geom)
        return stop_geoms

    def _calculate_route_stop_distances(self, stop_geoms: List[shapely.geometry.Point]) -> List[float]:
        """
        Calculate distances between sequential stops in a route.
        
        Args:
            stop_geoms (List[shapely.geometry.Point]): List of stop geometries in route order
            
        Returns:
            List[float]: List of distances between consecutive stops (last stop distance is 0)
        """
        # Convert list to GeoSeries
        stop_geoms_series = gpd.GeoSeries(stop_geoms, crs=4326)
        # Get next stop geometries (shifted by 1)
        next_stop_geoms = stop_geoms_series.shift(-1)
        # Fill last stop with its own geometry
        next_stop_geoms.iloc[-1] = stop_geoms_series.iloc[-1]
        return GeoUtils.calculate_dist_vectorized(stop_geoms_series, next_stop_geoms).round(1)

    def save_static_data(self):
        """
        Save processed static data to output directory.
        
        Creates the output directory if it doesn't exist and saves:
        - Stops data as Shapefile format
        - Route sequences as CSV format
        """
        output_path = self.output_path / "static_data"
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving static data to {output_path}")
        
        # Save stops as shapefile
        stops_output = output_path / "stops_0.shp"
        gpd.GeoDataFrame(self.stops_df, crs=4326).to_file(stops_output)
        logger.info(f"Saved stops data to {stops_output}")
        
        # Save route sequences as CSV
        sequences_output = output_path / "route_to_stop_sequence.csv"
        self.route_sequences_df.to_csv(sequences_output, index=False)
        logger.info(f"Saved route sequences to {sequences_output}")

@Utils.timeit
def run(args):
    """
    Run the static data processor with timing.

    Args:
        args (argparse.Namespace): Command line arguments containing:
            - data_path (str): Path to input data directory
            - output_path (str): Path to output data directory

    Returns:
        None
    """
    data_path = args.data_path
    output_path = args.output_path
    data_processor = StaticDataProcessor(data_path, output_path)
    data_processor.save_static_data()

def main():
    """
    Main entry point for the static data processor.
    
    Parses command line arguments and runs the static data processing pipeline.
    """
    arg_parser = argparse.ArgumentParser(
        description="Process static reference data for bus arrival prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python static_data_processor.py --data-path ./data --output-path ./processed
  python static_data_processor.py --data-path /path/to/data --output-path /path/to/output
        """
    )
    arg_parser.add_argument(
        "--data-path", 
        type=str, 
        required=True,
        help="Path to the input data directory containing base_data folder"
    )
    arg_parser.add_argument(
        "--output-path", 
        type=str, 
        required=True,
        help="Path to the output directory for processed static data"
    )
    args = arg_parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()