from pathlib import Path
from utils.training_data_processor import TrainingDataProcessor
from utils.utils import Utils
from typing import List, Dict
import pandas as pd
import argparse
from utils.feature_pipeline import FeaturePipeline
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/training_data_generator.log'
)
logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    """
    Generating Features for the Historical Trip Data for the Bangalore Last Mile Challenge.
    
    This class handles the preprocessing of historical trip data to create
    training features for machine learning models. It processes trip data
    by route and date, extracts comprehensive features, and saves the
    processed data for model training.
    
    Attributes:
        data_path (Path): Path to the data directory
        data_processor (TrainingDataProcessor): Processor for trip data
        feature_pipeline (FeaturePipeline): Pipeline for feature extraction
        logger (logging.Logger): Logger instance for the class
    """
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        Initialize the TrainingDataGenerator.
        
        Args:
            raw_data_path (str): Path to the raw data directory
            processed_data_path (str): Path to the processed data directory
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
        logger.info("Initializing TrainingDataGenerator")
        # Initialize components
        self.data_processor = TrainingDataProcessor(raw_data_path, processed_data_path)
        self.feature_pipeline = FeaturePipeline()
    
    def generate_training_data(self, dates: List[str], exclude_routes: List[str] = [], include_routes: List[str] = [], max_routes: int = 5) -> Dict:
        """
        Prepare training data for a given list of dates.

        Args:
            dates (List[str]): List of dates in format ['date=2025-08-17', ...]
            exclude_routes (List[str], optional): List of route IDs to exclude. Defaults to [].
            include_routes (List[str], optional): List of route IDs to include (takes precedence over automatic selection). Defaults to [].
            max_routes (int, optional): Limit number of routes for speed (ignored if include_routes is provided). Defaults to 5.

        Returns:
            Dict: Dictionary of training data per route (currently returns empty dict)
        """
        logger.info(f"Starting training data generation for {len(dates)} dates")
        
        for date in dates:
            logger.info(f"Processing date: {date}")
            
            # Load raw data
            date_data = self.data_processor.load_trip_data_batch([date])

            if date_data.empty:
                logger.warning(f"No data present for date {date}")
                continue

            # Determine which routes to process
            if include_routes:
                # Use explicitly specified routes
                selected_routes = [r for r in include_routes if r not in exclude_routes]
                logger.info(f"Processing {len(selected_routes)} specified routes: {selected_routes}")
            else:
                # Select top routes by volume in train set
                route_counts_df = date_data.groupby('route_id', as_index=False).agg(count=('route_id', 'count')).sort_values('count', ascending=False)
                selected_routes = route_counts_df[~route_counts_df['route_id'].isin(exclude_routes)]['route_id'].tolist()
                selected_routes = selected_routes[:max_routes] if max_routes > 0 else selected_routes
                logger.info(f"Auto-selected {len(selected_routes)} routes by volume")

            for route_id in selected_routes:
                logger.info(f"Processing route {route_id} for date {date}")
                try:
                    route_data_all_trips_processed = self.data_processor.process_trips_by_route(date_data, route_id)
                    route_data_all_trips_processed_feats = self.feature_pipeline.fit_transform(route_data_all_trips_processed, route_id)
                    route_data_all_trips_processed_feats['geometry'] = route_data_all_trips_processed_feats['geometry'].apply(lambda x: x.wkt)
                    
                    out_dir = self.processed_data_path / "route_data_feats" / f"route_{route_id}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{date}.parquet"
                    route_data_all_trips_processed_feats.to_parquet(out_path)
                    
                    logger.info(f"Successfully processed route {route_id} for date {date}")
                except Exception as e:
                    logger.error(f"Error processing route {route_id} for date {date}: {e}")
                    continue

@Utils.timeit
def run(args):
    """
    Run the training data generator with timing.

    Args:
        args (argparse.Namespace): Command line arguments containing:
            - raw_data_path (str): Path to raw data directory
            - processed_data_path (str): Path to processed data directory
            - dates (str): Comma-separated list of dates
            - exclude_routes (str): Comma-separated list of route IDs to exclude
            - include_routes (str): Comma-separated list of route IDs to include
            - max_routes (int): Maximum number of routes to process

    Returns:
        None
    """
    raw_data_path = args.raw_data_path
    processed_data_path = args.processed_data_path
    max_routes = args.max_routes
    dates = args.dates.split(",")
    
    # Parse exclude routes
    exclude_routes = [r.strip() for r in args.exclude_routes.split(",") if r.strip()]
    
    # Parse include routes
    include_routes = [r.strip() for r in args.include_routes.split(",") if r.strip()]
    
    # Generate training data
    training_data_generator = TrainingDataGenerator(raw_data_path, processed_data_path)
    training_data_generator.generate_training_data(dates, exclude_routes, include_routes, max_routes)

def main():
    """
    Main entry point for the training data generator.
    
    Parses command line arguments and runs the training data generation pipeline.
    """
    arg_parser = argparse.ArgumentParser(
        description="Generate training data from historical trip data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training_data_generator.py --data-path /path/to/data --dates "date=2025-08-17,date=2025-08-18"
  python training_data_generator.py --data-path /path/to/data --dates "date=2025-08-17" --include-routes "1708,15889" --max-routes 10
        """
    )
    arg_parser.add_argument(
        "--raw-data-path", 
        type=str, 
        required=True, 
        help="Path to the Raw data directory"
    )
    arg_parser.add_argument(
        "--processed-data-path", 
        type=str, 
        required=True, 
        help="Path to the Processed data directory"
    )
    arg_parser.add_argument(
        "--dates", 
        type=str, 
        required=True, 
        default="date=2025-08-17,date=2025-08-18,date=2025-08-19",
        help="Comma-separated list of dates in format 'date=YYYY-MM-DD'"
    )
    arg_parser.add_argument(
        "--include-routes", 
        type=str, 
        required=False, 
        default="", 
        help="Comma-separated list of route IDs to include (takes precedence)"
    )
    arg_parser.add_argument(
        "--exclude-routes", 
        type=str, 
        required=False, 
        default="", 
        help="Comma-separated list of route IDs to exclude"
    )
    arg_parser.add_argument(
        "--max-routes", 
        type=int, 
        required=False, 
        default=5, 
        help="Maximum number of routes to process (ignored if include-routes is set)"
    )
    args = arg_parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()