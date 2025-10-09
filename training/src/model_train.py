from model import HeuristicModel
import pyarrow.dataset as ds
import argparse
from pathlib import Path
import pandas as pd
import logging
from utils.utils import Utils

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/model_train.log'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Model training pipeline for bus arrival prediction.
    
    This class handles the training of machine learning models using
    preprocessed feature data from multiple routes. It loads training
    data, trains the model, and saves the trained model for inference.
    
    Attributes:
        data_path (Path): Path to the data directory
        route_data_feats_path (Path): Path to route feature data
        available_routes (List[str]): List of available route IDs
        model (HeuristicModel): The model to be trained
        logger (logging.Logger): Logger instance for the class
    """
    
    def __init__(self, processed_data_path: str, model_path: str):
        """
        Initialize the ModelTrainer.
        
        Args:
            processed_data_path (str): Path to the data directory containing route_data_feats
        """
        self.data_path = Path(processed_data_path)
        self.route_data_feats_path = self.data_path / "route_data_feats"
        self.available_routes = [route.name.split("_")[1] for route in self.route_data_feats_path.iterdir() if route.is_dir()]
        self.model = HeuristicModel()
        self.model_path = Path(model_path) / "model.pkl"
        logger.info(f"Initialized ModelTrainer with {len(self.available_routes)} available routes")

    def train(self):
        """
        Train the model using all available route data.
        
        Loads training data from all available routes, combines them,
        and trains a unified model for bus arrival prediction.
        """
        logger.info("Starting model training")
        all_dfs = []
        
        for route_id in self.available_routes:
            # logger.info(f"Loading data for route {route_id}")
            try:
                # Try reading with schema inference disabled and ignore schema conflicts
                route_data = ds.dataset(
                    self.route_data_feats_path / f"route_{route_id}", 
                    format="parquet",
                    schema=None  # Let PyArrow infer schema
                )
                # Use safe conversion options
                route_data_df = route_data.to_table(
                    columns=None,  # Read all columns
                    filter=None    # No filtering at read time
                ).to_pandas(
                    ignore_metadata=True,  # Ignore metadata conflicts
                    safe=False            # Allow unsafe conversions
                )
            except Exception as e:
                logger.warning(f"Error reading route {route_id} with dataset: {e}")
                try:
                    # Fallback: read individual parquet files
                    import glob
                    parquet_files = glob.glob(str(self.route_data_feats_path / f"route_{route_id}" / "*.parquet"))
                    if not parquet_files:
                        logger.warning(f"No parquet files found for route {route_id}")
                        continue
                    
                    dfs = []
                    for file in parquet_files:
                        try:
                            df = pd.read_parquet(file, engine='pyarrow')
                            dfs.append(df)
                        except Exception as file_error:
                            logger.error(f"Error reading file {file}: {file_error}")
                            continue
                    
                    if not dfs:
                        logger.warning(f"No valid data found for route {route_id}")
                        continue
                        
                    route_data_df = pd.concat(dfs, ignore_index=True)
                except Exception as fallback_error:
                    logger.error(f"Fallback failed for route {route_id}: {fallback_error}")
                    continue
            
            # Filter out invalid data
            route_data_df = route_data_df[route_data_df.segment_duration > 0].reset_index(drop=True)
            all_dfs.append(route_data_df)
            # logger.info(f"Loaded {len(route_data_df)} records for route {route_id}")
        
        if not all_dfs:
            logger.error("No training data loaded")
            return
            
        # Combine all route data
        combined_data = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined training data: {len(combined_data)} total records")
        
        # Train the model
        logger.info("Training defaultmodel with combined data")
        self.model.train(combined_data, 'default')
        
        # Save the model
        self.model.save(self.model_path)
        logger.info(f"Model saved to {self.model_path}")

@Utils.timeit
def main():
    """
    Main entry point for model training.
    
    Parses command line arguments and runs the model training pipeline.
    """
    arg_parser = argparse.ArgumentParser(
        description="Train bus arrival prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_train.py --processed-data-path ../../preprocessing/data/processed --model-path ../models
        """
    )
    arg_parser.add_argument(
        "--processed-data-path", 
        type=str, 
        required=True, 
        default="../../preprocessing/data/processed",
        help="Path to the data directory containing route_data_feats"
    )
    arg_parser.add_argument(
        "--model-path", 
        type=str, 
        required=False, 
        default="../models",
        help="Path to the model file"
    )
    args = arg_parser.parse_args()
    data_path = args.processed_data_path
    model_path = args.model_path
    logger.info(f"Starting model training with data path: {data_path}")
    model_trainer = ModelTrainer(data_path, model_path)
    model_trainer.train()
    logger.info("Model training completed")

if __name__ == "__main__":
    main()