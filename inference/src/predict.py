from pathlib import Path
import os
import json
import argparse
import pandas as pd
import pyarrow.parquet as pq
import logging
from model import HeuristicModel
from utils.test_data_processor import TestDataProcessor
from utils.feature_pipeline import FeaturePipeline
from utils.utils import Utils

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/predict.log'
)
logger = logging.getLogger(__name__)

class Predictor:
    """
    Main prediction engine for bus arrival times.
    
    This class orchestrates the complete prediction pipeline including
    data loading, preprocessing, feature extraction, and model inference
    for real-time bus arrival predictions.
    
    Attributes:
        test_data_processor (TestDataProcessor): Data processing component
        feature_pipeline (FeaturePipeline): Feature extraction pipeline
        heuristic_model (HeuristicModel): Trained prediction model
        use_ml_model (bool): Flag for ML model usage (future feature)
        ml_model: ML model instance (future feature)
        logger (logging.Logger): Logger instance for the class
    """
    
    def __init__(self):
        """
        Initialize the Predictor with required components.
        """
        print("Initializing Predictor")
        print(os.getcwd())
        print(os.listdir('/app'))
        print(os.listdir('/app/data'),os.listdir('/app/out'))
        print(os.listdir('/app/data/eval_data'))
        print(os.listdir('/app/utils'))
        print(os.listdir('/app/logs'))
        print(os.path.exists('/app/static_data'))
        print(os.path.exists('/app/model.pkl'))
        print(os.listdir('/app/static_data'))
        self.static_data_path = Path('../../preprocessing/data/processed/static_data')
        print(os.path.exists(self.static_data_path))
        print(f"Loading static data from {self.static_data_path}")
        if not os.path.exists(self.static_data_path):
            self.static_data_path = '/app/static_data'
            print(f"Static data file not found, using docker path: {self.static_data_path}")
        self.test_data_processor = TestDataProcessor(self.static_data_path)
        self.feature_pipeline = FeaturePipeline()
        
        self.model_path = Path('../../training/models/model.pkl')
        print(f"Loading model from {self.model_path}")
        if not os.path.exists(self.model_path):
            self.model_path = '/app/model.pkl'
            print(f"Model file not found, using docker path: {self.model_path}")
        self.heuristic_model = HeuristicModel()
        self.heuristic_model.load(self.model_path)
        print("Predictor initialized successfully")

    def load_and_process_data(self, parquet_path: str) -> tuple[pd.DataFrame, str]:
        """
        Load and process trip data for prediction.
        
        Args:
            parquet_path (str): Path to the parquet file containing trip data
            
        Returns:
            tuple[pd.DataFrame, str]: Processed data and route ID
        """
        print(f"Loading and processing data from {parquet_path}")
        
        df = pq.read_table(parquet_path).to_pandas() 
        df = self.test_data_processor.clean_trip_data(df)
        current_data, route_id = self.test_data_processor.get_current_data(df)
        current_df = self.test_data_processor.process_data(current_data, route_id)
        current_df = self.feature_pipeline.fit_transform(current_df, route_id)
        
        print(f"Processed data for route {route_id}: {len(current_df)} records")
        return current_df, route_id

    def predict(self, input_json_path: str, output_json_path: str) -> dict:
        """
        Generate predictions for bus arrival times.
        
        Args:
            input_json_path (str): Path to input JSON file with parquet file paths
            output_json_path (str): Path to output JSON file for predictions
            
        Returns:
            dict: Dictionary of predictions by route ID
        """
        print(f"Starting prediction process")
        print(f"Input: {input_json_path}, Output: {output_json_path}")
        
        if not os.path.exists(input_json_path):
            input_json_path = '/app/data/input.json'
            logger.warning(f"Input file not found, using docker path: {input_json_path}")
        
        with open(input_json_path, 'r') as f:
            input_json = json.load(f)
        
        print(f"Processing {len(input_json)} input files")
        predictions = {}
        
        for idx, parquet_path in input_json.items():
            print(f"Processing file {idx}: {parquet_path}")
            
            if not os.path.exists(parquet_path):
                parquet_path = os.path.join('/app/data', parquet_path)
                print(f"Using Docker path: {parquet_path}")
            
            try:
                df, route_id = self.load_and_process_data(parquet_path)
                predicted_arrival_times, _ = self.heuristic_model.predict(df)
                predictions[route_id] = predicted_arrival_times
                print(f"Predicted arrival times for route {route_id}: {predicted_arrival_times}")
            except Exception as e:
                print(f"Error processing {parquet_path}: {e}")
                continue
        
        # Save predictions
        with open(output_json_path, 'w') as f:
            json.dump(predictions, f)
        
        print(f"Predictions saved to {output_json_path}")
        return predictions

@Utils.timeit
def main():
    """
    Main entry point for bus arrival prediction.
    
    Parses command line arguments and runs the prediction pipeline.
    """
    arg_parser = argparse.ArgumentParser(
        description="Predict bus arrival times using trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --input-json data/input.json --output-json data/output.json
  python predict.py --input-json /app/data/input.json --output-json /app/out/output.json
        """
    )
    arg_parser.add_argument(
        "--input-json", 
        type=str, 
        required=True,
        help="Path to input JSON file containing parquet file paths"
    )
    arg_parser.add_argument(
        "--output-json", 
        type=str, 
        required=True,
        help="Path to output JSON file for predictions"
    )
    args = arg_parser.parse_args()
    input_json_path = args.input_json
    output_json_path = args.output_json
    
    print(f"Starting prediction with input: {input_json_path}")
    predictor = Predictor()
    predictor.predict(input_json_path, output_json_path)
    print("Prediction completed")

if __name__ == "__main__":
    main()
