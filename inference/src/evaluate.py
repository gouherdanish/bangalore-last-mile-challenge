#!/usr/bin/env python3
"""
Evaluation script for bus arrival prediction model.

This script compares predicted vs actual arrival times and calculates
various metrics including Mean Absolute Error (MAE) across all routes
and stops for the Bangalore Last Mile Challenge.
"""

import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/evaluate.log'
)
logger = logging.getLogger(__name__)


class PredictionEvaluator:
    """
    Evaluator for bus arrival prediction model performance.
    
    This class handles the evaluation of predicted vs actual arrival times,
    calculating various metrics including MAE, RMSE, and accuracy statistics
    across different routes and stops.
    
    Attributes:
        predicted_data (Dict): Predicted arrival times by route and stop
        actual_data (Dict): Actual arrival times by route and stop
        logger (logging.Logger): Logger instance for the class
    """
    
    def __init__(self, predicted_file: str, actual_file: str):
        """
        Initialize the PredictionEvaluator.
        
        Args:
            predicted_file (str): Path to predicted arrival times JSON file
            actual_file (str): Path to actual arrival times JSON file
        """
        self.predicted_data = {}
        self.actual_data = {}
        self.logger = logging.getLogger(__name__)
        
        logger.info(f"Initializing evaluator with predicted: {predicted_file}, actual: {actual_file}")
        self._load_data(predicted_file, actual_file)
    
    def _load_data(self, predicted_file: str, actual_file: str):
        """
        Load predicted and actual data from JSON files.
        
        Args:
            predicted_file (str): Path to predicted data file
            actual_file (str): Path to actual data file
        """
        try:
            with open(predicted_file, 'r') as f:
                self.predicted_data = json.load(f)
            logger.info(f"Loaded predicted data with {len(self.predicted_data)} routes")
            
            with open(actual_file, 'r') as f:
                self.actual_data = json.load(f)
            logger.info(f"Loaded actual data with {len(self.actual_data)} routes")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """
        Parse timestamp string to datetime object.
        
        Args:
            timestamp_str (str): Timestamp in format 'YYYY-MM-DD HH:MM:SS'
            
        Returns:
            datetime: Parsed datetime object
        """
        try:
            return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
            raise
    
    def _calculate_time_difference(self, predicted_time: str, actual_time: str) -> float:
        """
        Calculate time difference between predicted and actual arrival times.
        
        Args:
            predicted_time (str): Predicted arrival time
            actual_time (str): Actual arrival time
            
        Returns:
            float: Time difference in seconds (positive = prediction late, negative = prediction early)
        """
        pred_dt = self._parse_timestamp(predicted_time)
        actual_dt = self._parse_timestamp(actual_time)
        
        # Calculate difference in seconds
        time_diff = (pred_dt - actual_dt).total_seconds()/60
        return time_diff
    
    def evaluate_route(self, route_id: str) -> float:
        """
        Evaluate prediction accuracy for a specific route.
        
        Args:
            route_id (str): Route identifier
            
        Returns:
            float: Average absolute time difference in minutes for the route
        """
        logger.info(f"Evaluating route {route_id}")
        
        # Check if route exists in actual data
        if route_id not in self.actual_data:
            logger.warning(f"Route {route_id} not found in actual data")
            return 0.0
        
        # If route not in predicted data, return default penalty of 100 minutes
        if route_id not in self.predicted_data:
            logger.warning(f"Route {route_id} not found in predicted data, using default penalty: 100 minutes")
            return 100.0
        
        predicted_stops = self.predicted_data[route_id]
        actual_stops = self.actual_data[route_id]
        
        # Get all actual stops for this route
        actual_stop_ids = set(actual_stops.keys())
        
        if not actual_stop_ids:
            logger.warning(f"No stops found for route {route_id}")
            return 0.0
        
        logger.info(f"Found {len(actual_stop_ids)} stops for route {route_id}")
        
        # Calculate absolute time differences for each stop
        absolute_differences = []
        
        for stop_id in actual_stop_ids:
            try:
                if stop_id not in predicted_stops:
                    # If stop not in predicted data, use default penalty of 10 minutes
                    logger.debug(f"Stop {stop_id} not found in predicted data, using default penalty: 10 minutes")
                    absolute_differences.append(10.0)
                else:
                    # Calculate time difference in minutes
                    time_diff_minutes = self._calculate_time_difference(
                        predicted_stops[stop_id], 
                        actual_stops[stop_id]
                    )
                    absolute_differences.append(abs(time_diff_minutes))
                    
                    logger.debug(f"Stop {stop_id}: pred={predicted_stops[stop_id]}, actual={actual_stops[stop_id]}, abs_diff={abs(time_diff_minutes):.2f} min")
                
            except Exception as e:
                logger.warning(f"Error processing stop {stop_id} in route {route_id}: {e}")
                # Use default penalty for error cases
                absolute_differences.append(10.0)
                continue
        
        if not absolute_differences:
            logger.warning(f"No valid time differences calculated for route {route_id}")
            return 0.0
        
        # Calculate average absolute difference for this route
        route_eta_score = np.mean(absolute_differences)
        
        logger.info(f"Route {route_id} ETA score: {route_eta_score:.2f} minutes")
        return route_eta_score
    
    def evaluate_all_routes(self) -> Dict:
        """
        Evaluate prediction accuracy across all routes.
        
        Returns:
            Dict: Overall ETA score and per-route results
        """
        logger.info("Starting evaluation of all routes")
        
        # Get all routes from actual data (ground truth)
        all_routes = set(self.actual_data.keys())
        
        if not all_routes:
            logger.error("No routes found in actual data")
            return {}
        
        logger.info(f"Evaluating {len(all_routes)} routes: {sorted(all_routes)}")
        
        route_scores = []
        route_details = []
        
        for route_id in sorted(all_routes):
            route_eta_score = self.evaluate_route(route_id)
            route_scores.append(route_eta_score)
            route_details.append({
                'route_id': route_id,
                'eta_score_minutes': route_eta_score
            })
        
        if not route_scores:
            logger.error("No valid route evaluations completed")
            return {}
        
        # Calculate overall ETA score (average across all routes)
        overall_eta_score = np.mean(route_scores)
        
        results = {
            'total_routes': len(route_scores),
            'overall_eta_score_minutes': float(overall_eta_score),
            'route_details': route_details
        }
        
        logger.info(f"Overall ETA Score: {overall_eta_score:.2f} minutes")
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results (Dict): Evaluation results
            output_file (str): Output file path
        """
        logger.info(f"Saving results to {output_file}")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved successfully to {output_file}")
    
    def print_summary(self, results: Dict):
        """
        Print a summary of evaluation results.
        
        Args:
            results (Dict): Evaluation results
        """
        print("\n" + "="*60)
        print("BUS ARRIVAL PREDICTION ETA SCORE EVALUATION")
        print("="*60)
        
        print(f"Total Routes Evaluated: {results['total_routes']}")
        print()
        
        print("OVERALL ETA SCORE:")
        print(f"  Average Absolute Time Difference: {results['overall_eta_score_minutes']:.2f} minutes")
        print()
        
        print("PER-ROUTE ETA SCORES:")
        print("-" * 60)
        for route_detail in results['route_details']:
            route_id = route_detail['route_id']
            eta_score = route_detail['eta_score_minutes']
            print(f"  Route {route_id}: ETA Score = {eta_score:.2f} minutes")
        
        print("="*60)
        print(f"FINAL ETA SCORE: {results['overall_eta_score_minutes']:.2f} minutes")
        print("="*60)


def main():
    """
    Main entry point for the evaluation script.
    
    Parses command line arguments and runs the evaluation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate bus arrival prediction model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --predicted data/output.json --actual data/actual.json
  python evaluate.py --predicted data/output.json --actual data/actual.json --output results/evaluation.json
  python evaluate.py --predicted data/output.json --actual data/actual.json --verbose
        """
    )
    
    parser.add_argument(
        "--predicted", 
        type=str, 
        default="../data/output.json",
        help="Path to predicted arrival times JSON file (default: ../data/output.json)"
    )
    parser.add_argument(
        "--actual", 
        type=str, 
        default="../data/actual.json",
        help="Path to actual arrival times JSON file (default: ../data/actual.json)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="logs/evaluation_results.json",
        help="Path to output results JSON file (default: logs/evaluation_results.json)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting evaluation with predicted: {args.predicted}, actual: {args.actual}")
    
    try:
        # Initialize evaluator
        evaluator = PredictionEvaluator(args.predicted, args.actual)
        
        # Run evaluation
        results = evaluator.evaluate_all_routes()
        
        if not results:
            logger.error("Evaluation failed - no results generated")
            return 1
        
        # Save results
        evaluator.save_results(results, args.output)
        
        # Print summary
        evaluator.print_summary(results)
        
        logger.info("Evaluation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
