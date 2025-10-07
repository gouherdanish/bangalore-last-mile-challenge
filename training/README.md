# Training Module

This module handles the training of machine learning models for bus arrival prediction in the Bangalore Last Mile Challenge.

## Overview

The training pipeline consists of:
1. **Model Training** - Train heuristic-based models using preprocessed feature data
2. **Model Persistence** - Save trained models for inference

## Directory Structure

```
training/
├── src/
│   ├── model_train.py          # Model training pipeline
│   ├── model.py               # Heuristic model implementation
│   └── logs/                  # Training logs
│       └── model_train.log    # Training process logs
├── models/                    # Trained model storage
│   ├── model.pkl             # Main trained model
│   └── ml_model.pkl          # Alternative ML model
└── README.md                 # This file
```

## Model Architecture

### HeuristicModel

The `HeuristicModel` class implements a heuristic-based approach for bus arrival prediction:

- **Historical Pattern Analysis**: Analyzes travel times between stops under different temporal conditions
- **Statistical Modeling**: Uses mean, standard deviation, and count statistics for predictions
- **Temporal Grouping**: Groups data by stop pairs, hour, and day of week for contextual predictions
- **Route-Specific Training**: Supports both route-specific and default (all-routes) models

### Key Features

- **Pattern Recognition**: Identifies travel time patterns based on historical data
- **Temporal Awareness**: Considers time-of-day and day-of-week variations
- **Robust Statistics**: Uses multiple statistical measures for reliable predictions
- **Scalable Design**: Handles both individual routes and combined route data

## Usage

### Prerequisites

Ensure you have completed the preprocessing step and have processed feature data available in the `route_data_feats` directory.

### Training a Model

#### Basic Training

```bash
cd training/src
python model_train.py --processed-data-path ../../preprocessing/data/processed --model-path ../models
```

#### With Custom Paths

```bash
python model_train.py \
  --processed-data-path /path/to/processed/data \
  --model-path /path/to/model/output
```

### Parameters

- `--processed-data-path`: Path to the processed data directory containing `route_data_feats`
- `--model-path`: Path where the trained model will be saved

## Training Process

### 1. Data Loading

The training process loads feature data from all available routes:

```
processed_data/
└── route_data_feats/
    ├── route_1708/
    │   ├── date=2025-08-17.parquet
    │   ├── date=2025-08-18.parquet
    │   └── date=2025-08-19.parquet
    ├── route_15889/
    │   └── date=2025-08-17.parquet
    └── route_4247/
        └── date=2025-08-17.parquet
```

### 2. Data Processing

- **Route Discovery**: Automatically discovers available routes from directory structure
- **Data Combination**: Combines data from all routes for unified training
- **Quality Filtering**: Filters out invalid records (e.g., zero duration segments)
- **Schema Handling**: Robust handling of parquet schema variations

### 3. Pattern Calculation

The model calculates statistical patterns for:

- **Stop Pairs**: Travel times between consecutive stops
- **Temporal Conditions**: Hour of day and day of week variations
- **Speed Analysis**: Average and standard deviation of speeds
- **Distance Metrics**: Segment distances for normalization

### 4. Model Persistence

Trained models are saved as pickle files containing:
- Historical patterns dictionary
- Statistical summaries for each stop-time combination
- Model metadata and configuration

## Model Output

### Trained Model Structure

```python
{
    'default': {
        'stop_travel_times': DataFrame with columns:
            - current_stop_id: Stop identifier
            - hour: Hour of day (0-23)
            - day_of_week: Day of week (0-6)
            - segment_duration: [mean, std, count]
            - speed: [mean, std]
            - segment_distance: mean
    }
}
```

### Usage in Inference

The trained model can be loaded and used for predictions:

```python
from model import HeuristicModel

# Load trained model
model = HeuristicModel()
model.load('models/model.pkl')

# Use for predictions (see inference module)
```

## Logging and Monitoring

### Log Files

Training progress is logged to `logs/model_train.log` with:
- Route discovery and data loading progress
- Training statistics and data quality metrics
- Error handling and fallback operations
- Model saving confirmation

### Log Format

```
2025-01-XX XX:XX:XX - __main__ - INFO - Initialized ModelTrainer with 5 available routes
2025-01-XX XX:XX:XX - __main__ - INFO - Loading data for route 1708
2025-01-XX XX:XX:XX - __main__ - INFO - Loaded 1250 records for route 1708
2025-01-XX XX:XX:XX - __main__ - INFO - Combined training data: 5420 total records
2025-01-XX XX:XX:XX - __main__ - INFO - Model saved to models/model.pkl
```

## Error Handling

The training pipeline includes robust error handling:

### Data Loading Errors
- **Schema Conflicts**: Automatic fallback to individual file reading
- **Missing Files**: Graceful handling of missing route data
- **Corrupted Data**: Skip problematic files with detailed logging

### Training Errors
- **Empty Data**: Validation of training data availability
- **Invalid Records**: Filtering of invalid or corrupted records
- **Memory Issues**: Efficient data processing for large datasets

## Performance Considerations

### Optimization Features
- **Vectorized Operations**: Efficient pandas operations for large datasets
- **Memory Management**: Streaming data processing for large files
- **Parallel Processing**: Potential for multi-route parallel processing

### Scalability
- **Route Scaling**: Handles varying numbers of routes automatically
- **Data Volume**: Efficient processing of large historical datasets
- **Model Size**: Compact model storage with essential patterns only

## Integration with Other Modules

### Preprocessing Integration
- **Input**: Uses processed feature data from preprocessing module
- **Dependencies**: Requires `route_data_feats` directory structure
- **Data Format**: Expects parquet files with specific feature columns

### Inference Integration
- **Output**: Produces model files for inference module
- **Compatibility**: Model format compatible with inference pipeline
- **Versioning**: Supports model versioning and updates

## Troubleshooting

### Common Issues

1. **No Training Data Found**
   ```
   ERROR: No training data loaded
   ```
   - Ensure preprocessing has been completed
   - Check that `route_data_feats` directory exists
   - Verify processed data contains valid parquet files

2. **Schema Conflicts**
   ```
   WARNING: Error reading route XXXX with dataset: ...
   ```
   - Normal behavior, system will fallback to individual file reading
   - Check logs for successful fallback processing

3. **Empty Route Data**
   ```
   WARNING: No valid data found for route XXXX
   ```
   - Route may have insufficient historical data
   - Check preprocessing logs for data quality issues

### Performance Tips

- **Data Quality**: Ensure preprocessing produces clean, valid data
- **Route Selection**: Focus on routes with sufficient historical data
- **Memory Management**: Monitor memory usage for large datasets
- **Log Monitoring**: Check logs for data quality and processing issues

## Next Steps

After training, the model can be used for:

- **Live Prediction** (see `../inference/` directory)
- **Model Evaluation** and performance analysis
- **Feature Importance** analysis
- **Model Updates** with new training data

## Example Workflow

```bash
# 1. Complete preprocessing
cd ../preprocessing/src
python static_data_processor.py --data-path ../data/raw --output-path ../data/processed
python training_data_generator.py --raw-data-path ../data/raw --processed-data-path ../data/processed --dates "date=2025-08-17,date=2025-08-18"

# 2. Train model
cd ../../training/src
python model_train.py --processed-data-path ../../preprocessing/data/processed --model-path ../models

# 3. Use for inference
cd ../../inference/src
python predict.py --input-json data/input.json --output-json data/output.json
```

This completes the training pipeline for the Bangalore Last Mile Challenge bus arrival prediction system.
