# Inference Module

This module handles real-time bus arrival prediction using trained models for the Bangalore Last Mile Challenge.

## Overview

The inference pipeline provides real-time prediction capabilities for bus arrival times using pre-trained models and live trip data. It processes incoming trip data, extracts features, and generates arrival time predictions.

## Directory Structure

```
inference/
├── src/
│   ├── predict.py                    # Main prediction pipeline
│   ├── model.py                     # Model interface
│   ├── utils/                       # Utility modules package
│   │   ├── __init__.py              # Package initialization
│   │   ├── test_data_processor.py   # Real-time data processing
│   │   ├── feature_engineering.py   # Feature extraction for inference
│   │   ├── feature_pipeline.py      # Feature processing pipeline
│   │   └── geoutils.py              # Geographic utilities
│   └── logs/                        # Inference logs
│       └── predict.log              # Prediction process logs
├── data/
│   ├── input.json                   # Input configuration
│   ├── eval_data/                   # Evaluation data
│   └── static_data/                 # Static reference data
├── notebooks/                       # Jupyter notebooks for analysis
├── Dockerfile                       # Container configuration
├── docker-push.sh                   # Docker deployment script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Architecture

### Prediction Pipeline

The inference system follows a multi-stage pipeline:

1. **Data Loading** - Load live trip data from parquet files
2. **Data Processing** - Clean and validate incoming data
3. **Feature Extraction** - Extract temporal and spatial features
4. **Model Inference** - Generate predictions using trained models
5. **Output Generation** - Format and save predictions

### Core Components

#### Predictor Class
- **Main orchestrator** for the prediction pipeline
- Handles input/output management
- Coordinates between data processing and model inference
- Manages error handling and logging

#### TestDataProcessor Class
- **Real-time data processing** for live trip data
- Data cleaning and validation
- Route position detection and mapping
- Current state extraction

#### Feature Pipeline
- **Feature extraction** for inference data
- Temporal and spatial feature engineering
- Data preparation for model input

## Usage

### Prerequisites

1. **Trained Model**: Ensure you have a trained model file (`model.pkl`)
2. **Static Data**: Processed static data in `static_data/` directory
3. **Input Data**: Trip data in parquet format

### Basic Prediction

```bash
cd inference/src
python predict.py --input-json ../data/input.json --output-json ../data/output.json
```

### Docker Deployment

```bash
# Build Docker image
docker build -t bus-prediction .

# Run container
docker run -v $(pwd)/data:/app/data -v $(pwd)/out:/app/out bus-prediction
```

### Input Format

The input JSON file should contain paths to parquet files:

```json
{
  "0": "route_1708_trip_68185898.parquet",
  "1": "route_15889_trip_12345678.parquet"
}
```

### Output Format

The output JSON file contains predictions by route:

```json
{
  "1708": [120, 240, 360],
  "15889": [90, 180, 270]
}
```

## Data Processing

### Input Data Requirements

Trip data parquet files should contain:
- `route_id`: Route identifier
- `trip_id`: Trip identifier
- `vehicle_id`: Vehicle identifier
- `stop_id`: Stop identifier
- `vehicle_timestamp`: Unix timestamp
- `latitude`: GPS latitude
- `longitude`: GPS longitude
- `schedule_relationship`: Trip status

### Data Cleaning

The system performs automatic data cleaning:
- **String normalization**: Removes quotes and standardizes formats
- **Timestamp conversion**: Converts Unix timestamps to datetime
- **Coordinate validation**: Filters valid GPS coordinates
- **Trip filtering**: Keeps only scheduled trips

### Route Position Detection

Advanced algorithms determine bus position on route:
- **Geometric projection**: Projects GPS points onto route segments
- **Distance calculations**: Uses UTM projections for accuracy
- **Position validation**: Ensures logical route progression

## Feature Engineering

### Temporal Features
- Hour of day, day of week, month
- Rush hour indicators
- Cyclical time encodings (sin/cos)

### Spatial Features
- Route position and progress
- Distance to next stops
- Route complexity metrics

### Route-Specific Features
- Historical travel patterns
- Stop sequence information
- Route characteristics

## Model Integration

### Model Loading
- Automatic model loading from `model.pkl`
- Fallback mechanisms for missing models
- Model validation and error handling

### Prediction Process
- Feature preparation for model input
- Batch processing for multiple routes
- Confidence scoring and validation

### Output Generation
- Arrival time predictions in seconds
- Route-specific predictions
- JSON format for easy integration

## Logging and Monitoring

### Log Files

Prediction progress is logged to `logs/predict.log`:
- Data loading and processing status
- Feature extraction progress
- Model inference results
- Error handling and recovery

### Log Format

```
2025-01-XX XX:XX:XX - __main__ - INFO - Initializing Predictor
2025-01-XX XX:XX:XX - __main__ - INFO - Loading model from model.pkl
2025-01-XX XX:XX:XX - __main__ - INFO - Processing file 0: route_1708_trip_68185898.parquet
2025-01-XX XX:XX:XX - __main__ - INFO - Predicted arrival times for route 1708: [120, 240, 360]
```

### Monitoring

Key metrics tracked:
- **Processing time** per route
- **Data quality** indicators
- **Prediction accuracy** (when ground truth available)
- **Error rates** and recovery

## Error Handling

### Data Errors
- **Missing files**: Graceful handling with logging
- **Corrupted data**: Skip problematic records
- **Invalid coordinates**: Filter and continue processing

### Model Errors
- **Missing models**: Clear error messages
- **Prediction failures**: Fallback to default values
- **Feature errors**: Robust feature extraction

### System Errors
- **Memory issues**: Efficient data processing
- **File I/O errors**: Retry mechanisms
- **Network issues**: Offline operation support

## Performance Optimization

### Processing Efficiency
- **Vectorized operations**: Fast pandas operations
- **Memory management**: Streaming data processing
- **Parallel processing**: Multi-route concurrent processing

### Scalability Features
- **Batch processing**: Handle multiple routes simultaneously
- **Incremental updates**: Process new data efficiently
- **Caching**: Static data caching for performance

## Docker Deployment

### Container Configuration

The Dockerfile creates a production-ready container:
- **Base image**: Python 3.9 slim
- **Dependencies**: All required packages
- **Data volumes**: Mounted data and output directories
- **Health checks**: Container health monitoring

### Deployment Commands

```bash
# Build image
docker build -t bus-prediction:latest .

# Run with data volumes
docker run -d \
  -v /path/to/data:/app/data \
  -v /path/to/output:/app/out \
  --name bus-prediction \
  bus-prediction:latest

# Push to registry
./docker-push.sh
```

## Integration Examples

### API Integration

```python
from predict import Predictor

# Initialize predictor
predictor = Predictor()

# Generate predictions
predictions = predictor.predict(
    input_json_path="data/input.json",
    output_json_path="data/output.json"
)

# Access results
for route_id, arrival_times in predictions.items():
    print(f"Route {route_id}: {arrival_times} seconds")
```

### Batch Processing

```bash
# Process multiple input files
for input_file in data/inputs/*.json; do
    output_file="data/outputs/$(basename $input_file)"
    python predict.py --input-json "$input_file" --output-json "$output_file"
done
```

## Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   ERROR: Model file not found
   ```
   - Ensure `model.pkl` exists in the correct location
   - Check file permissions and paths

2. **Static Data Missing**
   ```
   ERROR: Static data not found
   ```
   - Verify `static_data/` directory exists
   - Ensure required files are present

3. **Invalid Input Data**
   ```
   WARNING: No valid data found
   ```
   - Check input parquet file format
   - Verify required columns are present

### Performance Issues

- **Slow Processing**: Check data size and complexity
- **Memory Usage**: Monitor memory consumption
- **File I/O**: Ensure fast storage access

## Configuration

### Environment Variables

- `MODEL_PATH`: Path to model file
- `STATIC_DATA_PATH`: Path to static data
- `LOG_LEVEL`: Logging verbosity
- `BATCH_SIZE`: Processing batch size

### Configuration Files

- `input.json`: Input file configuration
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container configuration

## Testing

### Unit Tests

```bash
# Run tests
python -m pytest tests/

# Test specific components
python -c "from utils.test_data_processor import TestDataProcessor; tdp = TestDataProcessor(); print('TestDataProcessor OK')"
```

### Integration Tests

```bash
# Test full pipeline
python predict.py --input-json test_data/input.json --output-json test_data/output.json

# Validate output
python -c "import json; print(json.load(open('test_data/output.json')))"
```

## Next Steps

After inference, results can be used for:

- **Real-time APIs**: Serve predictions via REST API
- **Dashboard Integration**: Display predictions in web interfaces
- **Mobile Apps**: Provide arrival times to mobile users
- **Analytics**: Analyze prediction accuracy and patterns

## Example Workflow

```bash
# 1. Prepare input data
echo '{"0": "route_1708_trip_68185898.parquet"}' > data/input.json

# 2. Run prediction
python predict.py --input-json data/input.json --output-json data/output.json

# 3. Check results
cat data/output.json
# {"1708": [120, 240, 360]}

# 4. Deploy with Docker
docker build -t bus-prediction .
docker run -v $(pwd)/data:/app/data bus-prediction
```

This completes the inference pipeline for the Bangalore Last Mile Challenge bus arrival prediction system.
