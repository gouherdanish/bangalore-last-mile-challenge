# Preprocessing Module

This module handles the preprocessing of static data and historical trip data for the Bangalore Last Mile Challenge.

## Overview

The preprocessing pipeline consists of two main steps:
1. **Static Data Preprocessing** - Process reference data (stops, routes, sequences)
2. **Training Data Generation** - Process historical trip data and create features

## Directory Structure

```
preprocessing/
├── src/
│   ├── static_data_processor.py      # Static data preprocessing
│   ├── training_data_generator.py     # Training data generation
│   └── utils/                        # Utility modules
│       ├── geoutils.py               # Geographic utilities
│       ├── feature_engineering.py   # Feature extraction
│       ├── feature_pipeline.py      # Feature pipeline
│       └── training_data_processor.py # Trip data processing
├── data/
│   ├── raw/                         # Raw input data
│   │   ├── other_data/              # Static reference data
│   │   │   ├── stops_0.csv
│   │   │   ├── route_to_stop_sequence_v2.csv
│   │   │   └── routes.csv
│   │   └── trip_data/               # Historical trip data
│   │       └── BMTC_*_august_*/     # Date-based trip data folders
│   └── processed/                   # Processed output data
│       ├── static_data/             # Processed static data
│       └── route_data_feats/        # Processed training features
└── README.md                        # This file
```

## Step 1: Static Data Preprocessing

### Purpose
Process static reference data including bus stops, route sequences, and route information.

### Input Files
- `raw/other_data/stops_0.csv` - Bus stop locations and metadata
- `raw/other_data/route_to_stop_sequence_v2.csv` - Route stop sequences
- `raw/other_data/routes.csv` - Route information

### Output Files
- `processed/static_data/stops_0.shp` - Processed stop geometries
- `processed/static_data/route_to_stop_sequence.csv` - Processed route sequences with distances

### Usage

```bash
cd preprocessing/src
python static_data_processor.py --data-path ../data
```

Example:
```
python static_data_processor.py --data-path /Users/gouher/Documents/personal/codes/bangalore-last-mile-challenge/preprocessing/data/raw --output-path /Users/gouher/Documents/personal/codes/bangalore-last-mile-challenge/preprocessing/data/processed
```

### Parameters
- `--data-path`: Path to the data directory (default: `./data`)

### What it does
1. Loads stop data and creates geometric points
2. Processes route sequences and calculates distances between consecutive stops
3. Saves processed data in formats suitable for downstream processing

## Step 2: Training Data Generation

### Purpose
Process historical trip data to create training features for machine learning models.

### Input Files
- `raw/trip_data/BMTC_*_august_*/` - Historical trip data organized by date ranges

### Output Files
- `processed/route_data_feats/route_{route_id}/` - Feature data for each route
  - `{date}.parquet` - Processed features for specific dates

### Usage

```bash
cd preprocessing/src
python training_data_generator.py --data-path ../data --dates "date=2025-08-17,date=2025-08-18,date=2025-08-19"
```

Example
```
python training_data_generator.py --raw-data-path /Users/gouher/Documents/personal/codes/bangalore-last-mile-challenge/preprocessing/data/raw/trip_data/BMTC_05_august_to_07_august --processed-data-path /Users/gouher/Documents/personal/codes/bangalore-last-mile-challenge/preprocessing/data/processed --dates "date=2025-08-07" --include-routes "15889"
```

### Parameters
- `--data-path`: Path to the data directory (default: `./data`)
- `--dates`: Comma-separated list of dates to process (required)
- `--exclude-routes`: Comma-separated list of route IDs to exclude (optional)
- `--include-routes`: Comma-separated list of route IDs to include (optional, takes precedence)
- `--max-routes`: Maximum number of routes to process (default: 5, ignored if include-routes is set)

### What it does
1. Loads historical trip data for specified dates
2. Processes trips by route to create sequential data
3. Extracts temporal, spatial, and route-specific features
4. Saves processed features for each route and date

## Feature Engineering

The preprocessing pipeline extracts several types of features:

### Temporal Features
- Hour, day of week, month, week of year
- Rush hour indicators (morning/evening peak)
- Cyclical encoding (sin/cos transformations)

### Spatial Features
- Segment distances between consecutive GPS points
- Movement patterns (speed, acceleration, bearing changes)
- Route complexity metrics

### Route Characteristics
- Average trip length and speed
- Stop density and route complexity
- Progress indicators (stops remaining, progress ratio)

## Data Flow

```
Raw Data → Static Processing → Training Data Generation → Feature Engineering → Processed Data
    ↓              ↓                    ↓                      ↓                ↓
stops_0.csv → static_data/ → trip_data/ → route_data_feats/ → ML Training
routes.csv
sequences.csv
```

## Requirements

Ensure you have the following Python packages installed:
- pandas
- geopandas
- pyarrow
- numpy
- shapely
- pyproj
- scikit-learn
- scipy

## Example Workflow

1. **Prepare static data:**
   ```bash
   cd preprocessing/src
   python static_data_processor.py --data-path ../data
   ```

2. **Generate training data for specific dates:**
   ```bash
   python training_data_generator.py \
     --data-path ../data \
     --dates "date=2025-08-17,date=2025-08-18,date=2025-08-19" \
     --max-routes 10
   ```

3. **Process specific routes:**
   ```bash
   python training_data_generator.py \
     --data-path ../data \
     --dates "date=2025-08-17" \
     --include-routes "1708,15889,4247"
   ```

## Output Structure

After preprocessing, you'll have:

```
processed/
├── static_data/
│   ├── stops_0.shp                    # Stop geometries
│   ├── stops_0.dbf                    # Stop attributes
│   ├── stops_0.prj                    # Projection info
│   ├── stops_0.shx                    # Shape index
│   └── route_to_stop_sequence.csv     # Route sequences with distances
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

## Troubleshooting

### Common Issues
1. **Missing data files**: Ensure raw data is in the correct directory structure
2. **Memory issues**: Use `--max-routes` to limit processing for large datasets
3. **Date format**: Use the exact format `date=YYYY-MM-DD` for date parameters

### Performance Tips
- Process data in smaller batches using `--max-routes`
- Use `--include-routes` to focus on specific routes of interest
- Monitor disk space as processed data can be large

## Next Steps

After preprocessing, the processed data can be used for:
- Model training (see `../training/` directory)
- Live prediction (see `../inference/` directory)
- Feature analysis and exploration
