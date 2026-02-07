# Root Tracker

A tool for tracking and analyzing plant root growth from time-series images.

## Features

- **Automatic image preprocessing**: Cropping, rotation, and background removal
- **Image registration**: Aligns time-series images using template matching
- **Root tracking**: Segments and tracks individual roots across time
- **Plant identification**: Clusters roots by plant using stem detection
- **Statistical analysis**: Computes growth rates and root metrics
- **Batch processing**: Parallel processing of multiple image series

## Architecture

The code is organized into modular components ready for GUI integration:

```
root_tracker/
├── config.py           # Type-safe configuration (Python dataclasses)
├── pipeline.py         # Main orchestration class
├── models/             # Data models (Image, Root, Statistics)
├── preprocessing/      # Cropping, green detection, background removal
├── registration/       # Image alignment
├── tracking/           # Root segmentation and linking
├── analysis/           # Statistics calculation
└── io/                 # Loading and exporting
```

## Installation

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line

```bash
# Activate your virtual environment first!
source venv/bin/activate

# Run with default config
python scripts/run_pipeline.py --config configs/in_vitro.yaml

# Run without parallelization (useful for debugging)
python scripts/run_pipeline.py --config configs/in_vitro.yaml --no-parallel

# Verbose output
python scripts/run_pipeline.py --config configs/in_vitro.yaml -v
```

### Python API

```python
from root_tracker import Config, RootTrackingPipeline

# Load configuration
config = Config.from_yaml("configs/in_vitro.yaml")

# Create and run pipeline
pipeline = RootTrackingPipeline(config)
output_path = pipeline.run()
print(f"Results saved to: {output_path}")
```

### Step-by-Step Processing (for GUI integration)

```python
from root_tracker import Config, RootTrackingPipeline

config = Config.from_yaml("configs/in_vitro.yaml")
pipeline = RootTrackingPipeline(config)

# Step 1: Load images
series_dict = pipeline.load_images()

# Step 2: Process each series
all_stats = []
for series in series_dict.values():
    # Preprocess (crop, rotate, detect plants)
    pipeline.preprocess_series(series)

    # Register (align time series)
    pipeline.register_series(series)

    # Here a GUI could show results for user review/editing
    # Users could:
    # - Adjust crop boundaries
    # - Add/remove/edit roots
    # - Mark regions to ignore

    # Track and analyze
    stats = pipeline.track_and_analyze_series(series)
    all_stats.extend(stats)

# Step 3: Export results
pipeline.export_results(all_stats)
```

## Configuration

Configuration uses Python dataclasses for type safety. You can:

1. **Load from YAML** (recommended for configuration files):

   ```python
   config = Config.from_yaml("configs/in_vitro.yaml")
   ```

2. **Create programmatically** (for full type hints and IDE support):

   ```python
   from root_tracker.config import Config, DataConfig

   config = Config(
       n_clusters=6,
       rotation=180,
       data=DataConfig(
           input="my_data/",
           output="my_results/"
       )
   )
   ```

## Output

- **Annotated images**: Saved to the output directory with root visualization
- **Statistics CSV**: Contains per-plant metrics including:
  - Root total length and area
  - Main root depth and length
  - Relative growth rates (RGR)
  - Green (stem) area

## Future Development

The architecture is designed to support:

- GUI integration with step-by-step workflow
- Manual root editing and correction
- Batch operations (e.g., ignore regions across all images)
- RSML (Root System Markup Language) export

## License

See [LICENSE](LICENSE) file for details.
