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

## License

See [LICENSE](LICENSE) file for details.

### RSML replacements and export

Right-click an image and choose **Replace with RSML…**. This makes the imported
roots authoritative for that image: previews, statistics, CSV export, summaries,
and RSML export use the replacement. Replaced images show a purple **RSML** label
in the tree instead of a checkmark. Automatic preprocessing, registration, and
tracking skip replaced frames; they never use imported roots to seed other frames.

The replacement and its image background are saved in `.root_tracker_rsml/`
next to the photographs, separately from the processing cache. Replacements
survive restarting, clearing the cache, changing settings, masking, and moving
images to/from Aside. The photograph is not overwritten. Selecting another RSML
file explicitly replaces the previous replacement.

Replacement requires 2D polylines in the selected image's pixel coordinates
(the current processed image, or the photograph if it has not been processed).
The saved background keeps the original coordinate frame even if crop settings
later change. Invalid files and out-of-bounds geometry leave the current result
untouched. The standalone inspection command below can also read 3D documents.

Imported measurements use the RSML geometry:

- Total length is the sum of root polyline lengths in pixels, including laterals.
- Root count is the number of RSML root axes.
- Main-root length and depth use the longest primary axis; depth is its vertical extent.
- Relative growth rates use the updated lengths, including the next image's rate.
  Standard 1-based plant IDs match tracker slots. Other IDs match identical IDs
  in adjacent RSML documents; ambiguous cross-source matches have no growth rate.
- Segmented root area, green area, new pixel area/parts, and area change are
  unavailable for imported polylines and export as blank values, never old tracker
  measurements. Growth segmentation immediately after an import is also unavailable.
- CSVs containing replacements include `root_source` and `rsml_plant_id` columns.
  Native-only CSV output retains its existing fields and measurement definitions.

Right-click an image and choose **Export to RSML…** to export only that image.
For replaced images, export preserves the original XML byte-for-byte, including
unknown metadata, functions, annotations, namespaces, and comments. A frozen
background is included; safe image filenames referenced in the XML are retained.
External image paths are preserved as metadata but never followed. If a reference
cannot be used safely, the background is supplied as `preview.png` instead.

For native tracking results, tracking must be current; old caches without export
geometry require tracking again. Export produces an RSML file and matching
processed PNG in a distinct image folder. Geometry uses processed pixel
coordinates after cropping and registration (unit `pixel`, resolution `1`).
Disconnected segments and junction branches remain separate roots; biological
parentage and persistent temporal root IDs are not inferred. Isolated detections
become zero-length, repeated-point polylines. Re-importing such a file therefore
uses its explicit geometry rather than recovering the tracker's pixel metrics.

Choose a fresh output folder for repeated exports: existing destinations are
refused. Series names are sanitized and given a digest; frame indexes avoid name
collisions. A failed series is cleaned up; earlier completed series remain.

```bash
source venv/bin/activate
# Inspect or copy an external document without changing any image
python scripts/rsml.py roots.rsml
python scripts/rsml.py roots.rsml --output roots-copy.rsml
# Run the pipeline, honoring saved replacements, with optional RSML export
python scripts/run_pipeline.py --config configs/in_vitro.yaml --rsml-output rsml-results
```

Malformed XML, DTD/entity declarations, invalid coordinates, and RSML documents
over 64 MiB are rejected. There is no RSML geometry editor; unknown XML is preserved.
