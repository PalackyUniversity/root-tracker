# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/claude-code) when working with code in this repository.

## Project Overview

Root Tracker is a Python tool for tracking and analyzing plant root growth from time-series images. It processes images through a pipeline: cropping, rotation, plant detection, background removal, registration, root segmentation/linking, and statistical analysis.

## Common Commands

### Running the Pipeline

```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Unix

# Run with config file
python scripts/run_pipeline.py --config configs/in_vitro.yaml

# Run without parallelization (useful for debugging)
python scripts/run_pipeline.py --config configs/in_vitro.yaml --no-parallel

# Verbose output
python scripts/run_pipeline.py --config configs/in_vitro.yaml -v

# Run the GUI
python scripts/run_gui.py
```

### Installation

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Architecture

### Modular Pipeline Design

The codebase is organized into modular components in `root_tracker/`:

- **config.py** - Type-safe configuration using Python dataclasses with YAML loading
- **pipeline.py** - `RootTrackingPipeline` orchestrates all steps with step-by-step execution for GUI integration
- **models/** - `ImageData`, `ImageSeries`, `PipelineState`, `PlantStatistics`, `Root`
- **preprocessing/** - `ImageCropper`, `GreenAreaDetector`, `BackgroundRemover`
- **registration/** - `ImageRegistrator` (aligns time-series images)
- **tracking/** - `RootThresholder`, `RootSkeletonizer`, `CornerDetector`, `RootLinker`
- **analysis/** - `StatisticsCalculator`
- **io/** - `ImageLoader`, `ResultExporter`, `BarcodeReader`, `series_cache`, `mask_io`
- **gui/** - PySide6 GUI with 4-step workflow

### Pipeline Flow

1. **Load** - Images discovered and grouped by barcode/filename pattern
2. **Preprocess** - Crop, rotate, detect green areas (stems), remove background gradient
3. **Register** - Align time-series images using template matching
4. **Track** - Threshold, skeletonize, find corners, link segments to plants
5. **Export** - Save annotated images and CSV statistics

Each step can be run independently via pipeline methods (`preprocess_series()`, `register_series()`, `track_and_analyze_series()`), designed for GUI workflow where users can review/edit between steps.

### Configuration System

Configuration uses nested Python dataclasses for type safety:
- `Config` - Main config with margins, rotation, n_clusters
- `DataConfig` - Input/output paths, filename templates
- `GreenConfig` - HSV ranges for plant stem detection
- `ThresholdConfig` - Root segmentation parameters
- `CropConfig` - Blue background detection for auto-cropping

Load from YAML: `config = Config.from_yaml("configs/in_vitro.yaml")`

The config provides hash methods (`preprocess_config_hash()`, `tracking_config_hash()`) used by the cache system to invalidate results when parameters change.

### Caching System

Processed images are cached to disk in `{input_dir}/.root_tracker_cache/` as compressed `.npz` files. This:
- Allows freeing large numpy arrays from memory during GUI operation
- Preserves processed results across sessions
- Invalidates automatically when config parameters change (via hash comparison)

Key functions in `io/series_cache.py`:
- `save_series()` - Save all arrays and metadata
- `load_series_state()` - Restore metadata only (fast, for UI state)
- `load_series()` - Restore full arrays when needed
- `has_valid_cache()` - Check if cache matches current config

### Image Data Models

- `ImageData` - Single image with path, date, barcode, and all processed arrays (image, process, canny, diff) and results (positions, lengths, areas)
- `ImageSeries` - Collection of images sorted by date, with pipeline state and user/working masks
- `PipelineState` - Tracks which steps completed and config hashes

Note: `ImageData.image_annotated` should never be mutated - always copy before drawing.

### Barcode Detection

Barcodes are read from original images (independent of preprocessing rotation) using `BarcodeReader.read_fast()`. Results stored as `barcode_read`, `barcode_rect`, with flags `barcode_mismatch` and `barcode_not_found` for UI warnings.

### Parallel Processing

The pipeline supports multiprocessing via `process_map()` from tqdm. Worker functions `preprocess_and_cache_worker()` and `track_and_cache_worker()` are pickle-friendly standalone functions that create new pipeline instances per worker.

## Configuration Files

- `configs/in_vitro.yaml` - Default configuration for in vitro experiments
- Modify `data.input`, `data.output`, `n_clusters`, `rotation`, thresholds for different setups
