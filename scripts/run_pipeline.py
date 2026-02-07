#!/usr/bin/env python3
"""
Run the Root Tracking pipeline from command line.

Usage:
    python run_pipeline.py --config configs/in_vitro.yaml
    
Remember to activate your virtual environment before running!
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from root_tracker import Config, RootTrackingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run the Root Tracking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    python run_pipeline.py --config configs/in_vitro.yaml
    
    # Run without parallelization (for debugging)
    python run_pipeline.py --config configs/in_vitro.yaml --no-parallel
    
    # Override number of workers
    python run_pipeline.py --config configs/in_vitro.yaml --workers 4
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/in_vitro.yaml",
        help="Path to the configuration file (default: configs/in_vitro.yaml)"
    )
    
    parser.add_argument(
        "--no-parallel", 
        action="store_true",
        help="Disable parallel processing (useful for debugging)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.verbose:
        print(f"Loading configuration from: {args.config}")
    
    try:
        config = Config.from_yaml(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Input directory: {config.data.input}")
        print(f"Output directory: {config.data.output}")
        print(f"Statistics file: {config.data.statistics}")
        print(f"Number of plants: {config.n_clusters}")
    
    # Create and run pipeline
    pipeline = RootTrackingPipeline(config)
    
    parallel = not args.no_parallel
    
    if args.verbose:
        mode = "parallel" if parallel else "sequential"
        print(f"Running pipeline in {mode} mode...")
    
    try:
        output_path = pipeline.run(parallel=parallel, max_workers=args.workers)
        print(f"Pipeline completed successfully!")
        print(f"Statistics saved to: {output_path}")
    except Exception as e:
        print(f"Error running pipeline: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
