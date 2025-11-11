#!/usr/bin/env python3
"""
OPERA Displacement-Coherence Averaging Script

This script processes OPERA GeoTIFF files containing displacement and coherence data.
It performs multi-level averaging (within frames and across frames), handles spatial
overlaps intelligently, and clips results to county boundaries.

Features:
- Frame-based grouping and averaging of multi-temporal GeoTIFF files
- Memory-efficient windowed processing for large files
- Intelligent overlap handling when merging frames
- County-based clipping using shapefiles
- Saves displacement and coherence as separate single-band GeoTIFF files

Author: Generated with Claude Code
Date: 2025
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as rasterio_mask
import geopandas as gpd
from pathlib import Path
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DisplacementCoherenceAverager:
    """
    Processes OPERA displacement and coherence GeoTIFF files by:
    1. Grouping files by frame
    2. Averaging within each frame
    3. Merging frame averages with overlap handling
    4. Clipping to county boundaries
    5. Saving as separate displacement and coherence files
    """

    def __init__(self, geotiff_root, shapefile_root, output_root, chunk_size=1024):
        """
        Parameters:
        - geotiff_root: Root directory containing county GeoTIFF subfolders (REQUIRED)
        - shapefile_root: Root directory containing county shapefile subfolders (REQUIRED)
        - output_root: Output folder for processed files
        - chunk_size: Window size for chunked processing (default: 1024)
        """
        self.geotiff_root = Path(geotiff_root)
        self.shapefile_root = Path(shapefile_root)
        self.output_root = Path(output_root)
        self.chunk_size = chunk_size

        # Early validation: ensure both input directories exist
        if not self.geotiff_root.exists():
            raise ValueError(f"GeoTIFF root directory not found: {self.geotiff_root}")

        if not self.shapefile_root.exists():
            raise ValueError(f"Shapefile root directory not found: {self.shapefile_root}")

        # Create output directory
        self.output_root.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"OPERA Displacement-Coherence Averaging Processor")
        print(f"{'='*80}")
        print(f"Input Configuration:")
        print(f"  GeoTIFF root: {self.geotiff_root}")
        print(f"  Shapefile root: {self.shapefile_root}")
        print(f"  Output root: {self.output_root}")
        print(f"  Chunk size: {self.chunk_size}x{self.chunk_size}")
        print(f"{'='*80}\n")

    def parse_county_folders(self):
        """
        Discover county folders from GeoTIFF root directory.
        Returns list of county Path objects.
        """
        counties = []

        print(f"Discovering counties from: {self.geotiff_root}")

        # Scan GeoTIFF root for county folders
        for item in self.geotiff_root.iterdir():
            if item.is_dir():
                # Check if this county has the expected subfolder structure
                vertical_folder = item / "Vertical-Mask-Reproject"
                if vertical_folder.exists() and vertical_folder.is_dir():
                    counties.append(item)
                    print(f"  ✓ Found county: {item.name}")
                else:
                    print(f"  ✗ Skipping {item.name}: No Vertical-Mask-Reproject folder")

        if not counties:
            print("WARNING: No valid county folders found")
        else:
            print(f"\nTotal counties discovered: {len(counties)}")

        return counties

    def parse_frame_id(self, filename):
        """Extract frame ID from filename using regex"""
        match = re.search(r'F(\d{5})', str(filename))
        if match:
            return f"F{match.group(1)}"
        return None

    def group_files_by_frame(self, county_path):
        """Group GeoTIFF files by frame ID"""
        input_dir = county_path / "Vertical-Mask-Reproject"
        frames = defaultdict(list)

        for tif_file in input_dir.glob("*.tif"):
            frame_id = self.parse_frame_id(tif_file.name)
            if frame_id:
                frames[frame_id].append(tif_file)
            else:
                print(f"    Warning: Could not parse frame ID from {tif_file.name}")

        return frames

    def get_windows(self, height, width):
        """Generate windows for chunked processing"""
        for row in range(0, height, self.chunk_size):
            for col in range(0, width, self.chunk_size):
                window_height = min(self.chunk_size, height - row)
                window_width = min(self.chunk_size, width - col)
                yield Window(col, row, window_width, window_height)

    def validate_frame_files(self, file_list):
        """Ensure all files in frame have consistent metadata"""
        reference = None

        print(f"      Validating {len(file_list)} files...")
        for idx, file_path in enumerate(file_list, 1):
            try:
                print(f"         [{idx}/{len(file_list)}] Checking: {file_path.name}")
                with rasterio.open(file_path) as src:
                    meta = {
                        'height': src.height,
                        'width': src.width,
                        'count': src.count,
                        'crs': src.crs,
                        'dtype': src.dtypes[0],  # Get dtype of first band
                        'transform': src.transform,
                        'nodata': src.nodata
                    }

                    print(f"                  Size: {meta['width']}x{meta['height']}, Bands: {meta['count']}, Dtype: {meta['dtype']}")

                    if reference is None:
                        reference = meta
                        print(f"                  ✓ Set as reference")
                    else:
                        # Check consistency
                        if meta['height'] != reference['height'] or \
                           meta['width'] != reference['width']:
                            print(f"      ERROR: Dimension mismatch in {file_path.name}")
                            print(f"             Expected: {reference['width']}x{reference['height']}")
                            print(f"             Got: {meta['width']}x{meta['height']}")
                            return None

                        if meta['count'] != 2:
                            print(f"      ERROR: Expected 2 bands, found {meta['count']} in {file_path.name}")
                            return None

                        print(f"                  ✓ Consistent with reference")

            except Exception as e:
                print(f"      ERROR: Cannot read {file_path.name}: {e}")
                print(f"             Full path: {file_path}")
                import traceback
                traceback.print_exc()
                return None

        print(f"      ✓ All files validated successfully")
        return reference

    def average_frame(self, frame_id, file_list, output_path):
        """
        Average all GeoTIFF files for a single frame using windowed processing.

        Returns: True if successful, False otherwise
        """
        # Special case: single file (no averaging needed)
        if len(file_list) == 1:
            print(f"      Single file in frame, copying to output")
            shutil.copy(file_list[0], output_path)
            return True

        print(f"      Averaging {len(file_list)} files...")

        # Validate consistency
        reference_meta = self.validate_frame_files(file_list)
        if reference_meta is None:
            return False

        try:
            # Get dimensions and create output
            with rasterio.open(file_list[0]) as src:
                profile = src.profile.copy()
                height, width = src.height, src.width
                nodata = src.nodata if src.nodata is not None else -9999
                profile.update(nodata=nodata, compress='lzw')

            # Calculate total windows
            total_windows = 0
            for _ in self.get_windows(height, width):
                total_windows += 1

            print(f"      Processing {height}x{width} pixels in {total_windows} windows...")

            # Create output file
            with rasterio.open(output_path, 'w', **profile) as dst:
                # Generate windows with progress
                window_count = 0
                last_progress = 0

                for window in self.get_windows(height, width):
                    window_count += 1

                    # Print progress every 10% or every 5 windows (whichever is more frequent)
                    current_progress = int((window_count / total_windows) * 100)
                    if current_progress >= last_progress + 10 or window_count % 5 == 0 or window_count == total_windows:
                        print(f"         Progress: {window_count}/{total_windows} windows ({current_progress}%)")
                        last_progress = current_progress

                    # Process each band
                    for band_idx in [1, 2]:
                        # Accumulate data from all files
                        band_data = []
                        for file_path in file_list:
                            with rasterio.open(file_path) as src:
                                data = src.read(band_idx, window=window)
                                band_data.append(data)

                        # Stack and average
                        stacked = np.stack(band_data, axis=0)  # Shape: (n_files, rows, cols)

                        # Replace nodata with NaN for averaging
                        stacked_float = stacked.astype(np.float64)
                        stacked_float[stacked == nodata] = np.nan

                        # Compute mean, ignoring NaN
                        averaged = np.nanmean(stacked_float, axis=0)

                        # Replace NaN back to nodata
                        averaged[np.isnan(averaged)] = nodata

                        # Write to output
                        dst.write(averaged.astype(profile['dtype']), band_idx, window=window)

                # Set band descriptions
                dst.set_band_description(1, 'Vertical Displacement (frame-averaged)')
                dst.set_band_description(2, 'Temporal Coherence (frame-averaged)')

            print(f"      ✓ Frame average completed: {output_path.name}")
            return True

        except Exception as e:
            print(f"      ERROR: Frame averaging failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def compute_union_bounds(self, bounds_list):
        """Compute union of all bounds"""
        min_x = min(b[0] for b in bounds_list)
        min_y = min(b[1] for b in bounds_list)
        max_x = max(b[2] for b in bounds_list)
        max_y = max(b[3] for b in bounds_list)
        return (min_x, min_y, max_x, max_y)

    def compute_target_resolution(self, frame_metadata):
        """Determine target resolution (use finest)"""
        resolutions = [abs(meta['transform'].a) for meta in frame_metadata]
        return min(resolutions)  # Finest resolution

    def bounds_intersect(self, bounds1, bounds2):
        """Check if two bounds rectangles intersect"""
        return not (bounds1[2] < bounds2[0] or bounds1[0] > bounds2[2] or
                    bounds1[3] < bounds2[1] or bounds1[1] > bounds2[3])

    def read_window_from_frame(self, frame_path, band_idx, target_bounds,
                               target_height, target_width, target_transform):
        """
        Read data from a frame that corresponds to target window bounds.
        Handles coordinate transformation if needed.
        """
        try:
            with rasterio.open(frame_path) as src:
                # Calculate window in source coordinates
                from rasterio.windows import from_bounds as window_from_bounds
                src_window = window_from_bounds(*target_bounds, transform=src.transform)

                # Read data with resampling
                data = src.read(band_idx, window=src_window,
                              out_shape=(target_height, target_width),
                              resampling=Resampling.bilinear)

                return data
        except Exception as e:
            # Silent failure - window may be outside frame bounds
            return None

    def merge_frame_averages(self, frame_avg_files, output_path):
        """
        Merge frame averages with intelligent overlap handling.

        Returns: True if successful, False otherwise
        """
        if len(frame_avg_files) == 1:
            # Single frame, no merging needed
            print(f"      Single frame, copying to output")
            shutil.copy(frame_avg_files[0], output_path)
            return True

        print(f"      Merging {len(frame_avg_files)} frame averages...")

        try:
            # Step 1: Analyze all frames
            frame_metadata = []
            for file_path in frame_avg_files:
                with rasterio.open(file_path) as src:
                    frame_metadata.append({
                        'path': file_path,
                        'bounds': src.bounds,
                        'crs': src.crs,
                        'transform': src.transform,
                        'shape': (src.height, src.width),
                        'nodata': src.nodata if src.nodata is not None else -9999
                    })

            # Step 2: Determine target CRS (use first)
            target_crs = frame_metadata[0]['crs']

            # Step 3: Determine union bounds
            all_bounds = [meta['bounds'] for meta in frame_metadata]
            union_bounds = self.compute_union_bounds(all_bounds)
            print(f"      Union bounds: {union_bounds}")

            # Step 4: Determine target resolution
            target_resolution = self.compute_target_resolution(frame_metadata)
            print(f"      Target resolution: {target_resolution}")

            # Step 5: Calculate output dimensions
            target_width = int((union_bounds[2] - union_bounds[0]) / target_resolution)
            target_height = int((union_bounds[3] - union_bounds[1]) / target_resolution)
            print(f"      Output dimensions: {target_width} x {target_height}")

            # Step 6: Create target transform
            target_transform = from_bounds(
                union_bounds[0], union_bounds[1],
                union_bounds[2], union_bounds[3],
                target_width, target_height
            )

            # Step 7: Create output profile
            profile = {
                'driver': 'GTiff',
                'dtype': 'float32',
                'count': 2,
                'width': target_width,
                'height': target_height,
                'crs': target_crs,
                'transform': target_transform,
                'nodata': -9999,
                'compress': 'lzw'
            }

            # Step 8: Process windows
            # Calculate total windows first
            total_windows = 0
            for _ in self.get_windows(target_height, target_width):
                total_windows += 1

            print(f"      Processing {target_height}x{target_width} pixels in {total_windows} windows...")
            print(f"      Merging {len(frame_avg_files)} frames with overlap handling...")

            with rasterio.open(output_path, 'w', **profile) as dst:
                window_count = 0
                last_progress = 0

                for window in self.get_windows(target_height, target_width):
                    window_count += 1

                    # Print progress every 10% or every 5 windows
                    current_progress = int((window_count / total_windows) * 100)
                    if current_progress >= last_progress + 10 or window_count % 5 == 0 or window_count == total_windows:
                        print(f"         Merging progress: {window_count}/{total_windows} windows ({current_progress}%)")
                        last_progress = current_progress

                    # Get window bounds in target CRS
                    from rasterio.windows import bounds as window_bounds
                    w_bounds = window_bounds(window, target_transform)

                    # Process each band
                    for band_idx in [1, 2]:
                        # Initialize accumulators
                        sum_array = np.zeros((window.height, window.width), dtype=np.float64)
                        count_array = np.zeros((window.height, window.width), dtype=np.uint16)

                        # Accumulate from each frame
                        for meta in frame_metadata:
                            # Check if frame intersects this window
                            if not self.bounds_intersect(w_bounds, meta['bounds']):
                                continue

                            # Read corresponding pixels from this frame
                            frame_data = self.read_window_from_frame(
                                meta['path'], band_idx, w_bounds,
                                window.height, window.width, target_transform
                            )

                            if frame_data is not None:
                                # Accumulate valid pixels
                                valid_mask = (frame_data != meta['nodata']) & np.isfinite(frame_data)
                                sum_array[valid_mask] += frame_data[valid_mask]
                                count_array[valid_mask] += 1

                        # Compute average
                        averaged = np.full((window.height, window.width), -9999, dtype=np.float32)
                        valid_pixels = count_array > 0
                        averaged[valid_pixels] = (sum_array[valid_pixels] / count_array[valid_pixels]).astype(np.float32)

                        # Write to output
                        dst.write(averaged, band_idx, window=window)

                        # Cleanup
                        del sum_array, count_array, averaged

                # Set band descriptions
                dst.set_band_description(1, 'Vertical Displacement (merged)')
                dst.set_band_description(2, 'Temporal Coherence (merged)')

            print(f"      ✓ Merged {window_count} windows from {len(frame_avg_files)} frames")
            return True

        except Exception as e:
            print(f"      ERROR: Frame merging failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def clip_and_save_separate(self, merged_file, county_name, output_dir):
        """
        Clip merged GeoTIFF to county boundary and save displacement and coherence as separate files.

        Parameters:
        - merged_file: Path to merged GeoTIFF (2 bands)
        - county_name: Name of the county
        - output_dir: Output directory for clipped files

        Returns: True if successful, False otherwise
        """
        # Construct shapefile path from shapefile root directory
        shapefile_path = self.shapefile_root / county_name / f"{county_name}.shp"

        # Define output paths for separate files
        displacement_output = output_dir / "displacement-average.tif"
        coherence_output = output_dir / "coherence-average.tif"

        if not shapefile_path.exists():
            print(f"      WARNING: Shapefile not found at {shapefile_path}")
            print(f"               Skipping clipping step, using merged file as-is")
            # Still split into two files even without clipping
            try:
                with rasterio.open(merged_file) as src:
                    # Read both bands
                    displacement_data = src.read(1)
                    coherence_data = src.read(2)

                    # Prepare profile for single-band output
                    profile = src.profile.copy()
                    profile.update(count=1)

                    # Save displacement
                    with rasterio.open(displacement_output, 'w', **profile) as dst:
                        dst.write(displacement_data, 1)
                        dst.set_band_description(1, 'Vertical Displacement (averaged)')

                    # Save coherence
                    with rasterio.open(coherence_output, 'w', **profile) as dst:
                        dst.write(coherence_data, 1)
                        dst.set_band_description(1, 'Temporal Coherence (averaged)')

                print(f"      ✓ Saved as separate files (without clipping):")
                print(f"         - {displacement_output.name}")
                print(f"         - {coherence_output.name}")
                return True
            except Exception as e:
                print(f"      ERROR: Failed to split bands: {e}")
                return False

        try:
            # Load shapefile
            gdf = gpd.read_file(shapefile_path)
            print(f"      Loaded shapefile: {len(gdf)} features")

            # Open raster
            with rasterio.open(merged_file) as src:
                # Reproject shapefile to raster CRS if needed
                if gdf.crs != src.crs:
                    print(f"      Reprojecting shapefile from {gdf.crs} to {src.crs}")
                    gdf = gdf.to_crs(src.crs)

                # Check if shapefile intersects raster
                raster_bounds = src.bounds
                shapefile_bounds = gdf.total_bounds

                if not self.bounds_intersect(raster_bounds, shapefile_bounds):
                    print(f"      WARNING: County shapefile does not intersect with raster")
                    print(f"               Raster bounds: {raster_bounds}")
                    print(f"               Shapefile bounds: {shapefile_bounds}")
                    return False

                # Perform clipping (clips all bands at once)
                clipped_data, clipped_transform = rasterio_mask(
                    src,
                    gdf.geometry,
                    crop=True,
                    filled=True,
                    nodata=src.nodata
                )

                # Extract individual bands from clipped data
                # clipped_data shape: (bands, height, width)
                displacement_clipped = clipped_data[0, :, :]
                coherence_clipped = clipped_data[1, :, :]

                # Update profile for single-band output
                clipped_profile = src.profile.copy()
                clipped_profile.update({
                    'count': 1,
                    'height': clipped_data.shape[1],
                    'width': clipped_data.shape[2],
                    'transform': clipped_transform
                })

                # Write displacement file
                with rasterio.open(displacement_output, 'w', **clipped_profile) as dst:
                    dst.write(displacement_clipped, 1)
                    dst.set_band_description(1, 'Vertical Displacement (county-averaged)')

                # Write coherence file
                with rasterio.open(coherence_output, 'w', **clipped_profile) as dst:
                    dst.write(coherence_clipped, 1)
                    dst.set_band_description(1, 'Temporal Coherence (county-averaged)')

            print(f"      ✓ Successfully clipped and saved as separate files:")
            print(f"         - {displacement_output.name}")
            print(f"         - {coherence_output.name}")
            return True

        except Exception as e:
            print(f"      ERROR: Clipping failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_county(self, county_path):
        """
        Process a single county through complete workflow.

        Returns: Dictionary with processing statistics
        """
        county_name = county_path.name
        start_time = time.time()

        print(f"\n{'='*80}")
        print(f"Processing County: {county_name}")
        print(f"{'='*80}")

        result = {
            'county': county_name,
            'frames_processed': 0,
            'files_averaged': 0,
            'frame_avg_success': 0,
            'merge_success': False,
            'clip_success': False,
            'processing_time': 0,
            'errors': []
        }

        try:
            # Step 1: Group files by frame
            print(f"\nStep 1: Grouping files by frame...")
            frame_groups = self.group_files_by_frame(county_path)

            if not frame_groups:
                print(f"  No valid GeoTIFF files found")
                return result

            print(f"  Found {len(frame_groups)} frames:")
            for frame_id, files in frame_groups.items():
                print(f"    {frame_id}: {len(files)} files")
                for f in files:
                    print(f"       - {f.name}")
                result['files_averaged'] += len(files)

            # Step 2: Average each frame
            print(f"\nStep 2: Averaging frames...")
            frame_avg_dir = self.output_root / "Frame-Averages"
            frame_avg_dir.mkdir(parents=True, exist_ok=True)

            frame_avg_files = []
            for frame_id, files in tqdm(frame_groups.items(), desc=f"  Averaging {county_name} frames"):
                output_path = frame_avg_dir / f"frame_{frame_id}_average.tif"

                if self.average_frame(frame_id, files, output_path):
                    frame_avg_files.append(output_path)
                    result['frame_avg_success'] += 1
                else:
                    result['errors'].append(f"Failed to average frame {frame_id}")

            result['frames_processed'] = len(frame_avg_files)

            if not frame_avg_files:
                print(f"  No frame averages created")
                return result

            # Step 3: Merge frame averages
            print(f"\nStep 3: Merging frame averages...")
            merged_dir = self.output_root / "Merged"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_path = merged_dir / f"{county_name}_merged_average.tif"

            if self.merge_frame_averages(frame_avg_files, merged_path):
                result['merge_success'] = True
            else:
                result['errors'].append("Frame merging failed")
                return result

            # Step 4: Clip to county boundary and save as separate files
            print(f"\nStep 4: Clipping to county boundary and saving separate files...")
            # Output directory is directly the output_root (e.g., G:\processed\Baldwin\Average)
            output_dir = self.output_root
            output_dir.mkdir(parents=True, exist_ok=True)

            if self.clip_and_save_separate(merged_path, county_name, output_dir):
                result['clip_success'] = True
            else:
                result['errors'].append("Clipping/saving failed")

            # Success!
            result['processing_time'] = time.time() - start_time

            print(f"\n{'─'*80}")
            print(f"County {county_name} processed in {result['processing_time']/60:.1f} minutes")
            print(f"  Frames averaged: {result['frame_avg_success']}/{len(frame_groups)}")
            print(f"  Merge: {'✓' if result['merge_success'] else '✗'}")
            print(f"  Separate files saved: {'✓' if result['clip_success'] else '✗'}")
            print(f"{'─'*80}")

        except Exception as e:
            print(f"ERROR: Unexpected error processing {county_name}: {e}")
            import traceback
            traceback.print_exc()
            result['errors'].append(str(e))

        return result


def main():
    """Main function - Supports both command-line arguments and direct configuration"""
    import sys
    import argparse

    # Check if called with command-line arguments
    if len(sys.argv) > 1:
        # Command-line mode (called from automated_comprehensive_processor.py)
        parser = argparse.ArgumentParser(
            description='OPERA Displacement-Coherence Averaging Script'
        )
        parser.add_argument('input_dir', help='Input directory containing GeoTIFF files (e.g., G:\\processed\\Baldwin\\Vertical-Mask-Reproject)')
        parser.add_argument('shapefile_dir', help='Directory containing county shapefile (e.g., E:\\UH\\...\\counties\\Baldwin)')
        parser.add_argument('output_dir', help='Output directory for averaged files (e.g., G:\\processed\\Baldwin\\Average)')
        parser.add_argument('--chunk-size', type=int, default=1024, help='Window size for processing (default: 1024)')

        args = parser.parse_args()

        INPUT_DIR = Path(args.input_dir)
        SHAPEFILE_DIR = Path(args.shapefile_dir)
        OUTPUT_DIR = Path(args.output_dir)
        CHUNK_SIZE = args.chunk_size

        # Extract county name from input directory or shapefile directory
        # The county name is the parent folder of Vertical-Mask-Reproject
        if INPUT_DIR.name == "Vertical-Mask-Reproject":
            COUNTY_NAME = INPUT_DIR.parent.name
        else:
            # Try to get from shapefile directory
            COUNTY_NAME = SHAPEFILE_DIR.name

        print(f"Command-line mode activated")
        print(f"Configuration:")
        print(f"  Input directory: {INPUT_DIR}")
        print(f"  Shapefile directory: {SHAPEFILE_DIR}")
        print(f"  Output directory: {OUTPUT_DIR}")
        print(f"  County name: {COUNTY_NAME}")
        print(f"  Chunk size: {CHUNK_SIZE}x{CHUNK_SIZE}")
        print()

        try:
            # Create a temporary structure for processing
            # The processor expects: geotiff_root/county_name/Vertical-Mask-Reproject
            # We have: INPUT_DIR which IS Vertical-Mask-Reproject
            # So geotiff_root should be the parent of the parent
            GEOTIFF_ROOT = INPUT_DIR.parent
            SHAPEFILE_ROOT = SHAPEFILE_DIR.parent

            # Create processor
            processor = DisplacementCoherenceAverager(
                geotiff_root=GEOTIFF_ROOT,
                shapefile_root=SHAPEFILE_ROOT,
                output_root=OUTPUT_DIR,
                chunk_size=CHUNK_SIZE
            )

            # Process the county
            county_path = INPUT_DIR.parent
            result = processor.process_county(county_path)

            if result['merge_success'] and result['clip_success']:
                print("\n✓ Processing completed successfully!")
                print(f"\nOutput files saved in: {OUTPUT_DIR}")
                print(f"  - displacement-average.tif")
                print(f"  - coherence-average.tif")
                sys.exit(0)
            else:
                print("\n✗ Processing failed")
                if result['errors']:
                    print(f"Errors: {', '.join(result['errors'])}")
                sys.exit(1)

        except Exception as e:
            print(f"\n✗ FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        # Direct configuration mode (backward compatibility)
        # ========================================================================
        # CONFIGURATION: Set your paths here
        # ========================================================================

        # Input paths
        GEOTIFF_ROOT = r"G:\processed"
        SHAPEFILE_ROOT = r"E:\UH\graduate class\Applied Gepspatial Computations\data\CoastalCounties\counties"

        # Output path - directly set to Baldwin\Average as requested
        OUTPUT_ROOT = r"G:\processed\Baldwin\Average"

        # Processing parameters
        COUNTY_TO_PROCESS = "Baldwin"  # Set to specific county name, or None to process all
        CHUNK_SIZE = 1024  # Window size for processing

        # ========================================================================

        print(f"Direct configuration mode")
        print(f"Configuration:")
        print(f"  GeoTIFF root: {GEOTIFF_ROOT}")
        print(f"  Shapefile root: {SHAPEFILE_ROOT}")
        print(f"  Output root: {OUTPUT_ROOT}")
        print(f"  County: {COUNTY_TO_PROCESS if COUNTY_TO_PROCESS else 'All counties'}")
        print(f"  Chunk size: {CHUNK_SIZE}x{CHUNK_SIZE}")
        print()

        try:
            # Create processor
            processor = DisplacementCoherenceAverager(
                geotiff_root=GEOTIFF_ROOT,
                shapefile_root=SHAPEFILE_ROOT,
                output_root=OUTPUT_ROOT,
                chunk_size=CHUNK_SIZE
            )

            # Process specific county
            if COUNTY_TO_PROCESS:
                county_path = processor.geotiff_root / COUNTY_TO_PROCESS
                if not county_path.exists():
                    print(f"ERROR: County folder not found: {county_path}")
                    return

                result = processor.process_county(county_path)

                if result['merge_success'] and result['clip_success']:
                    print("\n✓ Processing completed successfully!")
                    print(f"\nOutput files saved in: {OUTPUT_ROOT}")
                    print(f"  - displacement-average.tif")
                    print(f"  - coherence-average.tif")
                else:
                    print("\n✗ Processing failed")
                    if result['errors']:
                        print(f"Errors: {', '.join(result['errors'])}")
            else:
                print("ERROR: COUNTY_TO_PROCESS not set. Please specify a county name in main()")

        except Exception as e:
            print(f"\n✗ FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
