import os
import re
import random
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import rasterio
from rasterio import mask, transform, merge
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import fiona


class InSARSpatioTemporalProcessor:
    """
    Process InSAR vertical displacement data with spatio-temporal analysis.

    Handles frame mosaicking, intersection analysis, buffer zone creation,
    and time series extraction from OPERA vertical displacement data.
    """

    def __init__(self, input_dir, shapefile_path, output_dir):
        """
        Initialize the processor.

        Args:
            input_dir (str): Directory containing OPERA multi-band TIF files (displacement + coherence)
            shapefile_path (str): Path to shapefile with point data
            output_dir (str): Directory for output files
        """
        self.input_dir = Path(input_dir)
        self.shapefile_path = Path(shapefile_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Store multi-band data by frame
        self.frames_data = defaultdict(list)  # multi-band data (displacement + coherence)
        self.mosaic_raster = None
        self.mosaic_transform = None
        self.mosaic_crs = None
        self.overlap_polygons = []

        # Track failed files for logging
        self.failed_files = []

        # Track towers with retry attempts
        self.retry_log = []
        self.failed_towers = []

    def validate_file(self, filepath):
        """
        Validate that a TIFF file can be opened and read.

        Args:
            filepath (Path): Path to the file to validate

        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            with rasterio.open(filepath) as src:
                # Try to read basic metadata
                _ = src.count
                _ = src.width
                _ = src.height
                _ = src.crs
            return True
        except Exception as e:
            self.failed_files.append({
                'filepath': str(filepath),
                'error': str(e),
                'stage': 'validation'
            })
            print(f"Validation failed for {filepath.name}: {e}")
            return False

    def parse_filename(self, filename):
        """
        Parse OPERA filename to extract metadata.

        Expected formats for multi-band files based on input directory:
        - *No_Mask: OPERA_VERTICAL_COHERENCE_UNMASKED_REPROJECTED_F{frame}_{start_date}_{end_date}.tif
        - *Recommended_Mask: OPERA_VERTICAL_COHERENCE_MASKED_REPROJECTED_F{frame}_{start_date}_{end_date}.tif
        - *Coherence_0.5: OPERA_VERTICAL_COHERENCE_COH05_REPROJECTED_F{frame}_{start_date}_{end_date}.tif
        - *Coherence_0.2: OPERA_VERTICAL_COHERENCE_COH02_REPROJECTED_F{frame}_{start_date}_{end_date}.tif

        Args:
            filename (str): The filename to parse

        Returns:
            dict: Parsed metadata including frame, start_date, end_date
        """
        # Define patterns for different mask types
        patterns = [
            r'OPERA_VERTICAL_COHERENCE_UNMASKED_REPROJECTED_F(\d+)_(\d{8})_(\d{8})\.tif',
            r'OPERA_VERTICAL_COHERENCE_MASKED_REPROJECTED_F(\d+)_(\d{8})_(\d{8})\.tif',
            r'OPERA_VERTICAL_COHERENCE_COH05_REPROJECTED_F(\d+)_(\d{8})_(\d{8})\.tif',
            r'OPERA_VERTICAL_COHERENCE_COH02_REPROJECTED_F(\d+)_(\d{8})_(\d{8})\.tif'
        ]

        # Try each pattern
        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                frame = match.group(1)
                start_date = datetime.strptime(match.group(2), '%Y%m%d')
                end_date = datetime.strptime(match.group(3), '%Y%m%d')

                return {
                    'frame': frame,
                    'start_date': start_date,
                    'end_date': end_date,
                    'filename': filename,
                    'filepath': self.input_dir / filename,
                    'file_type': 'multi_band'
                }

        return None

    def group_files_by_frame(self):
        """
        Group all multi-band TIF files by frame number.
        Automatically detects the naming pattern based on input directory.
        """
        print("Grouping multi-band OPERA files by frame...")

        # Determine the naming pattern based on input directory
        input_dir_name = str(self.input_dir).lower()

        if "no_mask" in input_dir_name:
            pattern = "OPERA_VERTICAL_COHERENCE_UNMASKED_REPROJECTED_*.tif"
            print(f"Detected No_Mask directory - searching for UNMASKED files")
        elif "recommended_mask" in input_dir_name:
            pattern = "OPERA_VERTICAL_COHERENCE_MASKED_REPROJECTED_*.tif"
            print(f"Detected Recommended_Mask directory - searching for MASKED files")
        elif "coherence_0.5" in input_dir_name:
            pattern = "OPERA_VERTICAL_COHERENCE_COH05_REPROJECTED_*.tif"
            print(f"Detected Coherence_0.5 directory - searching for COH05 files")
        elif "coherence_0.2" in input_dir_name:
            pattern = "OPERA_VERTICAL_COHERENCE_COH02_REPROJECTED_*.tif"
            print(f"Detected Coherence_0.2 directory - searching for COH02 files")
        else:
            # Fallback: try all patterns
            print(f"Unknown directory type - searching all OPERA VERTICAL COHERENCE files")
            pattern = "OPERA_VERTICAL_COHERENCE_*.tif"

        # Search for OPERA TIF files with the determined pattern
        valid_count = 0
        invalid_count = 0
        for file in self.input_dir.glob(pattern):
            # Validate file before adding to collection
            if self.validate_file(file):
                metadata = self.parse_filename(file.name)
                if metadata:
                    self.frames_data[metadata['frame']].append(metadata)
                    valid_count += 1
            else:
                invalid_count += 1

        print(f"Found {len(self.frames_data)} frames ({valid_count} valid files, {invalid_count} invalid files):")
        for frame, files in self.frames_data.items():
            print(f"  Frame {frame}: {len(files)} multi-band images")

    def select_random_images_per_frame(self):
        """
        Randomly select one image per frame for mosaicking.

        Returns:
            list: Selected image metadata for mosaicking
        """
        selected_images = []

        for frame, images in self.frames_data.items():
            selected = random.choice(images)
            selected_images.append(selected)
            print(f"Frame {frame}: Selected {selected['filename']}")

        return selected_images

    def create_mosaic(self, selected_images):
        """
        Create a mosaic from selected images and determine overlapping areas.

        Args:
            selected_images (list): List of selected image metadata
        """
        print("Creating mosaic...")

        # Open all rasters
        src_files = []
        for img in selected_images:
            src = rasterio.open(img['filepath'])
            src_files.append(src)

        # Get nodata value from first file
        nodata_value = src_files[0].nodata
        print(f"Source nodata value: {nodata_value}")

        # Create mosaic with nodata value
        mosaic, out_trans = merge.merge(src_files, nodata=nodata_value)

        # Get CRS from first file
        self.mosaic_crs = src_files[0].crs
        self.mosaic_transform = out_trans

        # Check valid pixels
        valid_pixels_band1 = np.sum(mosaic[0] != nodata_value) if nodata_value is not None else np.sum(~np.isnan(mosaic[0]))
        valid_pixels_band2 = np.sum(mosaic[1] != nodata_value) if nodata_value is not None else np.sum(~np.isnan(mosaic[1]))
        print(f"Mosaic band 1 valid pixels: {valid_pixels_band1}/{mosaic[0].size}")
        print(f"Mosaic band 2 valid pixels: {valid_pixels_band2}/{mosaic[1].size}")

        # Save mosaic
        mosaic_path = self.output_dir / "mosaic.tif"
        with rasterio.open(
            mosaic_path, 'w',
            driver='GTiff',
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=mosaic.shape[0],
            dtype=mosaic.dtype,
            crs=self.mosaic_crs,
            transform=out_trans,
            nodata=nodata_value,
            compress='lzw'
        ) as dst:
            dst.write(mosaic)
            # Set band descriptions
            dst.set_band_description(1, 'Vertical Displacement')
            dst.set_band_description(2, 'Coherence')

        self.mosaic_raster = mosaic
        print(f"Mosaic saved to {mosaic_path} with nodata={nodata_value}")

        # Determine overlapping areas
        self._find_overlapping_areas(src_files)

        # Close source files
        for src in src_files:
            src.close()

    def _find_overlapping_areas(self, src_files):
        """
        Find overlapping areas between frames.

        Args:
            src_files (list): List of opened raster sources
        """
        print("Finding overlapping areas...")

        # Convert raster bounds to polygons
        frame_polygons = []
        for src in src_files:
            bounds = src.bounds
            poly = Polygon([
                (bounds.left, bounds.bottom),
                (bounds.right, bounds.bottom),
                (bounds.right, bounds.top),
                (bounds.left, bounds.top)
            ])
            frame_polygons.append(poly)

        # Find intersections between all pairs
        overlaps = []
        for i in range(len(frame_polygons)):
            for j in range(i + 1, len(frame_polygons)):
                intersection = frame_polygons[i].intersection(frame_polygons[j])
                if not intersection.is_empty and intersection.area > 0:
                    overlaps.append(intersection)

        self.overlap_polygons = overlaps
        print(f"Found {len(overlaps)} overlapping areas")

    def find_shapefile_intersections(self):
        """
        Find intersection points between shapefile and mosaiced area.

        Returns:
            gpd.GeoDataFrame: Points that intersect with the mosaic
        """
        print("Finding shapefile intersections...")

        # Load shapefile
        gdf = gpd.read_file(self.shapefile_path)

        # Reproject to mosaic CRS if needed
        if gdf.crs != self.mosaic_crs:
            gdf = gdf.to_crs(self.mosaic_crs)

        # Create polygon from mosaic bounds
        height, width = self.mosaic_raster.shape[1], self.mosaic_raster.shape[2]
        bounds = rasterio.transform.array_bounds(height, width, self.mosaic_transform)
        mosaic_polygon = Polygon([
            (bounds[0], bounds[1]),  # left, bottom
            (bounds[2], bounds[1]),  # right, bottom
            (bounds[2], bounds[3]),  # right, top
            (bounds[0], bounds[3])   # left, top
        ])

        # Find intersecting points
        intersecting_points = gdf[gdf.geometry.intersects(mosaic_polygon)]

        print(f"Found {len(intersecting_points)} intersecting points")
        return intersecting_points

    def prepare_points_for_analysis(self, intersecting_points):
        """
        Prepare intersection points for time series analysis.

        Args:
            intersecting_points (gpd.GeoDataFrame): Points to analyze

        Returns:
            gpd.GeoDataFrame: Points ready for analysis
        """
        print("Preparing points for time series analysis...")

        # Ensure tower_id column exists
        if 'tower_id' not in intersecting_points.columns:
            intersecting_points['tower_id'] = range(1, len(intersecting_points) + 1)
            print("Added tower_id column")

        return intersecting_points

    def classify_points(self, points):
        """
        Classify points as inside or outside mosaic scope.

        Args:
            points (gpd.GeoDataFrame): Point features

        Returns:
            dict: Classification results
        """
        print("Classifying points...")

        # Create mosaic polygon
        height, width = self.mosaic_raster.shape[1], self.mosaic_raster.shape[2]
        bounds = rasterio.transform.array_bounds(height, width, self.mosaic_transform)
        mosaic_polygon = Polygon([
            (bounds[0], bounds[1]),
            (bounds[2], bounds[1]),
            (bounds[2], bounds[3]),
            (bounds[0], bounds[3])
        ])

        classification = {
            'inside': [],
            'outside': []
        }

        for idx, point in points.iterrows():
            point_geom = point.geometry

            if mosaic_polygon.contains(point_geom):
                classification['inside'].append(idx)
            else:
                classification['outside'].append(idx)

        print(f"Classification: {len(classification['inside'])} inside, "
              f"{len(classification['outside'])} outside")

        return classification

    def check_overlap_intersection(self, point_geom):
        """
        Check if point intersects with any overlapping areas.

        Args:
            point_geom: Point geometry

        Returns:
            bool: True if intersects with overlap areas
        """
        for overlap in self.overlap_polygons:
            if point_geom.intersects(overlap):
                return True
        return False

    def group_images_by_time_periods(self, frame):
        """
        Group multi-band images by time periods within a frame.

        Args:
            frame (str): Frame identifier

        Returns:
            dict: Time periods grouped by start date
        """
        frame_images = self.frames_data[frame]
        periods = defaultdict(list)

        for img in frame_images:
            start_key = img['start_date'].strftime('%Y%m%d')
            periods[start_key].append(img)

        # Sort by end date within each period
        for period in periods.values():
            period.sort(key=lambda x: x['end_date'])

        return periods

    def extract_time_series_single_frame(self, frame, point_geom, tower_id):
        """
        Extract cumulative time series for both displacement and coherence data
        from multi-band files for a point within a single frame.

        Displacement: Cumulative values from the first period's start date.
        Coherence: Direct values (no cumulative calculation).

        Args:
            frame (str): Frame identifier
            point_geom: Point geometry
            tower_id: Tower ID for reference

        Returns:
            dict: Time series data with metadata for both displacement and coherence
        """
        # Extract time series from multi-band images
        periods = self.group_images_by_time_periods(frame)
        displacement_time_series = []
        coherence_time_series = []
        displacement_valid_count = 0
        displacement_total_count = 0
        coherence_valid_count = 0
        coherence_total_count = 0

        # Get all periods sorted by start date
        sorted_periods = sorted(periods.items())
        period_baselines = {}  # Store final displacement of each period

        # Process each period
        for period_idx, (start_date_key, images) in enumerate(sorted_periods):
            images_sorted = sorted(images, key=lambda x: x['end_date'])

            for img in images_sorted:
                displacement_total_count += 1
                coherence_total_count += 1

                # Extract displacement values from band 1
                pixel_displacements = self._extract_displacement_from_point(
                    img['filepath'], point_geom, band_number=1
                )

                # Extract coherence values from band 2
                pixel_coherences = self._extract_coherence_from_point(
                    img['filepath'], point_geom, band_number=2
                )

                # Process displacement data
                if pixel_displacements is not None:
                    displacement_valid_count += 1

                    # Store displacement data for all 9 pixels
                    pixel_data = {}
                    for pixel_num in range(1, 10):  # Pixels 1-9
                        pixel_displacement = pixel_displacements.get(pixel_num)

                        if pixel_displacement is not None:
                            # Calculate cumulative displacement from first period baseline
                            cumulative_displacement = 0

                            if period_idx == 0:
                                # First period: displacement is relative to first period start
                                cumulative_displacement = pixel_displacement
                            else:
                                # Subsequent periods: add previous period baselines + current displacement
                                cumulative_displacement = pixel_displacement

                                # Add baselines from all previous periods
                                for prev_idx in range(period_idx):
                                    prev_start_key = sorted_periods[prev_idx][0]
                                    if (prev_start_key in period_baselines and
                                        pixel_num in period_baselines[prev_start_key] and
                                        period_baselines[prev_start_key][pixel_num] is not None):
                                        cumulative_displacement += period_baselines[prev_start_key][pixel_num]

                            pixel_data[pixel_num] = cumulative_displacement
                        else:
                            pixel_data[pixel_num] = None

                    displacement_time_series.append({
                        'date': img['end_date'],
                        'pixel_displacements': pixel_data,
                        'frame': frame,
                    })

                # Process coherence data (no cumulative calculation)
                if pixel_coherences is not None:
                    coherence_valid_count += 1

                    coherence_time_series.append({
                        'date': img['end_date'],
                        'pixel_coherences': pixel_coherences,
                        'frame': frame,
                    })

            # Store the final displacement of this period as baseline for next period
            if images_sorted:
                final_img = images_sorted[-1]
                final_pixel_displacements = self._extract_displacement_from_point(
                    final_img['filepath'], point_geom, band_number=1
                )
                if final_pixel_displacements is not None:
                    period_baselines[start_date_key] = final_pixel_displacements

        # Sort time series data by date
        displacement_time_series.sort(key=lambda x: x['date'])
        coherence_time_series.sort(key=lambda x: x['date'])

        # Calculate integrity for both datasets
        displacement_integrity = displacement_valid_count / displacement_total_count if displacement_total_count > 0 else 0.0
        coherence_integrity = coherence_valid_count / coherence_total_count if coherence_total_count > 0 else 0.0

        # Get the first start time from all periods
        first_start_time = min([datetime.strptime(key, '%Y%m%d') for key, _ in sorted_periods]) if sorted_periods else None

        return {
            'tower_id': tower_id,
            'displacement_time_series': displacement_time_series,
            'coherence_time_series': coherence_time_series,
            'displacement_integrity': displacement_integrity,
            'coherence_integrity': coherence_integrity,
            'displacement_valid_count': displacement_valid_count,
            'displacement_total_count': displacement_total_count,
            'coherence_valid_count': coherence_valid_count,
            'coherence_total_count': coherence_total_count,
            'period_start_dates': [datetime.strptime(key, '%Y%m%d') for key, _ in sorted_periods],
            'first_start_time': first_start_time
        }

    def _extract_displacement_from_point(self, image_path, point_geom, band_number=1, max_retries=5):
        """
        Extract displacement values from 9-pixel neighborhood centered on point location.
        Pixels are numbered 1-9 from top to bottom, left to right:
        1 2 3
        4 5 6
        7 8 9

        Args:
            image_path: Path to the multi-band image
            point_geom: Point geometry
            band_number: Band number to read displacement from (default: 1)
            max_retries: Maximum number of retry attempts for file access (default: 5)

        Returns:
            dict: Dictionary with pixel numbers (1-9) as keys and displacement values as values,
                  or None if extraction fails
        """
        src = None
        for attempt in range(max_retries):
            try:
                # Add small delay before opening to allow any file locks to clear
                if attempt > 0:
                    time.sleep(0.5 * (2 ** attempt))  # Exponential backoff: 0.5s, 1s, 2s, 4s

                src = rasterio.open(image_path)

                # Verify band exists
                if band_number > src.count:
                    print(f"Warning: Band {band_number} does not exist in {image_path}. Available bands: {src.count}")
                    if src:
                        src.close()
                    return None

                # Convert to Point if Polygon (use centroid)
                if point_geom.geom_type != 'Point':
                    point_geom = point_geom.centroid

                # Get pixel coordinates from geographic coordinates
                center_row, center_col = src.index(point_geom.x, point_geom.y)

                # Define 3x3 neighborhood around center pixel
                # Check if 3x3 window is within raster bounds
                if (1 <= center_row < src.height - 1) and (1 <= center_col < src.width - 1):
                    # Read 3x3 window centered on the point from specified band
                    window = ((center_row - 1, center_row + 2), (center_col - 1, center_col + 2))
                    neighborhood = src.read(band_number, window=window)

                    # Extract 9 pixels and number them 1-9 (top to bottom, left to right)
                    pixel_values = {}
                    pixel_count = 1

                    for row_idx in range(3):
                        for col_idx in range(3):
                            value = neighborhood[row_idx, col_idx]

                            # Check if value is valid (not nodata)
                            if value != src.nodata and not np.isnan(value):
                                pixel_values[pixel_count] = float(value)
                            else:
                                pixel_values[pixel_count] = None

                            pixel_count += 1

                    # Explicitly close before returning
                    src.close()
                    return pixel_values
                else:
                    # Point is outside raster bounds or too close to edge
                    if attempt == 0:  # Only print once
                        print(f"Warning: Point ({point_geom.x}, {point_geom.y}) -> pixel ({center_row}, {center_col}) is outside valid bounds for {image_path.name}. Raster size: {src.height}x{src.width}")
                    if src:
                        src.close()
                    return None
            except rasterio.errors.RasterioIOError as e:
                # Ensure file is closed before retry
                if src:
                    try:
                        src.close()
                    except:
                        pass

                if attempt < max_retries - 1:
                    wait_time = 0.5 * (2 ** attempt)  # Exponential backoff
                    print(f"Warning: RasterioIOError on attempt {attempt + 1}/{max_retries} for {image_path.name}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    self.failed_files.append({
                        'filepath': str(image_path),
                        'error': str(e),
                        'stage': 'displacement_extraction',
                        'attempts': max_retries
                    })
                    print(f"ERROR: Cannot read displacement from {image_path.name} after {max_retries} attempts. Error: {str(e)}")
                    return None
            except Exception as e:
                # Ensure file is closed
                if src:
                    try:
                        src.close()
                    except:
                        pass
                print(f"Warning: Error extracting displacement from {image_path}: {e}. Skipping.")
                return None
            finally:
                # Always try to close the file handle
                if src and not src.closed:
                    try:
                        src.close()
                    except:
                        pass

        return None

    def _extract_coherence_from_point(self, image_path, point_geom, band_number=2, max_retries=5):
        """
        Extract coherence values from 9-pixel neighborhood centered on point location.
        Pixels are numbered 1-9 from top to bottom, left to right:
        1 2 3
        4 5 6
        7 8 9

        Args:
            image_path: Path to the multi-band image
            point_geom: Point geometry
            band_number: Band number to read coherence from (default: 2)
            max_retries: Maximum number of retry attempts for file access (default: 5)

        Returns:
            dict: Dictionary with pixel numbers (1-9) as keys and coherence values as values,
                  or None if extraction fails
        """
        src = None
        for attempt in range(max_retries):
            try:
                # Add small delay before opening to allow any file locks to clear
                if attempt > 0:
                    time.sleep(0.5 * (2 ** attempt))  # Exponential backoff: 0.5s, 1s, 2s, 4s

                src = rasterio.open(image_path)

                # Check if the requested band exists
                if band_number > src.count:
                    print(f"Warning: Band {band_number} does not exist in {image_path}. Available bands: {src.count}")
                    if src:
                        src.close()
                    return None

                # Convert to Point if Polygon (use centroid)
                if point_geom.geom_type != 'Point':
                    point_geom = point_geom.centroid

                # Get pixel coordinates from geographic coordinates
                center_row, center_col = src.index(point_geom.x, point_geom.y)

                # Define 3x3 neighborhood around center pixel
                # Check if 3x3 window is within raster bounds
                if (1 <= center_row < src.height - 1) and (1 <= center_col < src.width - 1):
                    # Read 3x3 window centered on the point from specified band
                    window = ((center_row - 1, center_row + 2), (center_col - 1, center_col + 2))
                    neighborhood = src.read(band_number, window=window)

                    # Extract 9 pixels and number them 1-9 (top to bottom, left to right)
                    pixel_values = {}
                    pixel_count = 1

                    for row_idx in range(3):
                        for col_idx in range(3):
                            value = neighborhood[row_idx, col_idx]

                            # Check if value is valid (not nodata)
                            if value != src.nodata and not np.isnan(value):
                                pixel_values[pixel_count] = float(value)
                            else:
                                pixel_values[pixel_count] = None

                            pixel_count += 1

                    # Explicitly close before returning
                    src.close()
                    return pixel_values
                else:
                    # Point is outside raster bounds or too close to edge
                    if attempt == 0:  # Only print once
                        print(f"Warning: Point ({point_geom.x}, {point_geom.y}) -> pixel ({center_row}, {center_col}) is outside valid bounds for {image_path.name}. Raster size: {src.height}x{src.width}")
                    if src:
                        src.close()
                    return None
            except rasterio.errors.RasterioIOError as e:
                # Ensure file is closed before retry
                if src:
                    try:
                        src.close()
                    except:
                        pass

                if attempt < max_retries - 1:
                    wait_time = 0.5 * (2 ** attempt)  # Exponential backoff
                    print(f"Warning: RasterioIOError on attempt {attempt + 1}/{max_retries} for {image_path.name}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    self.failed_files.append({
                        'filepath': str(image_path),
                        'error': str(e),
                        'stage': 'coherence_extraction',
                        'attempts': max_retries
                    })
                    print(f"ERROR: Cannot read coherence from {image_path.name} after {max_retries} attempts. Error: {str(e)}")
                    return None
            except Exception as e:
                # Ensure file is closed
                if src:
                    try:
                        src.close()
                    except:
                        pass
                print(f"Warning: Error extracting coherence from {image_path}: {e}. Skipping.")
                return None
            finally:
                # Always try to close the file handle
                if src and not src.closed:
                    try:
                        src.close()
                    except:
                        pass

        return None

    def process_points(self, points, classification):
        """
        Process all valid points and extract time series for both displacement and coherence.

        Args:
            points (gpd.GeoDataFrame): Point features
            classification (dict): Point classification results

        Returns:
            dict: Time series results for each point
        """
        results = {}

        # Process points that are inside the mosaic with progress bar
        inside_indices = classification['inside']

        with tqdm(total=len(inside_indices), desc="Processing towers (displacement+coherence)") as pbar:
            for idx in inside_indices:
                point_row = points.loc[idx]
                point_geom = point_row.geometry
                tower_id = point_row.get('tower_id', idx + 1)

                # Check if point intersects with overlap areas
                intersects_overlap = self.check_overlap_intersection(point_geom)

                if intersects_overlap:
                    # Multiple frames - randomly choose one
                    available_frames = list(self.frames_data.keys())
                    chosen_frame = random.choice(available_frames)
                else:
                    # Single frame - find which frame contains this point
                    chosen_frame = self._find_containing_frame(point_geom)

                if chosen_frame:
                    time_series_result = self.extract_time_series_single_frame(
                        chosen_frame, point_geom, tower_id
                    )
                    results[tower_id] = time_series_result

                    # Save individual tower CSV immediately after processing
                    self.save_individual_tower_csv(tower_id, time_series_result)

                # Update progress bar
                displacement_count = len(time_series_result.get('displacement_time_series', [])) if chosen_frame and tower_id in results else 0
                coherence_count = len(time_series_result.get('coherence_time_series', [])) if chosen_frame and tower_id in results else 0
                pbar.set_postfix({
                    'Tower': str(tower_id),
                    'Frame': chosen_frame if chosen_frame else 'N/A',
                    'Disp': displacement_count,
                    'Coh': coherence_count
                })
                pbar.update(1)

        print(f"Processed {len(results)} towers with valid time series data (displacement + coherence)")
        return results

    def _find_containing_frame(self, geometry):
        """
        Find which frame contains the given geometry using point-in-polygon testing.

        Args:
            geometry: Shapely geometry (Point)

        Returns:
            str: Frame identifier or None if point is not contained in any frame
        """
        # Check each frame to see if it contains the point
        for frame_id in self.frames_data.keys():
            # Get a representative image from this frame to determine bounds
            frame_images = self.frames_data[frame_id]
            if not frame_images:
                continue

            # Use the first image to get frame bounds
            sample_image = frame_images[0]

            try:
                with rasterio.open(sample_image['filepath']) as src:
                    bounds = src.bounds

                    # Create polygon from raster bounds
                    frame_polygon = Polygon([
                        (bounds.left, bounds.bottom),
                        (bounds.right, bounds.bottom),
                        (bounds.right, bounds.top),
                        (bounds.left, bounds.top)
                    ])

                    # Check if point is contained within this frame's polygon
                    if frame_polygon.contains(geometry):
                        return frame_id

            except Exception as e:
                print(f"Error checking frame {frame_id}: {e}")
                continue

        # If no frame contains the point, return None
        return None

    def save_individual_tower_csv(self, tower_id, tower_data):
        """
        Save individual tower time series results immediately after processing.

        Args:
            tower_id: Tower identifier
            tower_data: Time series data for the tower
        """
        # Create individual time series data for this tower
        tower_time_series = []

        # Create a complete date index from both displacement and coherence data
        all_dates = set()

        # Collect all dates from displacement data
        for ts_point in tower_data['displacement_time_series']:
            all_dates.add(ts_point['date'])

        # Collect all dates from coherence data
        for ts_point in tower_data['coherence_time_series']:
            all_dates.add(ts_point['date'])

        # Sort dates
        sorted_dates = sorted(all_dates)

        # Create dictionaries for quick lookup
        displacement_by_date = {ts['date']: ts for ts in tower_data['displacement_time_series']}
        coherence_by_date = {ts['date']: ts for ts in tower_data['coherence_time_series']}

        # Process each date
        for date in sorted_dates:
            # Create base row data
            row_data = {
                'tower_id': f"tower_id_{tower_id}",
                'start_time': tower_data['first_start_time'].strftime('%Y-%m-%d') if tower_data.get('first_start_time') else 'N/A',
                'date': date.strftime('%Y-%m-%d'),
                'frame': tower_data.get('displacement_time_series', [{}])[0].get('frame', 'N/A') if tower_data.get('displacement_time_series') else 'N/A',
            }

            # Add pixel displacement data (pixels 1-9)
            if date in displacement_by_date:
                pixel_displacements = displacement_by_date[date]['pixel_displacements']
                for pixel_num in range(1, 10):
                    displacement = pixel_displacements.get(pixel_num)
                    if displacement is not None:
                        row_data[f'pixel_{pixel_num}_displacement_m'] = round(displacement, 3)
                    else:
                        row_data[f'pixel_{pixel_num}_displacement_m'] = None
            else:
                # No displacement data for this date
                for pixel_num in range(1, 10):
                    row_data[f'pixel_{pixel_num}_displacement_m'] = None

            # Add pixel coherence data (pixels 1-9)
            if date in coherence_by_date:
                pixel_coherences = coherence_by_date[date]['pixel_coherences']
                for pixel_num in range(1, 10):
                    coherence = pixel_coherences.get(pixel_num)
                    if coherence is not None:
                        row_data[f'pixel_{pixel_num}_coherence'] = round(coherence, 4)
                    else:
                        row_data[f'pixel_{pixel_num}_coherence'] = None
            else:
                # No coherence data for this date
                for pixel_num in range(1, 10):
                    row_data[f'pixel_{pixel_num}_coherence'] = None

            tower_time_series.append(row_data)

        # Save individual tower CSV file
        if tower_time_series:
            tower_df = pd.DataFrame(tower_time_series)

            # Reorder columns for better readability
            base_cols = ['tower_id', 'start_time', 'date', 'frame']
            displacement_cols = [f'pixel_{i}_displacement_m' for i in range(1, 10)]
            coherence_cols = [f'pixel_{i}_coherence' for i in range(1, 10)]

            # Interleave displacement and coherence columns
            interleaved_cols = []
            for i in range(1, 10):
                interleaved_cols.append(f'pixel_{i}_displacement_m')
                interleaved_cols.append(f'pixel_{i}_coherence')

            tower_df = tower_df[base_cols + interleaved_cols]

            tower_path = self.output_dir / f"tower_{tower_id}_9pixel_displacement_coherence.csv"
            tower_df.to_csv(tower_path, index=False)
            print(f"Tower {tower_id} (9-pixel displacement+coherence) time series saved to {tower_path}")

    def check_tower_has_valid_data(self, tower_data):
        """
        Check if a tower has any valid (non-None) data across all timestamps.

        Args:
            tower_data (dict): Time series data for a tower

        Returns:
            bool: True if tower has at least one valid value, False otherwise
        """
        # Check displacement data
        for ts in tower_data.get('displacement_time_series', []):
            pixel_displacements = ts.get('pixel_displacements', {})
            if any(val is not None for val in pixel_displacements.values()):
                return True

        # Check coherence data
        for ts in tower_data.get('coherence_time_series', []):
            pixel_coherences = ts.get('pixel_coherences', {})
            if any(val is not None for val in pixel_coherences.values()):
                return True

        return False

    def get_all_frames_containing_point(self, point_geom):
        """
        Find all frames that contain the given point.

        Args:
            point_geom: Point geometry

        Returns:
            list: List of frame identifiers that contain the point
        """
        containing_frames = []

        for frame_id in self.frames_data.keys():
            frame_images = self.frames_data[frame_id]
            if not frame_images:
                continue

            # Use the first image to get frame bounds
            sample_image = frame_images[0]

            try:
                with rasterio.open(sample_image['filepath']) as src:
                    bounds = src.bounds

                    # Create polygon from raster bounds
                    frame_polygon = Polygon([
                        (bounds.left, bounds.bottom),
                        (bounds.right, bounds.bottom),
                        (bounds.right, bounds.top),
                        (bounds.left, bounds.top)
                    ])

                    # Check if point is contained within this frame's polygon
                    if frame_polygon.contains(point_geom):
                        containing_frames.append(frame_id)

            except Exception as e:
                print(f"Error checking frame {frame_id}: {e}")
                continue

        return containing_frames

    def retry_tower_with_different_frames(self, tower_id, point_geom, original_frame):
        """
        Retry extracting data for a tower using different frames.

        Args:
            tower_id: Tower identifier
            point_geom: Point geometry
            original_frame: The frame that was originally tried

        Returns:
            tuple: (success, frame_used, tower_data) or (False, None, None)
        """
        # Get all frames containing this point
        all_frames = self.get_all_frames_containing_point(point_geom)

        print(f"\nRetrying tower {tower_id}:")
        print(f"  Original frame: {original_frame}")
        print(f"  Available frames: {all_frames}")

        # Try each frame except the original one
        for frame in all_frames:
            if frame == original_frame:
                continue

            print(f"  Trying frame {frame}...")
            time_series_result = self.extract_time_series_single_frame(
                frame, point_geom, tower_id
            )

            # Check if this frame has valid data
            if self.check_tower_has_valid_data(time_series_result):
                print(f"  SUCCESS: Found valid data in frame {frame}!")
                return True, frame, time_series_result

        print(f"  FAILED: No valid data found in any available frame")
        return False, None, None

    def process_empty_towers(self, results, analysis_points):
        """
        Identify towers with all empty values and retry with different frames.

        Args:
            results (dict): Original processing results
            analysis_points (gpd.GeoDataFrame): Point features

        Returns:
            dict: Updated results with retried towers
        """
        print("\n" + "="*80)
        print("CHECKING FOR TOWERS WITH EMPTY VALUES")
        print("="*80)

        empty_towers = []
        for tower_id, tower_data in results.items():
            if not self.check_tower_has_valid_data(tower_data):
                empty_towers.append(tower_id)

        if not empty_towers:
            print("No towers with empty values found!")
            return results

        print(f"\nFound {len(empty_towers)} towers with all empty values: {empty_towers[:10]}{'...' if len(empty_towers) > 10 else ''}")
        print("Attempting to retry with different frames...\n")

        retry_success_count = 0
        retry_failed_count = 0

        for tower_id in empty_towers:
            # Get original tower data
            original_data = results[tower_id]
            original_frame = original_data.get('displacement_time_series', [{}])[0].get('frame', 'Unknown') if original_data.get('displacement_time_series') else 'Unknown'

            # Find tower geometry
            if 'tower_id' in analysis_points.columns:
                tower_row = analysis_points[analysis_points['tower_id'] == tower_id]
            else:
                tower_row = analysis_points[analysis_points.index == tower_id - 1]

            if len(tower_row) == 0:
                print(f"WARNING: Cannot find geometry for tower {tower_id}")
                continue

            point_geom = tower_row.geometry.iloc[0]

            # Try different frames
            success, new_frame, new_data = self.retry_tower_with_different_frames(
                tower_id, point_geom, original_frame
            )

            if success:
                # Update results
                results[tower_id] = new_data

                # Save new CSV
                self.save_individual_tower_csv(tower_id, new_data)

                # Log retry success
                self.retry_log.append({
                    'tower_id': tower_id,
                    'original_frame': original_frame,
                    'retry_frame': new_frame,
                    'status': 'SUCCESS',
                    'frames_checked': len(self.get_all_frames_containing_point(point_geom))
                })

                retry_success_count += 1
            else:
                # Log retry failure
                all_frames = self.get_all_frames_containing_point(point_geom)
                self.retry_log.append({
                    'tower_id': tower_id,
                    'original_frame': original_frame,
                    'retry_frame': 'N/A',
                    'status': 'FAILED',
                    'frames_checked': len(all_frames),
                    'available_frames': ','.join(all_frames) if all_frames else 'None'
                })

                self.failed_towers.append({
                    'tower_id': tower_id,
                    'frames_checked': ','.join(all_frames) if all_frames else 'None',
                    'reason': 'No valid data in any available frame'
                })

                retry_failed_count += 1

        print(f"\n{'='*80}")
        print(f"RETRY SUMMARY:")
        print(f"  Successfully recovered: {retry_success_count} towers")
        print(f"  Still failed: {retry_failed_count} towers")
        print(f"{'='*80}\n")

        return results

    def save_retry_and_failed_logs(self):
        """
        Save logs for retry attempts and permanently failed towers.
        """
        # Get county name from output directory
        county_name = self.output_dir.parent.parent.name if self.output_dir.parent.parent else "Unknown"

        # Save retry log
        if self.retry_log:
            retry_df = pd.DataFrame(self.retry_log)
            retry_path = self.output_dir / "tower_retry_log.csv"
            retry_df.to_csv(retry_path, index=False)
            print(f"\nRetry log saved to {retry_path}")

        # Save failed towers report
        if self.failed_towers:
            failed_df = pd.DataFrame(self.failed_towers)
            failed_df.insert(0, 'county', county_name)
            failed_path = self.output_dir / "permanently_failed_towers_report.csv"
            failed_df.to_csv(failed_path, index=False)

            print(f"\n{'='*80}")
            print(f"PERMANENTLY FAILED TOWERS REPORT")
            print(f"{'='*80}")
            print(f"County: {county_name}")
            print(f"Total failed towers: {len(self.failed_towers)}")
            print(f"\nTowers with no valid data in any frame:")
            for tower in self.failed_towers[:20]:  # Show first 20
                print(f"  Tower {tower['tower_id']}: checked frames [{tower['frames_checked']}]")
            if len(self.failed_towers) > 20:
                print(f"  ... and {len(self.failed_towers) - 20} more")
            print(f"\nDetailed report saved to: {failed_path}")
            print(f"{'='*80}\n")

    def save_failed_files_log(self):
        """
        Save a log of all failed files for post-processing review.
        """
        if self.failed_files:
            failed_df = pd.DataFrame(self.failed_files)
            failed_path = self.output_dir / "failed_files_log.csv"
            failed_df.to_csv(failed_path, index=False)
            print(f"\n{'='*60}")
            print(f"Failed files log saved to {failed_path}")
            print(f"Total failed files: {len(self.failed_files)}")
            print(f"{'='*60}\n")
        else:
            print("\nNo failed files encountered during processing.")

    def save_time_series_as_csv(self, results):
        """
        Save time series results with 9-pixel neighborhood data for both displacement and coherence.
        Each tower gets its own CSV with columns for all 9 pixels for both data types.

        Args:
            results (dict): Time series results for each tower
        """
        print("Saving comprehensive CSVs and summary data...")

        summary_data = []

        # Collect summary data for each tower (individual CSVs already saved)
        for tower_id, tower_data in results.items():
            summary_data.append({
                'tower_id': f"tower_id_{tower_id}",
                'displacement_integrity': tower_data['displacement_integrity'],
                'coherence_integrity': tower_data['coherence_integrity'],
                'displacement_valid_count': tower_data['displacement_valid_count'],
                'displacement_total_count': tower_data['displacement_total_count'],
                'coherence_valid_count': tower_data['coherence_valid_count'],
                'coherence_total_count': tower_data['coherence_total_count']
            })

        # Create separate comprehensive CSVs for displacement and coherence
        if results:
            print("Creating comprehensive CSVs...")

            # Displacement comprehensive CSV
            displacement_data = []
            for tower_id, tower_data in results.items():
                for ts_point in tower_data['displacement_time_series']:
                    pixel_displacements = ts_point['pixel_displacements']
                    for pixel_num in range(1, 10):
                        displacement = pixel_displacements.get(pixel_num)
                        if displacement is not None:
                            displacement_data.append({
                                'tower_id': f"tower_id_{tower_id}",
                                'start_time': tower_data['first_start_time'].strftime('%Y-%m-%d') if tower_data.get('first_start_time') else 'N/A',
                                'pixel_number': pixel_num,
                                'date': ts_point['date'].strftime('%Y-%m-%d'),
                                'displacement_m': round(displacement, 3),
                                'frame': ts_point['frame'],
                                'integrity': tower_data['displacement_integrity']
                            })

            if displacement_data:
                displacement_df = pd.DataFrame(displacement_data)
                displacement_path = self.output_dir / "all_towers_9pixel_displacement_comprehensive.csv"
                displacement_df.to_csv(displacement_path, index=False)
                print(f"Comprehensive displacement CSV saved to {displacement_path}")

            # Coherence comprehensive CSV
            coherence_data = []
            for tower_id, tower_data in results.items():
                for ts_point in tower_data['coherence_time_series']:
                    pixel_coherences = ts_point['pixel_coherences']
                    for pixel_num in range(1, 10):
                        coherence = pixel_coherences.get(pixel_num)
                        if coherence is not None:
                            coherence_data.append({
                                'tower_id': f"tower_id_{tower_id}",
                                'start_time': tower_data['first_start_time'].strftime('%Y-%m-%d') if tower_data.get('first_start_time') else 'N/A',
                                'pixel_number': pixel_num,
                                'date': ts_point['date'].strftime('%Y-%m-%d'),
                                'coherence': round(coherence, 4),
                                'frame': ts_point['frame'],
                                'integrity': tower_data['coherence_integrity']
                            })

            if coherence_data:
                coherence_df = pd.DataFrame(coherence_data)
                coherence_path = self.output_dir / "all_towers_9pixel_coherence_comprehensive.csv"
                coherence_df.to_csv(coherence_path, index=False)
                print(f"Comprehensive coherence CSV saved to {coherence_path}")

        # Save summary CSV with all towers
        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            # Add summary statistics
            if len(summary_data) > 0:
                displacement_integrities = [data['displacement_integrity'] for data in summary_data]
                coherence_integrities = [data['coherence_integrity'] for data in summary_data if data['coherence_integrity'] > 0]

                summary_stats = pd.DataFrame([{
                    'metric': 'Average Displacement Integrity',
                    'value': np.mean(displacement_integrities)
                }, {
                    'metric': 'Average Coherence Integrity',
                    'value': np.mean(coherence_integrities) if coherence_integrities else 0
                }, {
                    'metric': 'Min Displacement Integrity',
                    'value': np.min(displacement_integrities)
                }, {
                    'metric': 'Max Displacement Integrity',
                    'value': np.max(displacement_integrities)
                }, {
                    'metric': 'High Displacement Integrity Towers (>=0.8)',
                    'value': sum(1 for i in displacement_integrities if i >= 0.8)
                }, {
                    'metric': 'Medium Displacement Integrity Towers (0.5-0.8)',
                    'value': sum(1 for i in displacement_integrities if 0.5 <= i < 0.8)
                }, {
                    'metric': 'Low Displacement Integrity Towers (<0.5)',
                    'value': sum(1 for i in displacement_integrities if i < 0.5)
                }, {
                    'metric': 'Towers with Coherence Data',
                    'value': len(coherence_integrities)
                }])

                summary_path = self.output_dir / "powertower_9pixel_displacement_coherence_summary.csv"
                summary_df.to_csv(summary_path, index=False)
                print(f"Tower summary saved to {summary_path}")

                stats_path = self.output_dir / "powertower_9pixel_displacement_coherence_statistics.csv"
                summary_stats.to_csv(stats_path, index=False)
                print(f"Summary statistics saved to {stats_path}")

        print(f"Comprehensive CSV export complete! Created displacement and coherence comprehensive CSV files and summary statistics.")

    def run_analysis(self):
        """
        Run the complete spatio-temporal analysis workflow.
        """
        print("Starting InSAR Spatio-Temporal Analysis...")

        # Step 1: Group files by frame
        self.group_files_by_frame()

        if not self.frames_data:
            print("No valid OPERA files found!")
            return

        # Step 2: Select random images and create mosaic
        selected_images = self.select_random_images_per_frame()
        self.create_mosaic(selected_images)

        # Step 3: Find shapefile intersections
        intersecting_points = self.find_shapefile_intersections()

        if len(intersecting_points) == 0:
            print("No intersecting points found!")
            return

        # Step 4: Prepare points for analysis
        analysis_points = self.prepare_points_for_analysis(intersecting_points)

        # Step 5: Classify points
        classification = self.classify_points(analysis_points)

        # Step 6: Process valid points and extract time series (individual CSVs saved immediately)
        results = self.process_points(analysis_points, classification)

        # Step 7: Check for towers with empty values and retry with different frames
        if results:
            results = self.process_empty_towers(results, analysis_points)

        # Step 8: Save comprehensive results and summary data
        if results:
            self.save_time_series_as_csv(results)
            print(f"Analysis complete! Processed {len(results)} towers with immediate CSV saving.")
        else:
            print("No valid towers to process.")

        # Step 9: Save retry and failed tower logs
        self.save_retry_and_failed_logs()

        # Step 10: Save failed files log
        self.save_failed_files_log()


def main():
    """
    Main function to run the InSAR analysis.
    """
    # Configuration
    input_dir = r"G:\processed\Baldwin\Vertical-Mask-Reproject\Coherence_0.5"  # Directory with OPERA multi-band TIF files (displacement + coherence)
    shapefile_path = r"D:\project\InSAR OPERA\powertower_every_county\output\Baldwin\Baldwin_tower.shp"          # Path to your shapefile
    output_dir = r"G:\processed\Baldwin\Vertical-Time-Series2\Coherence_0.5"                       # Output directory

    # Initialize and run processor
    processor = InSARSpatioTemporalProcessor(input_dir, shapefile_path, output_dir)

    print("1. Group OPERA multi-band TIF files by frames, and create a mosaic area of different frames.")
    print("2. Find intersection points between shapefile and mosaiced area.")
    print("3. Classify these intersection points as inside or outside mosaic scope.")
    print("4. Extract cumulative vertical displacement (band 1) and coherence values (band 2) for 3×3 pixel neighborhood around each tower.")
    print("5. For displacement: Calculate cumulative time series from the first starting time.")
    print("6. For coherence: Extract direct values (no cumulative calculation).")
    print("7. Save both displacement and coherence data for all 9 pixels as CSV format with sequential pixel numbering (1-9).")
    print("8. Auto-detect towers with empty values and retry extraction using different frames.")
    print("9. Generate detailed reports for towers that failed in all available frames.")

    processor.run_analysis()


if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) == 4:
        # Called from automated_county_processor.py with arguments
        input_dir = sys.argv[1]
        shapefile_path = sys.argv[2]
        output_dir = sys.argv[3]

        print("Running with command line arguments:")
        print(f"  Input directory: {input_dir}")
        print(f"  Shapefile path: {shapefile_path}")
        print(f"  Output directory: {output_dir}")
        print()

        # Initialize and run processor with provided arguments
        processor = InSARSpatioTemporalProcessor(input_dir, shapefile_path, output_dir)

        print("1. Group OPERA multi-band TIF files by frames, and create a mosaic area of different frames.")
        print("2. Find intersection points between shapefile and mosaiced area.")
        print("3. Classify these intersection points as inside or outside mosaic scope.")
        print("4. Extract cumulative vertical displacement (band 1) and coherence values (band 2) for 3×3 pixel neighborhood around each tower.")
        print("5. For displacement: Calculate cumulative time series from the first starting time.")
        print("6. For coherence: Extract direct values (no cumulative calculation).")
        print("7. Save both displacement and coherence data for all 9 pixels as CSV format with sequential pixel numbering (1-9).")
        print("8. Auto-detect towers with empty values and retry extraction using different frames.")
        print("9. Generate detailed reports for towers that failed in all available frames.")

        processor.run_analysis()
    else:
        # No arguments provided, use default paths from main()
        print("No command line arguments provided, using default paths from main()")
        main()