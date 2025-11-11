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
from scipy.ndimage import binary_dilation

import rasterio
from rasterio import mask, transform, merge
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import geopandas as gpd
from shapely.geometry import Point, Polygon, mapping
from shapely.ops import unary_union
import fiona
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


class InSARSubstationProcessor:
    """
    Process InSAR vertical displacement data for substations (polygons).

    Handles frame mosaicking, polygon intersection analysis,
    and time series extraction from OPERA vertical displacement data for all pixels
    within each substation polygon.
    """

    def __init__(self, input_dir, shapefile_path, output_dir):
        """
        Initialize the processor.

        Args:
            input_dir (str): Directory containing OPERA multi-band TIF files (displacement + coherence)
            shapefile_path (str): Path to shapefile with polygon data (substations)
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

        # Track substations with retry attempts
        self.retry_log = []
        self.failed_substations = []

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
        Find intersection polygons between shapefile and mosaiced area.

        Returns:
            gpd.GeoDataFrame: Polygons that intersect with the mosaic
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

        # Find intersecting polygons
        intersecting_polygons = gdf[gdf.geometry.intersects(mosaic_polygon)]

        print(f"Found {len(intersecting_polygons)} intersecting substations")
        return intersecting_polygons

    def prepare_polygons_for_analysis(self, intersecting_polygons):
        """
        Prepare intersection polygons for time series analysis.

        Args:
            intersecting_polygons (gpd.GeoDataFrame): Polygons to analyze

        Returns:
            gpd.GeoDataFrame: Polygons ready for analysis
        """
        print("Preparing substations for time series analysis...")

        # Ensure fid column exists
        if 'fid' not in intersecting_polygons.columns:
            intersecting_polygons['fid'] = range(1, len(intersecting_polygons) + 1)
            print("Added fid column")
        else:
            print(f"Using existing fid column")

        return intersecting_polygons

    def classify_polygons(self, polygons):
        """
        Classify polygons as inside or outside mosaic scope.

        Args:
            polygons (gpd.GeoDataFrame): Polygon features

        Returns:
            dict: Classification results
        """
        print("Classifying polygons...")

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

        for idx, polygon in polygons.iterrows():
            polygon_geom = polygon.geometry

            if mosaic_polygon.intersects(polygon_geom):
                classification['inside'].append(idx)
            else:
                classification['outside'].append(idx)

        print(f"Classification: {len(classification['inside'])} inside, "
              f"{len(classification['outside'])} outside")

        return classification

    def check_overlap_intersection(self, polygon_geom):
        """
        Check if polygon intersects with any overlapping areas.

        Args:
            polygon_geom: Polygon geometry

        Returns:
            bool: True if intersects with overlap areas
        """
        for overlap in self.overlap_polygons:
            if polygon_geom.intersects(overlap):
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

    def extract_time_series_single_frame(self, frame, polygon_geom, substation_fid):
        """
        Extract cumulative time series for both displacement and coherence data
        from multi-band files for all pixels within a polygon.

        Displacement: Cumulative values from the first period's start date.
        Coherence: Direct values (no cumulative calculation).

        Args:
            frame (str): Frame identifier
            polygon_geom: Polygon geometry
            substation_fid: Substation FID for reference

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
        period_baselines = {}  # Store final displacement of each period for each pixel

        # Collect pixel metadata (only need to do this once)
        pixel_metadata_all = {}

        # Process each period
        for period_idx, (start_date_key, images) in enumerate(sorted_periods):
            images_sorted = sorted(images, key=lambda x: x['end_date'])

            for img in images_sorted:
                displacement_total_count += 1
                coherence_total_count += 1

                # Extract displacement values from band 1 for all pixels in polygon (including metadata)
                pixel_displacements, pixel_metadata_disp = self._extract_displacement_from_polygon(
                    img['filepath'], polygon_geom, band_number=1
                )

                # Extract coherence values from band 2 for all pixels in polygon (including metadata)
                pixel_coherences, pixel_metadata_coh = self._extract_coherence_from_polygon(
                    img['filepath'], polygon_geom, band_number=2
                )

                # Store pixel metadata (combine from both displacement and coherence, only once)
                if pixel_metadata_disp and len(pixel_metadata_all) == 0:
                    pixel_metadata_all.update(pixel_metadata_disp)
                if pixel_metadata_coh and len(pixel_metadata_all) == 0:
                    pixel_metadata_all.update(pixel_metadata_coh)

                # Process displacement data
                if pixel_displacements is not None and len(pixel_displacements) > 0:
                    displacement_valid_count += 1

                    # Store displacement data for all pixels
                    cumulative_pixel_data = {}

                    for pixel_id, pixel_displacement in pixel_displacements.items():
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
                                        pixel_id in period_baselines[prev_start_key] and
                                        period_baselines[prev_start_key][pixel_id] is not None):
                                        cumulative_displacement += period_baselines[prev_start_key][pixel_id]

                            cumulative_pixel_data[pixel_id] = cumulative_displacement
                        else:
                            cumulative_pixel_data[pixel_id] = None

                    displacement_time_series.append({
                        'date': img['end_date'],
                        'pixel_displacements': cumulative_pixel_data,
                        'frame': frame,
                    })

                # Process coherence data (no cumulative calculation)
                if pixel_coherences is not None and len(pixel_coherences) > 0:
                    coherence_valid_count += 1

                    coherence_time_series.append({
                        'date': img['end_date'],
                        'pixel_coherences': pixel_coherences,
                        'frame': frame,
                    })

            # Store the final displacement of this period as baseline for next period
            if images_sorted:
                final_img = images_sorted[-1]
                final_pixel_displacements, _ = self._extract_displacement_from_polygon(
                    final_img['filepath'], polygon_geom, band_number=1
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
            'substation_fid': substation_fid,
            'displacement_time_series': displacement_time_series,
            'coherence_time_series': coherence_time_series,
            'displacement_integrity': displacement_integrity,
            'coherence_integrity': coherence_integrity,
            'displacement_valid_count': displacement_valid_count,
            'displacement_total_count': displacement_total_count,
            'coherence_valid_count': coherence_valid_count,
            'coherence_total_count': coherence_total_count,
            'period_start_dates': [datetime.strptime(key, '%Y%m%d') for key, _ in sorted_periods],
            'first_start_time': first_start_time,
            'pixel_metadata': pixel_metadata_all
        }

    def _generate_polygon_and_outer_ring_masks(self, src, polygon_geom):
        """
        Generate polygon mask and outer-ring mask using morphological dilation.

        Step 1: Create buffered polygon to ensure outer-ring pixels are included in masked_data
        Step 2: Use original polygon rasterization to precisely identify intersected pixels
        Step 3: Dilate intersected mask to get outer-ring pixels

        Args:
            src: Open rasterio source
            polygon_geom: Polygon geometry

        Returns:
            tuple: (masked_data, masked_transform, polygon_mask, outer_ring_mask)
        """
        from rasterio import features

        # Step 1: Create buffered polygon to ensure outer-ring pixels are included
        pixel_size_x = abs(src.transform[0])
        pixel_size_y = abs(src.transform[4])
        pixel_size = max(pixel_size_x, pixel_size_y)
        buffered_polygon = polygon_geom.buffer(pixel_size * 1.5)

        # Use buffered polygon for mask.mask() to include outer-ring pixels in data
        masked_data, masked_transform = mask.mask(
            src, [mapping(buffered_polygon)], crop=True, filled=False, all_touched=True
        )

        # Step 2: Get the shape of masked data
        if masked_data.ndim == 3:
            mask_shape = masked_data.shape[1:]
        else:
            mask_shape = masked_data.shape

        # Rasterize the ORIGINAL (non-buffered) polygon to create precise intersected pixel mask
        polygon_mask = features.rasterize(
            [mapping(polygon_geom)],  # Use original polygon, not buffered
            out_shape=mask_shape,
            transform=masked_transform,
            fill=0,
            default_value=1,
            all_touched=True,
            dtype=np.uint8
        ).astype(bool)

        # Convert to binary (1 for intersected, 0 for outside)
        polygon_binary = polygon_mask.astype(np.uint8)

        # Step 3: Dilate by 1 pixel to get outer-ring
        kernel_3x3 = np.ones((3, 3), dtype=np.uint8)
        dilated_mask = binary_dilation(polygon_binary, structure=kernel_3x3).astype(np.uint8)

        # Step 4: Get outer ring (dilated - original)
        outer_ring_mask = (dilated_mask - polygon_binary).astype(bool)

        return masked_data, masked_transform, polygon_mask, outer_ring_mask

    def _extract_displacement_from_polygon(self, image_path, polygon_geom, band_number=1, max_retries=5):
        """
        Extract displacement values from both intersected and outer-ring pixels.

        Args:
            image_path: Path to the multi-band image
            polygon_geom: Polygon geometry
            band_number: Band number to read displacement from (default: 1)
            max_retries: Maximum number of retry attempts for file access (default: 5)

        Returns:
            tuple: (pixel_values_dict, pixel_metadata_dict) where:
                - pixel_values_dict: {pixel_id: displacement_value}
                - pixel_metadata_dict: {pixel_id: {'lon': lon, 'lat': lat, 'row': row, 'col': col, 'label': 'intersected'/'outer'}}
                Or (None, None) if extraction fails
        """
        src = None
        for attempt in range(max_retries):
            try:
                # Add small delay before opening to allow any file locks to clear
                if attempt > 0:
                    time.sleep(0.5 * (2 ** attempt))  # Exponential backoff

                src = rasterio.open(image_path)

                # Verify band exists
                if band_number > src.count:
                    print(f"Warning: Band {band_number} does not exist in {image_path}. Available bands: {src.count}")
                    if src:
                        src.close()
                    return None, None

                # Generate polygon and outer-ring masks
                try:
                    masked_data, masked_transform, polygon_mask, outer_ring_mask = \
                        self._generate_polygon_and_outer_ring_masks(src, polygon_geom)

                    # Extract the specific band
                    if band_number <= masked_data.shape[0]:
                        band_data = masked_data[band_number - 1]
                    else:
                        raise ValueError(f"Band {band_number} not available")
                except ValueError as e:
                    # Polygon doesn't intersect with raster
                    if attempt == 0:
                        print(f"Warning: Polygon does not intersect with raster {image_path.name}")
                    if src:
                        src.close()
                    return None, None

                # Extract pixel values and metadata
                pixel_values = {}
                pixel_metadata = {}

                # Get the band data as array
                if band_data.ndim == 2:
                    data_array = band_data
                else:
                    data_array = band_data[0] if band_data.ndim == 3 else band_data

                # Process intersected pixels
                for row_idx in range(polygon_mask.shape[0]):
                    for col_idx in range(polygon_mask.shape[1]):
                        if polygon_mask[row_idx, col_idx]:
                            # Get pixel value
                            if not np.ma.is_masked(data_array[row_idx, col_idx]):
                                value = data_array[row_idx, col_idx]
                                if value != src.nodata and not np.isnan(value):
                                    pixel_id = f"{row_idx}_{col_idx}"
                                    pixel_values[pixel_id] = float(value)

                                    # Calculate lon/lat
                                    lon, lat = rasterio.transform.xy(masked_transform, row_idx, col_idx, offset='center')
                                    pixel_metadata[pixel_id] = {
                                        'lon': lon,
                                        'lat': lat,
                                        'row': row_idx,
                                        'col': col_idx,
                                        'label': 'intersected'
                                    }

                # Process outer-ring pixels
                for row_idx in range(outer_ring_mask.shape[0]):
                    for col_idx in range(outer_ring_mask.shape[1]):
                        if outer_ring_mask[row_idx, col_idx]:
                            # Get pixel value
                            if not np.ma.is_masked(data_array[row_idx, col_idx]):
                                value = data_array[row_idx, col_idx]
                                if value != src.nodata and not np.isnan(value):
                                    pixel_id = f"{row_idx}_{col_idx}"
                                    pixel_values[pixel_id] = float(value)

                                    # Calculate lon/lat
                                    lon, lat = rasterio.transform.xy(masked_transform, row_idx, col_idx, offset='center')
                                    pixel_metadata[pixel_id] = {
                                        'lon': lon,
                                        'lat': lat,
                                        'row': row_idx,
                                        'col': col_idx,
                                        'label': 'outer'
                                    }

                # Close before returning
                src.close()
                if len(pixel_values) > 0:
                    return pixel_values, pixel_metadata
                else:
                    return None, None

            except rasterio.errors.RasterioIOError as e:
                # Ensure file is closed before retry
                if src:
                    try:
                        src.close()
                    except:
                        pass

                if attempt < max_retries - 1:
                    wait_time = 0.5 * (2 ** attempt)
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
                    return None, None
            except Exception as e:
                # Ensure file is closed
                if src:
                    try:
                        src.close()
                    except:
                        pass
                print(f"Warning: Error extracting displacement from {image_path}: {e}. Skipping.")
                return None, None
            finally:
                # Always try to close the file handle
                if src and not src.closed:
                    try:
                        src.close()
                    except:
                        pass

        return None, None

    def _extract_coherence_from_polygon(self, image_path, polygon_geom, band_number=2, max_retries=5):
        """
        Extract coherence values from both intersected and outer-ring pixels.

        Args:
            image_path: Path to the multi-band image
            polygon_geom: Polygon geometry
            band_number: Band number to read coherence from (default: 2)
            max_retries: Maximum number of retry attempts for file access (default: 5)

        Returns:
            tuple: (pixel_values_dict, pixel_metadata_dict) where:
                - pixel_values_dict: {pixel_id: coherence_value}
                - pixel_metadata_dict: {pixel_id: {'lon': lon, 'lat': lat, 'row': row, 'col': col, 'label': 'intersected'/'outer'}}
                Or (None, None) if extraction fails
        """
        src = None
        for attempt in range(max_retries):
            try:
                # Add small delay before opening to allow any file locks to clear
                if attempt > 0:
                    time.sleep(0.5 * (2 ** attempt))  # Exponential backoff

                src = rasterio.open(image_path)

                # Check if the requested band exists
                if band_number > src.count:
                    print(f"Warning: Band {band_number} does not exist in {image_path}. Available bands: {src.count}")
                    if src:
                        src.close()
                    return None, None

                # Generate polygon and outer-ring masks
                try:
                    masked_data, masked_transform, polygon_mask, outer_ring_mask = \
                        self._generate_polygon_and_outer_ring_masks(src, polygon_geom)

                    # Extract the specific band
                    if band_number <= masked_data.shape[0]:
                        band_data = masked_data[band_number - 1]
                    else:
                        raise ValueError(f"Band {band_number} not available")
                except ValueError as e:
                    # Polygon doesn't intersect with raster
                    if attempt == 0:
                        print(f"Warning: Polygon does not intersect with raster {image_path.name}")
                    if src:
                        src.close()
                    return None, None

                # Extract pixel values and metadata
                pixel_values = {}
                pixel_metadata = {}

                # Get the band data as array
                if band_data.ndim == 2:
                    data_array = band_data
                else:
                    data_array = band_data[0] if band_data.ndim == 3 else band_data

                # Process intersected pixels
                for row_idx in range(polygon_mask.shape[0]):
                    for col_idx in range(polygon_mask.shape[1]):
                        if polygon_mask[row_idx, col_idx]:
                            # Get pixel value
                            if not np.ma.is_masked(data_array[row_idx, col_idx]):
                                value = data_array[row_idx, col_idx]
                                if value != src.nodata and not np.isnan(value):
                                    pixel_id = f"{row_idx}_{col_idx}"
                                    pixel_values[pixel_id] = float(value)

                                    # Calculate lon/lat
                                    lon, lat = rasterio.transform.xy(masked_transform, row_idx, col_idx, offset='center')
                                    pixel_metadata[pixel_id] = {
                                        'lon': lon,
                                        'lat': lat,
                                        'row': row_idx,
                                        'col': col_idx,
                                        'label': 'intersected'
                                    }

                # Process outer-ring pixels
                for row_idx in range(outer_ring_mask.shape[0]):
                    for col_idx in range(outer_ring_mask.shape[1]):
                        if outer_ring_mask[row_idx, col_idx]:
                            # Get pixel value
                            if not np.ma.is_masked(data_array[row_idx, col_idx]):
                                value = data_array[row_idx, col_idx]
                                if value != src.nodata and not np.isnan(value):
                                    pixel_id = f"{row_idx}_{col_idx}"
                                    pixel_values[pixel_id] = float(value)

                                    # Calculate lon/lat
                                    lon, lat = rasterio.transform.xy(masked_transform, row_idx, col_idx, offset='center')
                                    pixel_metadata[pixel_id] = {
                                        'lon': lon,
                                        'lat': lat,
                                        'row': row_idx,
                                        'col': col_idx,
                                        'label': 'outer'
                                    }

                # Close before returning
                src.close()
                if len(pixel_values) > 0:
                    return pixel_values, pixel_metadata
                else:
                    return None, None

            except rasterio.errors.RasterioIOError as e:
                # Ensure file is closed before retry
                if src:
                    try:
                        src.close()
                    except:
                        pass

                if attempt < max_retries - 1:
                    wait_time = 0.5 * (2 ** attempt)
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
                    return None, None
            except Exception as e:
                # Ensure file is closed
                if src:
                    try:
                        src.close()
                    except:
                        pass
                print(f"Warning: Error extracting coherence from {image_path}: {e}. Skipping.")
                return None, None
            finally:
                # Always try to close the file handle
                if src and not src.closed:
                    try:
                        src.close()
                    except:
                        pass

        return None, None

    def process_polygons(self, polygons, classification):
        """
        Process all valid polygons and extract time series for both displacement and coherence.

        Args:
            polygons (gpd.GeoDataFrame): Polygon features
            classification (dict): Polygon classification results

        Returns:
            dict: Time series results for each substation
        """
        results = {}

        # Process polygons that are inside the mosaic with progress bar
        inside_indices = classification['inside']

        with tqdm(total=len(inside_indices), desc="Processing substations (displacement+coherence)") as pbar:
            for idx in inside_indices:
                polygon_row = polygons.loc[idx]
                polygon_geom = polygon_row.geometry
                substation_fid = polygon_row.get('fid', idx + 1)

                # Check if polygon intersects with overlap areas
                intersects_overlap = self.check_overlap_intersection(polygon_geom)

                if intersects_overlap:
                    # Multiple frames - randomly choose one
                    available_frames = list(self.frames_data.keys())
                    chosen_frame = random.choice(available_frames)
                else:
                    # Single frame - find which frame contains this polygon
                    chosen_frame = self._find_containing_frame(polygon_geom)

                if chosen_frame:
                    time_series_result = self.extract_time_series_single_frame(
                        chosen_frame, polygon_geom, substation_fid
                    )
                    results[substation_fid] = time_series_result

                    # Save individual substation CSV immediately after processing
                    self.save_individual_substation_csv(substation_fid, time_series_result)

                # Update progress bar
                displacement_count = len(time_series_result.get('displacement_time_series', [])) if chosen_frame and substation_fid in results else 0
                coherence_count = len(time_series_result.get('coherence_time_series', [])) if chosen_frame and substation_fid in results else 0
                pbar.set_postfix({
                    'Substation': str(substation_fid),
                    'Frame': chosen_frame if chosen_frame else 'N/A',
                    'Disp': displacement_count,
                    'Coh': coherence_count
                })
                pbar.update(1)

        print(f"Processed {len(results)} substations with valid time series data (displacement + coherence)")
        return results

    def _find_containing_frame(self, geometry):
        """
        Find which frame contains the given geometry.

        Args:
            geometry: Shapely geometry (Polygon)

        Returns:
            str: Frame identifier or None if polygon is not contained in any frame
        """
        # Check each frame to see if it contains/intersects the polygon
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

                    # Check if polygon intersects with this frame's polygon
                    if frame_polygon.intersects(geometry):
                        return frame_id

            except Exception as e:
                print(f"Error checking frame {frame_id}: {e}")
                continue

        # If no frame contains the polygon, return None
        return None

    def save_individual_substation_csv(self, substation_fid, substation_data):
        """
        Save individual substation time series results as Excel with two sheets:
        Sheet 1: Time-series displacement and coherence data
        Sheet 2: Pixel metadata (lon, lat, row, col, label)

        Args:
            substation_fid: Substation FID
            substation_data: Time series data for the substation
        """
        # Convert FID to int to remove .0
        fid_int = int(substation_fid) if isinstance(substation_fid, float) else substation_fid

        # Create individual time series data for this substation
        substation_time_series = []

        # Create a complete date index from both displacement and coherence data
        all_dates = set()

        # Collect all dates from displacement data
        for ts_point in substation_data['displacement_time_series']:
            all_dates.add(ts_point['date'])

        # Collect all dates from coherence data
        for ts_point in substation_data['coherence_time_series']:
            all_dates.add(ts_point['date'])

        # Sort dates
        sorted_dates = sorted(all_dates)

        # Create dictionaries for quick lookup
        displacement_by_date = {ts['date']: ts for ts in substation_data['displacement_time_series']}
        coherence_by_date = {ts['date']: ts for ts in substation_data['coherence_time_series']}

        # Get all unique pixel IDs from all timestamps
        all_pixel_ids = set()
        for ts in substation_data['displacement_time_series']:
            all_pixel_ids.update(ts['pixel_displacements'].keys())
        for ts in substation_data['coherence_time_series']:
            all_pixel_ids.update(ts['pixel_coherences'].keys())

        # Sort pixel IDs for consistent ordering
        sorted_pixel_ids = sorted(all_pixel_ids)

        # Process each date
        for date in sorted_dates:
            # Create base row data
            row_data = {
                'substation_fid': f"substation_{fid_int}",
                'start_time': substation_data['first_start_time'].strftime('%Y-%m-%d') if substation_data.get('first_start_time') else 'N/A',
                'date': date.strftime('%Y-%m-%d'),
                'frame': substation_data.get('displacement_time_series', [{}])[0].get('frame', 'N/A') if substation_data.get('displacement_time_series') else 'N/A',
            }

            # Add displacement and coherence data for each pixel
            if date in displacement_by_date:
                pixel_displacements = displacement_by_date[date]['pixel_displacements']
                for pixel_id in sorted_pixel_ids:
                    displacement = pixel_displacements.get(pixel_id)
                    if displacement is not None:
                        row_data[f'pixel_{pixel_id}_displacement_m'] = round(displacement, 3)
                    else:
                        row_data[f'pixel_{pixel_id}_displacement_m'] = None
            else:
                # No displacement data for this date
                for pixel_id in sorted_pixel_ids:
                    row_data[f'pixel_{pixel_id}_displacement_m'] = None

            if date in coherence_by_date:
                pixel_coherences = coherence_by_date[date]['pixel_coherences']
                for pixel_id in sorted_pixel_ids:
                    coherence = pixel_coherences.get(pixel_id)
                    if coherence is not None:
                        row_data[f'pixel_{pixel_id}_coherence'] = round(coherence, 4)
                    else:
                        row_data[f'pixel_{pixel_id}_coherence'] = None
            else:
                # No coherence data for this date
                for pixel_id in sorted_pixel_ids:
                    row_data[f'pixel_{pixel_id}_coherence'] = None

            substation_time_series.append(row_data)

        # Create pixel metadata list for Sheet 2
        pixel_metadata_list = []
        pixel_metadata_dict = substation_data.get('pixel_metadata', {})

        for pixel_id in sorted_pixel_ids:
            if pixel_id in pixel_metadata_dict:
                metadata = pixel_metadata_dict[pixel_id]
                pixel_metadata_list.append({
                    'pixel_id': pixel_id,
                    'longitude': round(metadata['lon'], 6),
                    'latitude': round(metadata['lat'], 6),
                    'row': metadata['row'],
                    'col': metadata['col'],
                    'label': metadata['label']
                })

        # Save as Excel file with two sheets
        if substation_time_series:
            # Create Excel file path
            excel_path = self.output_dir / f"substation_{fid_int}_displacement_coherence.xlsx"

            # Create Excel writer
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Time-series data
                time_series_df = pd.DataFrame(substation_time_series)
                time_series_df.to_excel(writer, sheet_name='Time_Series', index=False)

                # Sheet 2: Pixel metadata
                if pixel_metadata_list:
                    metadata_df = pd.DataFrame(pixel_metadata_list)
                    metadata_df.to_excel(writer, sheet_name='Pixel_Metadata', index=False)

            print(f"Substation {fid_int} time series and pixel metadata saved to {excel_path}")

    def check_substation_has_valid_data(self, substation_data):
        """
        Check if a substation has any valid (non-None) data across all timestamps.

        Args:
            substation_data (dict): Time series data for a substation

        Returns:
            bool: True if substation has at least one valid value, False otherwise
        """
        # Check displacement data
        for ts in substation_data.get('displacement_time_series', []):
            pixel_displacements = ts.get('pixel_displacements', {})
            if any(val is not None for val in pixel_displacements.values()):
                return True

        # Check coherence data
        for ts in substation_data.get('coherence_time_series', []):
            pixel_coherences = ts.get('pixel_coherences', {})
            if any(val is not None for val in pixel_coherences.values()):
                return True

        return False

    def get_all_frames_containing_polygon(self, polygon_geom):
        """
        Find all frames that contain the given polygon.

        Args:
            polygon_geom: Polygon geometry

        Returns:
            list: List of frame identifiers that contain the polygon
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

                    # Check if polygon intersects with this frame's polygon
                    if frame_polygon.intersects(polygon_geom):
                        containing_frames.append(frame_id)

            except Exception as e:
                print(f"Error checking frame {frame_id}: {e}")
                continue

        return containing_frames

    def retry_substation_with_different_frames(self, substation_fid, polygon_geom, original_frame):
        """
        Retry extracting data for a substation using different frames.

        Args:
            substation_fid: Substation FID
            polygon_geom: Polygon geometry
            original_frame: The frame that was originally tried

        Returns:
            tuple: (success, frame_used, substation_data) or (False, None, None)
        """
        # Get all frames containing this polygon
        all_frames = self.get_all_frames_containing_polygon(polygon_geom)

        print(f"\nRetrying substation {substation_fid}:")
        print(f"  Original frame: {original_frame}")
        print(f"  Available frames: {all_frames}")

        # Try each frame except the original one
        for frame in all_frames:
            if frame == original_frame:
                continue

            print(f"  Trying frame {frame}...")
            time_series_result = self.extract_time_series_single_frame(
                frame, polygon_geom, substation_fid
            )

            # Check if this frame has valid data
            if self.check_substation_has_valid_data(time_series_result):
                print(f"  SUCCESS: Found valid data in frame {frame}!")
                return True, frame, time_series_result

        print(f"  FAILED: No valid data found in any available frame")
        return False, None, None

    def process_empty_substations(self, results, analysis_polygons):
        """
        Identify substations with all empty values and retry with different frames.

        Args:
            results (dict): Original processing results
            analysis_polygons (gpd.GeoDataFrame): Polygon features

        Returns:
            dict: Updated results with retried substations
        """
        print("\n" + "="*80)
        print("CHECKING FOR SUBSTATIONS WITH EMPTY VALUES")
        print("="*80)

        empty_substations = []
        for substation_fid, substation_data in results.items():
            if not self.check_substation_has_valid_data(substation_data):
                empty_substations.append(substation_fid)

        if not empty_substations:
            print("No substations with empty values found!")
            return results

        print(f"\nFound {len(empty_substations)} substations with all empty values: {empty_substations[:10]}{'...' if len(empty_substations) > 10 else ''}")
        print("Attempting to retry with different frames...\n")

        retry_success_count = 0
        retry_failed_count = 0

        for substation_fid in empty_substations:
            # Get original substation data
            original_data = results[substation_fid]
            original_frame = original_data.get('displacement_time_series', [{}])[0].get('frame', 'Unknown') if original_data.get('displacement_time_series') else 'Unknown'

            # Find substation geometry
            if 'fid' in analysis_polygons.columns:
                substation_row = analysis_polygons[analysis_polygons['fid'] == substation_fid]
            else:
                substation_row = analysis_polygons[analysis_polygons.index == substation_fid - 1]

            if len(substation_row) == 0:
                print(f"WARNING: Cannot find geometry for substation {substation_fid}")
                continue

            polygon_geom = substation_row.geometry.iloc[0]

            # Try different frames
            success, new_frame, new_data = self.retry_substation_with_different_frames(
                substation_fid, polygon_geom, original_frame
            )

            if success:
                # Update results
                results[substation_fid] = new_data

                # Save new CSV
                self.save_individual_substation_csv(substation_fid, new_data)

                # Log retry success
                self.retry_log.append({
                    'substation_fid': substation_fid,
                    'original_frame': original_frame,
                    'retry_frame': new_frame,
                    'status': 'SUCCESS',
                    'frames_checked': len(self.get_all_frames_containing_polygon(polygon_geom))
                })

                retry_success_count += 1
            else:
                # Log retry failure
                all_frames = self.get_all_frames_containing_polygon(polygon_geom)
                self.retry_log.append({
                    'substation_fid': substation_fid,
                    'original_frame': original_frame,
                    'retry_frame': 'N/A',
                    'status': 'FAILED',
                    'frames_checked': len(all_frames),
                    'available_frames': ','.join(all_frames) if all_frames else 'None'
                })

                self.failed_substations.append({
                    'substation_fid': substation_fid,
                    'frames_checked': ','.join(all_frames) if all_frames else 'None',
                    'reason': 'No valid data in any available frame'
                })

                retry_failed_count += 1

        print(f"\n{'='*80}")
        print(f"RETRY SUMMARY:")
        print(f"  Successfully recovered: {retry_success_count} substations")
        print(f"  Still failed: {retry_failed_count} substations")
        print(f"{'='*80}\n")

        return results

    def save_retry_and_failed_logs(self):
        """
        Save logs for retry attempts and permanently failed substations.
        """
        # Get county name from output directory
        county_name = self.output_dir.parent.parent.name if self.output_dir.parent.parent else "Unknown"

        # Save retry log
        if self.retry_log:
            retry_df = pd.DataFrame(self.retry_log)
            retry_path = self.output_dir / "substation_retry_log.csv"
            retry_df.to_csv(retry_path, index=False)
            print(f"\nRetry log saved to {retry_path}")

        # Save failed substations report
        if self.failed_substations:
            failed_df = pd.DataFrame(self.failed_substations)
            failed_df.insert(0, 'county', county_name)
            failed_path = self.output_dir / "permanently_failed_substations_report.csv"
            failed_df.to_csv(failed_path, index=False)

            print(f"\n{'='*80}")
            print(f"PERMANENTLY FAILED SUBSTATIONS REPORT")
            print(f"{'='*80}")
            print(f"County: {county_name}")
            print(f"Total failed substations: {len(self.failed_substations)}")
            print(f"\nSubstations with no valid data in any frame:")
            for substation in self.failed_substations[:20]:  # Show first 20
                print(f"  Substation {substation['substation_fid']}: checked frames [{substation['frames_checked']}]")
            if len(self.failed_substations) > 20:
                print(f"  ... and {len(self.failed_substations) - 20} more")
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
        Save time series results for all substations.

        Args:
            results (dict): Time series results for each substation
        """
        print("Saving comprehensive CSVs and summary data...")

        summary_data = []

        # Collect summary data for each substation (individual CSVs already saved)
        for substation_fid, substation_data in results.items():
            # Count total pixels
            total_pixels = 0
            if substation_data['displacement_time_series']:
                total_pixels = len(substation_data['displacement_time_series'][0].get('pixel_displacements', {}))

            # Convert FID to int to remove .0
            fid_int = int(substation_fid) if isinstance(substation_fid, float) else substation_fid
            summary_data.append({
                'substation_fid': f"substation_{fid_int}",
                'total_pixels': total_pixels,
                'displacement_integrity': substation_data['displacement_integrity'],
                'coherence_integrity': substation_data['coherence_integrity'],
                'displacement_valid_count': substation_data['displacement_valid_count'],
                'displacement_total_count': substation_data['displacement_total_count'],
                'coherence_valid_count': substation_data['coherence_valid_count'],
                'coherence_total_count': substation_data['coherence_total_count']
            })

        # Save summary CSV with all substations
        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            # Add summary statistics
            if len(summary_data) > 0:
                displacement_integrities = [data['displacement_integrity'] for data in summary_data]
                coherence_integrities = [data['coherence_integrity'] for data in summary_data if data['coherence_integrity'] > 0]

                summary_stats = pd.DataFrame([{
                    'metric': 'Total Substations Processed',
                    'value': len(summary_data)
                }, {
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
                    'metric': 'High Displacement Integrity Substations (>=0.8)',
                    'value': sum(1 for i in displacement_integrities if i >= 0.8)
                }, {
                    'metric': 'Medium Displacement Integrity Substations (0.5-0.8)',
                    'value': sum(1 for i in displacement_integrities if 0.5 <= i < 0.8)
                }, {
                    'metric': 'Low Displacement Integrity Substations (<0.5)',
                    'value': sum(1 for i in displacement_integrities if i < 0.5)
                }, {
                    'metric': 'Substations with Coherence Data',
                    'value': len(coherence_integrities)
                }])

                summary_path = self.output_dir / "substation_displacement_coherence_summary.csv"
                summary_df.to_csv(summary_path, index=False)
                print(f"Substation summary saved to {summary_path}")

                stats_path = self.output_dir / "substation_displacement_coherence_statistics.csv"
                summary_stats.to_csv(stats_path, index=False)
                print(f"Summary statistics saved to {stats_path}")

        print(f"CSV export complete! Created summary statistics and individual substation files.")

    def run_analysis(self):
        """
        Run the complete spatio-temporal analysis workflow for substations.
        """
        print("Starting InSAR Substation Analysis...")

        # Step 1: Group files by frame
        self.group_files_by_frame()

        if not self.frames_data:
            print("No valid OPERA files found!")
            return

        # Step 2: Select random images and create mosaic
        selected_images = self.select_random_images_per_frame()
        self.create_mosaic(selected_images)

        # Step 3: Find shapefile intersections
        intersecting_polygons = self.find_shapefile_intersections()

        if len(intersecting_polygons) == 0:
            print("No intersecting substations found!")
            return

        # Step 4: Prepare polygons for analysis
        analysis_polygons = self.prepare_polygons_for_analysis(intersecting_polygons)

        # Step 5: Classify polygons
        classification = self.classify_polygons(analysis_polygons)

        # Step 6: Process valid polygons and extract time series (individual Excel files saved immediately)
        results = self.process_polygons(analysis_polygons, classification)

        # Step 7: Check for substations with empty values and retry with different frames
        if results:
            results = self.process_empty_substations(results, analysis_polygons)

        # Step 8: Save comprehensive results and summary data
        if results:
            self.save_time_series_as_csv(results)
            print(f"Analysis complete! Processed {len(results)} substations with immediate Excel file saving.")
        else:
            print("No valid substations to process.")

        # Step 9: Save retry and failed substation logs
        self.save_retry_and_failed_logs()

        # Step 10: Save failed files log
        self.save_failed_files_log()


def main():
    """
    Main function to run the InSAR substation analysis.
    """
    # Configuration
    input_dir = r"G:\processed\Baldwin\Vertical-Mask-Reproject\Coherence_0.2"  # Directory with OPERA multi-band TIF files
    shapefile_path = r"D:\project\InSAR OPERA\substations\Baldwin_substations.shp"  # Path to substation shapefile
    output_dir = r"G:\processed\Baldwin\Vertical-Time-Series2\Coherence_0.2_Substations"  # Output directory

    # Initialize and run processor
    processor = InSARSubstationProcessor(input_dir, shapefile_path, output_dir)

    print("1. Group OPERA multi-band TIF files by frames, and create a mosaic area of different frames.")
    print("2. Identify intersection polygons between shapefile and mosaiced area.")
    print("3. Find all intersected pixels inside each polygon AND outer-ring pixels (1-pixel dilation).")
    print("4. Extract cumulative vertical displacement (band 1) and coherence values (band 2) for all pixels (intersected + outer).")
    print("5. For displacement: Calculate cumulative time series from the first starting time.")
    print("6. For coherence: Extract direct values (no cumulative calculation).")
    print("7. Save Excel file for each substation (substation_{fid}_displacement_coherence.xlsx):")
    print("   - Sheet 1: Time-series displacement and coherence data")
    print("   - Sheet 2: Pixel metadata (lon, lat, row, col, label='intersected'/'outer')")
    print("8. Auto-detect substations with empty values and retry extraction using different frames.")
    print("9. Generate detailed reports for substations that failed in all available frames.")

    processor.run_analysis()


if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) == 4:
        # Called with arguments
        input_dir = sys.argv[1]
        shapefile_path = sys.argv[2]
        output_dir = sys.argv[3]

        print("Running with command line arguments:")
        print(f"  Input directory: {input_dir}")
        print(f"  Shapefile path: {shapefile_path}")
        print(f"  Output directory: {output_dir}")
        print()

        # Initialize and run processor with provided arguments
        processor = InSARSubstationProcessor(input_dir, shapefile_path, output_dir)

        print("1. Group OPERA multi-band TIF files by frames, and create a mosaic area of different frames.")
        print("2. Identify intersection polygons between shapefile and mosaiced area.")
        print("3. Find all intersected pixels inside each polygon AND outer-ring pixels (1-pixel dilation).")
        print("4. Extract cumulative vertical displacement (band 1) and coherence values (band 2) for all pixels (intersected + outer).")
        print("5. For displacement: Calculate cumulative time series from the first starting time.")
        print("6. For coherence: Extract direct values (no cumulative calculation).")
        print("7. Save Excel file for each substation (substation_{fid}_displacement_coherence.xlsx):")
        print("   - Sheet 1: Time-series displacement and coherence data")
        print("   - Sheet 2: Pixel metadata (lon, lat, row, col, label='intersected'/'outer')")
        print("8. Auto-detect substations with empty values and retry extraction using different frames.")
        print("9. Generate detailed reports for substations that failed in all available frames.")

        processor.run_analysis()
    else:
        # No arguments provided, use default paths from main()
        print("No command line arguments provided, using default paths from main()")
        main()
