import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import re
import sys
from datetime import datetime
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import Point
import matplotlib.pyplot as plt
import pyproj
import warnings
warnings.filterwarnings('ignore')

class OPERADisplacementProcessor:
    """OPERA Displacement Data Processor (H5 format) - Supports LOS to Vertical Conversion"""
    
    def __init__(self, input_folder, output_folder, shapefile_path=None, mask_flag=0):
        """
        Initialize the processor

        Parameters:
        input_folder: Input folder containing OPERA H5 files
        output_folder: Intermediate results output folder
        shapefile_path: Optional path to shapefile for validation/comparison
        mask_flag: Masking behavior control
            - 0: No mask applied
            - 1: Apply recommended mask
            - 0 < mask_flag < 1: Use as coherence threshold (mask pixels with coherence < mask_flag)
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.shapefile_path = Path(shapefile_path) if shapefile_path else None
        self.mask_flag = mask_flag

        # Create output folder
        self.output_folder.mkdir(exist_ok=True, parents=True)

        # Store processing result information
        self.processing_log = []
        self.shapefile_data = None

        print(f"Initialize OPERA processor (vertical conversion mode)")
        print(f"Input folder: {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Mask flag: {self.mask_flag}")
        if mask_flag == 0:
            print(f"  Masking: Disabled (no mask applied)")
        elif mask_flag == 1:
            print(f"  Masking: Recommended mask")
        elif 0 < mask_flag < 1:
            print(f"  Masking: Coherence threshold = {mask_flag}")
        else:
            print(f"  Warning: Invalid mask_flag value, using default (1)")
            self.mask_flag = 1

        # Load shapefile if provided
        if self.shapefile_path:
            self.load_shapefile()
    
    def extract_metadata_from_filename(self, filename):
        """Extract metadata information from filename"""
        metadata = {
            'filename': filename,
            'frame': None,
            'start_date': None,
            'end_date': None,
            'track': None
        }
        
        filename_str = str(filename)
        
        # Extract Frame information (F24455 format)
        frame_pattern = r'F(\d{5})'
        frame_match = re.search(frame_pattern, filename_str)
        if frame_match:
            metadata['frame'] = f"F{frame_match.group(1)}"
        
        # If F format not found, try T format (Track)
        if not metadata['frame']:
            track_pattern = r'T(\d{3})'
            track_match = re.search(track_pattern, filename_str)
            if track_match:
                metadata['track'] = f"T{track_match.group(1)}"
                metadata['frame'] = f"T{track_match.group(1)}"
        
        # Extract time information
        time_pattern = r'(\d{8}T\d{6}Z)'
        time_matches = re.findall(time_pattern, filename_str)
        
        if len(time_matches) >= 2:
            try:
                start_time = datetime.strptime(time_matches[0], '%Y%m%dT%H%M%SZ')
                end_time = datetime.strptime(time_matches[1], '%Y%m%dT%H%M%SZ')
                metadata['start_date'] = start_time.strftime('%Y%m%d')
                metadata['end_date'] = end_time.strftime('%Y%m%d')
            except ValueError as e:
                print(f"Time parsing error {filename}: {e}")
        
        # If the above method doesn't work, try other time formats
        if not metadata['start_date']:
            date_pattern = r'(\d{8})'
            date_matches = re.findall(date_pattern, filename_str)
            if len(date_matches) >= 2:
                metadata['start_date'] = date_matches[0]
                metadata['end_date'] = date_matches[1]
        
        return metadata

    def detect_coordinate_system(self, x_coords, y_coords, h5_data):
        """Detect the coordinate system using spatial_ref from H5 file"""
        print("\nDetecting coordinate system...")
        print("   OPERA DISP-S1: Frame-based products on projected UTM coordinate system")
        print("   Pixel spacing: 30x30 meters with 'pixel is area' convention")
        print("   Coordinates represent pixel centers following CF-conventions")

        # Get coordinate bounds
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        print(f"   X range: {x_min:.2f} to {x_max:.2f} (Easting)")
        print(f"   Y range: {y_min:.2f} to {y_max:.2f} (Northing)")

        # OPERA DISP-S1 products are ALWAYS on UTM projection according to specifications
        print("   → OPERA DISP-S1 confirmed: UTM projected coordinate system")

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        print(f"   Frame center: X={center_x:.0f}, Y={center_y:.0f}")

        # PRIORITY 1: Check spatial_ref dataset for CRS information (most reliable)
        estimated_zone = None
        if 'spatial_ref' in h5_data:
            print("   Checking spatial_ref dataset for CRS information...")
            spatial_ref = h5_data['spatial_ref']

            try:
                # Check for crs_wkt attribute
                if 'crs_wkt' in spatial_ref.attrs:
                    crs_wkt = spatial_ref.attrs['crs_wkt']
                    print(f"      Found crs_wkt in spatial_ref")

                    # Extract zone from WKT string or EPSG code
                    import re
                    crs_wkt_str = str(crs_wkt)

                    # Look for ALL EPSG codes in WKT using proper ID["EPSG",CODE] format
                    # The WKT contains multiple EPSG codes; we need the UTM zone one (32601-32760)
                    epsg_matches = re.findall(r'ID\["EPSG",(\d+)\]', crs_wkt_str)

                    if epsg_matches:
                        print(f"      Found EPSG codes in WKT: {epsg_matches}")

                        # Find the UTM zone EPSG code (32601-32760)
                        utm_epsg = None
                        for epsg_str in epsg_matches:
                            epsg_code = int(epsg_str)
                            if 32601 <= epsg_code <= 32660:  # UTM North
                                utm_epsg = epsg_code
                                break
                            elif 32701 <= epsg_code <= 32760:  # UTM South
                                utm_epsg = epsg_code
                                break

                        if utm_epsg:
                            print(f"      Selected UTM EPSG code: {utm_epsg}")

                            # Extract zone from EPSG code
                            if 32601 <= utm_epsg <= 32660:  # UTM North
                                estimated_zone = utm_epsg - 32600
                                print(f"      Extracted UTM Zone from EPSG: {estimated_zone}N")
                            elif 32701 <= utm_epsg <= 32760:  # UTM South
                                estimated_zone = utm_epsg - 32700
                                print(f"      Extracted UTM Zone from EPSG: {estimated_zone}S")
                        else:
                            print(f"      No UTM zone EPSG found in: {epsg_matches}")

                # Also check projected_crs_name attribute as fallback
                if 'projected_crs_name' in spatial_ref.attrs and estimated_zone is None:
                    crs_name = str(spatial_ref.attrs['projected_crs_name'])
                    print(f"      Found projected_crs_name: {crs_name}")

                    # Extract zone number from name (e.g., "WGS 84 / UTM zone 16N")
                    zone_match = re.search(r'zone\s*(\d{1,2})', crs_name.lower())
                    if zone_match:
                        estimated_zone = int(zone_match.group(1))
                        print(f"      Extracted UTM Zone from CRS name: {estimated_zone}")

            except Exception as e:
                print(f"      Error reading spatial_ref: {e}")

        # PRIORITY 2: Check metadata for UTM zone information
        if estimated_zone is None and 'metadata' in h5_data and h5_data['metadata']:
            metadata = h5_data['metadata']
            print("   Checking metadata for UTM zone information...")

            for key, value in metadata.items():
                key_str = str(key).lower()
                value_str = str(value).lower()

                # Look for UTM zone in metadata
                if any(keyword in key_str for keyword in ['utm', 'zone', 'epsg', 'crs', 'frame']):
                    print(f"      Found UTM info: {key} = {value}")

                    # Try to extract zone number from metadata
                    import re
                    # Look for patterns like "UTM 16N", "Zone 16", "EPSG:32616", etc.
                    zone_patterns = [
                        r'utm[\s]*(\d{1,2})[ns]?',  # UTM 16N, UTM16N
                        r'zone[\s]*(\d{1,2})',       # Zone 16
                        r'326(\d{2})',               # EPSG:32616 -> zone 16
                        r'327(\d{2})',               # EPSG:32716 -> zone 16 (south)
                        r'(\d{1,2})[ns]',            # 16N, 16S
                    ]

                    for pattern in zone_patterns:
                        zone_match = re.search(pattern, value_str)
                        if zone_match:
                            estimated_zone = int(zone_match.group(1))
                            print(f"      Extracted UTM zone from metadata: {estimated_zone}")
                            break

                    if estimated_zone:
                        break

        # PRIORITY 3 (FALLBACK): If no zone from H5 file, use shapefile as fallback
        if estimated_zone is None:
            print("   No UTM zone in H5 file, using shapefile as fallback...")
            estimated_zone = self._determine_utm_zone_from_shapefile(center_x, center_y)

        # Ensure valid UTM zone
        if estimated_zone:
            estimated_zone = max(1, min(60, estimated_zone))

            # Determine hemisphere from northing coordinate
            # UTM northings: 0-10,000,000m for Northern hemisphere
            #               0-10,000,000m for Southern hemisphere (but with different false northing)
            if center_y > 0 and center_y < 10000000:  # Northern hemisphere
                utm_crs = CRS.from_epsg(32600 + estimated_zone)
                print(f"   → OPERA DISP-S1 UTM Zone: {estimated_zone}N (EPSG:{32600 + estimated_zone})")
                print(f"   → 30m pixel spacing, pixel-area convention, UTM projection")
                return utm_crs, "utm"
            else:  # Southern hemisphere or edge case
                utm_crs = CRS.from_epsg(32700 + estimated_zone)
                print(f"   → OPERA DISP-S1 UTM Zone: {estimated_zone}S (EPSG:{32700 + estimated_zone})")
                print(f"   → 30m pixel spacing, pixel-area convention, UTM projection")
                return utm_crs, "utm"

        # Fallback - should rarely happen for OPERA DISP-S1 products
        print("   → Warning: Could not determine UTM zone")
        print("   → Using UTM Zone 16N as default for OPERA DISP-S1")
        return CRS.from_epsg(32616), "utm_default"

    def _determine_utm_zone_from_shapefile(self, center_x, center_y):
        """Determine UTM zone using inverse projection from shapefile CRS"""
        from rasterio.warp import transform as rio_transform
        import pyproj

        # If shapefile is available, use its CRS for proper zone determination
        if self.shapefile_data is not None:
            shapefile_crs = self.shapefile_data.crs
            print(f"      Using shapefile CRS: {shapefile_crs}")

            try:
                # Use shapefile center for UTM zone calculation (much more reliable than round-trip testing)
                shapefile_bounds = self.shapefile_data.total_bounds
                shapefile_center_lon = (shapefile_bounds[0] + shapefile_bounds[2]) / 2
                shapefile_center_lat = (shapefile_bounds[1] + shapefile_bounds[3]) / 2

                # Calculate UTM zone from shapefile center longitude using standard formula
                zone = int((shapefile_center_lon + 180) / 6) + 1
                zone = max(1, min(60, zone))  # Ensure valid range

                print(f"      → Shapefile center: ({shapefile_center_lon:.6f}, {shapefile_center_lat:.6f})")
                print(f"      → Calculated UTM zone: {zone}N")
                return zone

            except Exception as e:
                print(f"      → Shapefile center UTM zone calculation failed: {e}")

        # Fallback: Calculate UTM zone from longitude if we can estimate it
        # This is a simplified approach when shapefile method fails
        print("      Using simplified longitude-based estimation...")

        # Rough estimation: assume we're in continental US
        # UTM zone = floor((longitude + 180) / 6) + 1
        # For US: Zone 10 (west coast) to Zone 19 (east coast)

        # Estimate longitude from UTM easting (very rough approximation)
        # This is not ideal but provides a fallback
        estimated_lon = None

        # Try to get a geographic coordinate from Opera data for better estimation
        if self.shapefile_data is not None:
            try:
                # Get shapefile bounds in geographic coordinates
                shapefile_bounds = self.shapefile_data.total_bounds
                shapefile_center_lon = (shapefile_bounds[0] + shapefile_bounds[2]) / 2

                # Calculate UTM zone from shapefile center longitude
                zone = int((shapefile_center_lon + 180) / 6) + 1
                zone = max(1, min(60, zone))  # Ensure valid range

                print(f"      → Estimated from shapefile center longitude {shapefile_center_lon:.6f}: Zone {zone}")
                return zone

            except Exception as e:
                print(f"      → Shapefile bounds estimation failed: {e}")

        # Final fallback for continental US based on typical coordinate ranges
        # NOTE: This fallback is UNRELIABLE and should rarely be used!
        # UTM eastings vary significantly even within the same zone depending on location
        # This is kept only as last resort when no other information is available

        print("      → WARNING: Using unreliable fallback zone estimation")
        print("      → Shapefile should be provided for accurate zone detection!")

        # Very rough estimation based on central meridian
        # Assume center_x represents distance from central meridian
        # For zones 10-19 (US), typical easting ranges from ~155,000 to ~850,000

        if center_x < 200000:
            estimated_zone = 10  # Far west coast
        elif center_x < 300000:
            estimated_zone = 11  # West coast
        elif center_x < 400000:
            estimated_zone = 12  # Mountain west
        elif center_x < 500000:
            estimated_zone = 13  # Rocky Mountains
        elif center_x < 600000:
            estimated_zone = 14  # Central plains
        elif center_x < 750000:
            estimated_zone = 15  # Mississippi valley / Gulf Coast
        elif center_x < 850000:
            estimated_zone = 16  # East of Mississippi
        else:
            estimated_zone = 17  # Eastern seaboard

        print(f"      → Fallback zone estimation: {estimated_zone} (UNRELIABLE!)")
        print(f"      → Based on easting: {center_x:.0f}")
        print(f"      → RECOMMENDATION: Provide shapefile for accurate detection!")

        return estimated_zone

    def read_opera_h5(self, h5_path):
        """Read OPERA H5 file"""
        try:
            print(f"Reading H5 file: {h5_path.name}")

            h5_data = {
                'file': None,
                'displacement': None,
                'temporal_coherence': None,
                'mask': None,
                'metadata': {},
                'incidence_angles': {}
            }
            
            # Open H5 file
            h5_file = h5py.File(h5_path, 'r')
            h5_data['file'] = h5_file
            
            # Recursively explore H5 file structure
            def explore_h5_structure(group, path=""):
                items = []
                for key in group.keys():
                    item_path = f"{path}/{key}" if path else key
                    item = group[key]
                    
                    if isinstance(item, h5py.Group):
                        print(f"   Group: {item_path}")
                        items.extend(explore_h5_structure(item, item_path))
                    elif isinstance(item, h5py.Dataset):
                        items.append((item_path, item))
                
                return items
            
            print(f"   Exploring H5 file structure:")
            datasets = explore_h5_structure(h5_file)

            print(f"   All datasets found:")
            for path, dataset in datasets:
                print(f"      {path}: {dataset.shape} {dataset.dtype}")
            
            # Find required data layers
            displacement_dataset = None
            temporal_coherence_dataset = None
            mask_dataset = None
            x_coordinate = None
            y_coordinate = None
            far_range_incidence_angle = None
            near_range_incidence_angle = None
            spatial_ref_dataset = None

            print(f"   Searching for required data layers:")
            print(f"      - displacement")
            print(f"      - temporal_coherence")
            print(f"      - recommended_mask")
            print(f"      - x")
            print(f"      - y")
            print(f"      - spatial_ref (CRS information)")
            print(f"      - /identification/far_range_incidence_angle")
            print(f"      - /identification/near_range_incidence_angle")

            for path, dataset in datasets:
                layer_name = path.split('/')[-1]

                # Look for displacement data (can be in various paths)
                if layer_name == 'displacement':
                    displacement_dataset = dataset
                    print(f"   Found displacement data: {path}")

                elif layer_name == 'temporal_coherence':
                    temporal_coherence_dataset = dataset
                    print(f"   Found temporal_coherence data: {path}")

                elif layer_name == 'recommended_mask':
                    mask_dataset = dataset
                    print(f"   Found mask data: {path}")

                elif layer_name == 'x':
                    if path == 'x' or path.count('/') <= 1:
                        if x_coordinate is None:
                            x_coordinate = dataset
                            print(f"   Found X coordinates: {path}")

                elif layer_name == 'y':
                    if path == 'y' or path.count('/') <= 1:
                        if y_coordinate is None:
                            y_coordinate = dataset
                            print(f"   Found Y coordinates: {path}")

                elif layer_name == 'far_range_incidence_angle':
                    if 'identification' in path:
                        far_range_incidence_angle = dataset
                        print(f"   Found far range incidence angle: {path}")

                elif layer_name == 'near_range_incidence_angle':
                    if 'identification' in path:
                        near_range_incidence_angle = dataset
                        print(f"   Found near range incidence angle: {path}")

                elif layer_name == 'spatial_ref':
                    spatial_ref_dataset = dataset
                    print(f"   Found spatial_ref (CRS information): {path}")

            # Check required data
            missing_layers = []
            if displacement_dataset is None:
                missing_layers.append('displacement')
            if temporal_coherence_dataset is None:
                missing_layers.append('temporal_coherence')
            if mask_dataset is None:
                missing_layers.append('recommended_mask')
            if x_coordinate is None:
                missing_layers.append('x (root directory)')
            if y_coordinate is None:
                missing_layers.append('y (root directory)')

            if missing_layers:
                error_msg = f"Missing required data layers: {missing_layers}"
                print(f"   {error_msg}")
                raise ValueError(error_msg)
            
            # 读取全局属性
            for attr_name, attr_value in h5_file.attrs.items():
                h5_data['metadata'][attr_name] = attr_value
            
            # Store data
            h5_data['displacement'] = displacement_dataset
            h5_data['temporal_coherence'] = temporal_coherence_dataset
            h5_data['mask'] = mask_dataset
            h5_data['coordinates'] = {'x': x_coordinate, 'y': y_coordinate}

            # Store spatial_ref if available
            if spatial_ref_dataset is not None:
                h5_data['spatial_ref'] = spatial_ref_dataset
                print(f"   Spatial reference (CRS) information found")
            else:
                print(f"   Warning: spatial_ref not found - will use fallback zone detection")

            # Store incidence angles if available
            if far_range_incidence_angle is not None and near_range_incidence_angle is not None:
                h5_data['incidence_angles'] = {
                    'far_range': far_range_incidence_angle,
                    'near_range': near_range_incidence_angle
                }
                print(f"   Incidence angles found for LOS to vertical conversion")
            else:
                print(f"   Warning: Incidence angles not found - vertical conversion not available")

            print(f"   Basic data layers found!")

            # Analyze coordinate system
            x_coords_sample = x_coordinate[:10] if len(x_coordinate) > 10 else x_coordinate[:]
            y_coords_sample = y_coordinate[:10] if len(y_coordinate) > 10 else y_coordinate[:]

            detected_crs, crs_type = self.detect_coordinate_system(x_coords_sample, y_coords_sample, h5_data)
            h5_data['detected_crs'] = detected_crs
            h5_data['crs_type'] = crs_type

            return h5_data
            
        except Exception as e:
            print(f"Failed to read H5 file {h5_path.name}: {e}")
            return None
    def convert_to_wgs84(self, data_array, x_coords, y_coords, source_crs, target_crs=None):
        """Convert data from source CRS to target CRS (default: WGS84)"""
        if target_crs is None:
            target_crs = CRS.from_epsg(4326)  # WGS84

        print(f"\nConverting coordinates...")
        print(f"   Source CRS: {source_crs}")
        print(f"   Target CRS: {target_crs}")

        try:
            # Get source bounds
            if x_coords.ndim == 1 and y_coords.ndim == 1:
                # 1D coordinates
                west, east = float(np.min(x_coords)), float(np.max(x_coords))
                south, north = float(np.min(y_coords)), float(np.max(y_coords))
            else:
                # 2D coordinates
                west, east = float(np.min(x_coords)), float(np.max(x_coords))
                south, north = float(np.min(y_coords)), float(np.max(y_coords))

            print(f"   Source bounds: W={west:.6f}, E={east:.6f}, S={south:.6f}, N={north:.6f}")

            # Create source transform
            height, width = data_array.shape
            source_transform = from_bounds(west, south, east, north, width, height)

            # Calculate target transform and dimensions
            target_transform, target_width, target_height = calculate_default_transform(
                source_crs, target_crs, width, height, west, south, east, north
            )

            print(f"   Target dimensions: {target_width} x {target_height}")

            # Create target array
            target_array = np.full((target_height, target_width), np.nan, dtype=data_array.dtype)

            # Reproject data using nearest neighbor to preserve exact values and prevent interpolation artifacts
            reproject(
                source=data_array,
                destination=target_array,
                src_transform=source_transform,
                src_crs=source_crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
                src_nodata=np.nan,
                dst_nodata=np.nan
            )

            print(f"   Coordinate conversion completed")
            return target_array, target_transform, target_crs

        except Exception as e:
            print(f"   Coordinate conversion failed: {e}")
            print(f"   Using original coordinates")
            # Return original data with source transform
            height, width = data_array.shape
            west, east = float(np.min(x_coords)), float(np.max(x_coords))
            south, north = float(np.min(y_coords)), float(np.max(y_coords))
            original_transform = from_bounds(west, south, east, north, width, height)
            return data_array, original_transform, source_crs

    def apply_mask(self, displacement_data, mask_data, coherence_data):
        """Apply mask to displacement data based on mask_flag"""

        if self.mask_flag == 0:
            print("No mask applied (mask_flag = 0)")
            if hasattr(displacement_data, 'shape'):
                displacement = displacement_data[:]
            else:
                displacement = np.array(displacement_data)
            print(f"   Total pixels: {displacement.size}")
            return displacement

        elif self.mask_flag == 1:
            print("Apply recommended mask (mask_flag = 1)")
            if hasattr(displacement_data, 'shape'):
                displacement = displacement_data[:]
            else:
                displacement = np.array(displacement_data)

            if mask_data is not None:
                if hasattr(mask_data, 'shape'):
                    mask = mask_data[:]
                else:
                    mask = np.array(mask_data)

                if displacement.shape != mask.shape:
                    print(f"   Shape mismatch: displacement{displacement.shape} vs mask{mask.shape}")
                    if displacement.ndim == 3 and mask.ndim == 2:
                        mask = np.broadcast_to(mask, displacement.shape)
                    elif displacement.ndim == 2 and mask.ndim == 3:
                        mask = mask[0, :, :]

                masked_displacement = np.where(mask != 0, displacement, np.nan)
            else:
                print("   No mask data, using original displacement data")
                masked_displacement = displacement

            total_pixels = displacement.size
            valid_pixels = np.sum(~np.isnan(masked_displacement))
            valid_percentage = (valid_pixels / total_pixels) * 100

            print(f"   Total pixels: {total_pixels}")
            print(f"   Valid pixels: {valid_pixels}")
            print(f"   Valid rate: {valid_percentage:.2f}%")

            return masked_displacement

        elif 0 < self.mask_flag < 1:
            print(f"Apply coherence threshold mask (mask_flag = {self.mask_flag})")

            if hasattr(displacement_data, 'shape'):
                displacement = displacement_data[:]
            else:
                displacement = np.array(displacement_data)

            if hasattr(coherence_data, 'shape'):
                coherence = coherence_data[:]
            else:
                coherence = np.array(coherence_data)

            # Check shape compatibility
            if displacement.shape != coherence.shape:
                print(f"   Shape mismatch: displacement{displacement.shape} vs coherence{coherence.shape}")
                if displacement.ndim == 3 and coherence.ndim == 2:
                    coherence = np.broadcast_to(coherence, displacement.shape)
                elif displacement.ndim == 2 and coherence.ndim == 3:
                    coherence = coherence[0, :, :]

            # Mask pixels where coherence < threshold
            masked_displacement = np.where(coherence >= self.mask_flag, displacement, np.nan)

            total_pixels = displacement.size
            valid_pixels = np.sum(~np.isnan(masked_displacement))
            valid_percentage = (valid_pixels / total_pixels) * 100

            print(f"   Coherence threshold: {self.mask_flag}")
            print(f"   Total pixels: {total_pixels}")
            print(f"   Valid pixels (coherence >= {self.mask_flag}): {valid_pixels}")
            print(f"   Valid rate: {valid_percentage:.2f}%")

            return masked_displacement

        else:
            print(f"Invalid mask_flag value: {self.mask_flag}, using no mask")
            if hasattr(displacement_data, 'shape'):
                displacement = displacement_data[:]
            else:
                displacement = np.array(displacement_data)
            return displacement

    def apply_mask_to_temporal_coherence(self, temporal_coherence_data, mask_data):
        """Apply mask to temporal coherence data based on mask_flag"""

        if self.mask_flag == 0:
            print("No mask applied to temporal coherence (mask_flag = 0)")
            if hasattr(temporal_coherence_data, 'shape'):
                coherence = temporal_coherence_data[:]
            else:
                coherence = np.array(temporal_coherence_data)
            print(f"   Total pixels: {coherence.size}")
            return coherence

        elif self.mask_flag == 1:
            print("Apply recommended mask to temporal coherence (mask_flag = 1)")
            if hasattr(temporal_coherence_data, 'shape'):
                coherence = temporal_coherence_data[:]
            else:
                coherence = np.array(temporal_coherence_data)

            if mask_data is not None:
                if hasattr(mask_data, 'shape'):
                    mask = mask_data[:]
                else:
                    mask = np.array(mask_data)

                if coherence.shape != mask.shape:
                    print(f"   Shape mismatch: temporal_coherence{coherence.shape} vs mask{mask.shape}")
                    if coherence.ndim == 3 and mask.ndim == 2:
                        mask = np.broadcast_to(mask, coherence.shape)
                    elif coherence.ndim == 2 and mask.ndim == 3:
                        mask = mask[0, :, :]

                masked_coherence = np.where(mask != 0, coherence, np.nan)
            else:
                print("   No mask data, using original temporal coherence data")
                masked_coherence = coherence

            total_pixels = coherence.size
            valid_pixels = np.sum(~np.isnan(masked_coherence))
            valid_percentage = (valid_pixels / total_pixels) * 100

            print(f"   Total pixels: {total_pixels}")
            print(f"   Valid pixels: {valid_pixels}")
            print(f"   Valid rate: {valid_percentage:.2f}%")

            return masked_coherence

        elif 0 < self.mask_flag < 1:
            print(f"Apply coherence threshold mask to temporal coherence (mask_flag = {self.mask_flag})")

            if hasattr(temporal_coherence_data, 'shape'):
                coherence = temporal_coherence_data[:]
            else:
                coherence = np.array(temporal_coherence_data)

            # Mask pixels where coherence < threshold
            masked_coherence = np.where(coherence >= self.mask_flag, coherence, np.nan)

            total_pixels = coherence.size
            valid_pixels = np.sum(~np.isnan(masked_coherence))
            valid_percentage = (valid_pixels / total_pixels) * 100

            print(f"   Coherence threshold: {self.mask_flag}")
            print(f"   Total pixels: {total_pixels}")
            print(f"   Valid pixels (coherence >= {self.mask_flag}): {valid_pixels}")
            print(f"   Valid rate: {valid_percentage:.2f}%")

            return masked_coherence

        else:
            print(f"Invalid mask_flag value: {self.mask_flag}, using no mask")
            if hasattr(temporal_coherence_data, 'shape'):
                coherence = temporal_coherence_data[:]
            else:
                coherence = np.array(temporal_coherence_data)
            return coherence


    def save_dual_band_geotiff(self, masked_displacement, masked_coherence, coordinates, output_path, displacement_type="Vertical displacement", detected_crs=None, convert_to_wgs84=True):
        """Save both displacement and temporal coherence as separate bands in a single GeoTIFF file"""
        print(f"Save dual-band GeoTIFF: {displacement_type} and temporal coherence to {output_path}")

        try:
            # Ensure both datasets are 2D
            if masked_displacement.ndim == 3:
                masked_displacement = masked_displacement[0, :, :]
            elif masked_displacement.ndim == 1:
                print("Displacement data is 1D, cannot process")
                return False

            if masked_coherence.ndim == 3:
                masked_coherence = masked_coherence[0, :, :]
            elif masked_coherence.ndim == 1:
                print("Temporal coherence data is 1D, cannot process")
                return False

            # Check that both datasets have the same shape
            if masked_displacement.shape != masked_coherence.shape:
                print(f"Shape mismatch: displacement{masked_displacement.shape} vs coherence{masked_coherence.shape}")
                return False

            if 'x' not in coordinates or 'y' not in coordinates:
                print("Missing x and y coordinate information")
                return False

            x_coords = coordinates['x'][:]
            y_coords = coordinates['y'][:]

            print(f"   Coordinate information:")
            print(f"      X coordinate shape: {x_coords.shape}")
            print(f"      Y coordinate shape: {y_coords.shape}")
            print(f"      X range: {np.min(x_coords):.2f} ~ {np.max(x_coords):.2f}")
            print(f"      Y range: {np.min(y_coords):.2f} ~ {np.max(y_coords):.2f}")
            print(f"      Data shape: {masked_displacement.shape}")

            # Calculate bounds using OPERA pixel-area convention
            if x_coords.ndim == 1 and y_coords.ndim == 1:
                print(f"   1D coordinate array, creating boundaries")

                if len(x_coords) == masked_displacement.shape[1] and len(y_coords) == masked_displacement.shape[0]:
                    print(f"   Standard coordinate correspondence")
                    x_1d, y_1d = x_coords, y_coords
                elif len(x_coords) == masked_displacement.shape[0] and len(y_coords) == masked_displacement.shape[1]:
                    print(f"   Coordinates need to be swapped")
                    x_1d, y_1d = y_coords, x_coords
                else:
                    print(f"   Coordinate dimensions do not match data")
                    return False

                # OPERA uses "pixel is area" convention with coordinates at pixel center
                pixel_size_x = abs(x_1d[1] - x_1d[0]) if len(x_1d) > 1 else 30.0
                pixel_size_y = abs(y_1d[1] - y_1d[0]) if len(y_1d) > 1 else 30.0

                west = float(np.min(x_1d)) - pixel_size_x / 2
                east = float(np.max(x_1d)) + pixel_size_x / 2
                south = float(np.min(y_1d)) - pixel_size_y / 2
                north = float(np.max(y_1d)) + pixel_size_y / 2

                print(f"   OPERA pixel-area convention applied:")
                print(f"      Pixel size: {pixel_size_x:.1f}m x {pixel_size_y:.1f}m")

            elif x_coords.ndim == 2 and y_coords.ndim == 2:
                print(f"   2D coordinate array")
                if x_coords.shape == masked_displacement.shape and y_coords.shape == masked_displacement.shape:
                    # For 2D coordinates, estimate pixel size from coordinate spacing
                    if x_coords.shape[1] > 1:
                        pixel_size_x = abs(x_coords[0, 1] - x_coords[0, 0])
                    else:
                        pixel_size_x = 30.0

                    if y_coords.shape[0] > 1:
                        pixel_size_y = abs(y_coords[1, 0] - y_coords[0, 0])
                    else:
                        pixel_size_y = 30.0

                    # Apply pixel-area convention
                    west = float(np.min(x_coords)) - pixel_size_x / 2
                    east = float(np.max(x_coords)) + pixel_size_x / 2
                    south = float(np.min(y_coords)) - pixel_size_y / 2
                    north = float(np.max(y_coords)) + pixel_size_y / 2

                    print(f"   OPERA pixel-area convention applied:")
                    print(f"      Pixel size: {pixel_size_x:.1f}m x {pixel_size_y:.1f}m")
                else:
                    print(f"   2D coordinate shape does not match data")
                    return False
            else:
                print(f"   Unsupported coordinate dimension combination")
                return False

            print(f"   Data boundaries:")
            print(f"      West: {west:.6f}")
            print(f"      East: {east:.6f}")
            print(f"      South: {south:.6f}")
            print(f"      North: {north:.6f}")

            # Handle CRS conversion if needed
            if detected_crs is not None and convert_to_wgs84:
                print(f"   Converting to WGS84...")
                try:
                    # Convert displacement data
                    converted_displacement, final_transform, final_crs = self.convert_to_wgs84(
                        masked_displacement, x_coords, y_coords, detected_crs
                    )
                    # Convert coherence data using same parameters
                    converted_coherence, _, _ = self.convert_to_wgs84(
                        masked_coherence, x_coords, y_coords, detected_crs
                    )

                    displacement_array = converted_displacement
                    coherence_array = converted_coherence
                    transform = final_transform
                    crs = final_crs
                    print(f"   Successfully converted to WGS84")
                except Exception as e:
                    print(f"   CRS conversion failed: {e}")
                    print(f"   Using original coordinate system")
                    height, width = masked_displacement.shape
                    transform = from_bounds(west, south, east, north, width, height)
                    crs = detected_crs
                    displacement_array = masked_displacement
                    coherence_array = masked_coherence
            else:
                # Use original coordinate system
                height, width = masked_displacement.shape
                transform = from_bounds(west, south, east, north, width, height)
                crs = detected_crs
                displacement_array = masked_displacement
                coherence_array = masked_coherence

            # Convert NaN to -9999 only at final write stage for GeoTIFF compatibility
            final_displacement = np.where(np.isnan(displacement_array), -9999, displacement_array)
            final_coherence = np.where(np.isnan(coherence_array), -9999, coherence_array)

            # Write the dual-band GeoTIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=final_displacement.shape[0],
                width=final_displacement.shape[1],
                count=2,  # Two bands
                dtype=final_displacement.dtype,
                transform=transform,
                crs=crs,
                nodata=-9999,
                compress='lzw'
            ) as dst:
                # Write displacement as band 1
                dst.write(final_displacement, 1)
                dst.set_band_description(1, f'OPERA {displacement_type}')

                # Write temporal coherence as band 2
                dst.write(final_coherence, 2)
                dst.set_band_description(2, 'OPERA Temporal Coherence')

                # Set file-level tags
                dst.update_tags(
                    DESCRIPTION=f'OPERA {displacement_type} and Temporal Coherence',
                    BAND_1=f'{displacement_type}',
                    BAND_2='Temporal Coherence',
                    PROCESSING_DATE=datetime.now().isoformat()
                )

            print(f"Dual-band GeoTIFF save completed: {output_path}")
            print(f"   Band 1: {displacement_type}")
            print(f"   Band 2: Temporal Coherence")

            return True

        except Exception as e:
            print(f"Dual-band GeoTIFF save failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # NOTE: save_masked_data_as_geotiff() function has been removed (deprecated, never used)
    # All processing now uses save_dual_band_geotiff() for dual-band GeoTIFF output

    def convert_los_to_vertical(self, los_displacement_data, incidence_angles):
        """Convert LOS displacement to vertical displacement using incidence angle interpolation"""
        print("Converting LOS displacement to vertical displacement")

        try:
            if 'far_range' not in incidence_angles or 'near_range' not in incidence_angles:
                print("   Error: Missing incidence angle data")
                return None, None

            far_range_angle = float(incidence_angles['far_range'][()])
            near_range_angle = float(incidence_angles['near_range'][()])

            print(f"   Far range incidence angle: {far_range_angle:.3f} degrees")
            print(f"   Near range incidence angle: {near_range_angle:.3f} degrees")

            # Convert degrees to radians for calculation
            far_range_rad = np.deg2rad(far_range_angle)
            near_range_rad = np.deg2rad(near_range_angle)

            # Get data dimensions (rows = range direction, columns = azimuth direction)
            num_rows, num_cols = los_displacement_data.shape
            print(f"   Data dimensions: {num_rows} rows x {num_cols} columns")
            print(f"   Interpolating incidence angles across {num_rows} rows (range direction)")

            # Create incidence angle array for each row
            # First row (near range) gets near_range_angle
            # Last row (far range) gets far_range_angle
            # Intermediate rows get linearly interpolated angles
            incidence_angle_array = np.zeros((num_rows, num_cols))

            for row in range(num_rows):
                # Linear interpolation: row 0 = near range, row (num_rows-1) = far range
                interpolation_factor = row / (num_rows - 1) if num_rows > 1 else 0
                interpolated_angle_rad = near_range_rad + interpolation_factor * (far_range_rad - near_range_rad)

                # Same incidence angle for all columns in the same row (azimuth direction)
                incidence_angle_array[row, :] = interpolated_angle_rad

            print(f"   Row 0 (near range) angle: {np.rad2deg(incidence_angle_array[0, 0]):.3f} degrees")
            print(f"   Row {num_rows-1} (far range) angle: {np.rad2deg(incidence_angle_array[-1, 0]):.3f} degrees")

            # Calculate vertical displacement: vertical = los / cos(incidence_angle)
            print("   Calculating vertical displacement: vertical = los / cos(incidence_angle)")

            # Avoid division by zero or very small cosine values
            cos_incidence = np.cos(incidence_angle_array)

            # Check for problematic angles (close to 90 degrees)
            min_cos_threshold = 0.01  # corresponds to ~89.4 degrees
            problematic_mask = np.abs(cos_incidence) < min_cos_threshold
            problematic_count = np.sum(problematic_mask)

            if problematic_count > 0:
                print(f"   Warning: {problematic_count} pixels have incidence angles close to 90° (problematic for conversion)")

            # Calculate vertical displacement
            vertical_displacement = los_displacement_data / cos_incidence

            # Set problematic pixels to NaN
            vertical_displacement[problematic_mask] = np.nan

            print(f"   LOS range: {np.nanmin(los_displacement_data):.6f} ~ {np.nanmax(los_displacement_data):.6f}")
            print(f"   Vertical range: {np.nanmin(vertical_displacement):.6f} ~ {np.nanmax(vertical_displacement):.6f}")

            # Calculate amplification factor for reference
            mean_cos_incidence = np.nanmean(cos_incidence)
            amplification_factor = 1 / mean_cos_incidence
            print(f"   Average amplification factor: {amplification_factor:.3f}")

            return vertical_displacement, incidence_angle_array

        except Exception as e:
            print(f"   LOS to vertical conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def process_single_file(self, h5_path):
        """Process single OPERA H5 file"""
        print(f"\n{'='*60}")
        print(f"Processing file: {h5_path.name}")
        print(f"{'='*60}")
        
        metadata = self.extract_metadata_from_filename(h5_path.name)
        
        h5_data = self.read_opera_h5(h5_path)
        if h5_data is None:
            return {'status': 'failed', 'error': 'Cannot read H5 file'}
        
        try:
            if h5_data['displacement'] is None:
                raise ValueError('displacement data layer not found')

            if h5_data['temporal_coherence'] is None:
                raise ValueError('temporal_coherence data layer not found')

            if h5_data['mask'] is None:
                raise ValueError('recommended_mask data layer not found')

            if not h5_data['coordinates'] or 'x' not in h5_data['coordinates'] or 'y' not in h5_data['coordinates']:
                raise ValueError('x and y coordinate data layers not found')

            displacement_dataset = h5_data['displacement']
            temporal_coherence_dataset = h5_data['temporal_coherence']
            mask_dataset = h5_data['mask']
            coordinates = h5_data['coordinates']
            
            print(f"Displacement data shape: {displacement_dataset.shape}")
            print(f"Displacement data type: {displacement_dataset.dtype}")
            print(f"Temporal coherence data shape: {temporal_coherence_dataset.shape}")
            print(f"Temporal coherence data type: {temporal_coherence_dataset.dtype}")

            # Check dataset attributes for nodata/fill values
            # if hasattr(displacement_dataset, 'attrs'):
            #     print(f"Displacement dataset attributes: {dict(displacement_dataset.attrs)}")

            # if hasattr(temporal_coherence_dataset, 'attrs'):
            #     print(f"Temporal coherence dataset attributes: {dict(temporal_coherence_dataset.attrs)}")

            los_displacement_data = displacement_dataset[:]
            temporal_coherence_data = temporal_coherence_dataset[:]
            print(f"Displacement matrix extracted - shape: {los_displacement_data.shape}, dtype: {los_displacement_data.dtype}")
            print(f"Temporal coherence matrix extracted - shape: {temporal_coherence_data.shape}, dtype: {temporal_coherence_data.dtype}")

            # Check for nodata values and valid data
            total_pixels = los_displacement_data.size
            nan_count = np.sum(np.isnan(los_displacement_data))
            inf_count = np.sum(np.isinf(los_displacement_data))
            zero_count = np.sum(los_displacement_data == 0)

            print(f"Data analysis:")
            print(f"   Total pixels: {total_pixels}")
            print(f"   NaN pixels: {nan_count}")
            print(f"   Inf pixels: {inf_count}")
            print(f"   Zero pixels: {zero_count}")

            # Look for common OPERA nodata values
            common_nodata_values = [-9999, -32767, -32768, 9999, 32767]
            for nodata_val in common_nodata_values:
                count = np.sum(los_displacement_data == nodata_val)
                if count > 0:
                    print(f"   {nodata_val} pixels: {count}")

            # Show some sample values
            # print(f"   Sample values (first 5x5): ")
            # if len(los_displacement_data.shape) >= 2:
            #     print(los_displacement_data[:5, :5])
            # else:
            #     print(los_displacement_data[:10])

            # Calculate range only for non-infinite, non-NaN values
            finite_mask = np.isfinite(los_displacement_data)
            finite_data = los_displacement_data[finite_mask]

            if len(finite_data) > 0:
                print(f"STEP 1 - Original LOS data range: {np.min(finite_data):.6f} ~ {np.max(finite_data):.6f}")
                print(f"   Valid (finite) pixels: {len(finite_data)} ({len(finite_data)/total_pixels*100:.2f}%)")
            else:
                print(f"STEP 1 - WARNING: No finite displacement data found!")
                print(f"   All {total_pixels} pixels are NaN, Inf, or other invalid values")

            # Check for interpolation artifacts in original data
            problematic_mask = (los_displacement_data < -9998) & (los_displacement_data > -9999) & ~np.isnan(los_displacement_data)
            problematic_count = np.sum(problematic_mask)
            if problematic_count > 0:
                print(f"   Warning: Found {problematic_count} potential interpolation artifacts in original data")
                print(f"   Sample problematic values: {los_displacement_data[problematic_mask][:5]}")

            # Check if vertical conversion is possible
            incidence_angles = h5_data.get('incidence_angles', {})
            if incidence_angles and 'far_range' in incidence_angles and 'near_range' in incidence_angles:
                print(f"STEP 2 - Converting LOS to vertical displacement")
                vertical_displacement_data, incidence_angle_array = self.convert_los_to_vertical(
                    los_displacement_data, incidence_angles
                )

                if vertical_displacement_data is not None:
                    print(f"STEP 2 - Vertical conversion successful")
                    final_displacement_data = vertical_displacement_data
                    data_type = "Vertical displacement"
                else:
                    print(f"STEP 2 - Vertical conversion failed, using LOS data")
                    final_displacement_data = los_displacement_data
                    data_type = "LOS displacement"
            else:
                print(f"STEP 2 - Incidence angles not available, using LOS displacement data")
                final_displacement_data = los_displacement_data
                data_type = "LOS displacement"
            
            # apply mask to both displacement and temporal coherence based on mask_flag
            print(f"Mask data shape: {mask_dataset.shape}")
            masked_displacement = self.apply_mask(final_displacement_data, mask_dataset, temporal_coherence_data)
            masked_coherence = self.apply_mask_to_temporal_coherence(temporal_coherence_data, mask_dataset)

            if masked_displacement is None:
                raise ValueError('Displacement mask application failed')

            if masked_coherence is None:
                raise ValueError('Temporal coherence mask application failed')
            
            # Generate output filename
            frame_info = metadata['frame'] or 'unknown'
            start_date = metadata['start_date'] or 'unknown'
            end_date = metadata['end_date'] or 'unknown'

            # Generate mask type suffix for filename
            if self.mask_flag == 0:
                mask_suffix = "UNMASKED"
            elif self.mask_flag == 1:
                mask_suffix = "MASKED"
            else:
                mask_suffix = f"COH{str(self.mask_flag).replace('.', '')}"

            # Generate output filename for dual-band file
            if "Vertical" in data_type:
                output_filename = f"OPERA_VERTICAL_COHERENCE_{mask_suffix}_REPROJECTED_{frame_info}_{start_date}_{end_date}.tif"
            else:
                output_filename = f"OPERA_LOS_COHERENCE_{mask_suffix}_REPROJECTED_{frame_info}_{start_date}_{end_date}.tif"

            output_path = self.output_folder / output_filename
            print(f"Output file: {output_filename}")

            # Save both displacement and temporal coherence as dual-band GeoTIFF
            detected_crs = h5_data.get('detected_crs', None)
            success = self.save_dual_band_geotiff(
                masked_displacement, masked_coherence, coordinates, output_path,
                displacement_type=data_type, detected_crs=detected_crs, convert_to_wgs84=True
            )

            # Additional shapefile analysis if available
            
            # Record processing results
            result = {
                'status': 'success' if success else 'failed',
                'input_file': str(h5_path),
                'output_file': str(output_path) if success else None,
                'metadata': metadata,
                'displacement_shape': displacement_dataset.shape,
                'temporal_coherence_shape': temporal_coherence_dataset.shape,
                'mask_available': mask_dataset is not None,
                'coordinates_available': list(coordinates.keys()),
                'data_type': data_type,
                'has_temporal_coherence': True,
                'processing_time': datetime.now().isoformat()
            }
            
            if success:
                print("File processing completed")
                print(f"   Final data type: {data_type}")
                print(f"   Temporal coherence included: Yes")
                print(f"   Output format: Dual-band GeoTIFF")
            else:
                print("File processing failed")
                
            return result
            
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e), 'input_file': str(h5_path)}
        
        finally:
            # close H5 file
            if h5_data and h5_data['file'] is not None:
                h5_data['file'].close()
    
    def process_all_files(self):
        """process all OPERA H5 files in the input folder"""
        print(f"\nStarting batch processing of OPERA files...")
        print(f"Search folder: {self.input_folder}")
        
        # look for .h5 and .hdf5 files
        h5_files = list(self.input_folder.glob("*.h5"))
        hdf5_files = list(self.input_folder.glob("*.hdf5"))
        all_files = h5_files + hdf5_files
        
        if not all_files:
            print("No H5 or HDF5 files found")
            return []
        
        print(f"Found {len(all_files)} H5 files")
        
        # process each file
        for i, h5_file in enumerate(all_files, 1):
            print(f"\nProgress: {i}/{len(all_files)}")
            
            result = self.process_single_file(h5_file)
            self.processing_log.append(result)
        
        # generate processing report
        self.generate_processing_report()
        
        print(f"\nBatch processing completed!")
        print(f"Processing report: {self.output_folder / 'processing_report.txt'}")
        
        return self.processing_log
    
    def generate_processing_report(self):
        """Generate processing report"""
        report_path = self.output_folder / 'processing_report.txt'
        
        successful = [r for r in self.processing_log if r['status'] == 'success']
        failed = [r for r in self.processing_log if r['status'] == 'failed']
        
        # Count processing statistics
        los_processed = [r for r in successful if 'LOS' in r.get('data_type', '')]
        vertical_processed = [r for r in successful if 'Vertical' in r.get('data_type', '')]
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("OPERA Displacement Data Processing Report (LOS and Vertical Mode)\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input folder: {self.input_folder}\n")
            f.write(f"Output folder: {self.output_folder}\n\n")
            
            f.write(f"Processing statistics:\n")
            f.write(f"  Total files: {len(self.processing_log)}\n")
            f.write(f"  Successfully processed: {len(successful)}\n")
            f.write(f"  Failed processing: {len(failed)}\n")
            f.write(f"  Success rate: {len(successful)/len(self.processing_log)*100:.1f}%\n\n")
            
            f.write(f"Data type statistics:\n")
            f.write(f"  LOS displacement processed: {len(los_processed)}\n")
            f.write(f"  Vertical displacement processed: {len(vertical_processed)}\n\n")
            
            if successful:
                f.write("Successfully processed files:\n")
                f.write("-" * 30 + "\n")
                for result in successful:
                    metadata = result['metadata']
                    f.write(f"Input: {Path(result['input_file']).name}\n")
                    f.write(f"Output: {Path(result['output_file']).name}\n")
                    f.write(f"Frame: {metadata['frame']}\n")
                    f.write(f"Time: {metadata['start_date']} - {metadata['end_date']}\n")
                    f.write(f"Data shape: {result['displacement_shape']}\n")
                    f.write(f"Data type: {result.get('data_type', 'Unknown')}\n")
                    f.write(f"Mask available: {result['mask_available']}\n\n")
            
            if failed:
                f.write("Failed processing files:\n")
                f.write("-" * 30 + "\n")
                for result in failed:
                    f.write(f"File: {Path(result['input_file']).name}\n")
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n\n")
        
        print(f"Processing report generated: {report_path}")
    
    def get_frame_summary(self):
        """Get Frame summary information"""
        frame_summary = {}
        
        for result in self.processing_log:
            if result['status'] == 'success':
                metadata = result['metadata']
                frame = metadata['frame']
                
                if frame not in frame_summary:
                    frame_summary[frame] = {
                        'files': [],
                        'time_ranges': [],
                        'file_count': 0,
                        'los_processed': 0
                    }
                
                frame_summary[frame]['files'].append(Path(result['input_file']).name)
                frame_summary[frame]['time_ranges'].append(
                    f"{metadata['start_date']}-{metadata['end_date']}"
                )
                frame_summary[frame]['file_count'] += 1
                
                # Count processing type
                if 'LOS' in result.get('data_type', ''):
                    frame_summary[frame]['los_processed'] += 1
                elif 'Vertical' in result.get('data_type', ''):
                    if 'vertical_processed' not in frame_summary[frame]:
                        frame_summary[frame]['vertical_processed'] = 0
                    frame_summary[frame]['vertical_processed'] += 1
        
        return frame_summary

    def load_shapefile(self):
        """Load and analyze the comparison shapefile"""
        try:
            print(f"\nLoading shapefile: {self.shapefile_path}")

            # Find the .shp file if directory is provided
            if self.shapefile_path.is_dir():
                shp_files = list(self.shapefile_path.glob("*.shp"))
                if not shp_files:
                    print("   No .shp files found in directory")
                    return
                self.shapefile_path = shp_files[0]
                print(f"   Using: {self.shapefile_path.name}")

            # Load the shapefile
            self.shapefile_data = gpd.read_file(self.shapefile_path)

            print(f"   Shapefile info:")
            print(f"      Features: {len(self.shapefile_data)}")
            print(f"      CRS: {self.shapefile_data.crs}")
            print(f"      Bounds: {self.shapefile_data.total_bounds}")
            print(f"      Columns: {list(self.shapefile_data.columns)}")

            # Check if shapefile has displacement-related columns
            displacement_cols = [col for col in self.shapefile_data.columns
                               if any(term in col.lower() for term in ['displ', 'deform', 'subsid', 'uplift', 'vertical'])]

            if displacement_cols:
                print(f"      Potential displacement columns: {displacement_cols}")

            return True

        except Exception as e:
            print(f"   Failed to load shapefile: {e}")
            self.shapefile_data = None
            return False


# Data analysis tools
def analyze_masked_geotiff(tif_file_path):
    """Analyze masked GeoTIFF data (supports both single-band and dual-band files)"""
    try:
        print(f"Analyzing GeoTIFF: {tif_file_path}")

        with rasterio.open(tif_file_path) as src:
            transform = src.transform
            bounds = src.bounds
            crs = src.crs
            band_count = src.count

            # Use correct method to read tags
            tags = src.tags()
            data_type = tags.get('DATA_TYPE', "Unknown type")
            description = tags.get('DESCRIPTION', "No description")

            print(f"   File information:")
            print(f"      Data type: {data_type}")
            print(f"      Description: {description}")
            print(f"      Number of bands: {band_count}")
            print(f"      Coordinate system: {crs if crs else 'Not specified'}")
            print(f"      Boundaries: {bounds}")
            print(f"      Pixel size: {transform.a:.6f} × {-transform.e:.6f}")

            if band_count == 1:
                # Single band analysis (legacy files)
                data = src.read(1, masked=True)
                print(f"      Data shape: {data.shape}")

                valid_mask = ~data.mask if hasattr(data, 'mask') else ~np.isnan(data)
                valid_data = data.compressed() if hasattr(data, 'compressed') else data[valid_mask]

                print(f"   Data statistics:")
                print(f"      Total pixels: {data.size}")
                print(f"      Valid pixels: {len(valid_data)}")
                print(f"      Valid rate: {len(valid_data)/data.size*100:.2f}%")

                if len(valid_data) > 0:
                    print(f"      Value range: {np.min(valid_data):.3f} ~ {np.max(valid_data):.3f}")
                    print(f"      Average value: {np.mean(valid_data):.3f}")
                    print(f"      Standard deviation: {np.std(valid_data):.3f}")

                return data

            elif band_count == 2:
                # Dual band analysis (new files with displacement + coherence)
                displacement_data = src.read(1, masked=True)
                coherence_data = src.read(2, masked=True)

                print(f"      Data shape: {displacement_data.shape}")

                # Get band descriptions
                band1_desc = src.get_band_description(1)
                band2_desc = src.get_band_description(2)
                print(f"      Band 1: {band1_desc}")
                print(f"      Band 2: {band2_desc}")

                # Analyze displacement band (Band 1)
                valid_mask_disp = ~displacement_data.mask if hasattr(displacement_data, 'mask') else ~np.isnan(displacement_data)
                valid_data_disp = displacement_data.compressed() if hasattr(displacement_data, 'compressed') else displacement_data[valid_mask_disp]

                print(f"   Displacement statistics (Band 1):")
                print(f"      Total pixels: {displacement_data.size}")
                print(f"      Valid pixels: {len(valid_data_disp)}")
                print(f"      Valid rate: {len(valid_data_disp)/displacement_data.size*100:.2f}%")

                if len(valid_data_disp) > 0:
                    print(f"      Displacement range: {np.min(valid_data_disp):.3f} ~ {np.max(valid_data_disp):.3f}")
                    print(f"      Average displacement: {np.mean(valid_data_disp):.3f}")
                    print(f"      Standard deviation: {np.std(valid_data_disp):.3f}")

                # Analyze coherence band (Band 2)
                valid_mask_coh = ~coherence_data.mask if hasattr(coherence_data, 'mask') else ~np.isnan(coherence_data)
                valid_data_coh = coherence_data.compressed() if hasattr(coherence_data, 'compressed') else coherence_data[valid_mask_coh]

                print(f"   Temporal Coherence statistics (Band 2):")
                print(f"      Total pixels: {coherence_data.size}")
                print(f"      Valid pixels: {len(valid_data_coh)}")
                print(f"      Valid rate: {len(valid_data_coh)/coherence_data.size*100:.2f}%")

                if len(valid_data_coh) > 0:
                    print(f"      Coherence range: {np.min(valid_data_coh):.3f} ~ {np.max(valid_data_coh):.3f}")
                    print(f"      Average coherence: {np.mean(valid_data_coh):.3f}")
                    print(f"      Standard deviation: {np.std(valid_data_coh):.3f}")

                return displacement_data, coherence_data

            else:
                print(f"   Unsupported number of bands: {band_count}")
                return None

    except Exception as e:
        print(f"Analysis failed: {e}")
        return None

def batch_analyze_results(output_folder):
    """Batch analyze processing results"""
    output_path = Path(output_folder)

    # Updated patterns for dual-band files
    los_files = list(output_path.glob("OPERA_LOS_COHERENCE_MASKED_*.tif"))
    vertical_files = list(output_path.glob("OPERA_VERTICAL_COHERENCE_MASKED_*.tif"))

    # Also look for legacy single-band files for backward compatibility
    legacy_los_files = list(output_path.glob("OPERA_LOS_MASKED_*.tif"))
    legacy_vertical_files = list(output_path.glob("OPERA_VERTICAL_MASKED_*.tif"))

    all_files = los_files + vertical_files + legacy_los_files + legacy_vertical_files

    if not all_files:
        print("No processing result files found")
        return

    print(f"Batch analyze processing results")
    print(f"   LOS displacement + coherence files: {len(los_files)} files")
    print(f"   Vertical displacement + coherence files: {len(vertical_files)} files")
    print(f"   Legacy single-band files: {len(legacy_los_files + legacy_vertical_files)} files")
    print(f"   Total: {len(all_files)} files")

    for tif_file in all_files:
        print(f"\n{'='*40}")
        analyze_masked_geotiff(tif_file)

# Usage examples and main function
def main():
    """Main function: demonstrate how to use OPERA processor"""
    print("OPERA Displacement Mask, Reprojection and Subset Processing Tool")
    print("=" * 50)

    # Configure paths
    input_folder = r'D:\project\InSAR OPERA\Bay\original_data\raw_files'
    output_folder = r'D:\project\InSAR OPERA\Bay\Vertical-Mask-Reproject'
    shapefile_path = r'E:\UH\graduate class\Applied Gepspatial Computations\data\CoastalCounties\counties\Bay'
    mask_flag = 0  # 0: no mask, 1: recommended mask, 0-1: coherence threshold

    print("Features:")
    print("1. Read displacement, temporal_coherence, recommended_mask, x/y coordinates, and incidence angles from OPERA H5 files")
    print("2. Convert LOS displacement to vertical displacement using incidence angle interpolation")
    print("3. Apply mask to filter invalid pixels for both displacement and temporal coherence")
    print("   - mask_flag = 0: No mask")
    print("   - mask_flag = 1: Recommended mask")
    print("   - mask_flag = 0-1: Coherence threshold")
    print("4. Identify UTM projection Zone from shp and convert CRS to WGS84 EPSG:4326")
    print("5. Save displacement and temporal coherence data as dual-band GeoTIFF files")
    print("6. Generate detailed processing reports and statistics")
    print()

    # Create processor
    processor = OPERADisplacementProcessor(
        input_folder=input_folder,
        output_folder=output_folder,
        shapefile_path=shapefile_path,
        mask_flag=mask_flag
    )
    
    # Process all files
    results = processor.process_all_files()
    
    # Display Frame summary
    frame_summary = processor.get_frame_summary()
    print("\nFrame summary:")
    for frame, info in frame_summary.items():
        los_count = info['los_processed']
        vertical_count = info.get('vertical_processed', 0)
        total_count = info['file_count']
        print(f"{frame}: {total_count} files (LOS: {los_count}, Vertical: {vertical_count})")
    
    # Analyze processing results
    print(f"\n{'='*50}")
    print("Analyze processing results:")
    batch_analyze_results(output_folder)

def process_opera_files_with_los_displacement(input_folder, output_folder=None, shapefile_path=None, mask_flag=1):
    """Quick function to process OPERA files for LOS displacement

    Parameters:
    input_folder: Input folder containing OPERA H5 files
    output_folder: Output folder for processed files
    shapefile_path: Optional shapefile path for validation
    mask_flag: Masking behavior control
        - 0: No mask applied
        - 1: Apply recommended mask (default)
        - 0 < mask_flag < 1: Use as coherence threshold
    """
    if output_folder is None:
        output_folder = Path(input_folder).parent / "opera_los_processed"

    processor = OPERADisplacementProcessor(input_folder, output_folder, shapefile_path, mask_flag)
    results = processor.process_all_files()
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    los_processed = [r for r in successful if 'LOS' in r.get('data_type', '')]
    vertical_processed = [r for r in successful if 'Vertical' in r.get('data_type', '')]
    
    print(f"\nProcessing results summary:")
    print(f"   Success: {len(successful)} files")
    print(f"   Failed: {len(failed)} files")
    print(f"   LOS processed: {len(los_processed)} files")
    print(f"   Vertical processed: {len(vertical_processed)} files")
    
    if failed:
        print(f"\nFailed files:")
        for result in failed:
            print(f"   {Path(result['input_file']).name}: {result.get('error', 'Unknown error')}")
    
    return processor

# H5 file structure exploration tool
def explore_h5_structure(h5_path, max_depth=3):
    """Independent tool for exploring H5 file structure"""
    print(f"Explore H5 file structure: {h5_path}")
    print("=" * 50)
    
    try:
        with h5py.File(h5_path, 'r') as f:
            def print_structure(group, prefix="", depth=0):
                if depth > max_depth:
                    return
                
                for key in group.keys():
                    item = group[key]
                    full_path = f"{prefix}/{key}" if prefix else key
                    
                    if isinstance(item, h5py.Group):
                        print(f"{'  ' * depth}📁 {key}/")
                        print_structure(item, full_path, depth + 1)
                    elif isinstance(item, h5py.Dataset):
                        print(f"{'  ' * depth}📄 {key}: {item.shape} {item.dtype}")
                        
                        if item.attrs:
                            for attr_name, attr_value in item.attrs.items():
                                print(f"{'  ' * (depth + 1)}🏷️ {attr_name}: {attr_value}")
            
            if f.attrs:
                print("🌐 Global attributes:")
                for attr_name, attr_value in f.attrs.items():
                    print(f"  {attr_name}: {attr_value}")
                print()
            
            print("📂 File structure:")
            print_structure(f)
            
    except Exception as e:
        print(f"Exploration failed: {e}")

if __name__ == "__main__":
    print("OPERA Displacement Data Processing Tool - LOS-Mask-Reproject-Subset")
    print("="*50)

    # Check if command line arguments are provided
    if len(sys.argv) >= 4:
        # Called from automated_county_processor.py or automated_comprehensive_processor.py with arguments
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        shapefile_path = sys.argv[3]

        # mask_flag is optional (default to 0 if not provided)
        if len(sys.argv) >= 5:
            mask_flag = int(sys.argv[4])
        else:
            mask_flag = 0  # Default mask flag

        print("Running with command line arguments:")
        print(f"  Input folder: {input_folder}")
        print(f"  Output folder: {output_folder}")
        print(f"  Shapefile path: {shapefile_path}")
        print(f"  Mask flag: {mask_flag}")
        print()

        # Create processor with provided arguments
        processor = OPERADisplacementProcessor(
            input_folder=input_folder,
            output_folder=output_folder,
            shapefile_path=shapefile_path,
            mask_flag=mask_flag
        )

        # Process all files
        results = processor.process_all_files()

        # Display Frame summary
        frame_summary = processor.get_frame_summary()
        print("\nFrame summary:")
        for frame, info in frame_summary.items():
            los_count = info['los_processed']
            vertical_count = info.get('vertical_processed', 0)
            total_count = info['file_count']
            print(f"{frame}: {total_count} files (LOS: {los_count}, Vertical: {vertical_count})")

        # Analyze processing results
        print(f"\n{'='*50}")
        print("Analyze processing results:")
        batch_analyze_results(output_folder)
    else:
        # No arguments provided, use default paths from main()
        print("Quick usage:")
        print("processor = Opera-Vertical-Mask-Reproject-Processcer")
        print()
        print()
        print("Detailed usage:")
        print("main()")
        print()

        # Run main program with default paths
        main()