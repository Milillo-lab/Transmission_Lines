import requests
import geopandas as gpd
from pathlib import Path
import getpass
import time
import pandas as pd
from datetime import datetime
import asf_search as asf
from shapely.geometry import Polygon
from shapely.ops import unary_union

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent  
DATA_DIR = PROJECT_ROOT / "data"
COUNTIES_DIR = DATA_DIR / "counties_shapefile"
OUTPUT_BASE_DIR = DATA_DIR / "opera_displacement_data"
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"
MIN_COVERAGE_THRESHOLD = 2

def get_credentials():
    username = input("Username: ")
    password = getpass.getpass("Password: ")
    return username, password

def create_session(username, password):
    session = asf.ASFSession()
    session.auth_with_creds(username, password)
    return session

def find_county_shapefiles(counties_dir):
    """Find all county shapefiles in the directory structure"""
    counties_dir = Path(counties_dir)
    shapefiles = []
    
    for county_folder in counties_dir.iterdir():
        if county_folder.is_dir():
            county_name = county_folder.name
            
            # Find any .shp file in the folder
            shp_files = list(county_folder.glob("*.shp"))
            
            # Use the first .shp file found
            shapefile_path = shp_files[0]
            shapefiles.append({
                'name': county_name,
                'path': shapefile_path,
                'output_dir': Path(OUTPUT_BASE_DIR) / county_name
            })
    
    return shapefiles

def load_aoi_bbox(aoi_path):
    """Load AOI shapefile and return bounding box and geometry"""
    aoi_gdf = gpd.read_file(aoi_path)
    if aoi_gdf.crs != 'EPSG:4326':
        aoi_gdf = aoi_gdf.to_crs('EPSG:4326')
    
    bbox = aoi_gdf.total_bounds
    geometry = unary_union(aoi_gdf.geometry)
    
    return bbox, geometry

def parse_granule_footprint(granule):
    """Extract spatial footprint from granule metadata"""
    try:
        if 'polygons' in granule and granule['polygons']:
            for polygon_entry in granule['polygons']:
                if isinstance(polygon_entry, list) and polygon_entry:
                    polygon_str = polygon_entry[0]
                    coords = [float(x) for x in polygon_str.split()]
                    coord_pairs = [(coords[i+1], coords[i]) for i in range(0, len(coords), 2)]
                    
                    if len(coord_pairs) >= 3:
                        return Polygon(coord_pairs)
        
        if 'boxes' in granule and granule['boxes']:
            box = granule['boxes'][0]
            coords = [float(x) for x in box.split()]
            if len(coords) == 4:
                south, north, west, east = coords
                return Polygon([
                    (west, south), (east, south), 
                    (east, north), (west, north), 
                    (west, south)
                ])
    except:
        pass
    
    return None

def calculate_coverage_percentage(granule_geometry, aoi_geometry):
    """Calculate spatial coverage percentage for a granule"""
    try:
        if granule_geometry is None or aoi_geometry is None:
            return 0.0
        
        intersection = granule_geometry.intersection(aoi_geometry)
        aoi_area = aoi_geometry.area
        
        if aoi_area > 0:
            return (intersection.area / aoi_area) * 100
        else:
            return 0.0
    except:
        return 0.0

class OPERADownloader:
    def __init__(self, bbox, aoi_geometry, county_name, output_dir, session):
        self.bbox = bbox
        self.aoi_geometry = aoi_geometry
        self.county_name = county_name
        self.output_dir = Path(output_dir)
        self.session = session
        
        self.raw_files_dir = self.output_dir / 'raw_files'
        self.metadata_dir = self.output_dir / 'metadata'
        
        self.raw_files_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
    def make_granule_list(self,start_date, end_date,flight_direction):
        """Search for OPERA granules"""
        cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        params = {
            'collection_concept_id': 'C3294057315-ASF',
            'bounding_box': f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}",
            'temporal': f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
            'page_size': 2000,
            'sort_key': 'start_date',
            'attribute[]': f'string,ASCENDING_DESCENDING,{flight_direction}'
        }
        response = requests.get(cmr_url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        granules = data['feed']['entry']
        granule_list = []
        for granule in granules:
            granule_info = {
                'id': granule['id'],
                'title': granule['title'],
                'start_date': granule['time_start'],
                'end_date': granule['time_end'],
                'download_urls': [],
                'footprint': parse_granule_footprint(granule),
                'coverage_percentage': 0.0
            }
            
            if 'links' in granule:
                for link in granule['links']:
                    if link.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#':
                        granule_info['download_urls'].append(link['href'])
            
            granule_list.append(granule_info)
        return granule_list
    
    def search_granules(self, start_date, end_date , flight_direction = 'ASC/DSC'):
        """ Chose the proper flight direction, then Search for OPERA granules """
        if flight_direction == 'ASC/DSC':
            flight_direction = 'ASCENDING'
            granule_list = self.make_granule_list(start_date, end_date , flight_direction)
            granule_df = pd.DataFrame(granule_list)
            pattern = r'IW_(.*?)_VV'
            granule_df['frame'] =  granule_df['title'].str.extract(pattern, expand=False)
            len_ASC = len(granule_df) / len(granule_df['frame'].unique()) / len(granule_df['frame'].unique()) #gives priority to the direction with higher temporal and lower spatial frames
            
            flight_direction = 'DESCENDING'
            granule_list = self.make_granule_list(start_date, end_date , flight_direction)
            granule_df = pd.DataFrame(granule_list)
            pattern = r'IW_(.*?)_VV'
            granule_df['frame'] =  granule_df['title'].str.extract(pattern, expand=False)
            len_DSC = len(granule_df) / len(granule_df['frame'].unique()) / len(granule_df['frame'].unique()) #gives priority to the direction with higher temporal and lower spatial frames
            
            if len_ASC> len_DSC:
                flight_direction = 'ASCENDING'
        granule_list = self.make_granule_list(start_date, end_date , flight_direction)
        
        return granule_list
    
    def filter_by_coverage(self, granule_list):
        """Filter granules by spatial coverage threshold"""
        
        filtered_granules = []
        
        for granule in granule_list:
            coverage_pct = calculate_coverage_percentage(
                granule['footprint'], 
                self.aoi_geometry
            )
            
            granule['coverage_percentage'] = coverage_pct
            
            # Only keep granules with >2% coverage
            if coverage_pct > MIN_COVERAGE_THRESHOLD:
                filtered_granules.append(granule)
        
        return filtered_granules
    
    def process_county(self, start_date, end_date):
        """Complete processing workflow for a county"""
        
        print(f"Processing {self.county_name}...")
        
        # Search for all granules in date range
        granule_list = self.search_granules(start_date, end_date)
        if not granule_list:
            print(f"No granules found for {self.county_name}")
            return []
        
        # Filter by spatial coverage threshold
        filtered_granules = self.filter_by_coverage(granule_list)
        if not filtered_granules:
            print(f"No granules with >{MIN_COVERAGE_THRESHOLD}% coverage for {self.county_name}")
            return []
        
        print(f"Found {len(filtered_granules)} granules with >{MIN_COVERAGE_THRESHOLD}% coverage for {self.county_name}")
        
        # Save metadata and download
        self.save_search_metadata(filtered_granules, start_date, end_date)
        downloaded_files = self.download_granules(filtered_granules)
        print(f"Downloaded {len(downloaded_files)} files for {self.county_name}")
        
        return filtered_granules
    
    def save_search_metadata(self, granule_list, start_date, end_date):
        """Save search results metadata"""
        
        df_data = []
        for granule in granule_list:
            df_data.append({
                'title': granule['title'],
                'start_date': granule['start_date'],
                'end_date': granule['end_date'],
                'coverage_percentage': granule.get('coverage_percentage', 'Unknown'),
                'download_url': granule['download_urls'][0] if granule['download_urls'] else 'No URL'
            })
        
        df = pd.DataFrame(df_data)
        csv_path = self.metadata_dir / f'{self.county_name}_granules.csv'
        df.to_csv(csv_path, index=False)
    
    def download_granules(self, granule_list):
        """Download all granules"""
        
        if not granule_list:
            return []
        
        downloaded_files = []
        
        for granule in granule_list:
            if not granule['download_urls']:
                continue
            
            filename = f"{granule['title']}.h5"
            output_path = self.raw_files_dir / filename
            
            if output_path.exists():
                print(f"File already exists: {filename}")
                downloaded_files.append(output_path)
                continue
            
            print(f"Downloading: {filename}")
            success = self.download_file(granule['download_urls'][0], output_path)
            
            if success:
                downloaded_files.append(output_path)
                print(f"Successfully downloaded: {filename}")
            else:
                print(f"Failed to download: {filename}")
            
            time.sleep(1)
        
        self.save_download_summary(downloaded_files)
        return downloaded_files
    
    def download_file(self, url, output_path):
        """Download a single file"""
        
        try:
            response = self.session.get(url, stream=True, allow_redirects=True, timeout=300)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"Download error: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def save_download_summary(self, downloaded_files):
        """Save download summary to file"""
        
        summary_path = self.metadata_dir / f'{self.county_name}_download_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write(f"OPERA Download Summary - {self.county_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Download completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Successfully downloaded: {len(downloaded_files)} files\n")
            f.write(f"Coverage threshold: >{MIN_COVERAGE_THRESHOLD}%\n")
            f.write(f"Date range: {START_DATE} to {END_DATE}\n\n")
            
            if downloaded_files:
                f.write("Successfully downloaded files:\n")
                for file_path in downloaded_files:
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    f.write(f"  {file_path.name} ({file_size_mb:.1f} MB)\n")

def main():
    print(f"OPERA Data Downloader")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Minimum coverage threshold: >{MIN_COVERAGE_THRESHOLD}%")
    
    county_shapefiles = find_county_shapefiles(COUNTIES_DIR)
    
    if not county_shapefiles:
        print(f"No shapefiles found in {COUNTIES_DIR}")
        return
    
    print(f"Found {len(county_shapefiles)} counties")
    
    username, password = get_credentials()
    session = create_session(username, password)
    
    if not session:
        print("Failed to authenticate. Exiting.")
        return
    
    all_results = {}
    
    for i, county in enumerate(county_shapefiles):
        print(f"\n[{i+1}/{len(county_shapefiles)}] Processing {county['name']}")
        
        try:
            bbox, geometry = load_aoi_bbox(county['path'])
            
            downloader = OPERADownloader(
                bbox, geometry, county['name'], county['output_dir'], session
            )
            
            filtered_granules = downloader.process_county(START_DATE, END_DATE)
            
            all_results[county['name']] = len(filtered_granules)
                
        except Exception as e:
            print(f"Error processing {county['name']}: {e}")
            all_results[county['name']] = 0
    
    print(f"\nFinal Summary:")
    total_granules = 0
    for county_name, granule_count in all_results.items():
        print(f"{county_name}: {granule_count} granules")
        total_granules += granule_count
    
    print(f"Total: {total_granules} granules processed")
    print(f"Data location: {OUTPUT_BASE_DIR}/")

if __name__ == "__main__":
    main()