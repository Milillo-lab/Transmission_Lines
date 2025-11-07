import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import ListedColormap
from pathlib import Path
import glob
import rasterio
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class TowerAnalysis:
    """
    Main class for analyzing transmission tower deformation risks across multiple regions.
    """
    
    RISK_LEVELS = {
        'no_risk': 0,
        'slight_risk': 1,
        'moderate_risk': 2,
        'severe_risk': 3
    }
    
    def __init__(self, csv_folder_path, shapefile_path, study_area_path, output_folder, env_rasters=None):
        """
        Initialize the tower analysis system.
        """
        self.csv_folder = Path(csv_folder_path)
        self.study_area_path = Path(study_area_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True, parents=True)

        self.towers_gdf = gpd.read_file(shapefile_path)
        self.study_area = self._load_counties()
        
        if self.towers_gdf.crs != self.study_area.crs:
            self.towers_gdf = self.towers_gdf.to_crs(self.study_area.crs)
        
        self.towers_gdf = gpd.sjoin(self.towers_gdf, self.study_area, predicate='within', how='inner')
        self.towers_gdf = self.towers_gdf.drop(columns=[col for col in self.towers_gdf.columns if col.startswith('index_')])
        self.towers_gdf = self.towers_gdf.reset_index(drop=True)
        
        if 'county_name' not in self.towers_gdf.columns and 'NAME' in self.towers_gdf.columns:
            self.towers_gdf['county_name'] = self.towers_gdf['NAME']
        
        self.env_rasters = env_rasters or {}
        
        print(f"\nLoaded {len(self.towers_gdf)} towers across {len(self.study_area)} counties\n")
    
    def _load_counties(self):
        """
        Load county shapefiles from the study area directory.
        """
        county_gdfs = []
        county_dirs = [d for d in self.study_area_path.iterdir() if d.is_dir()]
        
        for county_dir in county_dirs:
            county_name = county_dir.name
            shp_path = county_dir / f"{county_name}.shp"
            
            if shp_path.exists():
                gdf = gpd.read_file(shp_path)
                if 'county_name' not in gdf.columns:
                    gdf['county_name'] = county_name
                county_gdfs.append(gdf)
            else:
                shp_files = list(county_dir.glob("*.shp"))
                if shp_files:
                    gdf = gpd.read_file(shp_files[0])
                    if 'county_name' not in gdf.columns:
                        gdf['county_name'] = county_name
                    county_gdfs.append(gdf)
        
        if not county_gdfs:
            raise ValueError("No county shapefiles could be loaded")
        
        return pd.concat(county_gdfs, ignore_index=True)
    
    def _get_csv_files(self):
        """
        Get all CSV files from the region's county directories.
        """
        csv_files = []
        county_dirs = [d for d in self.csv_folder.iterdir() if d.is_dir()]
        
        for county_dir in county_dirs:
            county_csvs = list(county_dir.glob("*.csv"))
            csv_files.extend(county_csvs)
        
        print(f"Total CSV files found: {len(csv_files)}")
        return csv_files
    
    def _get_tower_id(self, csv_file, df):
        """
        Extract tower ID from CSV filename or dataframe.
        """
        try:
            return int(Path(csv_file).stem.split('_')[1])
        except (ValueError, IndexError):
            if 'tower_id' in df.columns:
                tower_id_val = df['tower_id'].iloc[0]
                if isinstance(tower_id_val, str):
                    return int(tower_id_val.split('_')[-1])
                return int(tower_id_val)
        return None
    
    def get_risk_colors(self):
        """
        Generate risk level colors from velocity colormap.
        """
        cmap = plt.cm.get_cmap('RdYlBu_r')
        vmin, vmax = 0.005, 0.016
        
        positions = {
            'no_risk': (0.011 - vmin) / (vmax - vmin),
            'slight_risk': (0.008 - vmin) / (vmax - vmin),
            'moderate_risk': (0.013 - vmin) / (vmax - vmin),
            'severe_risk': 1.0
        }
        
        import matplotlib.colors as mcolors
        colors = []
        for risk_level in ['no_risk', 'slight_risk', 'moderate_risk', 'severe_risk']:
            rgba = cmap(positions[risk_level])
            hex_color = mcolors.to_hex(rgba[:3])
            if risk_level == 'no_risk':
                hex_color += '80'
            colors.append(hex_color)
        
        return colors
    
    def get_raster_values(self, raster_paths, towers_gdf):
        """
        Extract raster values at tower locations.
        """
        if isinstance(raster_paths, str):
            raster_path_obj = Path(raster_paths)
            
            if '*' in str(raster_paths):
                raster_files = glob.glob(raster_paths)
            elif raster_path_obj.is_dir():
                raster_files = [str(f) for f in raster_path_obj.rglob('*.tif')]
            else:
                raster_files = [raster_paths]
        else:
            raster_files = raster_paths
        
        if not raster_files:
            return np.full(len(towers_gdf), np.nan)
        
        values = np.full(len(towers_gdf), np.nan)
        
        for raster_file in raster_files:
            try:
                with rasterio.open(raster_file) as src:
                    if towers_gdf.crs != src.crs:
                        towers_proj = towers_gdf.to_crs(src.crs)
                    else:
                        towers_proj = towers_gdf
                    
                    raster_bounds = src.bounds
                    
                    for idx, (_, tower) in enumerate(towers_proj.iterrows()):
                        if not np.isnan(values[idx]):
                            continue
                        
                        geom = tower.geometry
                        
                        if (raster_bounds.left <= geom.x <= raster_bounds.right and 
                            raster_bounds.bottom <= geom.y <= raster_bounds.top):
                            
                            try:
                                for val in src.sample([(geom.x, geom.y)]):
                                    if src.nodata is not None and val[0] == src.nodata:
                                        continue
                                    elif val[0] < -9999:
                                        continue
                                    else:
                                        values[idx] = float(val[0])
                                        break
                            except:
                                continue
            except Exception as e:
                continue
        
        return values
    
    def process_env_factors(self, towers_analysis):
        """
        Process environmental factors by extracting raster values at tower locations.
        """
        if not self.env_rasters:
            return towers_analysis
        
        for factor_name, raster_paths in self.env_rasters.items():
            if isinstance(raster_paths, str):
                path_obj = Path(raster_paths)
                if not path_obj.exists() and '*' not in str(raster_paths):
                    towers_analysis[factor_name] = 0.0
                    continue
            
            try:
                values = self.get_raster_values(raster_paths, towers_analysis)
                values = np.nan_to_num(values, nan=0.0)
                towers_analysis[factor_name] = values
            except Exception as e:
                towers_analysis[factor_name] = 0.0
                
        return towers_analysis
    
    def classify_risk(self, value, p60, p85, p98):
        """
        Classify risk level based on percentile thresholds.
        """
        if value == 0:
            return self.RISK_LEVELS['no_risk']
        elif value <= p60:
            return self.RISK_LEVELS['no_risk']
        elif value <= p85:
            return self.RISK_LEVELS['slight_risk']
        elif value <= p98:
            return self.RISK_LEVELS['moderate_risk']
        else:
            return self.RISK_LEVELS['severe_risk']
    
    def classify_env_risks_selected(self, towers_analysis, best_storm_category):
        """
        Classify environmental risks for non-storm factors and selected storm surge category.
        """
        risk_classifications = {}
        
        for factor_name in self.env_rasters.keys():
            if 'storm_surge_cat' not in factor_name and factor_name in towers_analysis.columns:
                non_zero_values = towers_analysis[factor_name][towers_analysis[factor_name] > 0]
                
                if len(non_zero_values) == 0:
                    towers_analysis[f'{factor_name}_risk'] = self.RISK_LEVELS['no_risk']
                    risk_classifications[factor_name] = {'p60': 0, 'p85': 0, 'p98': 0}
                    continue
                
                p60 = np.percentile(non_zero_values, 60)
                p85 = np.percentile(non_zero_values, 85)
                p98 = np.percentile(non_zero_values, 98)
                
                risk_classifications[factor_name] = {'p60': p60, 'p85': p85, 'p98': p98}
                
                towers_analysis[f'{factor_name}_risk'] = towers_analysis[factor_name].apply(
                    lambda x: self.classify_risk(x, p60, p85, p98)
                )
        
        if best_storm_category and best_storm_category in towers_analysis.columns:
            non_zero_values = towers_analysis[best_storm_category][towers_analysis[best_storm_category] > 0]
            
            if len(non_zero_values) > 0:
                p60 = np.percentile(non_zero_values, 60)
                p85 = np.percentile(non_zero_values, 85)
                p98 = np.percentile(non_zero_values, 98)
                
                risk_classifications[best_storm_category] = {'p60': p60, 'p85': p85, 'p98': p98}
                
                towers_analysis[f'{best_storm_category}_risk'] = towers_analysis[best_storm_category].apply(
                    lambda x: self.classify_risk(x, p60, p85, p98)
                )
            else:
                towers_analysis[f'{best_storm_category}_risk'] = self.RISK_LEVELS['no_risk']
                risk_classifications[best_storm_category] = {'p60': 0, 'p85': 0, 'p98': 0}
        
        return towers_analysis, risk_classifications

    def classify_env_risks(self, towers_analysis):
        """
        Classify environmental risks for all environmental factors.
        """
        risk_classifications = {}
        
        for factor_name in self.env_rasters.keys():
            if factor_name not in towers_analysis.columns:
                continue
            
            non_zero_values = towers_analysis[factor_name][towers_analysis[factor_name] > 0]
            
            if len(non_zero_values) == 0:
                towers_analysis[f'{factor_name}_risk'] = self.RISK_LEVELS['no_risk']
                risk_classifications[factor_name] = {'p60': 0, 'p85': 0, 'p98': 0}
                continue
            
            p60 = np.percentile(non_zero_values, 60)
            p85 = np.percentile(non_zero_values, 85)
            p98 = np.percentile(non_zero_values, 98)
            
            risk_classifications[factor_name] = {'p60': p60, 'p85': p85, 'p98': p98}
            
            towers_analysis[f'{factor_name}_risk'] = towers_analysis[factor_name].apply(
                lambda x: self.classify_risk(x, p60, p85, p98)
            )
        
        return towers_analysis, risk_classifications
    
    def calc_union_risk(self, towers_analysis):
        """
        Calculate union risk as the maximum of all individual risk factors.
        """
        risk_columns = []
        
        deformation_risks = ['neighbor_risk', 'surrounding_risk', 'strain_risk']
        risk_columns.extend([col for col in deformation_risks if col in towers_analysis.columns])
        
        for factor_name in self.env_rasters.keys():
            risk_col = f'{factor_name}_risk'
            if risk_col in towers_analysis.columns:
                risk_columns.append(risk_col)
        
        if not risk_columns:
            towers_analysis['union_risk'] = self.RISK_LEVELS['no_risk']
            return towers_analysis
        
        towers_analysis['union_risk'] = towers_analysis[risk_columns].max(axis=1)
        
        return towers_analysis
    
    def load_timeseries(self, csv_file):
        """
        Load tower displacement time series data from CSV file.
        """
        df = pd.read_csv(csv_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        tower_id = self._get_tower_id(csv_file, df)
        if tower_id is None:
            return None, None
        
        displacement_cols_mm = [col for col in df.columns if col.startswith('pixel_') and 'displacement_mm' in col]
        for col in displacement_cols_mm:
            new_col = col.replace('_mm', '_cm')
            df[new_col] = df[col] * 100
        
        return df, tower_id
    
    def calc_velocity(self, df):
        """
        Calculate displacement velocity using linear regression on all data points.
        """
        if len(df) < 2:
            return 0.0
        
        time_days = (df['date'] - df['date'].min()).dt.days
        slope, _, _, _, _ = stats.linregress(time_days, df['pixel_5_displacement_cm'])
        velocity_cm_year = slope * 365.25
        
        return velocity_cm_year
    
    def calc_surrounding_velocity(self, df):
        """
        Calculate average velocity of surrounding pixels (30-42m range).
        """
        if len(df) < 2:
            return 0.0
        
        surrounding_cols = [f'pixel_{i}_displacement_cm' for i in [1,2,3,4,6,7,8,9] 
                        if f'pixel_{i}_displacement_cm' in df.columns]
        if not surrounding_cols:
            return 0.0
        
        avg_surrounding = df[surrounding_cols].mean(axis=1)
        time_days = (df['date'] - df['date'].min()).dt.days
        slope, _, _, _, _ = stats.linregress(time_days, avg_surrounding)
        velocity_cm_year = slope * 365.25
        
        return velocity_cm_year
    
    def process_towers(self):
        """
        Process all towers to calculate velocity metrics.
        """
        csv_files = self._get_csv_files()
        results = []
        
        for csv_file in tqdm(csv_files, desc="Processing towers"):
            try:
                df, tower_id = self.load_timeseries(csv_file)
                if df is None or tower_id is None or tower_id not in self.towers_gdf['tower_id'].values:
                    continue
                
                tower_velocity = self.calc_velocity(df)
                surrounding_velocity = self.calc_surrounding_velocity(df)
                
                results.append({
                    'tower_id': tower_id,
                    'velocity_cm_year': tower_velocity,
                    'surrounding_avg_velocity_cm_year': surrounding_velocity,
                    'relative_velocity_surrounding_cm_year': tower_velocity - surrounding_velocity,
                    'n_observations': len(df)
                })
            except Exception as e:
                continue
        
        return pd.DataFrame(results)
    
    def find_neighbors(self, max_distance_km=1.0):
        """
        Find nearest neighbor tower within specified distance for each tower.
        """
        towers_proj = (self.towers_gdf.to_crs(self.towers_gdf.estimate_utm_crs()) 
                      if self.towers_gdf.crs.is_geographic 
                      else self.towers_gdf.copy())
        
        coords = np.array([[geom.x, geom.y] for geom in towers_proj.geometry])
        tree = cKDTree(coords)
        max_distance_m = max_distance_km * 1000
        
        neighbors = {}
        
        for i in range(len(self.towers_gdf)):
            tower_id = self.towers_gdf.iloc[i]['tower_id']
            neighbor_indices = tree.query_ball_point(coords[i], max_distance_m)
            neighbor_indices = [n for n in neighbor_indices if n != i]
            
            if neighbor_indices:
                neighbor_idx = min(neighbor_indices, 
                                 key=lambda n: np.linalg.norm(coords[i] - coords[n]))
                neighbor_tower_id = self.towers_gdf.iloc[neighbor_idx]['tower_id']
                neighbors[tower_id] = {
                    'neighbor_id': neighbor_tower_id,
                    'distance_m': np.linalg.norm(coords[i] - coords[neighbor_idx])
                }
        
        return neighbors
    
    def calc_strain(self, tower_id1, tower_id2):
        """
        Calculate strain between two towers based on relative displacement.
        """
        if not hasattr(self, '_csv_lookup'):
            self._csv_lookup = {}
            for county_dir in self.csv_folder.iterdir():
                if county_dir.is_dir():
                    for csv_file in county_dir.glob("tower_*_9pixel_displacement_coherence.csv"):
                        tower_id = int(csv_file.stem.split('_')[1])
                        self._csv_lookup[tower_id] = csv_file
        
        csv_file1 = self._csv_lookup.get(tower_id1)
        csv_file2 = self._csv_lookup.get(tower_id2)
        
        if csv_file1 is None or csv_file2 is None:
            return None
        
        df1, _ = self.load_timeseries(str(csv_file1))
        df2, _ = self.load_timeseries(str(csv_file2))
        
        if df1 is None or df2 is None:
            return None
        
        merged = pd.merge(df1[['date', 'pixel_5_displacement_cm']], 
                         df2[['date', 'pixel_5_displacement_cm']], 
                         on='date', suffixes=('_t1', '_t2'), how='inner')
        
        if len(merged) < 2:
            return None
        
        delta_h_first = merged.iloc[0]['pixel_5_displacement_cm_t1'] - merged.iloc[0]['pixel_5_displacement_cm_t2']
        delta_h_last = merged.iloc[-1]['pixel_5_displacement_cm_t1'] - merged.iloc[-1]['pixel_5_displacement_cm_t2']
        
        tower1_geom = self.towers_gdf[self.towers_gdf['tower_id'] == tower_id1].geometry.iloc[0]
        tower2_geom = self.towers_gdf[self.towers_gdf['tower_id'] == tower_id2].geometry.iloc[0]
        
        if self.towers_gdf.crs.is_geographic:
            towers_proj = self.towers_gdf.to_crs(self.towers_gdf.estimate_utm_crs())
            tower1_proj = towers_proj[towers_proj['tower_id'] == tower_id1].geometry.iloc[0]
            tower2_proj = towers_proj[towers_proj['tower_id'] == tower_id2].geometry.iloc[0]
            horizontal_distance = tower1_proj.distance(tower2_proj)
        else:
            horizontal_distance = tower1_geom.distance(tower2_geom)
        
        if horizontal_distance == 0:
            return None
        
        horizontal_distance_cm = horizontal_distance * 100
        
        D_first = np.sqrt(delta_h_first**2 + horizontal_distance_cm**2)
        D_last = np.sqrt(delta_h_last**2 + horizontal_distance_cm**2)
        
        if D_first == 0:
            return None
        
        strain = (D_last - D_first) / D_first
        return {'strain': strain}
    
    def run_pca(self, towers_analysis):
        """
        Perform PCA analysis on storm surge categories to select most representative category.
        """
        print("\n" + "="*70)
        print("STORM SURGE PCA ANALYSIS - Category Selection")
        print("="*70)
        
        storm_surge_cols = [col for col in towers_analysis.columns 
                           if 'storm_surge_cat' in col.lower() and '_risk' not in col.lower()]
        
        if len(storm_surge_cols) == 0:
            print("No storm surge categories found")
            return towers_analysis, None, None, None
        
        print(f"\nFound {len(storm_surge_cols)} storm surge categories:")
        for col in storm_surge_cols:
            print(f"  - {col}")
        
        towers_pca = towers_analysis[storm_surge_cols].copy()
        towers_pca = towers_pca.fillna(0)
        
        scaler = StandardScaler()
        towers_scaled = scaler.fit_transform(towers_pca)
        towers_scaled_df = pd.DataFrame(towers_scaled, columns=storm_surge_cols, index=towers_pca.index)
        
        corr_matrix = towers_scaled_df.corr()
        
        print("\n--- Correlation Matrix ---")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                    vmin=-1, vmax=1)
        plt.title('Storm Surge Categories Correlation Matrix', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_folder / 'storm_surge_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        pca = PCA()
        pca_scores = pca.fit_transform(towers_scaled_df)
        
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print("\n--- PCA Results ---")
        print(f"{'PC':<5} {'Var Explained':<15} {'Cumulative':<15}")
        print("-" * 40)
        for i in range(len(explained_var)):
            print(f"PC{i+1:<3} {explained_var[i]:>12.1%}   {cumulative_var[i]:>12.1%}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, color='steelblue')
        ax1.plot(range(1, len(explained_var)+1), explained_var, 'o-', color='darkred', linewidth=2)
        ax1.set_xlabel('Principal Component', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Variance Explained', fontweight='bold', fontsize=12)
        ax1.set_title('Storm Surge PCA Scree Plot', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(range(1, len(cumulative_var)+1), cumulative_var, 'o-', linewidth=2, color='darkgreen')
        ax2.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='80%')
        ax2.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, label='90%')
        ax2.set_xlabel('Principal Component', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Cumulative Variance', fontweight='bold', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontweight='bold', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'storm_surge_pca_scree_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        loadings = pca.components_[0, :]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        loading_df = pd.DataFrame({
            'Category': storm_surge_cols,
            'Loading': loadings
        }).sort_values('Loading', ascending=False)
        
        colors = ['darkred' if abs(x) == abs(loading_df['Loading']).max() else 'steelblue' 
                 for x in loading_df['Loading']]
        bars = ax.bar(loading_df['Category'], loading_df['Loading'], color=colors, alpha=0.7)
        
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_xlabel('Storm Surge Category', fontweight='bold', fontsize=12)
        ax.set_ylabel('PC1 Loading', fontweight='bold', fontsize=12)
        ax.set_title(f'Storm Surge PC1 Loadings ({explained_var[0]:.1%} variance)\nHighest loading indicates most representative category', 
                    fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_folder / 'storm_surge_pc1_loadings.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        abs_loadings = np.abs(loadings)
        best_category_idx = np.argmax(abs_loadings)
        best_category = storm_surge_cols[best_category_idx]
        best_loading = loadings[best_category_idx]
        
        print("\n--- Category Selection Results ---")
        print(f"Most effective category: {best_category}")
        print(f"PC1 loading: {best_loading:.3f}")
        print(f"\nLoadings for all categories:")
        for cat, load in zip(storm_surge_cols, loadings):
            indicator = " <-- SELECTED" if cat == best_category else ""
            print(f"  {cat}: {load:.3f}{indicator}")
        
        print("\n" + "="*70)
        
        return towers_analysis, best_category, pca, explained_var
    
    def run_analysis(self):
        """
        Execute complete tower analysis workflow including velocity, strain, and environmental factors.
        """
        tower_metrics = self.process_towers()
        if tower_metrics.empty:
            return None, None

        towers_analysis = self.towers_gdf.merge(tower_metrics, on='tower_id', how='left')
        towers_analysis = towers_analysis[towers_analysis['velocity_cm_year'].notna()].copy()

        if len(towers_analysis) == 0:
            return None, None

        print("\nFinding nearest neighbors...")
        neighbors = self.find_neighbors(max_distance_km=1.0)
        velocity_lookup = tower_metrics.set_index('tower_id')['velocity_cm_year'].to_dict()

        print("Calculating strain between tower pairs...")
        strain_results = []
        
        for tower_id, neighbor_info in tqdm(neighbors.items(), desc="Calculating strain"):
            neighbor_id = neighbor_info['neighbor_id']
            strain_data = self.calc_strain(tower_id, neighbor_id)

            relative_velocity_neighbor = None
            if tower_id in velocity_lookup and neighbor_id in velocity_lookup:
                relative_velocity_neighbor = velocity_lookup[tower_id] - velocity_lookup[neighbor_id]

            strain_results.append({
                'tower_id': tower_id,
                'neighbor_tower_id': neighbor_info['neighbor_id'],
                'strain': strain_data['strain'] if strain_data else None,
                'distance_m': neighbor_info['distance_m'],
                'relative_velocity_neighbor_cm_year': relative_velocity_neighbor
            })

        strain_df = pd.DataFrame(strain_results)
        towers_analysis = towers_analysis.merge(
            strain_df[['tower_id', 'neighbor_tower_id', 'strain', 'distance_m', 'relative_velocity_neighbor_cm_year']],
            on='tower_id', how='left'
        )
        
        total_towers = len(towers_analysis)
        towers_with_strain = towers_analysis['strain'].notna().sum()
        towers_without_strain = total_towers - towers_with_strain
        
        print(f"\n--- Strain Calculation Summary ---")
        print(f"Total towers: {total_towers}")
        print(f"Towers with strain values: {towers_with_strain}")
        print(f"Towers without strain values: {towers_without_strain}")
        if towers_without_strain > 0:
            print(f"Reasons for missing strain values:")
            print(f"  - No nearest neighbor within 1km")
            print(f"  - CSV file not found for tower or neighbor")
            print(f"  - Insufficient overlapping time series data")
            print(f"  - Zero distance between towers")
        print("=" * 40)
        
        towers_analysis = self.process_env_factors(towers_analysis)
        
        print(f"\nAnalysis complete: {len(towers_analysis)} towers")
        
        return towers_analysis, strain_df
    
    def create_risk_maps(self, towers_analysis):
        """
        Create risk classification maps for deformation-based risk factors.
        """
        rel_vel_neighbor = towers_analysis['relative_velocity_neighbor_cm_year'].abs().dropna()
        rel_vel_surrounding = towers_analysis['relative_velocity_surrounding_cm_year'].abs().dropna()
        strain_values = towers_analysis['strain'].abs().dropna()
        
        neighbor_p60 = np.percentile(rel_vel_neighbor, 60)
        neighbor_p85 = np.percentile(rel_vel_neighbor, 85)
        neighbor_p98 = np.percentile(rel_vel_neighbor, 98)
        
        surrounding_p60 = np.percentile(rel_vel_surrounding, 60)
        surrounding_p85 = np.percentile(rel_vel_surrounding, 85)
        surrounding_p98 = np.percentile(rel_vel_surrounding, 98)
        
        strain_p60 = np.percentile(strain_values, 60)
        strain_p85 = np.percentile(strain_values, 85)
        strain_p98 = np.percentile(strain_values, 98)
        
        towers_analysis['neighbor_risk'] = towers_analysis['relative_velocity_neighbor_cm_year'].apply(
            lambda x: self.classify_risk(abs(x) if pd.notna(x) else 0, neighbor_p60, neighbor_p85, neighbor_p98)
        )
        towers_analysis['surrounding_risk'] = towers_analysis['relative_velocity_surrounding_cm_year'].apply(
            lambda x: self.classify_risk(abs(x) if pd.notna(x) else 0, surrounding_p60, surrounding_p85, surrounding_p98)
        )
        towers_analysis['strain_risk'] = towers_analysis['strain'].apply(
            lambda x: self.classify_risk(abs(x) if pd.notna(x) else 0, strain_p60, strain_p85, strain_p98)
        )
        
        colors = self.get_risk_colors()
        cmap = ListedColormap(colors)
        
        risk_metrics = [
            ('neighbor_risk', 'Relative Velocity Between Nearest Neighbor Towers', 'neighbor_risk_map.png'),
            ('surrounding_risk', 'Relative Velocity Between Tower and Surrounding 30-42 m Range', 'surrounding_risk_map.png'),
            ('strain_risk', 'Strain Between Nearest Neighbor Towers', 'strain_risk_map.png')
        ]
        
        for risk_col, title, filename in risk_metrics:
            self._plot_risk_map(towers_analysis, risk_col, title, filename, colors, cmap)
        
        return towers_analysis

    def _plot_risk_map(self, towers_analysis, risk_col, title, filename, colors, cmap):
        """
        Plot individual risk classification map.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        self.study_area.boundary.plot(ax=ax, edgecolor='black', linewidth=2)
        
        scatter = ax.scatter(
            towers_analysis.geometry.x,
            towers_analysis.geometry.y,
            c=towers_analysis[risk_col],
            cmap=cmap,
            s=15,
            alpha=0.8,
            vmin=0,
            vmax=3,
            edgecolors='none'
        )
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Severe Risk (>98th %ile)',
                markerfacecolor=colors[3], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Moderate Risk (85-98th %ile)',
                markerfacecolor=colors[2], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Slight Risk (60-85th %ile)',
                markerfacecolor=colors[1], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='No Risk (<60th %ile)',
                markerfacecolor=colors[0], markersize=10)
        ]

        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
        
        ax.set_xlabel('Longitude', fontweight='bold', fontsize=12)
        ax.set_ylabel('Latitude', fontweight='bold', fontsize=12)
        ax.set_title(f'Risk Classification Map\n{title}', fontweight='bold', fontsize=14)
        
        self._add_map_elements(ax)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _add_map_elements(self, ax):
        """
        Add scale bar and north arrow to map.
        """
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        scale_distance_km = 10
        scale_length_deg = scale_distance_km / 111
        ax.plot([x_max - 0.15, x_max - 0.15 + scale_length_deg], 
               [y_min + 0.05, y_min + 0.05], 'k-', linewidth=3)
        ax.text(x_max - 0.075, y_min + 0.08, f'{scale_distance_km} km',
               ha='center', fontweight='bold', fontsize=10)
        
        arrow_y = y_max - 0.15
        arrow_x = x_max - 0.05
        ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 0.1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                   fontsize=16, fontweight='bold', ha='center')

    def create_env_risk_maps(self, towers_analysis, risk_classifications):
        """
        Create risk classification maps for environmental factors.
        """
        colors = self.get_risk_colors()
        cmap = ListedColormap(colors)
        
        for factor_name, percentiles in risk_classifications.items():
            risk_col = f'{factor_name}_risk'
            if risk_col not in towers_analysis.columns:
                continue
            
            title_name = factor_name.replace('_', ' ').title()
            filename = f'{factor_name}_risk_map.png'
            
            self._plot_risk_map(towers_analysis, risk_col, title_name, filename, colors, cmap)
    
    def create_union_map(self, towers_analysis):
        """
        Create combined union risk map showing maximum risk across all factors.
        """
        colors = self.get_risk_colors()
        cmap = ListedColormap(colors)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        self.study_area.boundary.plot(ax=ax, edgecolor='black', linewidth=2)
        
        scatter = ax.scatter(
            towers_analysis.geometry.x,
            towers_analysis.geometry.y,
            c=towers_analysis['union_risk'],
            cmap=cmap,
            s=15,
            alpha=0.8,
            vmin=0,
            vmax=3,
            edgecolors='none'
        )
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Severe Risk',
                  markerfacecolor=colors[3], markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Moderate Risk',
                  markerfacecolor=colors[2], markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Slight Risk',
                  markerfacecolor=colors[1], markersize=12),
            Line2D([0], [0], marker='o', color='w', label='No Risk',
                  markerfacecolor=colors[0], markersize=12)
        ]

        ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
        
        ax.set_xlabel('Longitude', fontweight='bold', fontsize=13)
        ax.set_ylabel('Latitude', fontweight='bold', fontsize=13)
        ax.set_title('Union Risk Classification Map\nCombined All Risk Factors', 
                    fontweight='bold', fontsize=15)
        
        self._add_map_elements(ax)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'union_risk_map.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_vel_map(self, towers_analysis, tower_id1=None, tower_id2=None):
        """
        Create displacement velocity map with optional tower pair highlight.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        self.study_area.boundary.plot(ax=ax, edgecolor='black', linewidth=2)
        
        scatter = ax.scatter(
            towers_analysis.geometry.x,
            towers_analysis.geometry.y,
            c=towers_analysis['velocity_cm_year'],
            cmap='RdYlBu_r',
            s=15,
            alpha=0.8,
            edgecolors='none'
        )
        
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Velocity (cm/year)', fontweight='bold', fontsize=12)
        
        if tower_id1 is not None and tower_id2 is not None:
            tower1 = towers_analysis[towers_analysis['tower_id'] == tower_id1].iloc[0]
            tower2 = towers_analysis[towers_analysis['tower_id'] == tower_id2].iloc[0]
            
            x_coords = [tower1.geometry.x, tower2.geometry.x]
            y_coords = [tower1.geometry.y, tower2.geometry.y]
            
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            radius_deg = 0.050
            
            circle = plt.Circle((center_x, center_y), radius_deg, 
                               fill=False, edgecolor='red', linewidth=3, 
                               linestyle='-', zorder=5)
            ax.add_patch(circle)
        
        ax.set_xlabel('Longitude', fontweight='bold', fontsize=12)
        ax.set_ylabel('Latitude', fontweight='bold', fontsize=12)
        ax.set_title('Displacement Velocity Map', fontweight='bold', fontsize=14)
        
        self._add_map_elements(ax)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'velocity_map.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_zoom_view(self, towers_analysis, tower_id1, tower_id2):
        """
        Create zoomed view of specific tower pair with nearby towers.
        """
        tower1 = towers_analysis[towers_analysis['tower_id'] == tower_id1].iloc[0]
        tower2 = towers_analysis[towers_analysis['tower_id'] == tower_id2].iloc[0]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        x_coords = [tower1.geometry.x, tower2.geometry.x]
        y_coords = [tower1.geometry.y, tower2.geometry.y]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        padding_x = max(0.01, x_range * 0.3)
        padding_y = max(0.01, y_range * 0.3)
        
        x_min, x_max = min(x_coords) - padding_x, max(x_coords) + padding_x
        y_min, y_max = min(y_coords) - padding_y, max(y_coords) + padding_y
        
        nearby = towers_analysis.cx[x_min:x_max, y_min:y_max]
        
        vmin = nearby['velocity_cm_year'].min()
        vmax = nearby['velocity_cm_year'].max()
        
        ax.scatter(nearby.geometry.x, nearby.geometry.y,
                c=nearby['velocity_cm_year'], cmap='RdYlBu_r',
                s=100, alpha=0.8, vmin=vmin, vmax=vmax, zorder=2)
        
        ax.scatter([tower1.geometry.x], [tower1.geometry.y],
                s=200, c=[tower1['velocity_cm_year']], cmap='RdYlBu_r',
                vmin=vmin, vmax=vmax, zorder=3)
        ax.scatter([tower2.geometry.x], [tower2.geometry.y],
                s=200, c=[tower2['velocity_cm_year']], cmap='RdYlBu_r',
                vmin=vmin, vmax=vmax, zorder=3)
        
        offset_x = (x_max - x_min) * 0.05
        offset_y = (y_max - y_min) * 0.05
        
        ax.text(tower1.geometry.x, tower1.geometry.y + offset_y, 
                f'Tower {tower_id1}\n{tower1["velocity_cm_year"]:.3f} cm/yr',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.9))
        
        ax.text(tower2.geometry.x, tower2.geometry.y - offset_y, 
                f'Tower {tower_id2}\n{tower2["velocity_cm_year"]:.3f} cm/yr',
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='blue', alpha=0.9))
        
        ax.plot(x_coords, y_coords, 'k--', linewidth=2, alpha=0.5, zorder=1)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel('Longitude', fontweight='bold', fontsize=12)
        ax.set_ylabel('Latitude', fontweight='bold', fontsize=12)
        ax.set_title(f'Zoomed View: Towers {tower_id1} and {tower_id2}', 
                    fontweight='bold', fontsize=14)
        
        scatter = ax.scatter([], [], c=[], cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Velocity (cm/year)', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / f'zoomed_view_{tower_id1}_{tower_id2}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_displacement(self, tower_id1, tower_id2):
        """
        Plot displacement time series comparison for two towers.
        """
        csv_file1 = None
        csv_file2 = None
        
        for county_dir in self.csv_folder.iterdir():
            if county_dir.is_dir():
                potential_file1 = county_dir / f"tower_{tower_id1}_9pixel_displacement_coherence.csv"
                potential_file2 = county_dir / f"tower_{tower_id2}_9pixel_displacement_coherence.csv"
                
                if potential_file1.exists():
                    csv_file1 = potential_file1
                if potential_file2.exists():
                    csv_file2 = potential_file2
        
        if csv_file1 is None or csv_file2 is None:
            print(f"Warning: CSV files not found for towers {tower_id1} and/or {tower_id2}")
            return
        
        df1, _ = self.load_timeseries(str(csv_file1))
        df2, _ = self.load_timeseries(str(csv_file2))
        
        if df1 is None or df2 is None:
            print(f"Warning: Could not load timeseries for towers {tower_id1} and/or {tower_id2}")
            return
        
        velocity1 = self.calc_velocity(df1)
        velocity2 = self.calc_velocity(df2)
        
        start_time1 = pd.to_datetime(df1['start_time'].iloc[0])
        start_time2 = pd.to_datetime(df2['start_time'].iloc[0])
        
        start_row1 = pd.DataFrame({
            'date': [start_time1],
            'pixel_5_displacement_cm': [0]
        })
        df1_with_start = pd.concat([start_row1, df1[['date', 'pixel_5_displacement_cm']]], ignore_index=True)
        df1_with_start = df1_with_start.sort_values('date').reset_index(drop=True)
        
        start_row2 = pd.DataFrame({
            'date': [start_time2],
            'pixel_5_displacement_cm': [0]
        })
        df2_with_start = pd.concat([start_row2, df2[['date', 'pixel_5_displacement_cm']]], ignore_index=True)
        df2_with_start = df2_with_start.sort_values('date').reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        line1 = ax.plot(df1_with_start['date'], df1_with_start['pixel_5_displacement_cm'], 
                'o', label=f'Tower {tower_id1}', linewidth=2, markersize=4, alpha=0.7)
        line2 = ax.plot(df2_with_start['date'], df2_with_start['pixel_5_displacement_cm'], 
                's', label=f'Tower {tower_id2}', linewidth=2, markersize=4, alpha=0.7)
        
        color1 = line1[0].get_color()
        color2 = line2[0].get_color()
        
        velocity1_cm_day = velocity1 / 365.25
        velocity2_cm_day = velocity2 / 365.25
        
        time_days1_with_zero = (df1_with_start['date'] - df1_with_start['date'].min()).dt.days
        time_days2_with_zero = (df2_with_start['date'] - df2_with_start['date'].min()).dt.days
        
        df1_actual = df1[['date', 'pixel_5_displacement_cm']].copy()
        df2_actual = df2[['date', 'pixel_5_displacement_cm']].copy()
        
        time_days1_actual = (df1_actual['date'] - start_time1).dt.days
        time_days2_actual = (df2_actual['date'] - start_time2).dt.days
        
        mean_time1 = time_days1_actual.mean()
        mean_displacement1 = df1_actual['pixel_5_displacement_cm'].mean()
        intercept1 = mean_displacement1 - velocity1_cm_day * mean_time1
        
        mean_time2 = time_days2_actual.mean()
        mean_displacement2 = df2_actual['pixel_5_displacement_cm'].mean()
        intercept2 = mean_displacement2 - velocity2_cm_day * mean_time2
        
        trend1 = velocity1_cm_day * time_days1_with_zero + intercept1
        trend2 = velocity2_cm_day * time_days2_with_zero + intercept2
        
        from scipy.stats import pearsonr
        time_days1_original = (df1['date'] - df1['date'].min()).dt.days
        time_days2_original = (df2['date'] - df2['date'].min()).dt.days
        
        r_value1, _ = pearsonr(time_days1_original, df1['pixel_5_displacement_cm'])
        r_value2, _ = pearsonr(time_days2_original, df2['pixel_5_displacement_cm'])
        
        ax.plot(df1_with_start['date'], trend1, '--', linewidth=3, alpha=0.8, color=color1,
                label=f'Tower {tower_id1} trend: {velocity1:.2f} cm/yr (R²={r_value1**2:.3f})')
        ax.plot(df2_with_start['date'], trend2, '--', linewidth=3, alpha=0.8, color=color2,
                label=f'Tower {tower_id2} trend: {velocity2:.2f} cm/yr (R²={r_value2**2:.3f})')
        
        ax.set_xlabel('Date', fontweight='bold', fontsize=12)
        ax.set_ylabel('Displacement (cm)', fontweight='bold', fontsize=12)
        ax.set_title(f'Displacement Time Series\nTowers {tower_id1} and {tower_id2}', 
                    fontweight='bold', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / f'displacement_comparison_{tower_id1}_{tower_id2}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_relative_disp(self, tower_id1, tower_id2, distance_m):
        """
        Plot relative displacement between two towers over time.
        """
        csv_file1 = None
        csv_file2 = None
        
        for county_dir in self.csv_folder.iterdir():
            if county_dir.is_dir():
                potential_file1 = county_dir / f"tower_{tower_id1}_9pixel_displacement_coherence.csv"
                potential_file2 = county_dir / f"tower_{tower_id2}_9pixel_displacement_coherence.csv"
                
                if potential_file1.exists():
                    csv_file1 = potential_file1
                if potential_file2.exists():
                    csv_file2 = potential_file2
        
        if csv_file1 is None or csv_file2 is None:
            print(f"Warning: CSV files not found for towers {tower_id1} and/or {tower_id2}")
            return
        
        df1, _ = self.load_timeseries(str(csv_file1))
        df2, _ = self.load_timeseries(str(csv_file2))
        
        if df1 is None or df2 is None:
            print(f"Warning: Could not load timeseries for towers {tower_id1} and/or {tower_id2}")
            return
        
        velocity1 = self.calc_velocity(df1)
        velocity2 = self.calc_velocity(df2)
        relative_velocity_calculated = velocity1 - velocity2
        
        df1_merge = df1[['date', 'pixel_5_displacement_cm']].copy()
        df2_merge = df2[['date', 'pixel_5_displacement_cm']].copy()
        
        df1_merge['date'] = pd.to_datetime(df1_merge['date'])
        df2_merge['date'] = pd.to_datetime(df2_merge['date'])
        
        merged = pd.merge(df1_merge, df2_merge, on='date', suffixes=('_t1', '_t2'), how='inner')
        
        if len(merged) == 0:
            print(f"Warning: No overlapping dates for towers {tower_id1} and {tower_id2}")
            return
        
        merged['relative_displacement_cm'] = merged['pixel_5_displacement_cm_t1'] - merged['pixel_5_displacement_cm_t2']
        
        merged = merged.sort_values('date').reset_index(drop=True)
        
        earliest_date = merged['date'].min()
        start_row = pd.DataFrame({
            'date': [earliest_date],
            'relative_displacement_cm': [0.0]
        })
        merged_with_zero = pd.concat([start_row, merged], ignore_index=True)
        merged_with_zero = merged_with_zero.sort_values('date').reset_index(drop=True)
        merged_with_zero = merged_with_zero.drop_duplicates(subset=['date'], keep='first')
        
        merged_with_zero['date'] = pd.to_datetime(merged_with_zero['date'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        line = ax.plot(merged_with_zero['date'], merged_with_zero['relative_displacement_cm'], 
                'o', linewidth=2, markersize=5, alpha=0.7, label='Relative Displacement')
        data_color = line[0].get_color()
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        time_days_with_zero = (merged_with_zero['date'] - merged_with_zero['date'].min()).dt.days
        
        velocity_cm_day = relative_velocity_calculated / 365.25
        
        time_days_actual = (merged['date'] - merged['date'].min()).dt.days
        mean_time = time_days_actual.mean()
        mean_displacement = merged['relative_displacement_cm'].mean()
        intercept = mean_displacement - velocity_cm_day * mean_time
        
        trend = velocity_cm_day * time_days_with_zero + intercept
        
        from scipy.stats import pearsonr
        r_value, _ = pearsonr(time_days_actual, merged['relative_displacement_cm'])
        r_squared = r_value ** 2
        
        ax.plot(merged_with_zero['date'], trend, '--', linewidth=3, alpha=0.8, color=data_color,
                label=f'Trend: {relative_velocity_calculated:.2f} cm/yr (R²={r_squared:.3f})')
        
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax.set_xlabel('Date', fontweight='bold', fontsize=12)
        ax.set_ylabel('Relative Displacement (cm)', fontweight='bold', fontsize=12)
        ax.set_title(f'Relative Displacement\nTowers {tower_id1} - {tower_id2} (Towers Distance: {distance_m:.1f} m)', 
                    fontweight='bold', fontsize=14)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / f'relative_displacement_{tower_id1}_{tower_id2}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_histogram(self, data, title, xlabel, filename, color='steelblue', use_scientific=False):
        """
        Plot histogram with ridgeline density plot.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                        gridspec_kw={'height_ratios': [3, 1]}, 
                                        sharex=True)
        
        if 'storm_surge' in filename.lower() or 'storm surge' in title.lower():
            data_range = data.max() - data.min()
            num_bins = int(np.ceil(data_range / 0.5))
        else:
            num_bins = 50
        
        counts, bins, patches = ax1.hist(data, bins=num_bins, edgecolor='black', 
                                        alpha=0.7, color=color, rwidth=1.0)
        ax1.set_yscale('log')
        ax1.set_ylabel('Frequency (log scale)', fontweight='bold', fontsize=12)
        ax1.set_title(title, fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3, which='both')
        
        p60 = np.percentile(data, 60)
        p85 = np.percentile(data, 85)
        p98 = np.percentile(data, 98)
        
        if use_scientific:
            label_60 = f'60th %ile: {p60:.3e}'
            label_85 = f'85th %ile: {p85:.3e}'
            label_98 = f'98th %ile: {p98:.3e}'
        else:
            label_60 = f'60th %ile: {p60:.3f}'
            label_85 = f'85th %ile: {p85:.3f}'
            label_98 = f'98th %ile: {p98:.3f}'
        
        ax1.axvline(p60, color='green', linestyle='--', linewidth=3, label=label_60)
        ax1.axvline(p85, color='blue', linestyle='--', linewidth=3, label=label_85)
        ax1.axvline(p98, color='red', linestyle='--', linewidth=3, label=label_98)
        ax1.legend(fontsize=10, loc='upper right')
        
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 500)
        density = kde(x_range)
        
        ax2.fill_between(x_range, density, alpha=0.6, color=color)
        ax2.plot(x_range, density, color='black', linewidth=1.5)
        ax2.set_xlabel(xlabel, fontweight='bold', fontsize=12)
        ax2.set_ylabel('Density', fontweight='bold', fontsize=12)
        ax2.set_ylim(bottom=0)
        
        ax2.axvline(p60, color='green', linestyle='--', linewidth=3, alpha=0.7)
        ax2.axvline(p85, color='blue', linestyle='--', linewidth=3, alpha=0.7)
        ax2.axvline(p98, color='red', linestyle='--', linewidth=3, alpha=0.7)
        
        if use_scientific:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 2))
            ax1.xaxis.set_major_formatter(formatter)
            ax2.xaxis.set_major_formatter(formatter)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results(self, towers_analysis, strain_df):
        """
        Export all analysis results including maps, plots, and Excel summary.
        """
        towers_analysis, best_storm_category, pca, explained_var = self.run_pca(towers_analysis)
        
        if len(strain_df) > 0:
            max_rel_vel_row = strain_df.loc[strain_df['relative_velocity_neighbor_cm_year'].abs().idxmax()]
            tower_id1 = int(max_rel_vel_row['tower_id'])
            tower_id2 = int(max_rel_vel_row['neighbor_tower_id'])
            distance_m = max_rel_vel_row['distance_m']
            
            self.create_vel_map(towers_analysis, tower_id1, tower_id2)
            self.create_zoom_view(towers_analysis, tower_id1, tower_id2)
            self.plot_displacement(tower_id1, tower_id2)
            self.plot_relative_disp(tower_id1, tower_id2, distance_m)
        
        if len(strain_df) > 0 and strain_df['strain'].notna().sum() > 0:
            strain_values = strain_df['strain'].dropna()
            self.plot_histogram(
                strain_values, 'Distribution of Strain Between Neighboring Towers',
                'Strain (dimensionless)', 'strain_distribution_histogram_ridgeline.png',
                color='steelblue', use_scientific=True
            )
        
        if len(towers_analysis) > 0:
            rel_vel_neighbor = towers_analysis['relative_velocity_neighbor_cm_year'].dropna()
            rel_vel_surrounding = towers_analysis['relative_velocity_surrounding_cm_year'].dropna()
            
            self.plot_histogram(
                rel_vel_neighbor, 'Relative Velocity Between Nearest Neighbor Towers',
                'Relative Velocity (cm/year)', 'relative_velocity_neighbor_histogram_ridgeline.png',
                color='coral', use_scientific=False
            )
            
            self.plot_histogram(
                rel_vel_surrounding, 'Relative Velocity Between Tower and Surrounding 30-42 m Range',
                'Relative Velocity (cm/year)', 'relative_velocity_surrounding_histogram_ridgeline.png',
                color='mediumseagreen', use_scientific=False
            )
        
        env_factors_to_plot = []
        
        for factor_name in self.env_rasters.keys():
            if 'storm_surge_cat' not in factor_name and factor_name in towers_analysis.columns:
                env_factors_to_plot.append(factor_name)
        
        if best_storm_category and best_storm_category in towers_analysis.columns:
            env_factors_to_plot.append(best_storm_category)
        
        for factor_name in env_factors_to_plot:
            factor_data = towers_analysis[factor_name]
            non_zero_data = factor_data[factor_data > 0]
            
            if len(non_zero_data) > 1:
                title_name = factor_name.replace('_', ' ').title()
                
                if 'storm_surge' in factor_name:
                    xlabel = 'Maximum Envelopes of Water Inundation (feet)'
                elif 'sea_level_rise' in factor_name:
                    xlabel = f'{title_name} (m)'
                else:
                    xlabel = title_name
                
                self.plot_histogram(
                    non_zero_data, f'Distribution of {title_name}', xlabel,
                    f'{factor_name}_distribution_histogram_ridgeline.png',
                    color='steelblue', use_scientific=False
                )
        
        towers_analysis = self.create_risk_maps(towers_analysis)
        
        towers_analysis, risk_classifications = self.classify_env_risks_selected(towers_analysis, best_storm_category)
        self.create_env_risk_maps(towers_analysis, risk_classifications)
        towers_analysis = self.calc_union_risk(towers_analysis)
        self.create_union_map(towers_analysis)
        
        county_ranking = self._make_county_ranking(towers_analysis)
        if county_ranking is not None:
            print("\n" + "="*70)
            print("COUNTY RISK RANKING (by High Risk Towers)")
            print("="*70)
            print(county_ranking.to_string())
            print("="*70 + "\n")
        
        towers_export = towers_analysis.drop(columns=['geometry'])
        excel_path = self.output_folder / 'tower_analysis_results.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            towers_export.to_excel(writer, sheet_name='Tower Metrics', index=False)
            
            risk_summary = self._make_risk_summary(towers_analysis)
            risk_summary.to_excel(writer, sheet_name='Risk Summary', index=True)
            
            county_ranking = self._make_county_ranking(towers_analysis)
            if county_ranking is not None:
                county_ranking.to_excel(writer, sheet_name='County Risk Ranking', index=True)
            
            factor_risks = self._make_factor_summary(towers_analysis)
            factor_risks.to_excel(writer, sheet_name='Factor Risk Breakdown', index=True)
            
            if best_storm_category is not None:
                pca_summary = self._make_pca_summary(explained_var, best_storm_category)
                pca_summary.to_excel(writer, sheet_name='Storm Surge PCA', index=True)
        
        print(f"\nResults exported to: {self.output_folder}")
        print(f"Excel: tower_analysis_results.xlsx")
        print(f"Visualizations: {len(list(self.output_folder.glob('*.png')))} files\n")
    
    def _make_risk_summary(self, towers_analysis):
        """
        Create summary statistics of risk levels.
        """
        summary_data = {
            'Total Towers': [len(towers_analysis)],
            'Severe Risk': [(towers_analysis['union_risk'] == 3).sum()],
            'Moderate Risk': [(towers_analysis['union_risk'] == 2).sum()],
            'Slight Risk': [(towers_analysis['union_risk'] == 1).sum()],
            'No Risk': [(towers_analysis['union_risk'] == 0).sum()],
        }
        return pd.DataFrame(summary_data).T.rename(columns={0: 'Count'})
    
    def _make_factor_summary(self, towers_analysis):
        """
        Create risk breakdown by individual factors.
        """
        risk_columns = []
        
        for col in ['neighbor_risk', 'surrounding_risk', 'strain_risk']:
            if col in towers_analysis.columns:
                risk_columns.append(col)
        
        for factor_name in self.env_rasters.keys():
            risk_col = f'{factor_name}_risk'
            if risk_col in towers_analysis.columns:
                risk_columns.append(risk_col)
        
        summary = {}
        for col in risk_columns:
            factor_name = col.replace('_risk', '').replace('_', ' ').title()
            summary[factor_name] = {
                'Severe': (towers_analysis[col] == 3).sum(),
                'Moderate': (towers_analysis[col] == 2).sum(),
                'Slight': (towers_analysis[col] == 1).sum(),
                'No Risk': (towers_analysis[col] == 0).sum()
            }
        
        return pd.DataFrame(summary).T
    
    def _make_pca_summary(self, explained_var, best_category):
        """
        Create PCA analysis summary for storm surge category selection.
        """
        n_components = len(explained_var)
        cumulative = np.cumsum(explained_var)
        
        summary = pd.DataFrame({
            'Variance Explained': explained_var,
            'Cumulative Variance': cumulative
        }, index=[f'PC{i+1}' for i in range(n_components)])
        
        summary.loc['Selected Category'] = [best_category, '']
        
        return summary
    
    def _make_county_ranking(self, towers_analysis):
        """
        Create county-level risk ranking table.
        """
        if 'county_name' not in towers_analysis.columns:
            return None
        
        county_stats = []
        
        for county in towers_analysis['county_name'].unique():
            county_towers = towers_analysis[towers_analysis['county_name'] == county]
            
            total_towers = len(county_towers)
            severe_risk = (county_towers['union_risk'] == 3).sum()
            moderate_risk = (county_towers['union_risk'] == 2).sum()
            slight_risk = (county_towers['union_risk'] == 1).sum()
            no_risk = (county_towers['union_risk'] == 0).sum()
            
            county_stats.append({
                'County': county,
                'Total Towers': total_towers,
                'Severe Risk': severe_risk,
                'Moderate Risk': moderate_risk,
                'Slight Risk': slight_risk,
                'No Risk': no_risk,
                'Severe %': (severe_risk / total_towers * 100) if total_towers > 0 else 0,
                'Moderate %': (moderate_risk / total_towers * 100) if total_towers > 0 else 0
            })
        
        ranking_df = pd.DataFrame(county_stats)
        ranking_df = ranking_df.sort_values('Severe Risk', ascending=False)
        ranking_df = ranking_df.reset_index(drop=True)
        ranking_df.index = ranking_df.index + 1
        ranking_df.index.name = 'Rank'
        
        return ranking_df


if __name__ == "__main__":
    base_path = '/home/zchen66/2025Fall/CIVE 6381 - Applied Geospatial Computations/TransmissionLines_DALYMarg/NWS_StormSturge'
    
    env_rasters = {
        'sea_level_rise_1_5ft': '/home/zchen66/2025Fall/CIVE 6381 - Applied Geospatial Computations/TransmissionLines_DALYMarg/NOAASLR/*SLR/*_slr_depth_1.5ft.tif',
        'storm_surge_cat1': f'{base_path}/us_Category1_MOM_Inundation_HIGH.tif',
        'storm_surge_cat2': f'{base_path}/us_Category2_MOM_Inundation_HIGH.tif',
        'storm_surge_cat3': f'{base_path}/us_Category3_MOM_Inundation_HIGH.tif',
        'storm_surge_cat4': f'{base_path}/us_Category4_MOM_Inundation_HIGH.tif',
        'storm_surge_cat5': f'{base_path}/us_Category5_MOM_Inundation_HIGH.tif'
    }
    
    analyzer = TowerAnalysis(
        csv_folder_path="/data-new/zchen66/vertical_disp_nomask",
        shapefile_path="/home/zchen66/2025Fall/CIVE 6381 - Applied Geospatial Computations/PowerTower_WGS84_numbers/PowerTower_WGS84_numbers.shp",
        study_area_path="/home/zchen66/2025Fall/CIVE 6381 - Applied Geospatial Computations/counties_shapefile/",
        output_folder="/home/zchen66/2025Fall/CIVE 6381 - Applied Geospatial Computations/results/1106_whole_region",
        env_rasters=env_rasters
    )
    
    towers_analysis, strain_df = analyzer.run_analysis()
    
    if towers_analysis is not None:
        analyzer.export_results(towers_analysis, strain_df)