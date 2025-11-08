# Geospatial Analysis of Transmission Infrastructure Risk under Land Deformation and Environmental Change
Applied Geospatial Computation Lab Project

## 🎯 Big Picture: Why This Project Matters
The Gulf Coast is a critical hub for the United States' energy sector, facing increasing threats from climate change (e.g., severe storms, sea-level rise) and active geological processes (e.g., land subsidence). Maintaining the stability and resilience of its vast transmission line infrastructure is paramount for national energy security and the region's economy.

This project utilizes cutting-edge InSAR (Interferometric Synthetic Aperture Radar) data from the NASA/JPL OPERA mission to systematically assess and classify the deformation risk of transmission towers across the Gulf Coast.

The ultimate goal is to produce actionable recommendations for transmission line planning, siting, and maintenance to support the Gulf Coast's growing energy needs and enhance the resilience of the energy grid against environmental hazards.


## ⚙️ Project Workflow Overview
The analysis is structured into a three-step pipeline:

1. Data Acquisition: Automated search and download of OPERA Line-of-Sight (LOS) displacement data files.

2. Data Pre-Processing: Conversion of raw LOS displacement to vertical displacement, masking, and re-projection to WGS84.

3. Risk Classification & Analysis: Calculation of velocity, strain, and environmental risk factors, followed by comprehensive risk classification and mapping.


## 🚀 Getting Started
### 1. Environment Setup
The required packages for this project can be installed using the provided environment.yml file.

#### In terminal:

1. conda env create -f environment.yml  #Create the conda environment

2. conda activate transmission  #Activate the new environment


### 2. Execution
#### Step 1: Download OPERA Displacement Data (download_opera_disp_data.py)
This requires Earthdata Login credentials to access and download the data from the Alaska Satellite Facility (ASF).
#### Step 2: Extract Vertical Displacement from OPERA Data (Opera-Vertical-Mask-Reproject-Processcer.py)
This step processes the raw H5 files to extract usable vertical displacement data.
#### Step 3: Deformation Classification (deform_risk_classi(whole).py)
This script performs the core analysis, integrating time-series metrics and environmental data to assign a final risk score.


#### In terminal:
1. python download_opera_disp_data.py
2. python "Opera-Vertical-Mask-Reproject-Processcer.py" <input_dir> <output_dir> <shapefile_path> 1
3. python deform_risk_classi(whole).py
   
#### ⚡️ To run deform_risk_classi(whole).py in a Jupyter Notebook or terminal, please first modify the paths:

if __name__ == "__main__":
    base_path = '...' # Adjust path

    env_rasters = {
        # Adjust paths to your environmental data
        'sea_level_rise_1_5ft': '...', 
        'storm_surge_cat1': '...',
        # ... and so on
    }

    analyzer = TowerAnalysis(
        csv_folder_path="/data-new/zchen66/vertical_disp_nomask", # IMPORTANT: This is the directory containing the CSV files generated from the Time-Series Extraction Script (not provided but assumed to be run next)
        shapefile_path="...", # Path to tower shapefile
        study_area_path="...", # Path to county shapefile folder
        output_folder="...", # Path for output maps/reports
        env_rasters=env_rasters
    )

    towers_analysis, strain_df = analyzer.run_analysis()

    if towers_analysis is not None:
        analyzer.export_results(towers_analysis, strain_df)


