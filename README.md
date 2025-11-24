# Geospatial Analysis of Transmission Infrastructure Risk under Land Deformation and Environmental Change ⚡️

<p align="center">
  <strong>Zih-Syun Chen<sup>1</sup>, Peng Zhang<sup>1</sup>, Pietro Milillo<sup>1</sup></strong>
</p>

<p align="center">
  <sup>1</sup> Department of Civil and Environmental Engineering, University of Houston, Houston, TX, USA<br>
</p>

---

## Introduction

The **Gulf Coast** is a critical hub for the United States' energy sector, facing increasing threats from climate change (e.g., severe storms, sea-level rise) and active geological processes (e.g., land subsidence). Maintaining the stability and resilience of the transmission line is **important** for energy security and the economy.

This project utilizes **OPERA DISP Sentinel-1 products** from the NASA/JPL to systematically assess and classify the deformation risk of power towers across the Gulf Coast.

**The ultimate goal is to produce actionable recommendations for transmission line planning, siting, and maintenance to support the Gulf Coast's growing energy needs and enhance the resilience of the energy grid against environmental hazards.**


## Overview

The full pipeline includes three steps:

1. Download Data

2. Opera Data Process

3. Risk Classification


The full dataset is quite large, and the entire process (from the first to the last step) may take hours. Therefore, we provide the input files for each step.  **{3. Risk Classification}** can be run directly for quick review after setting up the environment.yml.


In Step 2,  the input for this step consists of the output generated in Step 1, as well as additional files including the geographic boundaries of 50 counties, and shapefiles for transmission towers and substations. You do not necessarily need to use the results from Step 1 — you can directly use the data ( [Opera-Processor-data-demo](https://uofh-my.sharepoint.com/:f:/r/personal/pzhang27_cougarnet_uh_edu/Documents/Opera-Processor-data-demo?csf=1&web=1&e=NvRfLW)) we provide to verify this step. However, since the Step 2 dataset contains some information that we prefer not to release publicly at this time, please send a request to **pzhang27@cougarnet.uh.edu** if you would like to access the demo data.

---


## Installation

1. Clone the repository:
```bash
   git clone https://github.com/Milillo-lab/Transmission_Lines.git
   cd Transmission_Lines
```

2. Create and activate the conda environment:
```bash
   conda env create -f environment.yml
   conda activate transmission
```
---


## Run the code

### 1. Download Data (An Earthdata account is required)

Downloading the data will take some time. It requires Earthdata's account and password for credentials(https://urs.earthdata.nasa.gov/).

Run this code in your terminal.

```Bash
cd code
python download_opera_disp_data.py
```

The outputs from  **1. Download Data** is the input for **2. Opera Data Process**.

.........................................................................................................................................................................


### 2. Opera Data Process

The content is divided into two sections. Section 1 presents a detailed explanation of the scripts, and Section 2 describes the execution process and usage instructions.

#### 2.1: scripts description

We use the **Opera-Processor** package to process the OPERA data. It contains five Python scripts and a *requirements.txt* file.

2.1.1 **automated_comprehensive_processor.py**

This script provides a comprehensive 4-step data processing pipeline capable of automatically processing multiple counties

(1)**Mask-Reproject-Process** - GeoTIFF file generation and preprocessing

(2)**Tower Time Series Extraction** - Extract power tower time series data

(3)**Substation Extraction** - Extract substation data (intersection + outer-ring pixels)

(4)**Displacement-Coherence Average** - Cross-frame averaging of displacement and coherence

2.1.2 **Opera-Vertical-Mask-Reproject-Processcer.py**

The **Opera-Vertical-Mask-Reproject-Processcer** is a processing tool designed to convert raw OPERA InSAR H5 files into standardized GeoTIFF format for subsequent analysis. 

This processor reads displacement and temporal coherence data from OPERA HDF5 files, converts Line-of-Sight (LOS) displacement to vertical displacement using incidence angle interpolation, applies configurable masking based on user-defined thresholds or recommended masks, automatically detects and converts coordinate systems from UTM to WGS84 EPSG:4326, and clips data to county boundaries using shapefiles. 

The output consists of dual-band GeoTIFF files with standardized naming format `OPERA_{DISPLACEMENT_TYPE}_COHERENCE_{MASK}_REPROJECTED_{FRAME}_{START_DATE}_{END_DATE}.tif` where each file contains Band 1 (displacement data in millimeters) and Band 2 (temporal coherence values 0.0-1.0), along with comprehensive processing logs and quality statistics. 

2.1.3 **Opera-TimeSeries-Tower-Processor.py**

The **Opera-TimeSeries-Tower-Processor** is a tool designed to extract and analyze time series data from OPERA GeoTIFF files specifically for power tower infrastructure monitoring. 

This processor reads dual-band GeoTIFF files containing vertical displacement and temporal coherence data, identifies power tower locations from shapefile data, extracts displacement and coherence values from a 9-pixel area surrounding each tower center (3x3 pixel window), processes temporal sequences across multiple satellite acquisition dates to create comprehensive time series datasets, and generates both individual tower data files and aggregated summary statistics for infrastructure monitoring applications. 

The output consists of detailed CSV files named `tower_{tower_id}_9pixel_displacement_coherence.csv` containing time series data with columns for dates, displacement values, coherence measurements, and coordinate information for each of the 9 pixels, plus a comprehensive summary file `powertower_9pixel_displacement_coherence_summary.csv` that aggregates data from all towers with statistical analysis including mean displacement, standard deviation, and quality metrics. 

Note: You will see 9-pixel number of each`tower_{tower_id}_9pixel_displacement_coherence.csv`. The **9-pixel** concept refers to a specific spatial sampling strategy used in geospatial analysis where each tower or infrastructure point is analyzed using a 3×3 pixel grid centered on the exact tower location. Here's what each of the 9 pixels represents

```
┌─────────┐
│ 1  2  3 │  ← Top row (pixels above the tower)
├─────────┤
│ 4  5  6 │  ← Middle row (pixel 5 = exact tower center)
├─────────┤
│ 7  8  9 │  ← Bottom row (pixels below the tower)
└─────────┘
```

- Pixel 1: Upper-left corner (northwest)
- Pixel 2: Upper-center (directly north)
- Pixel 3: Upper-right corner (northeast)
- Pixel 4: Middle-left (directly west)
- Pixel 5: **Center pixel** (exact tower location)
- Pixel 6: Middle-right (directly east)
- Pixel 7: Lower-left corner (southwest)
- Pixel 8: Lower-center (directly south)
- Pixel 9: Lower-right corner (southeast)

2.1.4 **Opera-TimeSeries-Substation-Processor.py**

The **Opera-TimeSeries-Substation-Processor** is tool designed specifically for extracting and analyzing time series data from OPERA GeoTIFF files at electrical substation locations. 

This processor combines intersected pixel analysis (capturing pixels directly at the substation location) with outer-ring pixel sampling (collecting data from the surrounding 3×3 pixel neighborhood) to provide both point-specific and spatially-contextual displacement and coherence measurements [you can see from the following figure.]. The system reads dual-band GeoTIFF files, identifies substation locations from shapefile data, applies the intersection and outer-ring sampling methodology to extract detailed time series data across multiple satellite acquisition dates, processes the temporal sequences to create comprehensive datasets for each substation, and generates both individual Excel workbooks for detailed analysis and a consolidated summary file for aggregate statistics. 

The output includes individual Excel files named `substation_{substation_id}_displacement_coherence.xlsx` containing time series data with displacement measurements, temporal coherence values, coordinate information, and multi-pixel spatial analysis results for each substation, plus a master summary file `substation_displacement_coherence_summary.csv` that aggregates data from all substations with statistical analysis including mean displacement, standard deviation, data quality metrics, and temporal trends.

2.1.5 **Opera-Displacement-Coherence-Average.py** 

This script is a tool designed to compute spatial averages of displacement and temporal coherence values from OPERA InSAR dual-band GeoTIFF files across multiple time frames, processing large raster datasets in memory-efficient chunks using windowed operations to generate county-wide averaged products that represent the mean surface deformation and measurement quality over time.

The output has two primary GeoTIFF files - displacement-average.tif containing the averaged vertical displacement measurements in millimeters and coherence-average.tif containing the corresponding averaged temporal coherence values ranging from 0.0 to 1.0, where the averaging process incorporates data from all available frames within the specified time range (2019 to 2023) while respecting county boundaries through shapefile-based clipping.

#### **2.2：How to run?**

You can access the demo data for one county from [Opera-Processor-data-demo](https://uofh-my.sharepoint.com/:f:/r/personal/pzhang27_cougarnet_uh_edu/Documents/Opera-Processor-data-demo?csf=1&web=1&e=NvRfLW). In general, processing data for a single county takes about 2–3 hours. However, due to the large size of the full dataset, it is not practical to upload all the data. Therefore, a smaller demo dataset is provided to demonstrate the functionality of the **Opera-Processor**.

2.2.1 you need to make sure you have set up appropriate environment. 

2.2.2 you need to change the directory of **automated_comprehensive_processor.py**

   When you access the demo data from [Opera-Processor-data-demo](https://uofh-my.sharepoint.com/:f:/r/personal/pzhang27_cougarnet_uh_edu/Documents/Opera-Processor-data-demo?csf=1&web=1&e=NvRfLW),  simply download all the files (including the five subfolders) to your local computer. After that, you only need to update the file paths in the scripts to match the locations of these five subfolders on your system.

   ```
       # Base path configuration
       processing_base = '/processing'  # the directory of raw OPERA satellite data (if you put Jefferson_22051 in directory ./processing/Jefferson_22051, then processing_base should be ./processing)
       processed_base = '/processed' # the directory of all processed results (if you want to put results in directory ./processed/Jefferson_22051, then processing_base should be ./processed)
       county_shapefile_base = '/counties' # the directory of county administrative boundaries
       tower_shapefile_base = '/powertower_every_county' # the directory of tower locations
       substation_shapefile_base = '/substation_every_county' # the directory of county electrical substations
   ```

2.2.3 command line

   Note: Please ensure you are in Opera-Processor now.

   ```
   cd Opera-Processor # note: cd is must
   
     # Process all counties with all 4 steps
     python automated_comprehensive_processor.py
   
     # Process specific counties only
     python automated_comprehensive_processor.py --counties Baldwin Mobile
   
     # Start from Step 2
     python automated_comprehensive_processor.py --start-step 2
   
     # Run Step 1 only
     python automated_comprehensive_processor.py --only-step 1
   
     # Resume previous incomplete processing
     python automated_comprehensive_processor.py --resume
   
     # Parallel processing (using 2 processes)
     python automated_comprehensive_processor.py --parallel 2
   
     # Parallel processing for all counties (using 4 processes)
     python automated_comprehensive_processor.py --parallel 4
   
     # Skip Step 4 (Displacement-Coherence Average)
     python automated_comprehensive_processor.py --skip-step4
   
     # Process specific counties without Step 4
     python automated_comprehensive_processor.py --counties Baldwin Mobile --skip-step4
   
     # Parallel processing without Step 4
     python automated_comprehensive_processor.py --parallel 2 --skip-step4
   
     # Resume mode, skipping Step 4
     python automated_comprehensive_processor.py --resume --skip-step4
   ```


2.2.4 Results

If you successfully run the code (the whole process will be like about 20 minutes), in your processed_base directory, you will see four subfolders named 'Vertical-Mask-Reproject', 'Vertical-Time-Series-Towers', 'Vertical-Time-Series-Substations' and 'Average'.

### 3. Risk Classification
(Script: deform_risk_classi(whole).py)

This script performs the core spatio-temporal analysis and requires path modifications within the script itself before execution.

Action Required: Open deform_risk_classi(whole).py and update the file paths inside the if __name__ == "__main__": block to match your local setup:

```Python

if __name__ == "__main__":
    base_path = '...' # Adjust path (used as base for environmental data)
    
    env_rasters = {
        # Adjust paths to your environmental data
        'sea_level_rise_1_5ft': '...', 
        'storm_surge_cat1': '...',
        # ... and so on
    }
    
    analyzer = TowerAnalysis(
        # IMPORTANT: This should point to the directory containing the CSV files 
        # generated by the previous time-series extraction step (not shown in the workflow but assumed to precede this step).
        csv_folder_path="/data-new/zchen66/vertical_disp_nomask", 
        shapefile_path="...", # Path to tower shapefile
        study_area_path="...", # Path to county shapefile folder
        output_folder="...", # Path for output maps/reports
        env_rasters=env_rasters
    )
    
    towers_analysis, strain_df = analyzer.run_analysis()
    
    if towers_analysis is not None:
        analyzer.export_results(towers_analysis, strain_df)
```
Once the paths are updated, run the script from your terminal:

```Bash
python deform_risk_classi(whole).py
```







| Step                       | Script                                                       | Function                                                     | Key Output                                                   |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **1. Data Acquisition**    | `download_opera_disp_data.py`                                | Connects to ASF DAAC, searches for OPERA InSAR displacement files for specified counties, and downloads the raw H5 files. | Raw OPERA H5 files                                           |
| **2. Opera data Process**  | `Opera-Processor/automated_comprehensive_processor.py`       | provides the following 4-step data processing pipeline capable of automatically processing multiple counties. | all the results from the following 4 scripts                 |
|                            | `Opera-Processor/Opera-Vertical-Mask-Reproject-Processcer.py` | convert raw OPERA InSAR H5 files into standardized GeoTIFF format (including vertical land motion and coherence) | each GeoTIFF contains Band 1 (displacement) and Band 2 (temporal coherence) |
|                            | `Opera-Processor/Opera-TimeSeries-Tower-Processor.py`        | extracts displacement and coherence values from a 9-pixel area surrounding each tower center (3x3 pixel window) | CSV filescontaining time series data with columns for dates, displacement values, coherence measurements, and coordinate information for each of the 9 pixels |
|                            | `Opera-Processor/Opera-TimeSeries-Substation-Processor.py`   | combines intersected pixel analysis with outer-ring pixel sampling  to provide substation displacement and coherence | Excel includes individual Excel files containing time series data with displacement measurements, temporal coherence values |
|                            | `Opera-Processor/Opera-Displacement-Coherence-Average.py`    | generate county-wide averaged products that represent the mean surface deformation and coherence | displacement-average.tif  and coherence-average.tif          |
| **3. Risk Classification** | `deform_risk_classi(whole).py`                               | Analyzes tower velocity and strain, integrates environmental factors (e.g., storm surge), classifies risk levels, runs PCA, and generates final maps/reports. | Risk Maps, Excel Report, Histograms                          |
