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

In terminal:

conda env create -f environment.yml # Create the conda environment
conda activate transmission # Activate the new environment




