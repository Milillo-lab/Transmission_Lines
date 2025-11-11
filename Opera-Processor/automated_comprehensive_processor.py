import os
import subprocess
import sys
import shutil
import time
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


def _process_county_in_subprocess(processing_base, processed_base, county_shapefile_base,
                                   tower_shapefile_base, substation_shapefile_base,
                                   county_name, start_step, only_step, skip_step4):
    """
    Independent function to process a single county in a subprocess.

    This function runs in a subprocess and needs to create a new processor instance.
    """
    processor = ComprehensiveProcessor(
        processing_base,
        processed_base,
        county_shapefile_base,
        tower_shapefile_base,
        substation_shapefile_base
    )

    try:
        return processor.process_county(county_name, start_step, only_step, skip_step4)
    except Exception as e:
        print(f"Error in subprocess for {county_name}: {e}")
        raise


class ComprehensiveProcessor:
    """
    Automated Comprehensive Processor - Complete workflow for processing all counties.

    Workflow:
    1. Mask-Reproject-Process (Generate GeoTIFF, mask_flag=0)
    2. Tower Time Series Extraction (Extract tower time series)
    3. Substation Extraction (Extract substation intersected + outer-ring pixels)
    4. Displacement-Coherence Average (Average displacement and coherence across frames)
    """

    def __init__(self, processing_base, processed_base, county_shapefile_base,
                 tower_shapefile_base, substation_shapefile_base):
        self.processing_base = Path(processing_base)
        self.processed_base = Path(processed_base)
        self.county_shapefile_base = Path(county_shapefile_base)
        self.tower_shapefile_base = Path(tower_shapefile_base)
        self.substation_shapefile_base = Path(substation_shapefile_base)

        # Processing results tracking
        self.results = {
            'step1': {'success': [], 'failed': []},
            'step2': {'success': [], 'failed': []},
            'step3': {'success': [], 'failed': []},
            'step4': {'success': [], 'failed': []}
        }

        # Timing tracking
        self.timing = {
            'start_time': None,
            'end_time': None,
            'county_times': {}
        }

    def find_county_shapefile(self, county_name):
        """Find county shapefile"""
        county_shapefile_dir = self.county_shapefile_base / county_name
        shapefile_path = county_shapefile_dir / f"{county_name}.shp"

        if shapefile_path.exists():
            return str(shapefile_path)
        else:
            print(f"  ⚠ Warning: County shapefile not found at {shapefile_path}")
            return None

    def find_tower_shapefile(self, county_name):
        """Find tower shapefile"""
        county_tower_dir = self.tower_shapefile_base / county_name

        if not county_tower_dir.exists():
            print(f"  ⚠ Warning: Tower directory not found at {county_tower_dir}")
            return None

        shp_files = list(county_tower_dir.glob("*.shp"))

        if not shp_files:
            print(f"  ⚠ Warning: No tower shapefile found in {county_tower_dir}")
            return None
        elif len(shp_files) > 1:
            print(f"  ⚠ Warning: Multiple shapefiles found, using first: {shp_files[0]}")

        return str(shp_files[0])

    def find_substation_shapefile(self, county_name):
        """Find substation shapefile"""
        county_substation_dir = self.substation_shapefile_base / county_name

        if not county_substation_dir.exists():
            print(f"  ⚠ Warning: Substation directory not found at {county_substation_dir}")
            return None

        shp_files = list(county_substation_dir.glob("*.shp"))

        if not shp_files:
            print(f"  ⚠ Warning: No substation shapefile found in {county_substation_dir}")
            return None
        elif len(shp_files) > 1:
            print(f"  ⚠ Warning: Multiple shapefiles found, using first: {shp_files[0]}")

        return str(shp_files[0])

    def verify_step1_output(self, output_dir):
        """Verify Step 1 output"""
        output_path = Path(output_dir)
        if not output_path.exists():
            return False

        tif_files = list(output_path.glob("*.tif"))
        return len(tif_files) > 0

    def verify_step2_output(self, output_dir):
        """Verify Step 2 output"""
        output_path = Path(output_dir)
        if not output_path.exists():
            return False

        # Check for tower CSV files and summary file
        csv_files = list(output_path.glob("tower_*_9pixel_displacement_coherence.csv"))
        summary_file = output_path / "powertower_9pixel_displacement_coherence_summary.csv"

        return len(csv_files) > 0 and summary_file.exists()

    def verify_step3_output(self, output_dir):
        """Verify Step 3 output"""
        output_path = Path(output_dir)
        if not output_path.exists():
            return False

        xlsx_files = list(output_path.glob("substation_*_displacement_coherence.xlsx"))
        summary_file = output_path / "substation_displacement_coherence_summary.csv"

        return len(xlsx_files) > 0 and summary_file.exists()

    def verify_step4_output(self, output_dir):
        """Verify Step 4 output"""
        output_path = Path(output_dir)
        if not output_path.exists():
            return False

        displacement_file = output_path / "displacement-average.tif"
        coherence_file = output_path / "coherence-average.tif"

        return displacement_file.exists() and coherence_file.exists()

    def run_mask_reproject_step(self, county_name, max_retries=3):
        """
        Step 1: Run Mask-Reproject-Process
        mask_flag is fixed to 0 (no mask applied)
        """
        print(f"\n{'─'*60}")
        print(f"Step 1: Mask-Reproject-Process")
        print(f"{'─'*60}")

        mask_flag = 0  # Fixed value: no mask applied

        input_folder = self.processing_base / county_name / "raw_files"
        output_folder = self.processed_base / county_name / "Vertical-Mask-Reproject"
        county_shapefile = self.find_county_shapefile(county_name)

        # Verify input
        if not input_folder.exists():
            print(f"  ✗ Input folder not found: {input_folder}")
            return False

        if not county_shapefile:
            print(f"  ✗ County shapefile not found")
            return False

        # Create output directory
        output_folder.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            "Opera-Vertical-Mask-Reproject-Processcer.py",
            str(input_folder),
            str(output_folder),
            county_shapefile,
            str(mask_flag)
        ]

        print(f"  Input: {input_folder}")
        print(f"  Output: {output_folder}")
        print(f"  Shapefile: {county_shapefile}")
        print(f"  Mask flag: {mask_flag} (UNMASKED)")

        # Execute processing with retry mechanism
        for attempt in range(max_retries):
            try:
                print(f"\n  Executing... (Attempt {attempt + 1}/{max_retries})")
                result = subprocess.run(cmd, check=True)

                # Verify output
                if self.verify_step1_output(output_folder):
                    print(f"  ✓ Step 1 completed successfully!")
                    return True
                else:
                    print(f"  ✗ Output verification failed")
                    if attempt < max_retries - 1:
                        print(f"  Retrying...")
                        time.sleep(5)
                        continue
                    return False

            except subprocess.CalledProcessError as e:
                print(f"  ✗ Error: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying...")
                    time.sleep(5)
                    continue
                return False

        return False

    def run_tower_extraction_step(self, county_name, max_retries=3):
        """Step 2: Run Tower Time Series Extraction"""
        print(f"\n{'─'*60}")
        print(f"Step 2: Tower Time Series Extraction")
        print(f"{'─'*60}")

        input_dir = self.processed_base / county_name / "Vertical-Mask-Reproject"
        output_dir = self.processed_base / county_name / "Vertical-Time-Series-Towers"
        tower_shapefile = self.find_tower_shapefile(county_name)

        # Verify input
        if not input_dir.exists() or not self.verify_step1_output(input_dir):
            print(f"  ✗ Step 1 output not found or invalid: {input_dir}")
            return False

        if not tower_shapefile:
            print(f"  ⚠ Tower shapefile not found, skipping Step 2")
            return False

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            "Opera-TimeSeries-Tower-Processor.py",
            str(input_dir),
            tower_shapefile,
            str(output_dir)
        ]

        print(f"  Input: {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Shapefile: {tower_shapefile}")

        # Execute processing
        for attempt in range(max_retries):
            try:
                print(f"\n  Executing... (Attempt {attempt + 1}/{max_retries})")
                result = subprocess.run(cmd, check=True)

                # Verify output
                if self.verify_step2_output(output_dir):
                    print(f"  ✓ Step 2 completed successfully!")
                    return True
                else:
                    print(f"  ✗ Output verification failed")
                    if attempt < max_retries - 1:
                        print(f"  Retrying...")
                        time.sleep(5)
                        continue
                    return False

            except subprocess.CalledProcessError as e:
                print(f"  ✗ Error: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying...")
                    time.sleep(5)
                    continue
                return False

        return False

    def run_substation_extraction_step(self, county_name, max_retries=3):
        """Step 3: Run Substation Extraction (Intersected + Outer-Ring)"""
        print(f"\n{'─'*60}")
        print(f"Step 3: Substation Extraction (Intersected + Outer-Ring)")
        print(f"{'─'*60}")

        input_dir = self.processed_base / county_name / "Vertical-Mask-Reproject"
        output_dir = self.processed_base / county_name / "Vertical-Time-Series-Substations"
        substation_shapefile = self.find_substation_shapefile(county_name)

        # Verify input
        if not input_dir.exists() or not self.verify_step1_output(input_dir):
            print(f"  ✗ Step 1 output not found or invalid: {input_dir}")
            return False

        if not substation_shapefile:
            print(f"  ⚠ Substation shapefile not found, skipping Step 3")
            return False

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            "Opera-TimeSeries-Substation-Processor.py",
            str(input_dir),
            substation_shapefile,
            str(output_dir)
        ]

        print(f"  Input: {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Shapefile: {substation_shapefile}")

        # Execute processing
        for attempt in range(max_retries):
            try:
                print(f"\n  Executing... (Attempt {attempt + 1}/{max_retries})")
                result = subprocess.run(cmd, check=True)

                # Verify output
                if self.verify_step3_output(output_dir):
                    print(f"  ✓ Step 3 completed successfully!")
                    return True
                else:
                    print(f"  ✗ Output verification failed")
                    if attempt < max_retries - 1:
                        print(f"  Retrying...")
                        time.sleep(5)
                        continue
                    return False

            except subprocess.CalledProcessError as e:
                print(f"  ✗ Error: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying...")
                    time.sleep(5)
                    continue
                return False

        return False

    def run_displacement_coherence_average_step(self, county_name, max_retries=3):
        """Step 4: Run Displacement-Coherence Average"""
        print(f"\n{'─'*60}")
        print(f"Step 4: Displacement-Coherence Average")
        print(f"{'─'*60}")

        input_dir = self.processed_base / county_name / "Vertical-Mask-Reproject"
        output_dir = self.processed_base / county_name / "Average"

        # Verify input
        if not input_dir.exists() or not self.verify_step1_output(input_dir):
            print(f"  ✗ Step 1 output not found or invalid: {input_dir}")
            return False

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            "Opera-Displacement-Coherence-Average.py",
            str(input_dir),
            str(self.county_shapefile_base / county_name),
            str(output_dir)
        ]

        print(f"  Input: {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Shapefile: {self.county_shapefile_base / county_name}")

        # Execute processing
        for attempt in range(max_retries):
            try:
                print(f"\n  Executing... (Attempt {attempt + 1}/{max_retries})")
                result = subprocess.run(cmd, check=True)

                # Verify output
                if self.verify_step4_output(output_dir):
                    print(f"  ✓ Step 4 completed successfully!")
                    return True
                else:
                    print(f"  ✗ Output verification failed")
                    if attempt < max_retries - 1:
                        print(f"  Retrying...")
                        time.sleep(5)
                        continue
                    return False

            except subprocess.CalledProcessError as e:
                print(f"  ✗ Error: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying...")
                    time.sleep(5)
                    continue
                return False

        return False

    def process_county(self, county_name, start_step=1, only_step=None, skip_step4=False):
        """Process complete workflow for a single county, return result dictionary"""
        print(f"\n{'='*80}")
        print(f"Processing County: {county_name}")
        print(f"{'='*80}")

        county_start_time = time.time()

        # Initialize result
        result = {
            'county': county_name,
            'step1': None,
            'step2': None,
            'step3': None,
            'step4': None,
            'duration': 0
        }

        # Step 1: Mask-Reproject-Process
        if (start_step <= 1 and only_step in [None, 1]):
            # Check if Step 1 output already exists
            step1_output_dir = self.processed_base / county_name / "Vertical-Mask-Reproject"
            if self.verify_step1_output(step1_output_dir):
                print(f"\n✓ Step 1 output already exists, skipping Step 1")
                step1_success = True
                result['step1'] = True
            else:
                step1_success = self.run_mask_reproject_step(county_name)
                result['step1'] = step1_success

                if not step1_success:
                    print(f"\n✗ Step 1 failed for {county_name}, skipping Steps 2 and 3")
                    result['duration'] = time.time() - county_start_time
                    return result
        else:
            print(f"\n⊳ Skipping Step 1 (start_step={start_step})")
            step1_success = True

        if only_step == 1:
            result['duration'] = time.time() - county_start_time
            return result

        # Step 2: Tower Extraction
        if (start_step <= 2 and only_step in [None, 2]):
            # Check if Step 2 output already exists
            step2_output_dir = self.processed_base / county_name / "Vertical-Time-Series-Towers"
            if self.verify_step2_output(step2_output_dir):
                print(f"\n✓ Step 2 output already exists, skipping Step 2")
                step2_success = True
                result['step2'] = True
            else:
                step2_success = self.run_tower_extraction_step(county_name)
                result['step2'] = step2_success

                if not step2_success:
                    print(f"\n⚠ Step 2 failed for {county_name}, but continuing to Step 3...")
        else:
            print(f"\n⊳ Skipping Step 2")
            step2_success = True

        if only_step == 2:
            result['duration'] = time.time() - county_start_time
            return result

        # Step 3: Substation Extraction
        if (start_step <= 3 and only_step in [None, 3]):
            # Check if Step 3 output already exists
            step3_output_dir = self.processed_base / county_name / "Vertical-Time-Series-Substations"
            if self.verify_step3_output(step3_output_dir):
                print(f"\n✓ Step 3 output already exists, skipping Step 3")
                step3_success = True
                result['step3'] = True
            else:
                step3_success = self.run_substation_extraction_step(county_name)
                result['step3'] = step3_success
        else:
            print(f"\n⊳ Skipping Step 3")
            step3_success = True

        if only_step == 3:
            result['duration'] = time.time() - county_start_time
            return result

        # Step 4: Displacement-Coherence Average
        if skip_step4:
            print(f"\n⊳ Skipping Step 4 (--skip-step4 flag enabled)")
            step4_success = True
        elif (start_step <= 4 and only_step in [None, 4]):
            # Check if Step 4 output already exists
            step4_output_dir = self.processed_base / county_name / "Average"
            if self.verify_step4_output(step4_output_dir):
                print(f"\n✓ Step 4 output already exists, skipping Step 4")
                step4_success = True
                result['step4'] = True
            else:
                step4_success = self.run_displacement_coherence_average_step(county_name)
                result['step4'] = step4_success
        else:
            print(f"\n⊳ Skipping Step 4")
            step4_success = True

        county_end_time = time.time()
        county_duration = county_end_time - county_start_time
        result['duration'] = county_duration

        print(f"\n{'─'*80}")
        print(f"Completed {county_name} in {county_duration/60:.1f} minutes")
        print(f"{'─'*80}")

        return result

    def get_all_counties(self):
        """Get all counties to be processed"""
        if not self.processing_base.exists():
            print(f"Error: Processing directory not found at {self.processing_base}")
            return []

        counties = [d.name for d in self.processing_base.iterdir() if d.is_dir()]

        if not counties:
            print(f"No county folders found in {self.processing_base}")
            return []

        return sorted(counties)

    def process_all_counties(self, county_list=None, start_step=1, only_step=None, resume=False, parallel=1, skip_step4=False):
        """Process all counties (supports parallel processing)"""
        self.timing['start_time'] = time.time()

        if county_list:
            counties = county_list
        else:
            counties = self.get_all_counties()

        if not counties:
            print("No counties to process!")
            return

        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE PROCESSING")
        print(f"{'='*80}")
        print(f"Counties to process: {len(counties)}")
        print(f"Counties: {', '.join(counties)}")
        print(f"Configuration: mask_flag=0 (UNMASKED)")
        if start_step > 1:
            print(f"Starting from Step {start_step}")
        if only_step:
            print(f"Only running Step {only_step}")
        if skip_step4:
            print(f"Step 4 will be skipped (--skip-step4 enabled)")
        if parallel > 1:
            print(f"Parallel processing: {parallel} counties at a time")
        print(f"{'='*80}\n")

        # If resume mode, detect completed counties
        if resume:
            completed_counties = self.detect_completed_counties(counties, skip_step4)
            counties = [c for c in counties if c not in completed_counties]
            print(f"\nResume mode: Skipping {len(completed_counties)} completed counties")
            print(f"Remaining: {len(counties)} counties\n")

        # Choose serial or parallel processing
        if parallel <= 1:
            # Serial processing
            self._process_serial(counties, start_step, only_step, skip_step4)
        else:
            # Parallel processing
            self._process_parallel(counties, start_step, only_step, parallel, skip_step4)

        self.timing['end_time'] = time.time()

        # Print final report
        self.print_summary()

    def _aggregate_result(self, result):
        """Aggregate processing results for a single county"""
        county = result['county']

        # Aggregate Step 1 results
        if result['step1'] is not None:
            if result['step1']:
                self.results['step1']['success'].append(county)
            else:
                self.results['step1']['failed'].append(county)

        # Aggregate Step 2 results
        if result['step2'] is not None:
            if result['step2']:
                self.results['step2']['success'].append(county)
            else:
                self.results['step2']['failed'].append(county)

        # Aggregate Step 3 results
        if result['step3'] is not None:
            if result['step3']:
                self.results['step3']['success'].append(county)
            else:
                self.results['step3']['failed'].append(county)

        # Aggregate Step 4 results
        if result['step4'] is not None:
            if result['step4']:
                self.results['step4']['success'].append(county)
            else:
                self.results['step4']['failed'].append(county)

        # Record processing time
        if result['duration'] > 0:
            self.timing['county_times'][county] = result['duration']

    def _process_serial(self, counties, start_step, only_step, skip_step4):
        """Process all counties serially"""
        for idx, county in enumerate(counties, 1):
            print(f"\n[{idx}/{len(counties)}] Processing {county}...")

            try:
                result = self.process_county(county, start_step, only_step, skip_step4)
                self._aggregate_result(result)
            except Exception as e:
                print(f"\n✗ Unexpected error processing {county}: {e}")
                self.results['step1']['failed'].append(county)

    def _process_parallel(self, counties, start_step, only_step, max_workers, skip_step4):
        """Process all counties in parallel"""
        print(f"\n{'='*80}")
        print(f"PARALLEL MODE: Processing {max_workers} counties simultaneously")
        print(f"{'='*80}\n")

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks, using independent function instead of instance method
            future_to_county = {
                executor.submit(
                    _process_county_in_subprocess,
                    str(self.processing_base),
                    str(self.processed_base),
                    str(self.county_shapefile_base),
                    str(self.tower_shapefile_base),
                    str(self.substation_shapefile_base),
                    county,
                    start_step,
                    only_step,
                    skip_step4
                ): county for county in counties
            }

            # Process completed tasks
            completed = 0
            total = len(counties)

            for future in as_completed(future_to_county):
                county = future_to_county[future]
                completed += 1

                try:
                    result = future.result()
                    self._aggregate_result(result)
                    print(f"\n[{completed}/{total}] ✓ Completed {county}")
                except Exception as e:
                    print(f"\n[{completed}/{total}] ✗ Failed {county}: {e}")
                    self.results['step1']['failed'].append(county)

    def detect_completed_counties(self, counties, skip_step4=False):
        """Detect completed counties"""
        completed = []
        for county in counties:
            step1_dir = self.processed_base / county / "Vertical-Mask-Reproject"
            step2_dir = self.processed_base / county / "Vertical-Time-Series-Towers"
            step3_dir = self.processed_base / county / "Vertical-Time-Series-Substations"
            step4_dir = self.processed_base / county / "Average"

            # Check steps 1-3
            steps_complete = (self.verify_step1_output(step1_dir) and
                            self.verify_step2_output(step2_dir) and
                            self.verify_step3_output(step3_dir))

            # If skip_step4, we only need steps 1-3 to be complete
            # Otherwise, we also need step 4
            if skip_step4:
                if steps_complete:
                    completed.append(county)
                    print(f"  ✓ {county}: Steps 1-3 completed (Step 4 skipped)")
            else:
                if steps_complete and self.verify_step4_output(step4_dir):
                    completed.append(county)
                    print(f"  ✓ {county}: All steps completed")

        return completed

    def print_summary(self):
        """Print processing summary"""
        total_time = self.timing['end_time'] - self.timing['start_time']

        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE PROCESSING SUMMARY")
        print(f"{'='*80}\n")

        print(f"Configuration: mask_flag=0 (UNMASKED)\n")

        # Step 1 statistics
        step1_total = len(self.results['step1']['success']) + len(self.results['step1']['failed'])
        step1_success = len(self.results['step1']['success'])
        step1_rate = (step1_success / step1_total * 100) if step1_total > 0 else 0

        print(f"Step 1: Mask-Reproject-Process")
        print(f"  Total runs: {step1_total}")
        print(f"  Successful: {step1_success}")
        print(f"  Failed: {len(self.results['step1']['failed'])}")
        print(f"  Success rate: {step1_rate:.1f}%")

        # Step 2 statistics
        step2_total = len(self.results['step2']['success']) + len(self.results['step2']['failed'])
        step2_success = len(self.results['step2']['success'])
        step2_rate = (step2_success / step2_total * 100) if step2_total > 0 else 0

        print(f"\nStep 2: Tower Time Series Extraction")
        print(f"  Total runs: {step2_total}")
        print(f"  Successful: {step2_success}")
        print(f"  Failed: {len(self.results['step2']['failed'])}")
        print(f"  Success rate: {step2_rate:.1f}%")

        # Step 3 statistics
        step3_total = len(self.results['step3']['success']) + len(self.results['step3']['failed'])
        step3_success = len(self.results['step3']['success'])
        step3_rate = (step3_success / step3_total * 100) if step3_total > 0 else 0

        print(f"\nStep 3: Substation Extraction (Intersected + Outer-Ring)")
        print(f"  Total runs: {step3_total}")
        print(f"  Successful: {step3_success}")
        print(f"  Failed: {len(self.results['step3']['failed'])}")
        print(f"  Success rate: {step3_rate:.1f}%")

        # Step 4 statistics
        step4_total = len(self.results['step4']['success']) + len(self.results['step4']['failed'])
        step4_success = len(self.results['step4']['success'])
        step4_rate = (step4_success / step4_total * 100) if step4_total > 0 else 0

        print(f"\nStep 4: Displacement-Coherence Average")
        print(f"  Total runs: {step4_total}")
        print(f"  Successful: {step4_success}")
        print(f"  Failed: {len(self.results['step4']['failed'])}")
        print(f"  Success rate: {step4_rate:.1f}%")

        # Failed counties
        if self.results['step1']['failed']:
            print(f"\nFailed Counties (Step 1):")
            for county in self.results['step1']['failed']:
                print(f"  - {county}")

        if self.results['step2']['failed']:
            print(f"\nFailed Counties (Step 2):")
            for county in self.results['step2']['failed']:
                print(f"  - {county}")

        if self.results['step3']['failed']:
            print(f"\nFailed Counties (Step 3):")
            for county in self.results['step3']['failed']:
                print(f"  - {county}")

        if self.results['step4']['failed']:
            print(f"\nFailed Counties (Step 4):")
            for county in self.results['step4']['failed']:
                print(f"  - {county}")

        # Successful counties
        all_success = [c for c in self.results['step1']['success']
                      if c in self.results['step2']['success']
                      and c in self.results['step3']['success']
                      and c in self.results['step4']['success']]

        if all_success:
            print(f"\nFully Completed Counties ({len(all_success)}):")
            for county in all_success:
                duration = self.timing['county_times'].get(county, 0)
                print(f"  ✓ {county} ({duration/60:.1f} minutes)")

        # Timing statistics
        print(f"\nTiming:")
        print(f"  Total processing time: {total_time/3600:.2f} hours")

        if self.timing['county_times']:
            avg_time = sum(self.timing['county_times'].values()) / len(self.timing['county_times'])
            print(f"  Average time per county: {avg_time/60:.1f} minutes")

        print(f"\n{'='*80}\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Automated Comprehensive Processor - Complete workflow for processing all counties',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
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
        """
    )

    parser.add_argument('--counties', nargs='+', help='Specify county names to process')
    parser.add_argument('--start-step', type=int, choices=[1, 2, 3, 4], default=1,
                       help='Start from specified step (default: 1)')
    parser.add_argument('--only-step', type=int, choices=[1, 2, 3, 4],
                       help='Run only the specified step')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process only first county')
    parser.add_argument('--resume', action='store_true',
                       help='Resume mode: skip completed counties')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel processes (default: 1, serial processing)')
    parser.add_argument('--skip-step4', action='store_true',
                       help='Skip Step 4 (Displacement-Coherence Average)')

    args = parser.parse_args()

    # Base path configuration
    processing_base = r"G:\processing"
    processed_base = r"G:\processed"
    county_shapefile_base = r"E:\UH\graduate class\Applied Gepspatial Computations\data\CoastalCounties\counties"
    tower_shapefile_base = r"D:\project\InSAR OPERA\powertower_every_county\output"
    substation_shapefile_base = r"D:\project\InSAR OPERA\substation_every_county"

    # Create processor
    processor = ComprehensiveProcessor(
        processing_base,
        processed_base,
        county_shapefile_base,
        tower_shapefile_base,
        substation_shapefile_base
    )

    # Determine county list to process
    if args.test:
        counties = processor.get_all_counties()
        county_list = [counties[0]] if counties else None
        print("TEST MODE: Processing only first county")
    else:
        county_list = args.counties

    # Start processing
    processor.process_all_counties(
        county_list=county_list,
        start_step=args.start_step,
        only_step=args.only_step,
        resume=args.resume,
        parallel=args.parallel,
        skip_step4=args.skip_step4
    )


if __name__ == "__main__":
    main()
