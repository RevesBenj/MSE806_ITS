# Dublin Bus GTFS ITS Analysis

**Author:** Benjelyn Reves Patiag  
**Course:** MSE806 Intelligent Transportation Systems Assessment 1  
**Script:** `DublinBus_GTFS_Analysis.py`  
**Purpose:** Analyse Dublin Bus GTFS timetable data and generate data evidence, CSV outputs, charts, and a written summary for an Intelligent Transportation Systems (ITS) case study paper.

---

## Table of Contents

- [1. Analysis Task Overview](#1-analysis-task-overview)
- [2. Main Features](#2-main-features)
- [3. Project Structure](#3-project-structure)
- [4. Installation Guide](#4-installation-guide)
  - [4.1 Prerequisites](#41-prerequisites)
  - [4.2 Create a Virtual Environment](#42-create-a-virtual-environment)
  - [4.3 Install Required Packages](#43-install-required-packages)
- [5. How to Run the Script](#5-how-to-run-the-script)
- [6. Command Line Arguments](#6-command-line-arguments)
- [7. Input Data Requirements](#7-input-data-requirements)
  - [Required GTFS Files](#required-gtfs-files)
  - [Optional GTFS Files](#optional-gtfs-files)
- [8. Output Files Explained](#8-output-files-explained)
  - [8.1 CSV Outputs](#81-csv-outputs)
  - [8.2 Chart Outputs](#82-chart-outputs)
- [9. Methodology](#9-methodology)
  - [Step 1: Data Acquisition](#step-1-data-acquisition)
  - [Step 2: Data Cleaning](#step-2-data-cleaning)
  - [Step 3: Feature Engineering](#step-3-feature-engineering)
  - [Step 4: Statistical Analysis](#step-4-statistical-analysis)
  - [Step 5: Visualisation](#step-5-visualisation)
- [10. Code Structure](#10-Code-Structure)
- [11. Academic Use in MSE806 ITS Paper](#11-academic-use-in-mse806-its-paper)
- [12. Important Limitation](#12-important-limitation)
- [13. Troubleshooting](#13-troubleshooting)
- [14. Future Improvements](#14-future-improvements)
- [15. Conclusion](#15-conclusion)
- [16. Technologies Used](#16-technologies-used)
- [17. License](#17-license)
- [18. Credits & Acknowledgements](#18-credits--acknowledgements)

---


## 1. Analysis Task Overview

This task analyses Dublin Bus public transport timetable data using the General Transit Feed Specification (GTFS).

The Python script loads and cleans GTFS files, calculates travel time between stops, compares peak and off-peak periods, estimates route speeds, analyses service frequency, and creates charts for reporting.

The results help support an Intelligent Transportation Systems (ITS) case study by providing evidence for:

* Public transport performance analysis
* Congestion and peak-hour patterns
* Route priority identification
* Service frequency and headway analysis
* GTFS data limitations
* AI, ML, and ITS improvement opportunities

This is important because ITS is not only about technology. It also depends on data, communication, analytics, and decision making. This project shows how transport timetable data can be converted into useful evidence for transport planning and ITS evaluation.


---

## 2. Main Features

The Python script performs the following tasks:

1. **Loads GTFS data from a ZIP file**
   - Required files:
     - `routes.txt`
     - `trips.txt`
     - `stops.txt`
     - `stop_times.txt`
     - `calendar.txt`
   - Optional files:
     - `calendar_dates.txt`
     - `shapes.txt`
     - `feed_info.txt`

2. **Cleans and prepares stop-time data**
   - Converts GTFS time format into seconds.
   - Handles GTFS extended time such as `24:10:00`.
   - Calculates hour of day.
   - Detects overnight records.
   - Creates inter-stop travel segments.
   - Removes invalid or extreme segment values.

3. **Creates transport analysis outputs**
   - Hourly stop events.
   - Hourly mean travel time.
   - Peak vs off-peak comparison.
   - Route distance summary.
   - Route average speed.
   - Scheduled headway by route and hour.
   - Stop density and route coverage.
   - Top congested routes.
   - Service day summary.
   - Calendar exception summary.
   - GTFS limitation note.

4. **Adds statistical and ML-style support**
   - Pearson correlation between peak flag and travel time.
   - Regression model using hour, peak flag, sine hour, and cosine hour.
   - Regression R² value.
   - Hourly actual vs predicted travel time output.

5. **Generates 12 charts**
   - Hourly stop event volume.
   - Travel time regression.
   - Top routes by scheduled trips.
   - Congestion impact by time period.
   - Scheduled trips by service day.
   - Weekday-hour congestion heatmap.
   - International ITS benchmarking.
   - AI/ML ITS roadmap.
   - Lowest scheduled route speed.
   - Scheduled headway pattern.
   - Stop density by route.
   - Top congested routes.

6. **Writes a plain-English summary report**
   - Creates `DUBLIN_BUS_GTFS_ANALYSIS_SUMMARY.txt`.
   - Explains dataset coverage, key findings, limitations, and final conclusion.

7. **Creates an output manifest**
   - Lists all generated CSV, chart, and summary files.

---

## 3. Project Structure

Recommended folder structure:

```text
DublinBus_GTFS_Project/
│
├── DublinBus_GTFS_Analysis.py
├── GTFS_Dublin_Bus (1).zip
├── requirements.txt
├── README.md
│
└── dublin_bus_outputs/              # Generated after running the script
    ├── csv_outputs/
    │   ├── data_quality_summary.csv
    │   ├── hourly_stop_events.csv
    │   ├── hourly_mean_travel_time.csv
    │   ├── top_routes_by_trips.csv
    │   ├── period_comparison.csv
    │   ├── service_day_summary.csv
    │   ├── calendar_exception_summary.csv
    │   ├── active_service_dates.csv
    │   ├── weekday_hour_heatmap.csv
    │   ├── route_distance_summary.csv
    │   ├── route_average_speed.csv
    │   ├── headway_by_route_hour.csv
    │   ├── stop_density_and_coverage.csv
    │   ├── top_congested_routes.csv
    │   ├── route_peak_offpeak_comparison.csv
    │   ├── gtfs_limitations_note.csv
    │   ├── international_benchmark.csv
    │   ├── regression_summary.csv
    │   └── regression_hourly_predictions.csv
    │
    ├── charts/
    │   ├── figure_1_hourly_stop_events.png
    │   ├── figure_2_travel_time_regression.png
    │   ├── figure_3_top_routes.png
    │   ├── figure_4_period_comparison.png
    │   ├── figure_5_service_day_trips.png
    │   ├── figure_6_weekday_hour_heatmap.png
    │   ├── figure_7_international_benchmark.png
    │   ├── figure_8_ai_ml_roadmap.png
    │   ├── figure_9_route_average_speed.png
    │   ├── figure_10_headway_high_frequency_routes.png
    │   ├── figure_11_stop_density_route_coverage.png
    │   └── figure_12_top_congested_routes.png
    │
    ├── DUBLIN_BUS_GTFS_ANALYSIS_SUMMARY.txt
    └── output_manifest.csv
```

---

## 4. Installation Guide

### 4.1 Prerequisites

Install Python 3.10 or later. Python 3.11 or 3.12 is recommended.

Check Python version:

```bash
python --version
```

or:

```bash
python3 --version
```

---

### 4.2 Create a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 4.3 Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 5. How to Run the Script

Run the script by passing the GTFS ZIP file and output folder.

```bash
python DublinBus_GTFS_Analysis.py --gtfs_zip "GTFS_Dublin_Bus (1).zip" --output_dir dublin_bus_outputs
```

Example with another GTFS file:

```bash
python DublinBus_GTFS_Analysis.py --gtfs_zip "my_gtfs_feed.zip" --output_dir outputs
```

When the script completes, it prints output paths similar to:

```text
Analysis complete.
CSV outputs: dublin_bus_outputs/csv_outputs
Charts: dublin_bus_outputs/charts
Summary: dublin_bus_outputs/DUBLIN_BUS_GTFS_ANALYSIS_SUMMARY.txt
Manifest: dublin_bus_outputs/output_manifest.csv
```

---

## 6. Command Line Arguments

| Argument | Required | Description | Example |
|---|---:|---|---|
| `--gtfs_zip` | Yes | Path to the GTFS ZIP file. | `"GTFS_Dublin_Bus (1).zip"` |
| `--output_dir` | No | Folder where outputs will be saved. Default is `dublin_bus_outputs`. | `dublin_bus_outputs` |

---

## 7. Input Data Requirements

The input must be a valid GTFS ZIP file.

### Required GTFS files

| File | Purpose |
|---|---|
| `routes.txt` | Contains route names and route IDs. |
| `trips.txt` | Links trips to routes and service IDs. |
| `stops.txt` | Contains bus stop information. |
| `stop_times.txt` | Contains arrival and departure times for stops. |
| `calendar.txt` | Contains normal service-day rules. |

### Optional GTFS files

| File | Purpose |
|---|---|
| `calendar_dates.txt` | Adds or removes service dates. |
| `shapes.txt` | Used for route distance and speed estimation. |
| `feed_info.txt` | Feed metadata. |

If any required file is missing, the script stops and raises a `FileNotFoundError`.

---

## 8. Output Files Explained

### 8.1 CSV Outputs

| CSV File | Description |
|---|---|
| `data_quality_summary.csv` | Summary of routes, stops, trips, records, missing values, valid segments, removed outliers, and overnight records. |
| `hourly_stop_events.csv` | Number of scheduled stop events per hour. |
| `hourly_mean_travel_time.csv` | Mean and median inter-stop travel time per hour. |
| `top_routes_by_trips.csv` | Top routes by number of scheduled trips in the GTFS feed. |
| `period_comparison.csv` | Travel-time comparison for off-peak, midday, morning peak, and evening peak. |
| `service_day_summary.csv` | Service coverage by day of week. |
| `calendar_exception_summary.csv` | Summary of service additions and removals from `calendar_dates.txt`. |
| `active_service_dates.csv` | Expanded active service dates after applying calendar exceptions. |
| `weekday_hour_heatmap.csv` | Weekday-hour mean segment seconds for heatmap visualisation. |
| `route_distance_summary.csv` | Route distance summary using `shapes.txt`. |
| `route_average_speed.csv` | Estimated scheduled route speed using trip duration and shape distance. |
| `headway_by_route_hour.csv` | Mean and median scheduled headway by route and hour. |
| `stop_density_and_coverage.csv` | Unique stops, route distance, and stops per kilometre. |
| `top_congested_routes.csv` | Routes with highest mean scheduled inter-stop segment time. |
| `route_peak_offpeak_comparison.csv` | Route-level peak vs off-peak travel-time difference. |
| `gtfs_limitations_note.csv` | Important notes explaining what GTFS can and cannot prove. |
| `international_benchmark.csv` | Indicative benchmark table for Dublin Bus, London, Singapore, and Seoul. |
| `regression_summary.csv` | Pearson correlation, p-value, R², and worst hour summary. |
| `regression_hourly_predictions.csv` | Actual and predicted hourly mean segment time. |

---

### 8.2 Chart Outputs

| Chart | Description |
|---|---|
| `figure_1_hourly_stop_events.png` | Shows scheduled bus activity by hour. |
| `figure_2_travel_time_regression.png` | Compares actual and predicted mean inter-stop travel time. |
| `figure_3_top_routes.png` | Shows the top 12 routes by scheduled trips. |
| `figure_4_period_comparison.png` | Compares congestion impact across time periods. |
| `figure_5_service_day_trips.png` | Shows scheduled trips by day type. |
| `figure_6_weekday_hour_heatmap.png` | Shows weekday-hour congestion pattern. |
| `figure_7_international_benchmark.png` | Shows indicative ITS benchmarking. |
| `figure_8_ai_ml_roadmap.png` | Shows AI/ML enhancement roadmap. |
| `figure_9_route_average_speed.png` | Shows lowest scheduled average speed by route. |
| `figure_10_headway_high_frequency_routes.png` | Shows scheduled headway pattern for high-frequency routes. |
| `figure_11_stop_density_route_coverage.png` | Shows highest stop density routes. |
| `figure_12_top_congested_routes.png` | Shows routes with highest mean segment time. |

---

## 9. Methodology

The script follows a simple data analytics workflow aligned with ITS analysis:

### Step 1: Data Acquisition

The script loads GTFS files from a ZIP archive. GTFS is a standard public transport data format that includes route, trip, stop, timetable, and service calendar data.

### Step 2: Data Cleaning

The script prepares `stop_times.txt` by:

- converting arrival and departure times into seconds;
- sorting stops by trip and stop sequence;
- calculating inter-stop segment time;
- removing invalid segment values below 5 seconds or above 600 seconds;
- preserving valid overnight GTFS records using 24+ hour format.

### Step 3: Feature Engineering

The script creates useful variables for ITS analysis:

- `hour_raw`
- `hour`
- `is_overnight`
- `segment_seconds`
- `valid_segment`
- `period`
- `peak_flag`
- `sin_hour`
- `cos_hour`

These features help transform raw timetable data into meaningful transport analysis indicators.

### Step 4: Statistical Analysis

The script applies:

- hourly mean travel-time analysis;
- Pearson correlation between peak flag and mean segment time;
- linear regression using hour, peak flag, sine hour, and cosine hour;
- R² score to explain how much the simple model explains hourly variation.

### Step 5: Visualisation

The script generates 12 charts using Matplotlib. These charts can be used in the assessment paper, appendix, presentation, or evidence package.


---

## 10. Code Structure

The script is structured using classes.

| Class | Responsibility |
|---|---|
| `AnalysisConfig` | Stores GTFS path, output path, and segment cleaning thresholds. |
| `GTFSLoader` | Loads required and optional GTFS files from ZIP. |
| `GTFSProcessor` | Cleans and enriches stop-time records. |
| `ITSAnalyzer` | Generates CSV summaries, statistics, and route analysis. |
| `ChartBuilder` | Creates the 12 PNG charts. |
| `SummaryWriter` | Writes the final plain-English summary text file. |
| `DublinBusGTFSApp` | Controls the full application workflow. |


---

## 11. Academic Use in MSE806 ITS Paper


| Paper Section | Useful Output |
|---|---|
| Dataset and Methodology | `data_quality_summary.csv`, `gtfs_limitations_note.csv` |
| Dublin Bus Service Pattern | `hourly_stop_events.csv`, `service_day_summary.csv` |
| Congestion Analysis | `hourly_mean_travel_time.csv`, `period_comparison.csv`, `top_congested_routes.csv` |
| Route Priority / TSP Recommendation | `top_routes_by_trips.csv`, `route_peak_offpeak_comparison.csv`, `headway_by_route_hour.csv` |
| AI/ML Recommendation | `regression_summary.csv`, `regression_hourly_predictions.csv` |
| Appendix Evidence | All CSV outputs and 12 charts |
| Presentation Visuals | PNG chart files from `charts/` |

---

## 12. Important Limitation

This task uses **scheduled GTFS data**, not real-time operational data.

This means the analysis can show:

- planned service patterns;
- expected peak pressure;
- scheduled travel-time variation;
- route coverage;
- timetable-based headway;
- planned route speed.

But it cannot directly prove:

- actual real-time bus delay;
- live congestion;
- passenger crowding;
- AVL vehicle movement;
- RTPI prediction accuracy;
- Traffic Signal Priority event success;
- real incident impact.

For stronger operational ITS evidence, future work should combine GTFS with:

- GTFS-Realtime;
- AVL data;
- RTPI data;
- passenger count / ticketing data;
- traffic signal event data;
- road congestion sensor data.

---

## 13. Troubleshooting

### Error: `FileNotFoundError: Required GTFS file missing`

The ZIP file does not contain one or more required GTFS files. Check that the ZIP includes:

```text
routes.txt
trips.txt
stops.txt
stop_times.txt
calendar.txt
```

### Error: `ModuleNotFoundError`

Install the required packages again:

```bash
pip install -r requirements.txt
```

### Charts are not generated

Make sure Matplotlib is installed:

```bash
pip install matplotlib
```

### Regression output is missing or weaker

The script can still run without SciPy or scikit-learn because it has fallback logic. However, for full statistical output, install all packages from `requirements.txt`.

### Output folder is empty

Check that the command includes the correct GTFS ZIP path:

```bash
python DublinBus_GTFS_Analysis.py --gtfs_zip "GTFS_Dublin_Bus (1).zip" --output_dir dublin_bus_outputs
```

---

## 14. Future Improvements

Possible future upgrades:

- Add GTFS-Realtime vehicle position analysis.
- Add AVL delay comparison against scheduled GTFS time.
- Add passenger demand and crowding analysis.
- Add machine learning models such as Random Forest, Gradient Boosting, or LSTM.
- Add interactive dashboard using Streamlit or Power BI.
- Add automated report generation in Word or PDF format.
- Add unit tests for loader, processor, and analyzer classes.
- Add configuration file for peak-period definitions.

---

## 15. Conclusion

This task provides a complete GTFS data analysis for Dublin Bus ITS study. It converts raw timetable data into useful outputs such as CSV files, charts, statistics, and summary reports.

The results help show public transport demand, peak-hour congestion, route priorities, and the need for better ITS integration using AVL, RTPI, TSP, GTFS-Realtime, and AI/ML.

This analysis is useful for academic work because it is data-driven, structured, and easy to understand. However, GTFS contains scheduled data only. The results should be used for planning and analysis, not as proof of actual real-time bus performance.



---

## 16 Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- SciPy
- GTFS Open Data

---

## 17 License
- This project is released under the **MIT License**. See `LICENSE`.
---

## 18 Credits & Acknowledgements
**Developer:** Benjelyn Reves Patiag  
**Course:** Master of Software Engineering (MSE)  
**Unit:** MSE806 - Intelligent Transportation Systems (ITS)
**Institution:** Yoobee College of Creative Innovation  
**Academic Support:** Dr. Mukesh Mishra, for guidance and feedback.