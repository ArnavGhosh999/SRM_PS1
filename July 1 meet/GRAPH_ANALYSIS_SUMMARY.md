# Electrical Data Analysis Summary
## Analysis of merged_data_20Hz.csv

This document provides a comprehensive overview of the graphs generated from the electrical measurement data.

### Dataset Overview
- **File**: merged_data_20Hz.csv
- **Size**: 3.3GB (1,678,506 rows)
- **Features**: A Current, A Voltage, B Current, B Voltage
- **Data Type**: Time series arrays stored as comma-separated strings
- **Additional Info**: Includes Time, Site Name, Point Machine Name, Direction, Type, Polling rates

---

## Generated Visualizations

### 1. Basic Analysis (`analysis_output/` directory)

#### 1.1 Time Series Samples (`time_series_samples.png`)
- **Purpose**: Shows sample time series for each electrical measurement
- **Insights**: 
  - Displays 5 random samples for each measurement type
  - Shows the temporal patterns and variations in electrical signals
  - Helps identify typical signal shapes and characteristics

#### 1.2 Statistical Distributions (`statistical_distributions.png`)
- **Purpose**: Box plots showing distribution of statistical measures
- **Measures**: Mean, Standard Deviation, Min, Max, Skewness, Kurtosis
- **Insights**:
  - Reveals the central tendency and spread of each measurement
  - Shows outliers and data distribution characteristics
  - Helps identify measurement patterns and anomalies

#### 1.3 Clustering Analysis (`*_clusters.png`)
- **Files**: A_Current_clusters.png, A_Voltage_clusters.png, B_Current_clusters.png, B_Voltage_clusters.png
- **Purpose**: Groups similar time series patterns using K-means clustering
- **Insights**:
  - Identifies distinct operational patterns
  - Shows mean patterns for each cluster with standard deviation bands
  - Helps categorize different operational states or conditions

#### 1.4 Summary Statistics (`summary_statistics.png`)
- **Purpose**: Comprehensive overview of data quality and characteristics
- **Components**:
  - Array length distribution
  - Data completeness (empty vs non-empty arrays)
  - Mean values comparison across features
  - Data quality heatmap

---

### 2. Advanced Analysis (`additional_analysis/` directory)

#### 2.1 Correlation Analysis
- **Correlation Heatmap** (`correlation_heatmap.png`)
  - Shows correlation coefficients between all measurement pairs
  - Uses color-coded matrix for easy interpretation
  - Values range from -1 (perfect negative) to +1 (perfect positive)

- **Correlation Scatter Plots** (`correlation_scatter_plots.png`)
  - Pairwise scatter plots with correlation coefficients
  - Shows relationship strength and direction
  - Helps identify linear and non-linear relationships

#### 2.2 Power Analysis (`power_analysis.png`)
- **Purpose**: Calculates and analyzes electrical power (P = V × I)
- **Components**:
  - A Power and B Power time series samples
  - Power distribution histograms
  - A Power vs B Power comparison
- **Insights**:
  - Shows power consumption patterns
  - Reveals power distribution characteristics
  - Compares power between A and B circuits

#### 2.3 Trend Analysis (`trend_analysis.png`)
- **Purpose**: Analyzes temporal trends in measurements
- **Features**:
  - Time series with error bars
  - Trend lines with slope indicators
  - Shows how measurements change over time
- **Insights**:
  - Identifies long-term trends
  - Shows measurement stability or drift
  - Helps detect systematic changes

#### 2.4 Direction Analysis (`direction_analysis.png`)
- **Purpose**: Compares measurements by operation direction
- **Categories**: Normal vs Reverse operation
- **Insights**:
  - Shows differences between operational modes
  - Reveals direction-specific patterns
  - Helps understand operational characteristics

---

## Key Insights from the Analysis

### Data Characteristics
1. **Time Series Nature**: All measurements are high-frequency time series (20Hz sampling)
2. **Array Format**: Data stored as comma-separated arrays in CSV format
3. **Variable Lengths**: Arrays have different lengths, indicating varying measurement durations
4. **Multiple Sites**: Data from different sites and point machines

### Electrical Patterns
1. **Current and Voltage Relationships**: Strong correlations expected between related measurements
2. **Power Calculations**: Derived power values show consumption patterns
3. **Operational Modes**: Different patterns for Normal vs Reverse operations
4. **Temporal Trends**: Long-term changes in electrical characteristics

### Data Quality
1. **Completeness**: Analysis of empty vs non-empty arrays
2. **Consistency**: Length distributions and measurement patterns
3. **Outliers**: Statistical analysis reveals unusual measurements
4. **Clustering**: Groups similar operational patterns

---

## Technical Details

### Analysis Parameters
- **Sample Size**: 1000 rows for basic analysis, 500 for advanced analysis
- **Clustering**: MiniBatchKMeans with 5 clusters
- **Dimensionality Reduction**: PCA for clustering
- **Statistical Measures**: Mean, Std, Min, Max, Skewness, Kurtosis

### File Structure
```
├── analysis_output/
│   ├── time_series_samples.png
│   ├── statistical_distributions.png
│   ├── A_Current_clusters.png
│   ├── A_Voltage_clusters.png
│   ├── B_Current_clusters.png
│   ├── B_Voltage_clusters.png
│   └── summary_statistics.png
├── additional_analysis/
│   ├── correlation_heatmap.png
│   ├── correlation_scatter_plots.png
│   ├── power_analysis.png
│   ├── trend_analysis.png
│   └── direction_analysis.png
├── analyze_20hz_data.py
├── additional_visualizations.py
└── GRAPH_ANALYSIS_SUMMARY.md
```

---

## Recommendations for Further Analysis

1. **Full Dataset Processing**: Consider analyzing the complete 1.6M rows for comprehensive insights
2. **Real-time Monitoring**: Implement real-time analysis for operational monitoring
3. **Anomaly Detection**: Develop algorithms to detect unusual electrical patterns
4. **Predictive Modeling**: Use historical patterns to predict future electrical behavior
5. **Site-specific Analysis**: Compare patterns across different sites and machines

---

## Usage Instructions

To regenerate the analysis:
```bash
python3 analyze_20hz_data.py          # Basic analysis
python3 additional_visualizations.py  # Advanced analysis
```

The scripts will create the output directories and generate all visualizations automatically. 