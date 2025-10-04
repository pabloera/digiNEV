# Stage 03 Deduplication Visualizations Documentation

## Overview

The Stage 03 Deduplication Visualizations Dashboard provides comprehensive analysis of cross-dataset deduplication patterns for the Brazilian political discourse research project (digiNEV v.final). This module implements five key visualization types to understand duplicate content propagation across the five political datasets.

## Implementation Details

### Files Created
- `/src/dashboard/stage03_deduplication_dashboard.py` - Main dashboard component
- `/src/dashboard/pages/3_🔄_Deduplication.py` - Streamlit page integration

### Data Requirements
The dashboard requires processed data files with deduplication metrics:
- `dupli_freq`: Frequency of duplication (integer)
- `channels_found`: Number of channels where content appears (integer)
- `date_span_days`: Propagation period in days (integer)
- `dataset_source`: Source dataset identifier (string)
- `datetime_parsed`: Parsed datetime for temporal analysis (datetime)
- `normalized_text`: Normalized text content for comparison (string)

### Datasets Analyzed
1. **Governo Bolsonaro (2019-2021)** - 135.9 MB
2. **Pandemia (2021-2022)** - 230.0 MB
3. **Pós-Eleição (2022-2023)** - 93.2 MB
4. **Eleições (2022-2023)** - 54.2 MB
5. **Eleições Extra (2022-2023)** - 25.2 MB

## Visualization Components

### 1. Duplicate Frequency Heatmap
**Function**: `create_duplicate_frequency_heatmap()`
**Purpose**: Visualizes cross-dataset duplicate frequency patterns

**Features**:
- Matrix view of duplication between datasets
- Diagonal shows internal duplicates
- Off-diagonal shows shared content
- Color intensity indicates frequency
- Hover tooltips with exact counts

**Interpretation**:
- Darker colors = higher duplication
- Diagonal values = internal dataset duplicates
- Cross-dataset patterns reveal content sharing

### 2. Duplicate Content Clustering
**Function**: `create_duplicate_content_clustering()`
**Purpose**: 3D clustering analysis of duplication patterns

**Features**:
- X-axis: Frequency of duplication
- Y-axis: Average channels where content appears
- Z-axis: Propagation period (days)
- Marker size: Quantity of texts
- Color: Number of datasets involved

**Interpretation**:
- Identifies dominant propagation patterns
- Shows relationship between frequency, reach, and time
- Reveals coordinated vs organic sharing

### 3. Temporal Duplicate Distribution
**Function**: `create_temporal_duplicate_distribution()`
**Purpose**: Time series analysis of duplication patterns

**Features**:
- Two-panel layout (quantity and frequency)
- Monthly aggregation by dataset
- Separate lines for each dataset
- Interactive hover with details

**Interpretation**:
- Seasonal patterns in duplication
- Dataset-specific temporal trends
- Correlation between volume and duplication rate

### 4. Shared Content Flow Diagram
**Function**: `create_shared_content_flow()`
**Purpose**: Sankey diagram of content flow between datasets

**Features**:
- Flow width indicates content volume
- Source and target datasets
- Bidirectional flow analysis
- Color-coded by dataset

**Interpretation**:
- Content propagation directions
- Dominant source vs recipient datasets
- Evidence of coordinated distribution

### 5. Duplicate Propagation Patterns
**Function**: `create_duplicate_propagation_patterns()`
**Purpose**: Heatmap analysis of propagation speed vs reach

**Features**:
- X-axis: Propagation speed categories
- Y-axis: Channel reach categories
- Cell values: Frequency of occurrence
- Annotations with exact counts

**Categories**:
- **Speed**: Instantânea, Mesmo Dia, 1 Semana, 1 Mês, Longo Prazo
- **Reach**: Canal Único, Poucos Canais, Vários Canais, Ampla Distribuição

## Dashboard Metrics

### Key Performance Indicators
- **Total Records**: Count of all processed records
- **Records with Duplicates**: Count and percentage of duplicated content
- **Deduplication Reduction**: Percentage reduction achieved
- **Datasets Processed**: Number of datasets analyzed

### Statistical Insights
- **Duplication Rate**: 9.8% of records are duplicates (based on test data)
- **Volume Reduction**: 40-50% expected reduction per Stage 03 design
- **Temporal Coverage**: 2019-2021 date range in current test data
- **Cross-Dataset Sharing**: Quantified content overlap between datasets

## Usage Instructions

### Standalone Execution
```bash
cd /Users/pabloalmada/development/project/dataanalysis-bolsonarismo
python src/dashboard/stage03_deduplication_dashboard.py
```

### Streamlit Integration
```bash
streamlit run src/dashboard/pages/3_🔄_Deduplication.py
```

### Programmatic Usage
```python
from src.dashboard.stage03_deduplication_dashboard import Stage03DeduplicationDashboard

dashboard = Stage03DeduplicationDashboard()
df = dashboard.load_deduplication_data()

if df is not None:
    # Generate specific visualizations
    heatmap = dashboard.create_duplicate_frequency_heatmap(df)
    clustering = dashboard.create_duplicate_content_clustering(df)
    temporal = dashboard.create_temporal_duplicate_distribution(df)
    flow = dashboard.create_shared_content_flow(df)
    patterns = dashboard.create_duplicate_propagation_patterns(df)
```

## Data Validation

### Input Requirements
- Files must be in `data/processed/` directory
- CSV format with semicolon separator
- Required columns: `dupli_freq`, `dataset_source`, `normalized_text`
- Optional temporal columns: `datetime_parsed`, `datetime`

### Quality Checks
- Automatic validation of data availability
- Graceful handling of missing columns
- Error reporting for data loading issues
- Fallback visualizations for incomplete data

## Technical Architecture

### Design Principles
- **Clean, minimalist aesthetics** with thin line styling
- **Academic-quality visualizations** suitable for research
- **Real data only** - no synthetic or invented metrics
- **Portuguese text compatibility** for Brazilian content
- **Interactive exploration** without visual clutter

### Color Scheme
- **Governo Bolsonaro**: #1f77b4 (blue)
- **Pandemia**: #ff7f0e (orange)
- **Pós-Eleição**: #2ca02c (green)
- **Eleições**: #d62728 (red)
- **Eleições Extra**: #9467bd (purple)

### Error Handling
- Comprehensive try-catch blocks
- User-friendly error messages
- Graceful degradation for missing data
- Detailed logging for debugging

## Research Applications

### Academic Use Cases
- **Coordinated Inauthentic Behavior**: Detect artificial amplification patterns
- **Information Propagation**: Track how political content spreads
- **Temporal Analysis**: Identify periods of high duplication activity
- **Cross-Platform Studies**: Compare sharing patterns across datasets
- **Content Authenticity**: Distinguish organic vs coordinated sharing

### Methodological Contributions
- **Quantitative Deduplication Metrics**: Objective measurement of content overlap
- **Multi-Dataset Analysis**: Comprehensive cross-platform perspective
- **Temporal Dynamics**: Understanding time-dependent propagation patterns
- **Network Effects**: Visualizing coordination and influence patterns

## Performance Characteristics

### Test Results (October 2025)
- **Data Loaded**: 2,686 records successfully processed
- **Duplicates Detected**: 262 records (9.8% duplication rate)
- **Datasets Covered**: 2 of 5 available datasets
- **Temporal Range**: July 2019 - October 2021
- **Visualization Generation**: All 5 charts created successfully

### Scalability Considerations
- Efficient pandas operations for large datasets
- Memory-conscious data loading with chunking support
- Optimized plotly visualizations for interactive performance
- Modular design allows selective visualization generation

## Future Enhancements

### Planned Features
- **Real-time Processing**: Live dashboard updates during pipeline execution
- **Export Capabilities**: PDF and PNG export for research publications
- **Advanced Clustering**: Machine learning-based content similarity analysis
- **Statistical Testing**: Significance tests for observed patterns
- **Comparative Analysis**: Side-by-side dataset comparison tools

### Integration Opportunities
- **Main Dashboard**: Embed as tab in primary analysis interface
- **Pipeline Monitoring**: Real-time deduplication progress tracking
- **Research Workflows**: Integration with academic reporting tools
- **API Endpoints**: RESTful access for external research tools

## Validation and Testing

### Test Coverage
✅ Data loading from processed files
✅ All visualization functions operational
✅ Error handling for missing data
✅ Streamlit page integration
✅ Real data processing (2,686 records)

### Quality Assurance
- **Academic Standards**: Follows research visualization best practices
- **Data Integrity**: Only displays real pipeline-generated data
- **User Experience**: Intuitive navigation and clear documentation
- **Performance**: Efficient processing of large datasets

---

**Version**: v.final | **Created**: October 2025 | **Author**: digiNEV Research Team
**Project**: Brazilian Political Discourse Analysis | **Component**: Stage 03 Deduplication Analysis