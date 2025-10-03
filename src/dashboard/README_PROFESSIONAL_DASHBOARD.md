# Professional Integrated Dashboard

## Overview

The Professional Integrated Dashboard is a clean, minimalist visualization system designed specifically for Brazilian political discourse analysis. It implements a professional design with academic research standards and focuses on data-driven insights.

## Design Specifications

### Color Scheme
- **Primary**: Dark Blue (#1B365D)
- **Secondary**: Dark Orange (#FF8C42)
- **Accents**: Light grays (#F5F5F5, #E0E0E0) for contrast
- **Text**: Dark (#2C2C2C) and White (#FFFFFF)

### Layout Structure
- **Single-page dashboard** (not multi-page)
- **Left sidebar** with light menu navigation
- **Top header** with extended color scheme
- **3-part content division**: main area + 2 side areas
- **Chart hierarchy**: Large main charts with small supplementary charts

### Key Features
- Professional minimalist design with clean lines
- Responsive grid layout
- Interactive elements with consistent styling
- Real-time data loading from pipeline outputs
- Multiple analysis views (Overview, Political, Sentiment, Temporal, Network)

## File Structure

```
src/dashboard/
├── integrated_professional_dashboard.py    # Main dashboard application
├── README_PROFESSIONAL_DASHBOARD.md        # This documentation
└── start_professional_dashboard.py         # Launcher script (in project root)
```

## Data Sources

The dashboard automatically loads processed data from:

```
pipeline_outputs/dashboard_ready/
├── *05_political_analysis*.csv     # Political classification results
├── *08_sentiment_analysis*.csv     # Sentiment analysis results
├── *14_temporal_analysis*.csv      # Temporal pattern analysis
├── *15_network_analysis*.csv       # Network coordination analysis
└── *summary*.json                  # Pipeline execution summaries
```

## Usage

### Quick Start

```bash
# From project root directory
python start_professional_dashboard.py
```

The launcher will:
- Automatically find available ports (8501-8504)
- Check for data availability
- Start the dashboard with proper configuration

### Manual Start

```bash
# Alternative manual start
python -m streamlit run src/dashboard/integrated_professional_dashboard.py --server.port 8502
```

### Access

Once started, access the dashboard at:
- Local: http://localhost:8501 (or 8502, 8503, 8504 if 8501 is busy)
- The launcher will display the exact URL

## Dashboard Views

### 1. Overview (Default)
**Layout**: 3-part content division
- **Main Area**: Large political distribution chart
- **Side Area 1**: Small content categories pie chart
- **Side Area 2**: Small sentiment trends line chart
- **Secondary Row**: Large temporal heatmap + small network summary

### 2. Political Analysis
- Detailed political alignment distributions
- Category breakdowns
- Sample classifications table

### 3. Sentiment Analysis
- Sentiment evolution timelines
- Emotional distribution charts
- Confidence metrics

### 4. Temporal Analysis
- Activity heatmaps (hour x day of week)
- Temporal pattern visualizations
- Peak activity identification

### 5. Network Analysis
- User activity rankings
- Coordination patterns
- Interaction summaries

## Technical Implementation

### Core Components

1. **DataLoader Class**
   - Manages data loading from pipeline outputs
   - Handles file timestamp sorting
   - Implements caching for performance

2. **ChartCreator Class**
   - Creates consistent professional charts
   - Applies color scheme uniformly
   - Handles different chart types (bar, pie, heatmap, timeline)

3. **Metrics Cards**
   - Display key statistics
   - Pipeline completion status
   - Data freshness indicators

### Chart Specifications

All charts follow professional standards:
- **Thin, clean lines** for borders and traces
- **Consistent color mapping** for political categories
- **Professional typography** (Arial family)
- **Minimal gridlines** for clarity
- **High contrast** for accessibility

### Performance Features

- **Automatic data refresh** button in sidebar
- **Lazy loading** of large datasets
- **Error handling** for missing data files
- **Responsive design** for different screen sizes

## Data Requirements

The dashboard expects pipeline output files with specific formats:

### Political Analysis Data
Required columns: `political_alignment`, `political_category`, `text_content`

### Sentiment Analysis Data
Required columns: `sentiment_score`, `sentiment_category`, `date`

### Temporal Analysis Data
Required columns: `date`, `hour`, `day_of_week` (auto-derived)

### Network Analysis Data
Required columns: `user_id`, activity metrics

## Troubleshooting

### Common Issues

1. **No data displayed**
   - Ensure pipeline has been run to generate data files
   - Check `pipeline_outputs/dashboard_ready/` directory exists
   - Verify file permissions

2. **Port conflicts**
   - Launcher automatically tries ports 8501-8504
   - Manually kill existing streamlit processes: `pkill -f streamlit`

3. **Chart rendering issues**
   - Clear browser cache
   - Refresh page (F5)
   - Check console for JavaScript errors

4. **Performance issues**
   - Use data refresh button to clear cache
   - Limit dataset size for faster rendering
   - Close other browser tabs

### Error Messages

- **"Political analysis data not available"**: Pipeline stage 05 needs to run
- **"Sentiment data not available"**: Pipeline stage 08 needs to run
- **"Data directory not found"**: Run pipeline first to generate outputs

## Academic Standards

This dashboard adheres to academic research standards:

- **No commercial language** or business metrics
- **Technical, neutral terminology** throughout
- **Research-focused visualizations**
- **Methodological transparency** in data presentation
- **Professional color schemes** suitable for academic publications

## Integration with Pipeline

The dashboard integrates seamlessly with the 22-stage analysis pipeline:

1. **Data Pipeline** generates analysis files
2. **Dashboard** automatically detects latest files
3. **Visualizations** update based on new analyses
4. **Metrics** reflect current pipeline status

## Development Guidelines

When modifying the dashboard:

1. **Maintain color scheme** consistency
2. **Preserve layout structure** (3-part division)
3. **Keep academic tone** in all text
4. **Test with real data** before deployment
5. **Update documentation** for any changes

## Version History

- **v5.1.0**: Initial professional dashboard implementation
- Professional color scheme (#1B365D, #FF8C42)
- Single-page layout with sidebar navigation
- Integrated data loading from pipeline outputs
- Minimalist design with clean traces