# DATASET INPUT/OUTPUT PATH CONFIGURATION

## DIRECTORY STRUCTURE

```
project/
├── data/
│   ├── uploads/           # Input datasets (primary source)
│   ├── DATASETS_FULL/     # Alternative input location (currently empty)
│   ├── interim/           # Intermediate processing files
│   └── dashboard_results/ # Dashboard-specific outputs
├── pipeline_outputs/      # Main pipeline outputs (CREATED)
├── checkpoints/          # Pipeline checkpoints and recovery
├── logs/                 # Execution logs
└── src/dashboard/data/   # Dashboard integration files
```

## PATH PRIORITIES (Fixed)

### Input Discovery:
1. `data/uploads/` (primary)
2. `data/DATASETS_FULL/` (fallback - currently empty)
3. Configured paths from config files

### Output Paths:
1. `pipeline_outputs/` (primary - newly created)
2. `data/interim/` (fallback)
3. `data/dashboard_results/` (final fallback)

## CRITICAL FIXES APPLIED

1. **Pipeline Stage Names**: Updated path_updating_stages to match v4.9 naming (01-20)
2. **Output Path Logic**: Added pipeline_outputs with proper fallbacks
3. **Dataset Discovery**: Added validation for non-empty CSV files
4. **Directory Creation**: Ensured all required directories exist

## DATA FLOW

```
INPUT: data/uploads/*.csv
  ↓
STAGE 01: pipeline_outputs/filename_01_chunk_processing.csv
  ↓
STAGE 02: pipeline_outputs/filename_02_encoding_validation.csv
  ↓
... (continues through 22 stages)
  ↓
FINAL: pipeline_outputs/filename_20_pipeline_validation.csv
```

## INTEGRATION POINTS

- **Checkpoints**: `/checkpoints/pipeline_XX_stagename_YYYYMMDD_HHMMSS.json`
- **Dashboard**: Auto-copy to `src/dashboard/data/dashboard_results/`
- **API Integration**: Results logged to `logs/pipeline_execution.log`