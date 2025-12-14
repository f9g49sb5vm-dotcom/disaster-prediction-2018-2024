# Raw Data Directory

## Dataset Required

Please download the dataset from Kaggle and place it in this directory:

**Dataset:** Global Disaster Response 2018-2024  
**URL:** https://www.kaggle.com/datasets/mubeenshehzadi/global-disaster-2018-2024

**Expected file:** `global_disaster_response_2018_2024.csv`

## Dataset Overview

- **Time Period:** 2018-01-01 to 2024-12-31
- **Features:** 12 columns
- **Countries:** 20 countries across 6 continents
- **Disaster Types:** 10 types of natural disasters

## Download Instructions

1. Visit the Kaggle dataset page (URL above)
2. Download the CSV file
3. Place it in this directory (`data/raw/`)
4. File should be named: `global_disaster_response_2018_2024.csv`

## After Downloading

Once the dataset is in place, you can run the preprocessing notebook:

```bash
jupyter notebook notebooks/phase2_preprocessing.ipynb
```

This will create the processed data files in `data/processed/` directory.
