# GitHub Copilot Instructions - Global Disaster Prediction System

## Project Overview

This is a Machine Learning project for analyzing and predicting global disasters from 2018-2024. The system uses Kaggle data with 12 features covering various natural disasters across 20 countries on 6 continents.

**Purpose:** Build a complete ML system with >85% classification accuracy and R² >0.80 for regression tasks.

## Technology Stack

- **Language:** Python 3.8+
- **Core Libraries:** pandas (>=1.5.0), numpy (>=1.23.0), scikit-learn (>=1.2.0)
- **ML Frameworks:** xgboost (>=1.7.0), tensorflow (>=2.11.0), imbalanced-learn (>=0.10.0)
- **Visualization:** matplotlib (>=3.6.0), seaborn (>=0.12.0), plotly (>=5.11.0), folium (>=0.14.0)
- **Development:** jupyter (>=1.0.0)

## Project Structure

```
disaster-prediction-2018-2024/
├── data/
│   ├── raw/                    # Original dataset (not in repo)
│   └── processed/              # Processed data (train.csv, test.csv, encoders.pkl)
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── phase2_preprocessing.ipynb
│   ├── phase2_eda.ipynb
│   ├── phase3_model_building.ipynb
│   ├── phase4_training_evaluation.ipynb
│   └── phase5_visualization.ipynb
├── src/                        # Python modules
│   ├── feature_engineering.py
│   ├── outlier_detection.py
│   ├── data_augmentation.py
│   └── data_split.py
├── models/                     # Trained model files (.pkl, .h5)
├── reports/                    # Documentation and visualizations
│   ├── figures/
│   └── final_report.md
└── requirements.txt
```

## Coding Standards

### Python Style
- Follow **PEP 8** conventions
- Use **bilingual documentation** (Vietnamese + English) in docstrings and comments
- Maximum line length: 88 characters (Black formatter compatible)
- Use type hints where appropriate
- Write comprehensive docstrings for all functions and classes

### Naming Conventions
- **Variables/Functions:** snake_case (e.g., `create_temporal_features`, `disaster_type`)
- **Classes:** PascalCase (e.g., `DisasterPredictor`)
- **Constants:** UPPER_SNAKE_CASE (e.g., `DEFAULT_RANDOM_STATE`)
- **Private methods:** Prefix with underscore (e.g., `_internal_helper`)

### Code Organization
- Group imports: standard library → third-party → local modules
- Add blank lines between logical sections
- Keep functions focused and under 50 lines when possible
- Use descriptive variable names (e.g., `severity_index` not `si`)

## Data Conventions

### Dataset Features (12 original)
1. `date` - Disaster occurrence date (2018-2024)
2. `country` - 20 countries across 6 continents:
   - **North America:** United States, Canada, Mexico
   - **South America:** Brazil, Chile
   - **Europe:** France, Germany, Greece, Spain, Italy, Turkey
   - **Asia:** India, China, Bangladesh, Japan
   - **Southeast Asia:** Indonesia, Philippines
   - **Africa:** Nigeria, South Africa
   - **Oceania:** Australia
3. `disaster_type` - 10 types: Earthquake, Extreme Heat, Hurricane, Wildfire, Flood, Storm Surge, Drought, Tornado, Landslide, Volcanic Eruption
4. `severity_index` - Severity level (1-10)
5. `casualties` - Number of people affected
6. `economic_loss_usd` - Economic damage (USD)
7. `response_time_hours` - Response time (hours)
8. `aid_amount_usd` - Aid amount (USD)
9. `response_efficiency_score` - Response efficiency (0-100)
10. `recovery_days` - Recovery days
11. `latitude` - Latitude
12. `longitude` - Longitude

### Engineered Features (20+)
- **Temporal (8):** year, month, day, quarter, season, day_of_week, is_weekend, days_since_start
- **Geographic (1):** continent (mapped from country)
- **Derived (5):** severity_category, aid_per_casualty, loss_per_casualty, recovery_efficiency, response_effectiveness

### Data Processing Rules
- Use **median** for filling missing numerical values
- Use **mode** for filling missing categorical values
- Handle outliers using **IQR method** for casualties, economic_loss_usd, aid_amount_usd
- Handle outliers using **Z-score (threshold=3)** for response_time_hours, recovery_days
- Apply **log/sqrt transformations** for highly skewed distributions
- Use **stratified split** by disaster_type (80/20 train/test)
- Set **random_state=42** for reproducibility

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Running Notebooks
Execute notebooks in sequential order:
1. `phase2_preprocessing.ipynb` - Data cleaning and feature engineering
2. `phase2_eda.ipynb` - Exploratory data analysis
3. `phase3_model_building.ipynb` - Model development
4. `phase4_training_evaluation.ipynb` - Training and evaluation
5. `phase5_visualization.ipynb` - Results visualization

### Testing
- No formal test suite currently exists
- Validate changes by running relevant notebook cells
- Check output visualizations in `reports/figures/`

## ML Model Guidelines

### Classification Models
- Target: `disaster_type` (10 classes)
- Baseline: Logistic Regression
- Advanced: Random Forest, XGBoost, SVM, Neural Network
- Target Accuracy: >85%
- Use **5-fold cross-validation**
- Handle class imbalance with **SMOTE** (from `src.data_augmentation`)

### Regression Models
- Targets: severity_index, casualties, economic_loss_usd, recovery_days
- Baseline: Linear Regression
- Advanced: Random Forest, XGBoost, Gradient Boosting, Neural Network
- Target R²: >0.80, RMSE: <10%
- Use **5-fold cross-validation**

### Hyperparameter Tuning
- Use **GridSearchCV** for systematic search
- Define comprehensive parameter grids
- Use appropriate scoring metrics (accuracy for classification, neg_mean_squared_error for regression)

## Visualization Standards

### Color Schemes
- Use **seaborn** default palettes for consistency
- Geographic maps: Use intuitive colors (red for severity, blue for water-related disasters)
- Time series: Use distinct colors for different disaster types

### Chart Requirements
- Include clear titles in English
- Label axes with units (e.g., "Economic Loss (USD)", "Recovery Time (days)")
- Add legends when multiple categories are shown
- Save figures to `reports/figures/` with descriptive names
- Use DPI=300 for publication-quality images

### Recommended Plot Types
- **Time series:** Line plots with markers
- **Geographic:** Folium/plotly maps with interactive features
- **Distributions:** Histograms, KDE plots, box plots
- **Relationships:** Scatter plots, correlation heatmaps
- **Model comparison:** Bar charts, ROC curves, confusion matrices

## Important Context

### Academic Standards
- This is a Vietnamese university ML project
- Documentation must be bilingual (Vietnamese + English)
- Follow Vietnamese academic presentation standards
- Target performance: >85% accuracy, R² >0.80

### Data Source
- Kaggle dataset: [Global Disaster Response 2018-2024](https://www.kaggle.com/datasets/mubeenshehzadi/global-disaster-2018-2024)
- Dataset is NOT included in repository (must be downloaded)
- Place downloaded CSV in `data/raw/global_disaster_response_2018_2024.csv`

### Key Design Decisions
- Modular code structure with reusable functions in `src/`
- Jupyter notebooks for interactive analysis and experimentation
- Separate preprocessing from modeling for clear workflow
- Emphasis on visualization for stakeholder communication

## Common Tasks

### Adding New Features
1. Implement feature engineering function in `src/feature_engineering.py`
2. Follow existing function structure (copy DataFrame, return modified copy)
3. Add comprehensive docstring (bilingual)
4. Update `engineer_all_features()` to include new features
5. Test in preprocessing notebook

### Adding New Models
1. Add model implementation to `phase3_model_building.ipynb`
2. Follow existing model structure (train, predict, evaluate)
3. Use consistent hyperparameter tuning approach
4. Add model comparison metrics
5. Save trained model to `models/` directory

### Creating Visualizations
1. Create visualization code in appropriate notebook
2. Follow naming convention: `{analysis_type}_{detail}.png`
3. Save to `reports/figures/`
4. Update `reports/figures/README.md` if adding new visualization type
5. Use consistent styling with existing visualizations

## Error Prevention

### Common Pitfalls to Avoid
- **Don't** modify raw data files - always work on copies
- **Don't** hard-code file paths - use relative paths from project root
- **Don't** commit large data files or model files to Git
- **Don't** skip data validation after preprocessing
- **Don't** forget to set random seeds for reproducibility

### Before Committing
- Verify notebooks can run from top to bottom
- Check that no absolute paths are hard-coded
- Ensure new dependencies are added to `requirements.txt`
- Update documentation if adding new features
- Clear notebook output cells before committing (optional, but recommended)

## Questions to Ask

When assigned a task, consider:
1. **Which phase does this belong to?** (Preprocessing, EDA, Modeling, Evaluation, Visualization)
2. **Which files need modification?** (notebooks vs src modules)
3. **Does this require new dependencies?** (Update requirements.txt)
4. **Are there existing similar implementations?** (Reuse patterns)
5. **Is bilingual documentation needed?** (Usually yes)
6. **Does this affect reproducibility?** (Set random seeds)
7. **Is this change properly validated?** (Run relevant notebook sections)

## Getting Started

For new contributors:
1. Read `README.md` for project overview
2. Review `QUICKSTART.md` for setup instructions
3. Examine existing code in `src/` for patterns
4. Run notebooks sequentially to understand workflow
5. Check `reports/final_report.md` for detailed methodology

## Contact & Support

- **Repository:** https://github.com/f9g49sb5vm-dotcom/disaster-prediction-2018-2024
- **Issues:** Use GitHub Issues for bugs and feature requests
- **Documentation:** See README.md and QUICKSTART.md for detailed guides
