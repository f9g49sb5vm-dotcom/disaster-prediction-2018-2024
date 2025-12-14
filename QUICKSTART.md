# Quick Start Guide / Hướng dẫn Bắt đầu Nhanh

## 📋 Prerequisites / Yêu cầu

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook
- At least 4GB RAM
- 2GB free disk space

## 🚀 Setup Instructions / Hướng dẫn Cài đặt

### Step 1: Clone Repository

```bash
git clone https://github.com/f9g49sb5vm-dotcom/disaster-prediction-2018-2024.git
cd disaster-prediction-2018-2024
```

### Step 2: Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- pandas (data manipulation)
- numpy (numerical computing)
- matplotlib, seaborn (visualization)
- plotly, folium (interactive visualizations)
- scikit-learn (machine learning)
- xgboost (gradient boosting)
- tensorflow (deep learning)
- imbalanced-learn (SMOTE)

### Step 4: Download Dataset

1. Go to Kaggle: https://www.kaggle.com/datasets/mubeenshehzadi/global-disaster-2018-2024
2. Download the CSV file
3. Place it in `data/raw/` directory
4. Rename to: `global_disaster_response_2018_2024.csv`

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

Your browser will open with the Jupyter interface.

## 📓 Execution Workflow / Quy trình Thực thi

Execute the notebooks **in sequential order**:

### Phase 2: Data Preprocessing & EDA

#### 2.1 Preprocessing
```
Open: notebooks/phase2_preprocessing.ipynb
```

**What it does:**
- Loads raw data
- Cleans data (missing values, duplicates, data types)
- Applies feature engineering
- Detects and handles outliers
- Splits data into train/test sets
- Saves processed data

**Output:**
- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/processed/encoders.pkl`
- `data/processed/full_engineered_data.csv`

#### 2.2 Exploratory Data Analysis
```
Open: notebooks/phase2_eda.ipynb
```

**What it does:**
- Analyzes all 12 features
- Creates 50+ visualizations
- Continental analysis
- Temporal trends
- Correlation analysis
- Geographic visualization

**Output:**
- Multiple PNG files in `reports/figures/eda/`

### Phase 3: Model Building

```
Open: notebooks/phase3_model_building.ipynb
```

**What it does:**
- Defines classification models (Logistic Regression, Random Forest, XGBoost, SVM, Neural Network)
- Defines regression models (Linear, Random Forest, XGBoost, Gradient Boosting, Neural Network)
- Sets hyperparameter grids

### Phase 4: Training & Evaluation

```
Open: notebooks/phase4_training_evaluation.ipynb
```

**What it does:**
- Trains all models with 5-fold cross-validation
- Performs hyperparameter tuning with GridSearchCV
- Evaluates models (accuracy, precision, recall, F1, ROC-AUC, R², RMSE, MAE)
- Creates comparison tables
- Saves best models

**Output:**
- Model files in `models/` directory
- Performance metrics
- Comparison tables

### Phase 5: Visualization & Results

```
Open: notebooks/phase5_visualization.ipynb
```

**What it does:**
- Creates 40+ professional visualizations
- Geographic world maps
- Economic analysis
- Response and recovery analysis
- Model performance visualization

**Output:**
- Multiple PNG files in `reports/figures/results/`

## 📊 Expected Results / Kết quả Mong đợi

### Classification Performance
- **Target:** Accuracy > 85%
- **Metrics:** Precision, Recall, F1-Score, ROC-AUC

### Regression Performance
- **Target:** R² > 0.80, RMSE < 10%
- **Metrics:** MSE, RMSE, MAE, MAPE

## 🔧 Using Python Modules Directly

You can also use the source modules in your own scripts:

```python
import sys
sys.path.append('src')

from feature_engineering import engineer_all_features
from outlier_detection import analyze_outliers
from data_augmentation import balance_dataset
from data_split import create_train_test_split

# Load your data
import pandas as pd
df = pd.read_csv('data/raw/global_disaster_response_2018_2024.csv')

# Apply feature engineering
df_engineered, encoders = engineer_all_features(df, fit=True)

# Analyze outliers
outlier_results = analyze_outliers(df_engineered)

# Balance dataset (optional)
df_balanced = balance_dataset(df_engineered, target_column='disaster_type')

# Split data
train_df, test_df, encoders = create_train_test_split(
    df_engineered,
    target_column='disaster_type',
    test_size=0.2,
    save=True
)
```

## 📝 Project Structure

```
disaster-prediction-2018-2024/
├── data/
│   ├── raw/                    # Place dataset here
│   └── processed/              # Generated processed data
├── notebooks/                  # Jupyter notebooks (run in order)
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
├── models/                     # Saved trained models
├── reports/                    # Documentation and figures
│   ├── figures/
│   └── final_report.md
└── requirements.txt            # Python dependencies
```

## ⚠️ Troubleshooting / Xử lý Sự cố

### Issue: ModuleNotFoundError

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Jupyter kernel not found

**Solution:**
```bash
python -m ipykernel install --user --name=venv
```

### Issue: Out of memory during training

**Solution:**
- Reduce hyperparameter grid size
- Use fewer cross-validation folds
- Train models one at a time

### Issue: Dataset not found

**Solution:**
- Verify dataset is in `data/raw/` directory
- Check filename: `global_disaster_response_2018_2024.csv`
- Download from Kaggle if missing

## 📚 Additional Resources

### Documentation
- `README.md` - Main project documentation
- `reports/final_report.md` - Complete analysis report
- `data/raw/README.md` - Dataset instructions
- `reports/figures/README.md` - Visualization catalog

### References
- Kaggle Dataset: https://www.kaggle.com/datasets/mubeenshehzadi/global-disaster-2018-2024
- Scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- TensorFlow: https://www.tensorflow.org/

## 🎯 Success Criteria

✅ **Data Preprocessing:**
- No missing values
- 20+ engineered features
- Outliers handled

✅ **EDA:**
- 50+ visualizations created
- All 12 features analyzed
- Continental analysis completed

✅ **Model Training:**
- 5+ classification models
- 5+ regression models
- Hyperparameter tuning completed

✅ **Performance:**
- Classification accuracy > 85%
- Regression R² > 0.80
- RMSE < 10%

✅ **Documentation:**
- All notebooks executed successfully
- Final report generated
- All visualizations saved

## 💡 Tips for Best Results

1. **Run notebooks in order** - Each phase depends on previous ones
2. **Save your work frequently** - Use File → Save in Jupyter
3. **Monitor memory usage** - Close unused notebooks
4. **Check output** - Review generated files after each phase
5. **Document findings** - Add your own observations in notebooks
6. **Experiment** - Try different hyperparameters
7. **Visualize everything** - More visualizations = better insights

## 🤝 Need Help?

- Check `README.md` for detailed information
- Review `reports/final_report.md` for methodology
- Open an issue on GitHub
- Review error messages carefully

## 📅 Estimated Time

- Setup: 15-30 minutes
- Phase 2 (Preprocessing + EDA): 1-2 hours
- Phase 3 (Model Building): 30 minutes
- Phase 4 (Training): 2-4 hours (depending on hardware)
- Phase 5 (Visualization): 1-2 hours
- **Total: 5-9 hours**

---

**Good luck with your disaster prediction analysis!**  
**Chúc bạn thành công với phân tích dự đoán thảm họa!** 🌍🔥💧
