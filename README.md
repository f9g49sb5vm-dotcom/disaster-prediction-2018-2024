# Global Disaster Prediction System 2018-2024 🌍🔥💧

## Hệ thống Dự đoán và Phân tích Thảm họa Toàn cầu

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11%2B-FF6F00)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.readthedocs.io/)

## 📖 Giới thiệu / Introduction

Dự án này xây dựng hệ thống Machine Learning hoàn chỉnh để phân tích và dự đoán thảm họa toàn cầu trong giai đoạn 2018-2024. Hệ thống sử dụng dữ liệu từ Kaggle bao gồm 12 đặc trưng về các loại thảm họa tự nhiên khác nhau tại 20 quốc gia trên 6 châu lục.

This project builds a complete Machine Learning system for analyzing and predicting global disasters from 2018-2024. The system uses Kaggle data including 12 features about various natural disasters across 20 countries on 6 continents.

### 🎯 Mục tiêu / Objectives

- Phân tích toàn diện dữ liệu thảm họa toàn cầu 2018-2024
- Xây dựng và so sánh nhiều mô hình Machine Learning
- Dự đoán loại thảm họa, mức độ nghiêm trọng, và các chỉ số khác
- Tạo ra các visualization chuyên nghiệp để hỗ trợ ra quyết định
- Đạt độ chính xác >85% cho classification và R² >0.80 cho regression

## 📊 Dataset Overview

### Nguồn dữ liệu / Data Source

**Kaggle:** [Global Disaster Response 2018-2024](https://www.kaggle.com/datasets/mubeenshehzadi/global-disaster-2018-2024)

### 12 Features:

1. **date** - Thời gian xảy ra thảm họa (2018-01-01 → 2024-12-31)
2. **country** - 20 quốc gia trên toàn cầu
3. **disaster_type** - 10 loại: Earthquake, Extreme Heat, Hurricane, Wildfire, Flood, Storm Surge, Drought, Tornado, Landslide, Volcanic Eruption
4. **severity_index** - Mức độ nghiêm trọng (1-10)
5. **casualties** - Số người bị ảnh hưởng
6. **economic_loss_usd** - Thiệt hại kinh tế (USD)
7. **response_time_hours** - Thời gian phản ứng (giờ)
8. **aid_amount_usd** - Số tiền viện trợ (USD)
9. **response_efficiency_score** - Điểm hiệu quả ứng phó (0-100)
10. **recovery_days** - Số ngày phục hồi
11. **latitude** - Vĩ độ
12. **longitude** - Kinh độ

### 20 Countries / 6 Continents:

- **North America:** United States, Canada, Mexico
- **South America:** Brazil, Chile
- **Europe:** France, Germany, Greece, Spain, Italy, Turkey
- **Asia:** India, China, Bangladesh, Japan
- **Southeast Asia:** Indonesia, Philippines
- **Africa:** Nigeria, South Africa
- **Oceania:** Australia

## 📁 Project Structure

```
disaster-prediction-2018-2024/
├── data/
│   ├── raw/                              # Dữ liệu gốc / Raw data
│   │   └── global_disaster_response_2018_2024.csv
│   └── processed/                        # Dữ liệu đã xử lý / Processed data
│       ├── train.csv
│       ├── test.csv
│       └── encoders.pkl
│
├── notebooks/                            # Jupyter notebooks
│   ├── phase2_preprocessing.ipynb        # Data cleaning & preprocessing
│   ├── phase2_eda.ipynb                  # Exploratory Data Analysis
│   ├── phase3_model_building.ipynb       # Model development
│   ├── phase4_training_evaluation.ipynb  # Training & evaluation
│   └── phase5_visualization.ipynb        # Advanced visualizations
│
├── src/                                  # Source code modules
│   ├── __init__.py
│   ├── feature_engineering.py            # Feature engineering functions
│   ├── outlier_detection.py              # Outlier detection & handling
│   ├── data_augmentation.py              # SMOTE & data augmentation
│   └── data_split.py                     # Train-test splitting
│
├── models/                               # Trained models
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── neural_network.h5
│   └── best_model.pkl
│
├── reports/                              # Reports & presentations
│   ├── figures/                          # Generated visualizations
│   ├── final_report.md                   # Full analysis report
│   ├── final_report.pdf                  # PDF version
│   └── presentation.pptx                 # Presentation slides
│
├── requirements.txt                      # Python dependencies
├── .gitignore                           # Git ignore rules
└── README.md                            # This file
```

## 🚀 Installation / Cài đặt

### Prerequisites / Yêu cầu

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/f9g49sb5vm-dotcom/disaster-prediction-2018-2024.git
cd disaster-prediction-2018-2024

# 2. Create virtual environment (recommended)
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook
```

## 📝 Usage / Sử dụng

### Step-by-Step Workflow

#### Phase 1: Data Preparation

1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/mubeenshehzadi/global-disaster-2018-2024)
2. Place CSV file in `data/raw/` directory

#### Phase 2: Preprocessing & EDA

```bash
# Run preprocessing notebook
jupyter notebook notebooks/phase2_preprocessing.ipynb

# Run EDA notebook
jupyter notebook notebooks/phase2_eda.ipynb
```

**What happens:**
- Data cleaning (missing values, duplicates, data types)
- Feature engineering (temporal, geographic, derived features)
- Outlier detection and handling
- Comprehensive exploratory data analysis with 50+ visualizations

#### Phase 3: Model Building

```bash
jupyter notebook notebooks/phase3_model_building.ipynb
```

**Models implemented:**
- **Classification:** Logistic Regression, Random Forest, XGBoost, SVM, Neural Network
- **Regression:** Linear Regression, Random Forest, XGBoost, Gradient Boosting, Neural Network

#### Phase 4: Training & Evaluation

```bash
jupyter notebook notebooks/phase4_training_evaluation.ipynb
```

**Evaluation metrics:**
- Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Regression: MSE, RMSE, MAE, R², MAPE

#### Phase 5: Visualization & Results

```bash
jupyter notebook notebooks/phase5_visualization.ipynb
```

**40+ professional visualizations including:**
- Geographic world maps
- Time series analysis
- Economic loss analysis
- Response efficiency analysis
- Model performance comparisons

### Using Python Modules

```python
# Import modules
from src.feature_engineering import engineer_all_features
from src.outlier_detection import analyze_outliers
from src.data_augmentation import balance_dataset
from src.data_split import create_train_test_split

# Load data
import pandas as pd
df = pd.read_csv('data/raw/global_disaster_response_2018_2024.csv')

# Feature engineering
df_engineered, encoders = engineer_all_features(df, fit=True)

# Outlier analysis
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

## 🎯 Key Features

### 1. Comprehensive Feature Engineering

- **Temporal Features:** year, month, day, quarter, season, day_of_week, is_weekend, days_since_start
- **Geographic Features:** continent mapping for all 20 countries
- **Derived Features:**
  - severity_category (Low, Medium, High, Critical)
  - aid_per_casualty
  - loss_per_casualty
  - recovery_efficiency
  - response_effectiveness

### 2. Advanced Preprocessing

- Missing value handling with documented strategies
- Duplicate removal
- Data type standardization
- Outlier detection (IQR + Z-score methods)
- Log and sqrt transformations for skewed data

### 3. Data Augmentation

- SMOTE for imbalanced disaster types
- Synthetic sample generation for rare classes
- Class balancing strategies

### 4. Multiple ML Models

- 5+ classification models
- 5+ regression models
- Hyperparameter tuning with GridSearchCV
- 5-fold cross-validation
- Model comparison and selection

### 5. Rich Visualizations

- Time series plots
- Geographic heat maps
- Correlation matrices
- Box plots and scatter plots
- Model performance visualizations
- Continental analysis charts

## 📈 Results Summary

### Target Performance Metrics

- **Classification Accuracy:** >85%
- **Regression RMSE:** <10%
- **Regression R²:** >0.80

### Best Model

*(To be filled after model training)*

```
Model: [Best Model Name]
Classification Accuracy: [XX.XX]%
Regression R²: [X.XX]
Training Time: [XX] seconds
```

## 📚 References / Tài liệu tham khảo

### GitHub Repositories

1. [Natural Disaster Prediction ML](https://github.com/ManishaLagisetty/Natural-Disaster-Prediction-Using-Machine-Learning) - Feature engineering techniques
2. [Disaster Recovery Time Prediction](https://github.com/haz4rl/Disaster-Recovery-Time-Prediction-using-Machine-Learning) - 2018-2024 analysis approaches
3. [ML Flood Prediction & Response](https://github.com/rfuadur/ML-Flood-Prediction-Disaster-Response) - Model comparison methodologies
4. [Global Natural Disasters Analysis](https://github.com/sri-maharagni-karrolla/Global-Natural-Disasters-Analysis) - EDA and visualization techniques

### Research Papers

5. [Hybrid NN-XGBoost for Disaster Prediction](https://www.diva-portal.org/smash/get/diva2:1961967/FULLTEXT01.pdf) - Achieving 94.8% accuracy
6. [MDPI Machine Learning Review](https://www.mdpi.com/2504-4990/4/2/20) - ML best practices for disaster prediction

### Datasets

7. [Global Disaster Response 2018-2024](https://www.kaggle.com/datasets/mubeenshehzadi/global-disaster-2018-2024) - Primary dataset

## 👥 Contributors

- **Project Team:** Disaster Prediction Team
- **Course:** Machine Learning Major Project
- **Institution:** [Your University Name]
- **Year:** 2024

## 📄 License

This project is created for educational purposes as part of a university Machine Learning course.

## 🤝 Contributing

This is an academic project. If you have suggestions or find issues, please feel free to open an issue or submit a pull request.

## 📞 Contact

For questions or discussions about this project:
- GitHub Issues: [Open an issue](https://github.com/f9g49sb5vm-dotcom/disaster-prediction-2018-2024/issues)

## 🙏 Acknowledgments

- Kaggle for providing the dataset
- All referenced GitHub repositories and research papers
- Open-source community for the excellent ML libraries

---

**Note:** This is an academic Machine Learning project following Vietnamese university standards. All code is production-ready, well-documented, and reproducible.

**Ghi chú:** Đây là dự án Machine Learning học thuật tuân theo tiêu chuẩn đại học Việt Nam. Tất cả code đều sẵn sàng sử dụng, được tài liệu hóa tốt và có thể tái tạo.