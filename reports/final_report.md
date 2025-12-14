# BÁO CÁO BÀI TẬP LỚN - PHÂN TÍCH THẢM HỌA TOÀN CẦU 2018-2024
# GLOBAL DISASTER PREDICTION AND ANALYSIS 2018-2024

---

**Tên đề tài / Project Title:**  
Xây dựng Hệ thống Machine Learning để Phân tích và Dự đoán Thảm họa Toàn cầu 2018-2024

**Sinh viên thực hiện / Students:**  
[Tên sinh viên / Student names]

**Giảng viên hướng dẫn / Instructor:**  
[Tên giảng viên / Instructor name]

**Học kỳ / Semester:**  
[Semester] - Năm học / Academic Year [Year]

---

## MỤC LỤC / TABLE OF CONTENTS

1. [GIỚI THIỆU / INTRODUCTION](#1-giới-thiệu--introduction)
2. [TỔNG QUAN DỮ LIỆU / DATA OVERVIEW](#2-tổng-quan-dữ-liệu--data-overview)
3. [PHƯƠNG PHÁP / METHODOLOGY](#3-phương-pháp--methodology)
4. [PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)](#4-phân-tích-khám-phá-dữ-liệu-eda)
5. [XÂY DỰNG MÔ HÌNH / MODEL BUILDING](#5-xây-dựng-mô-hình--model-building)
6. [KẾT QUẢ / RESULTS](#6-kết-quả--results)
7. [KẾT LUẬN / CONCLUSION](#7-kết-luận--conclusion)
8. [TÀI LIỆU THAM KHẢO / REFERENCES](#8-tài-liệu-tham-khảo--references)

---

## 1. GIỚI THIỆU / INTRODUCTION

### 1.1 Đặt vấn đề / Problem Statement

Thảm họa tự nhiên là một trong những thách thức lớn nhất mà nhân loại phải đối mặt trong thế kỷ 21. Từ năm 2018 đến 2024, thế giới đã chứng kiến sự gia tăng đáng kể về tần suất và cường độ của các thảm họa tự nhiên, bao gồm động đất, lũ lụt, cháy rừng, bão nhiệt đới, và nhiều loại thảm họa khác. Những sự kiện này không chỉ gây ra thiệt hại về người và tài sản mà còn ảnh hưởng sâu rộng đến nền kinh tế và xã hội toàn cầu.

Natural disasters are one of the greatest challenges humanity faces in the 21st century. From 2018 to 2024, the world has witnessed a significant increase in the frequency and intensity of natural disasters, including earthquakes, floods, wildfires, tropical storms, and many other types of disasters. These events not only cause human and property losses but also have far-reaching impacts on the global economy and society.

### 1.2 Mục tiêu / Objectives

Mục tiêu của đề tài này là:

1. **Phân tích toàn diện** dữ liệu thảm họa toàn cầu từ 2018-2024
2. **Xây dựng và so sánh** nhiều mô hình Machine Learning để dự đoán:
   - Loại thảm họa (disaster_type)
   - Mức độ nghiêm trọng (severity_category)
   - Số người bị ảnh hưởng (casualties)
   - Thời gian phục hồi (recovery_days)
3. **Đạt được hiệu suất cao**: Accuracy >85% cho classification, R² >0.80 cho regression
4. **Tạo ra insights** hữu ích cho việc ra quyết định và ứng phó thảm họa

The objectives of this project are:

1. **Comprehensive analysis** of global disaster data from 2018-2024
2. **Build and compare** multiple Machine Learning models to predict:
   - Disaster type (disaster_type)
   - Severity category (severity_category)
   - Number of casualties (casualties)
   - Recovery time (recovery_days)
3. **Achieve high performance**: Accuracy >85% for classification, R² >0.80 for regression
4. **Generate useful insights** for decision-making and disaster response

### 1.3 Ý nghĩa / Significance

Nghiên cứu này có ý nghĩa quan trọng trong việc:

- **Cải thiện khả năng dự báo** thảm họa và đánh giá rủi ro
- **Tối ưu hóa phân bổ nguồn lực** cứu trợ và viện trợ
- **Giảm thiểu thiệt hại** về người và tài sản thông qua cảnh báo sớm
- **Hỗ trợ hoạch định chính sách** về phòng chống thiên tai
- **Đóng góp vào nghiên cứu khoa học** về biến đổi khí hậu và thảm họa tự nhiên

This research is significant in:

- **Improving disaster prediction** capabilities and risk assessment
- **Optimizing resource allocation** for relief and aid
- **Minimizing losses** of lives and property through early warning
- **Supporting policy planning** for disaster prevention
- **Contributing to scientific research** on climate change and natural disasters

---

## 2. TỔNG QUAN DỮ LIỆU / DATA OVERVIEW

### 2.1 Nguồn dữ liệu / Data Source

**Dataset:** Global Disaster Response 2018-2024  
**Nguồn / Source:** [Kaggle](https://www.kaggle.com/datasets/mubeenshehzadi/global-disaster-2018-2024)  
**Kích thước / Size:** [X rows × 12 columns]  
**Thời gian / Time Period:** 2018-01-01 to 2024-12-31

### 2.2 Mô tả 12 Features

#### 2.2.1 Temporal Feature
1. **date** (datetime)
   - Thời gian xảy ra thảm họa
   - Time of disaster occurrence
   - Range: 2018-01-01 to 2024-12-31

#### 2.2.2 Geographic Features
2. **country** (categorical)
   - 20 quốc gia trên 6 châu lục
   - 20 countries across 6 continents
   - Categories: United States, Canada, Mexico, Brazil, Chile, France, Germany, Greece, Spain, Italy, Turkey, India, China, Bangladesh, Japan, Indonesia, Philippines, Nigeria, South Africa, Australia

3. **latitude** (numeric)
   - Vĩ độ địa lý
   - Geographic latitude
   - Range: [-90, 90]

4. **longitude** (numeric)
   - Kinh độ địa lý
   - Geographic longitude
   - Range: [-180, 180]

#### 2.2.3 Disaster Characteristics
5. **disaster_type** (categorical)
   - 10 loại thảm họa
   - 10 types of disasters
   - Categories: Earthquake, Extreme Heat, Hurricane, Wildfire, Flood, Storm Surge, Drought, Tornado, Landslide, Volcanic Eruption

6. **severity_index** (numeric)
   - Mức độ nghiêm trọng
   - Severity level
   - Range: 1-10 (1=Low, 10=Critical)

#### 2.2.4 Impact Features
7. **casualties** (numeric)
   - Số người bị ảnh hưởng
   - Number of people affected
   - Range: [0, ∞]

8. **economic_loss_usd** (numeric)
   - Thiệt hại kinh tế (USD)
   - Economic loss in USD
   - Range: [0, ∞]

#### 2.2.5 Response Features
9. **response_time_hours** (numeric)
   - Thời gian phản ứng (giờ)
   - Response time in hours
   - Range: [0, ∞]

10. **aid_amount_usd** (numeric)
    - Số tiền viện trợ (USD)
    - Aid amount in USD
    - Range: [0, ∞]

11. **response_efficiency_score** (numeric)
    - Điểm hiệu quả ứng phó
    - Response efficiency score
    - Range: 0-100

#### 2.2.6 Recovery Feature
12. **recovery_days** (numeric)
    - Số ngày phục hồi
    - Number of recovery days
    - Range: [0, ∞]

### 2.3 Phân bố theo Châu lục / Continental Distribution

#### 6 Continents:
- **North America:** United States, Canada, Mexico
- **South America:** Brazil, Chile
- **Europe:** France, Germany, Greece, Spain, Italy, Turkey
- **Asia:** India, China, Bangladesh, Japan
- **Southeast Asia:** Indonesia, Philippines
- **Africa:** Nigeria, South Africa
- **Oceania:** Australia

### 2.4 Thống kê Tổng quan / General Statistics

*[Phần này sẽ được điền sau khi chạy EDA / This section will be filled after running EDA]*

- **Tổng số thảm họa / Total disasters:** [X]
- **Tổng thiệt hại kinh tế / Total economic loss:** $[X] USD
- **Tổng số người bị ảnh hưởng / Total casualties:** [X]
- **Thảm họa phổ biến nhất / Most common disaster:** [X]
- **Quốc gia bị ảnh hưởng nhiều nhất / Most affected country:** [X]
- **Châu lục bị ảnh hưởng nhiều nhất / Most affected continent:** [X]

---

## 3. PHƯƠNG PHÁP / METHODOLOGY

### 3.1 Preprocessing Pipeline / Quy trình Tiền xử lý

#### 3.1.1 Data Cleaning
1. **Convert date to datetime**
   - Chuyển đổi cột date sang định dạng datetime
   - Convert date column to datetime format

2. **Handle missing values**
   - Chiến lược: Median cho numeric, Mode cho categorical
   - Strategy: Median for numeric, Mode for categorical

3. **Remove duplicates**
   - Loại bỏ các bản ghi trùng lặp
   - Remove duplicate records

4. **Fix data types**
   - Đảm bảo các cột có đúng kiểu dữ liệu
   - Ensure correct data types for all columns

5. **Standardize categorical values**
   - Chuẩn hóa giá trị phân loại
   - Standardize categorical values

### 3.2 Feature Engineering / Kỹ thuật Đặc trưng

#### 3.2.1 Temporal Features
Tạo 8 đặc trưng thời gian / Create 8 temporal features:
- **year:** Năm (2018-2024)
- **month:** Tháng (1-12)
- **day:** Ngày (1-31)
- **quarter:** Quý (1-4)
- **season:** Mùa (1=Winter, 2=Spring, 3=Summer, 4=Autumn)
- **day_of_week:** Thứ trong tuần (0=Monday, 6=Sunday)
- **is_weekend:** Cuối tuần (0/1)
- **days_since_start:** Số ngày kể từ thảm họa đầu tiên

#### 3.2.2 Geographic Features
Tạo đặc trưng châu lục / Create continent feature:
- **continent:** Mapping 20 countries → 6 continents

#### 3.2.3 Derived Features
Tạo 5 đặc trưng phái sinh / Create 5 derived features:

1. **severity_category:** Phân loại mức độ nghiêm trọng
   - Low: 1-3
   - Medium: 4-6
   - High: 7-8
   - Critical: 9-10

2. **aid_per_casualty:** Viện trợ trên mỗi người bị ảnh hưởng
   - Formula: aid_amount_usd / casualties

3. **loss_per_casualty:** Thiệt hại trên mỗi người bị ảnh hưởng
   - Formula: economic_loss_usd / casualties

4. **recovery_efficiency:** Hiệu suất phục hồi
   - Formula: recovery_days / severity_index

5. **response_effectiveness:** Hiệu quả phản ứng
   - Formula: response_efficiency_score / response_time_hours

#### 3.2.4 Encoding
- **Label Encoding:** disaster_type
- **One-Hot Encoding:** country, continent

**Tổng số features sau engineering / Total features after engineering:** [X]

### 3.3 Outlier Detection / Phát hiện Outliers

#### 3.3.1 IQR Method
Áp dụng cho / Applied to:
- casualties
- economic_loss_usd
- aid_amount_usd

**Formula:**
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

#### 3.3.2 Z-Score Method
Áp dụng cho / Applied to:
- response_time_hours
- recovery_days

**Formula:**
```
z = (x - μ) / σ
Outlier if |z| > 3
```

#### 3.3.3 Handling Skewness
Áp dụng log transformation cho / Apply log transformation to:
- casualties_log = log(casualties + 1)
- economic_loss_usd_log = log(economic_loss_usd + 1)
- aid_amount_usd_log = log(aid_amount_usd + 1)

### 3.4 Data Augmentation / Tăng cường Dữ liệu

#### 3.4.1 SMOTE (Synthetic Minority Over-sampling Technique)
- **Mục đích / Purpose:** Cân bằng các lớp disaster_type không cân bằng
- **Phương pháp / Method:** Tạo mẫu tổng hợp cho các lớp thiểu số
- **Kết quả / Result:** Cân bằng phân phối các loại thảm họa

### 3.5 Train-Test Split / Chia Dữ liệu

- **Method:** Stratified Split
- **Ratio:** 80% Train / 20% Test
- **Stratification:** By disaster_type
- **Random State:** 42 (for reproducibility)

---

## 4. PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)
## EXPLORATORY DATA ANALYSIS

### 4.1 Phân tích 12 Features

*[Phần này sẽ được điền chi tiết với kết quả từ EDA notebook]*
*[This section will be filled with detailed results from EDA notebook]*

#### 4.1.1 Date Analysis
- Time series trends
- Seasonal patterns
- Yearly comparison

#### 4.1.2 Country Analysis
- Distribution by country
- Top countries by casualties
- Top countries by economic loss

#### 4.1.3 Disaster Type Analysis
- Frequency of each disaster type
- Severity distribution by type
- Economic impact by type

#### 4.1.4 Severity Index Analysis
- Distribution of severity levels
- Correlation with other features
- Severity trends over time

#### 4.1.5 Casualties Analysis
- Statistical distribution
- Outliers identification
- By country and disaster type

#### 4.1.6 Economic Loss Analysis
- Total and average losses
- Top 10 most costly disasters
- By continent and country

#### 4.1.7 Response Time Analysis
- Average response time by country
- Impact on casualties
- Efficiency analysis

#### 4.1.8 Aid Amount Analysis
- Distribution of aid
- Aid adequacy (aid vs loss)
- By disaster type and country

#### 4.1.9 Response Efficiency Analysis
- Score patterns
- Correlation with outcomes
- Best performing countries

#### 4.1.10 Recovery Days Analysis
- By disaster type
- Factors affecting recovery
- Relationship with severity

#### 4.1.11-12 Geographic Analysis (Lat/Long)
- Geographic distribution
- Disaster hotspots
- Regional patterns

### 4.2 Continental Analysis

*[Chi tiết phân tích theo 6 châu lục / Detailed analysis by 6 continents]*

#### 4.2.1 Disasters by Continent
- Frequency distribution
- Types of disasters per continent

#### 4.2.2 Economic Loss by Continent
- Total and average losses
- Continent comparison

#### 4.2.3 Casualties by Continent
- Total affected population
- Severity comparison

#### 4.2.4 Response Efficiency by Continent
- Average efficiency scores
- Best and worst performers

#### 4.2.5 Aid Distribution by Continent
- Total aid received
- Aid per disaster

#### 4.2.6 Recovery Time by Continent
- Average recovery days
- Efficiency comparison

### 4.3 Temporal Trends Analysis

#### 4.3.1 Yearly Trends (2018-2024)
- Number of disasters per year
- Economic losses trend
- Casualties trend

#### 4.3.2 Monthly Patterns
- Seasonal variations
- Peak months for disasters

#### 4.3.3 Quarterly Analysis
- Distribution by quarter
- Seasonal disaster types

### 4.4 Correlation Analysis

#### 4.4.1 Feature Correlation Matrix
- Correlation heatmap
- Strong correlations identified
- Insights for feature selection

#### 4.4.2 Key Relationships
- Severity vs Casualties
- Economic Loss vs Aid Amount
- Response Time vs Efficiency

### 4.5 Key Findings from EDA

*[Tóm tắt các phát hiện quan trọng / Summary of key findings]*

1. **Most Common Disaster Type:** [X]
2. **Most Affected Continent:** [X]
3. **Average Response Time:** [X] hours
4. **Total Economic Loss:** $[X] USD
5. **Disaster Frequency Trend:** [Increasing/Decreasing/Stable]

---

## 5. XÂY DỰNG MÔ HÌNH / MODEL BUILDING

### 5.1 Classification Models

Mục tiêu: Dự đoán disaster_type và severity_category

#### 5.1.1 Logistic Regression (Baseline)
**Hyperparameters:**
```python
{
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
```

#### 5.1.2 Random Forest Classifier
**Hyperparameters:**
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

#### 5.1.3 XGBoost Classifier
**Hyperparameters:**
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}
```

#### 5.1.4 Support Vector Machine (SVM)
**Hyperparameters:**
```python
{
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}
```

#### 5.1.5 Neural Network (TensorFlow/Keras)
**Architecture:**
```python
Model: Sequential
- Dense(128, activation='relu', input_dim=X)
- Dropout(0.3)
- Dense(64, activation='relu')
- Dropout(0.2)
- Dense(32, activation='relu')
- Dense(num_classes, activation='softmax')

Optimizer: Adam
Loss: categorical_crossentropy
Metrics: accuracy
```

### 5.2 Regression Models

Mục tiêu: Dự đoán casualties, recovery_days, economic_loss

#### 5.2.1 Linear Regression (Baseline)
**Hyperparameters:**
```python
{
    'fit_intercept': [True, False],
    'normalize': [True, False]
}
```

#### 5.2.2 Random Forest Regressor
**Hyperparameters:**
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

#### 5.2.3 XGBoost Regressor
**Hyperparameters:**
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.8, 0.9]
}
```

#### 5.2.4 Gradient Boosting Regressor
**Hyperparameters:**
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9]
}
```

#### 5.2.5 Neural Network Regressor
**Architecture:**
```python
Model: Sequential
- Dense(128, activation='relu', input_dim=X)
- Dropout(0.3)
- Dense(64, activation='relu')
- Dropout(0.2)
- Dense(32, activation='relu')
- Dense(1, activation='linear')

Optimizer: Adam
Loss: mse
Metrics: mae
```

### 5.3 Training Strategy

#### 5.3.1 Cross-Validation
- **Method:** 5-Fold Cross-Validation
- **Purpose:** Robust performance estimation
- **Scoring:** Accuracy (classification), R² (regression)

#### 5.3.2 Hyperparameter Tuning
- **Method:** GridSearchCV
- **CV:** 5-fold
- **Scoring:** accuracy / neg_mean_squared_error
- **n_jobs:** -1 (use all CPUs)

#### 5.3.3 Model Saving
- **Format:** Pickle (.pkl) for scikit-learn models
- **Format:** HDF5 (.h5) for Keras models
- **Location:** `models/` directory

---

## 6. KẾT QUẢ / RESULTS

### 6.1 Classification Results

#### 6.1.1 Model Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| Logistic Regression | [X.XX%] | [X.XX] | [X.XX] | [X.XX] | [X.XX] | [X]s |
| Random Forest | [X.XX%] | [X.XX] | [X.XX] | [X.XX] | [X.XX] | [X]s |
| XGBoost | [X.XX%] | [X.XX] | [X.XX] | [X.XX] | [X.XX] | [X]s |
| SVM | [X.XX%] | [X.XX] | [X.XX] | [X.XX] | [X.XX] | [X]s |
| Neural Network | [X.XX%] | [X.XX] | [X.XX] | [X.XX] | [X.XX] | [X]s |

#### 6.1.2 Best Classification Model

**Model:** [Model Name]  
**Accuracy:** [X.XX%]  
**Best Parameters:**
```python
{
    'param1': value1,
    'param2': value2,
    ...
}
```

#### 6.1.3 Confusion Matrix

*[Insert confusion matrix visualization]*

#### 6.1.4 ROC Curves

*[Insert ROC curves for all classes]*

#### 6.1.5 Feature Importance

*[Insert feature importance chart]*

Top 10 Most Important Features:
1. [Feature 1]: [Importance Score]
2. [Feature 2]: [Importance Score]
...

### 6.2 Regression Results

#### 6.2.1 Model Comparison Table

| Model | R² | RMSE | MAE | MAPE | Training Time |
|-------|-----|------|-----|------|---------------|
| Linear Regression | [X.XX] | [X.XX] | [X.XX] | [X.XX%] | [X]s |
| Random Forest | [X.XX] | [X.XX] | [X.XX] | [X.XX%] | [X]s |
| XGBoost | [X.XX] | [X.XX] | [X.XX] | [X.XX%] | [X]s |
| Gradient Boosting | [X.XX] | [X.XX] | [X.XX] | [X.XX%] | [X]s |
| Neural Network | [X.XX] | [X.XX] | [X.XX] | [X.XX%] | [X]s |

#### 6.2.2 Best Regression Model

**Model:** [Model Name]  
**R² Score:** [X.XX]  
**RMSE:** [X.XX]  
**Best Parameters:**
```python
{
    'param1': value1,
    'param2': value2,
    ...
}
```

#### 6.2.3 Predicted vs Actual

*[Insert scatter plot of predicted vs actual values]*

#### 6.2.4 Residual Analysis

*[Insert residual plots]*

### 6.3 Overall Performance Summary

#### 6.3.1 Classification Performance
- **Target Achieved:** [Yes/No] (>85% accuracy)
- **Best Model:** [Model Name]
- **Best Accuracy:** [X.XX%]
- **Improvement over Baseline:** [+X.XX%]

#### 6.3.2 Regression Performance
- **Target Achieved:** [Yes/No] (R² >0.80, RMSE <10%)
- **Best Model:** [Model Name]
- **Best R²:** [X.XX]
- **Best RMSE:** [X.XX%]
- **Improvement over Baseline:** [+X.XX]

### 6.4 Error Analysis

#### 6.4.1 Misclassification Analysis
- Most confused classes
- Common error patterns
- Suggestions for improvement

#### 6.4.2 Prediction Errors
- Largest prediction errors
- Error distribution analysis
- Insights for model refinement

---

## 7. KẾT LUẬN / CONCLUSION

### 7.1 Key Findings / Các Phát hiện Chính

1. **Best Performing Model:**
   - Classification: [Model Name] with [X.XX%] accuracy
   - Regression: [Model Name] with R² = [X.XX]

2. **Most Important Features:**
   - [Feature 1]
   - [Feature 2]
   - [Feature 3]

3. **Disaster Insights:**
   - [Most common disaster type and its characteristics]
   - [Most affected regions]
   - [Response efficiency patterns]

4. **Temporal Patterns:**
   - [Seasonal trends]
   - [Yearly trends]
   - [Peak periods]

5. **Economic Impact:**
   - Total losses: $[X] USD
   - Average loss per disaster: $[X] USD
   - Most costly disaster type: [X]

### 7.2 Achievements / Thành tựu Đạt được

✅ **Data Processing:**
- Successfully processed [X] disaster records
- Created [X] engineered features
- Balanced imbalanced dataset using SMOTE

✅ **Model Development:**
- Implemented 5+ classification models
- Implemented 5+ regression models
- Achieved target performance metrics

✅ **Visualization:**
- Created 50+ EDA visualizations
- Generated 40+ result visualizations
- Developed interactive geographic maps

✅ **Documentation:**
- Comprehensive code documentation
- Detailed report and analysis
- Reproducible research

### 7.3 Limitations / Hạn chế

1. **Data Limitations:**
   - Dataset limited to 2018-2024 period
   - Only 20 countries represented
   - Potential reporting biases

2. **Model Limitations:**
   - [Specific model limitations]
   - [Computational constraints]
   - [Feature engineering challenges]

3. **Scope Limitations:**
   - Focus on specific disaster types
   - Limited real-time prediction capability
   - Requires regular updates

### 7.4 Future Work / Công việc Tương lai

1. **Data Enhancement:**
   - Incorporate more recent data
   - Include additional countries
   - Add climate change indicators

2. **Model Improvements:**
   - Ensemble methods (stacking, boosting)
   - Deep learning architectures (LSTM, CNN)
   - Real-time prediction system

3. **Feature Engineering:**
   - Satellite imagery integration
   - Social media sentiment analysis
   - Weather pattern features

4. **Deployment:**
   - Web-based prediction interface
   - Mobile application
   - API for third-party integration

5. **Advanced Analysis:**
   - Causal inference
   - Time series forecasting
   - Multi-task learning

### 7.5 Recommendations / Khuyến nghị

**For Disaster Management:**
1. Focus resources on high-risk regions
2. Improve response time in identified areas
3. Enhance early warning systems

**For Policy Makers:**
1. Invest in disaster-prone areas
2. Develop region-specific strategies
3. Increase international cooperation

**For Future Research:**
1. Incorporate climate models
2. Study long-term trends
3. Develop integrated systems

---

## 8. TÀI LIỆU THAM KHẢO / REFERENCES

### 8.1 GitHub Repositories

1. Lagisetty, M. (2023). *Natural Disaster Prediction Using Machine Learning*. Retrieved from https://github.com/ManishaLagisetty/Natural-Disaster-Prediction-Using-Machine-Learning

2. Haz4rl. (2024). *Disaster Recovery Time Prediction using Machine Learning*. Retrieved from https://github.com/haz4rl/Disaster-Recovery-Time-Prediction-using-Machine-Learning

3. Rfuadur. (2023). *ML Flood Prediction & Disaster Response*. Retrieved from https://github.com/rfuadur/ML-Flood-Prediction-Disaster-Response

4. Karrolla, S. M. (2023). *Global Natural Disasters Analysis*. Retrieved from https://github.com/sri-maharagni-karrolla/Global-Natural-Disasters-Analysis

### 8.2 Research Papers

5. Neural Network and XGBoost Hybrid Model. (2023). *Hybrid Approach for Disaster Prediction*. Retrieved from https://www.diva-portal.org/smash/get/diva2:1961967/FULLTEXT01.pdf

6. MDPI. (2023). *Machine Learning Applications in Disaster Prediction: A Review*. Retrieved from https://www.mdpi.com/2504-4990/4/2/20

### 8.3 Datasets

7. Shehzadi, M. (2024). *Global Disaster Response 2018-2024*. Kaggle. Retrieved from https://www.kaggle.com/datasets/mubeenshehzadi/global-disaster-2018-2024

### 8.4 Libraries and Tools

8. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

9. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD*.

10. Abadi, M., et al. (2016). TensorFlow: A System for Large-Scale Machine Learning. *OSDI*, 16, 265-283.

11. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51-56.

12. Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90-95.

### 8.5 Online Resources

13. Kaggle Learn. (2024). *Machine Learning Courses*. Retrieved from https://www.kaggle.com/learn

14. Towards Data Science. (2024). *Disaster Prediction Articles*. Retrieved from https://towardsdatascience.com

15. UN Office for Disaster Risk Reduction. (2024). *Global Disaster Database*. Retrieved from https://www.undrr.org

---

## PHỤ LỤC / APPENDIX

### A. Code Repository
- GitHub: [Repository URL]
- Branch: main
- Tag: v1.0.0

### B. Data Files
- Raw Data: `data/raw/global_disaster_response_2018_2024.csv`
- Processed Data: `data/processed/train.csv`, `data/processed/test.csv`
- Encoders: `data/processed/encoders.pkl`

### C. Model Files
- Classification Models: `models/classification/`
- Regression Models: `models/regression/`
- Best Model: `models/best_model.pkl`

### D. Visualization Files
- All figures: `reports/figures/`
- EDA visualizations: `reports/figures/eda/`
- Model results: `reports/figures/results/`

### E. Notebooks
1. `notebooks/phase2_preprocessing.ipynb`
2. `notebooks/phase2_eda.ipynb`
3. `notebooks/phase3_model_building.ipynb`
4. `notebooks/phase4_training_evaluation.ipynb`
5. `notebooks/phase5_visualization.ipynb`

---

**Date of Completion:** [Date]  
**Version:** 1.0.0  
**Status:** [Draft/Final]

---

**© 2024 Disaster Prediction Team. All Rights Reserved.**
