# Credit Risk Prediction System: EDA, Machine Learning & Deep Learning

**End-to-end Credit Risk Prediction System using the UCI Credit Card Default dataset. This project demonstrates data cleaning, exploratory data analysis (EDA), feature engineering, machine learning, and deep learning to predict customer credit default — simulating real-world financial risk modeling workflows used in banks like JPMorgan Chase.**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Objectives](#project-objectives)
4. [Technologies & Tools](#technologies--tools)
5. [Project Workflow](#project-workflow)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Data Preprocessing & Cleaning](#data-preprocessing--cleaning)
8. [Machine Learning Model](#machine-learning-model)
9. [Deep Learning Model](#deep-learning-model)
10. [Feature Importance](#feature-importance)
11. [Results & Evaluation](#results--evaluation)
12. [Project Artifacts](#project-artifacts)
13. [Future Improvements](#future-improvements)
14. [References](#references)

---

## Project Overview
Credit risk prediction is a core function in banking to assess the likelihood that a customer may default on their credit obligations. This project uses the **UCI Credit Card Default dataset** from Kaggle to build a robust **data-driven credit risk model** using both **machine learning** and **deep learning** techniques.

Key Highlights:
- End-to-end workflow: EDA → Cleaning → Modeling → Evaluation
- Comprehensive visualizations for insights
- Comparison of ML and DL approaches
- Production-ready code structure

---

## Dataset
- **Name:** UCI Credit Card Default Dataset  
- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)  
- **Size:** 30,000+ samples, 24 features + target  
- **Target Variable:** `default.payment.next.month` (0 = No Default, 1 = Default)

---

## Project Objectives
1. Understand the dataset and identify patterns in credit default.
2. Perform thorough **exploratory data analysis (EDA)** and visualization.
3. Clean and preprocess data to prepare it for modeling.
4. Build a **Random Forest Machine Learning model**.
5. Build a **Deep Neural Network** using TensorFlow/Keras.
6. Compare models using accuracy, classification report, and confusion matrix.
7. Analyze **feature importance** to understand the key factors influencing credit default.

---

## Technologies & Tools
- **Python** – primary programming language  
- **Pandas & NumPy** – data manipulation  
- **Matplotlib & Seaborn** – visualization  
- **Scikit-learn** – ML modeling & metrics  
- **TensorFlow & Keras** – deep learning  
- **Joblib** – model saving  
- **Jupyter Notebook / VSCode** – development environment  

---

## Project Workflow

### 1. Data Loading
- Load CSV dataset from Kaggle
- Inspect top and bottom rows using `head()` and `tail()`
- Check dataset shape, columns, and basic statistics

### 2. Exploratory Data Analysis (EDA)
- Display dataset info: `info()`, `describe()`, missing values
- Visualize:
  - Target distribution (`default.payment.next.month`)
  - Correlation heatmaps
  - Feature distributions (histograms, KDE)
  - Boxplots for outlier detection
  - Categorical feature analysis (`SEX`, `EDUCATION`, `MARRIAGE`)
  - Repayment status and bill/payment amounts

### 3. Data Cleaning
- Drop unnecessary columns (e.g., `ID`)
- Handle duplicate rows
- Detect and cap outliers using IQR method
- Scale numeric features using StandardScaler

### 4. Feature Engineering
- No additional encoding needed as all features are numeric
- Feature selection guided by correlation heatmaps and ML model importance

---

## Machine Learning Model
- **Model:** Random Forest Classifier
- **Parameters:** `n_estimators=300`, `max_depth=15`
- **Evaluation Metrics:**
  - Accuracy Score
  - Classification Report (Precision, Recall, F1-Score)
  - Confusion Matrix

---

## Deep Learning Model
- **Architecture:** Fully connected neural network with:
  - Input layer matching feature size
  - Two hidden layers (128 & 64 neurons) with ReLU and Dropout
  - Output layer with sigmoid activation
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy  
- **Training:** 20 epochs, batch size 32
- **Evaluation:** Accuracy, classification report, confusion matrix  
- **Training curves:** Plots for training & validation accuracy and loss

---

## Feature Importance
- Obtained from Random Forest
- Top predictors for credit default include:
  - Past repayment status (`PAY_0`, `PAY_2`, …)
  - Bill amount features (`BILL_AMT1`, `BILL_AMT2`, …)
  - Payment amounts (`PAY_AMT1`, …)
- Visualized using a horizontal bar chart for clarity

---

## Results & Evaluation

| Model                 | Accuracy | Notes                                      |
|-----------------------|----------|-------------------------------------------|
| Random Forest (ML)    | ~0.82    | Strong baseline, interpretable features    |
| Deep Neural Network   | ~0.83    | Slightly higher accuracy, flexible model  |

- Confusion matrices and classification reports printed for both models
- Models effectively distinguish high-risk customers

---

## Project Artifacts
- `rf_credit_default_model.pkl` → Saved Random Forest model  
- `dl_credit_default_model.h5` → Saved Deep Learning model  
- `scaler.pkl` → StandardScaler used for preprocessing  
- All visualizations generated during EDA for reporting

---

## Future Improvements
- Apply **SMOTE or class weighting** to handle class imbalance  
- Hyperparameter tuning using GridSearchCV / RandomizedSearchCV  
- Explore **XGBoost, LightGBM, or CatBoost** for improved ML performance  
- Build a **FastAPI / Flask API** for real-time predictions  
- Integrate **SHAP or LIME** for model explainability  
- Deploy the model as a **containerized application** (Docker)  

---

## References
- [Kaggle Dataset: UCI Credit Card Default](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)

---

## Author
**Your Name** – Aspiring Data Scientist / Python Developer  
- GitHub: [YourGitHub](https://github.com/YourUsername)  
- Email: your.email@example.com  

> This project demonstrates a professional, end-to-end workflow for credit risk prediction, combining **data analysis, ML, and DL models**, making it ideal for financial analytics and risk management roles at institutions like JPMorgan Chase.
