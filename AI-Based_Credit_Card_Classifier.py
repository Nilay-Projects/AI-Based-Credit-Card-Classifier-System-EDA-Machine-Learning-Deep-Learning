"""
Credit Risk Analysis & Prediction
Using: UCI Credit Card Default Dataset (Kaggle)
- Full EDA (visualizations, stats)
- Data cleaning
- Correlation analysis
- ML model: Random Forest
- DL model: Neural Network (Keras)
- Report metrics: Accuracy, Classification Report, Confusion Matrix
- Save trained models

Author: Nilay Rana

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import joblib

# ------------------------
# 1. Load the data
# ------------------------
def load_data(path="data/UCI_Credit_Card.csv"):
    # Read the CSV; sometimes Kaggle version has different header row
    df = pd.read_csv(path)
    print("Data loaded from:", path)
    print("Shape:", df.shape)
    return df

# ------------------------
# 2. Initial Data Inspection
# ------------------------
def inspect_data(df):
    print("\n--- HEAD OF DATA ---")
    print(df.head())
    print("\n--- TAIL OF DATA ---")
    print(df.tail())
    print("\n--- INFO ---")
    print(df.info())
    print("\n--- DESCRIPTIVE STATISTICS ---")
    print(df.describe(include='all'))
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

# ------------------------
# 3. Data Cleaning
# ------------------------
def clean_data(df):
    # Copy to avoid mutating original
    df_clean = df.copy()

    # Drop ID column if present
    if 'ID' in df_clean.columns:
        df_clean.drop('ID', axis=1, inplace=True)
        print("Dropped column: ID")

    # Check for duplicates
    dup_count = df_clean.duplicated().sum()
    print(f"Number of duplicated rows: {dup_count}")
    if dup_count > 0:
        df_clean = df_clean.drop_duplicates()
        print("Dropped duplicates.")

    # No missing values in this dataset per documentation :contentReference[oaicite:0]{index=0}

    # Check for outliers via basic method (IQR) for numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    print("Numeric cols for outlier detection:", numeric_cols)

    for col in numeric_cols:
        # Compute IQR
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        print(f"Outliers in {col}: {outliers}")
        # Optionally: cap or remove outliers — here we will cap
        df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])

    return df_clean

# ------------------------
# 4. Exploratory Data Analysis (EDA)
# ------------------------
def eda_plots(df):
    sns.set(style="whitegrid")

    # Distribution of target
    plt.figure(figsize=(6,4))
    sns.countplot(x='default.payment.next.month', data=df)
    plt.title('Count of Default vs Non-Default')
    plt.xlabel('Default Payment Next Month (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.show()

    # Correlation heatmap (numeric)
    plt.figure(figsize=(16,12))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.show()

    # Histograms for numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_plots = len(numeric_cols)
    plt.figure(figsize=(16, 4 * (num_plots // 4 + 1)))
    for i, col in enumerate(numeric_cols):
        plt.subplot((num_plots // 4) + 1, 4, i+1)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()

    # Boxplots for numeric features
    plt.figure(figsize=(16, 4 * (num_plots // 4 + 1)))
    for i, col in enumerate(numeric_cols):
        plt.subplot((num_plots // 4) + 1, 4, i+1)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()

    # Pairplot for a subset to avoid overload (choose smaller subset)
    subset = df[numeric_cols].sample(n=1000, random_state=42)
    sns.pairplot(subset, corner=True)
    plt.suptitle("Pairplot (sampled subset)", y=1.02)
    plt.show()

    # Categorical / discrete analysis: Education, Marriage, Sex
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    for col in cat_cols:
        if col in df.columns:
            plt.figure(figsize=(6,4))
            sns.countplot(x=col, hue='default.payment.next.month', data=df)
            plt.title(f"{col} vs Default Payment Next Month")
            plt.show()

    # Repayment status (PAY_0 .. PAY_6) vs default
    pay_cols = [c for c in df.columns if c.startswith('PAY_')]
    for col in pay_cols:
        plt.figure(figsize=(6,4))
        sns.countplot(x=col, hue='default.payment.next.month', data=df)
        plt.title(f"{col} vs Default")
        plt.show()

    # Bill amount features distributions
    bill_cols = [c for c in df.columns if c.startswith('BILL_AMT')]
    plt.figure(figsize=(16, 4 * (len(bill_cols)//4 + 1)))
    for i, col in enumerate(bill_cols):
        plt.subplot((len(bill_cols)//4) + 1, 4, i+1)
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(col)
    plt.tight_layout()
    plt.show()

    # Payment amount features distributions
    payamt_cols = [c for c in df.columns if c.startswith('PAY_AMT')]
    plt.figure(figsize=(16, 4 * (len(payamt_cols)//4 + 1)))
    for i, col in enumerate(payamt_cols):
        plt.subplot((len(payamt_cols)//4) + 1, 4, i+1)
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(col)
    plt.tight_layout()
    plt.show()

# ------------------------
# 5. Preprocess for Modeling
# ------------------------
def preprocess_for_model(df):
    # Separate features and target
    X = df.drop('default.payment.next.month', axis=1)
    y = df['default.payment.next.month']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, random_state=42,
                                                        stratify=y)

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Finished scaling features.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ------------------------
# 6. Train Machine Learning Model (Random Forest)
# ------------------------
def train_ml_model(X_train, X_test, y_train, y_test):
    print("\n=== Training ML Model (Random Forest) ===")
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print("Random Forest Accuracy:", acc)
    print("Classification Report (ML):")
    print(classification_report(y_test, preds))
    print("Confusion Matrix (ML):")
    print(confusion_matrix(y_test, preds))

    return rf, preds

# ------------------------
# 7. Train Deep Learning Model (Neural Network)
# ------------------------
def train_dl_model(X_train, X_test, y_train, y_test):
    print("\n=== Training DL Model (Neural Network) ===")
    input_dim = X_train.shape[1]

    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train
    history = model.fit(X_train, y_train, 
                        epochs=20, batch_size=32, 
                        validation_split=0.2, verbose=2)

    # Plot training history
    plt.figure(figsize=(12,4))

    # Accuracy plot
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    print("DL Model Accuracy:", acc)
    print("Classification Report (DL):")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix (DL):")
    print(confusion_matrix(y_test, y_pred))

    return model, y_pred, history

# ------------------------
# 8. Feature Importance (from ML model)
# ------------------------
def feature_importance(rf, X_train, df):
    importances = rf.feature_importances_
    feature_names = df.drop('default.payment.next.month', axis=1).columns
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x=fi.values, y=fi.index)
    plt.title("Feature Importance (Random Forest)")
    plt.show()

    print("\nTop 10 Important Features:")
    print(fi.head(10))

# ------------------------
# 9. Save Models & Scaler
# ------------------------
def save_artifacts(rf, scaler, dl_model):
    joblib.dump(rf, "rf_credit_default_model.pkl")
    print("Saved Random Forest model as rf_credit_default_model.pkl")

    joblib.dump(scaler, "scaler.pkl")
    print("Saved scaler as scaler.pkl")

    dl_model.save("dl_credit_default_model.h5")
    print("Saved DL model as dl_credit_default_model.h5")

# ------------------------
# MAIN FUNCTION
# ------------------------
def main():
    # Load
    df = load_data()

    # Inspect
    inspect_data(df)

    # Clean
    df_clean = clean_data(df)

    # EDA
    eda_plots(df_clean)

    # Preprocess for modeling
    X_train, X_test, y_train, y_test, scaler = preprocess_for_model(df_clean)

    # ML model
    rf_model, ml_preds = train_ml_model(X_train, X_test, y_train, y_test)

    # Feature importance
    feature_importance(rf_model, X_train, df_clean)

    # DL model
    dl_model_obj, dl_preds, history = train_dl_model(X_train, X_test, y_train, y_test)

    # Save artifacts
    save_artifacts(rf_model, scaler, dl_model_obj)

    print("\nAll done!")

if __name__ == "__main__":
    main()
