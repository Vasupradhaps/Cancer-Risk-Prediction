# Cancer Risk Level Prediction using Machine Learning

An end-to-end multiclass machine learning project to predict cancer risk levels (High, Medium, Low) using patient-related risk factors. This project explores data preprocessing, classification algorithms, class imbalance handling, hyperparameter optimisation and model deployment using Streamlit.

## Project Overview

Cancer risk is influenced by multiple factors, including lifestyle choices, family history, environmental exposure and genetic predisposition.

The objective of this project is to build a machine learning classification model that predicts a patient's risk category based on a set of input risk factors.

The project compares multiple classification approaches and investigates techniques to improve performance on minority classes, particularly the High-risk category.

**Note:** This is an educational machine learning project. Its predictions are not clinically validated and must not be used for medical diagnosis or treatment decisions.

## Objectives

- Perform exploratory data analysis and prepare the dataset for modelling.
- Build and compare multiple classification models.
- Address class imbalance using SMOTE and class weighting.
- Optimise model hyperparameters using Optuna.
- Evaluate model performance using class-specific precision, recall, F1-score and confusion matrices.

## Dataset

The dataset used in this project contains **2,000 records** with patient-related risk factors.

The target variable is `Risk_Level`, which contains three categories:

- High
- Medium
- Low

### Features

The modelling dataset includes 17 features after excluding identifiers, cancer type and the overall risk score.

| Feature category | Examples |
|---|---|
| Demographic | Age, Gender |
| Lifestyle | Smoking, Alcohol Use, Diet, Physical Activity |
| Health-related | BMI, Obesity, Family History |
| Environmental | Air Pollution, Occupational Hazards |
| Genetic and medical | BRCA Mutation, H. Pylori Infection, Calcium Intake |

**Target variable:** `Risk_Level`

`Patient_ID` and `Cancer_Type` were excluded from the predictive features. `Overall_Risk_Score` was also removed to reduce the risk of target leakage.

## Technology Stack

- **Programming:** Python
- **Data manipulation:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Classification:** Random Forest, Logistic Regression, XGBoost
- **Class imbalance handling:** Imbalanced-learn (SMOTE)
- **Hyperparameter optimisation:** Optuna

## Methodology

### 1. Data Preparation

- Loaded and explored the dataset using Pandas.
- Separated predictive features and the target variable.
- Encoded the target categories using LabelEncoder.
- Excluded non-predictive identifiers and potentially leakage-prone variables.
- Split the dataset into training and testing sets using an 80:20 stratified split.

### 2. Baseline Model Development

The following classification algorithms were explored:

- Random Forest Classifier
- Logistic Regression
- XGBoost Classifier

Initial model performance was evaluated using classification reports and confusion matrices.

### 3. Handling Class Imbalance

The dataset has an imbalanced target distribution:

| Risk level | Records | Distribution |
|---|---:|---:|
| Medium | 1,574 | 78.7% |
| Low | 324 | 16.2% |
| High | 102 | 5.1% |
| **Total** | **2,000** | **100%** |

Two approaches were investigated:

- **SMOTE:** Synthetic Minority Over-sampling Technique to increase representation of minority classes in the training data.
- **Class weighting:** Assigning higher weights to underrepresented classes during model training.

The experiments highlighted the trade-off between overall accuracy and the ability to identify minority-class observations.

### 4. Hyperparameter Optimisation using Optuna

Optuna was used to explore model hyperparameters and improve classification performance.

The optimisation experiments included:

- Random Forest optimisation using macro F1-score.
- XGBoost optimisation targeting recall for the High-risk class.
- Class-weighted XGBoost optimisation using macro F1-score.

Stratified cross-validation was used during model optimisation to preserve class distributions across folds.

### 5. Final Model Evaluation

The final model experiment recorded in the notebook was an **Optuna-tuned, class-weighted XGBoost classifier**.

Performance on the held-out test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 85% |
| Macro F1-score | 0.65 |
| Weighted F1-score | 0.85 |

#### Class-wise Performance

| Risk level | Precision | Recall | F1-score |
|---|---:|---:|---:|
| High | 0.50 | 0.45 | 0.47 |
| Low | 0.76 | 0.78 | 0.77 |
| Medium | 0.92 | 0.92 | 0.92 |

#### Confusion Matrix

Rows represent actual classes and columns represent predicted classes.

| Actual \ Predicted | High | Low | Medium |
|---|---:|---:|---:|
| High | 9 | 0 | 11 |
| Low | 0 | 51 | 14 |
| Medium | 9 | 16 | 290 |

**Key observation:** Although the model achieved 88% accuracy, High-risk recall remained at 45%. This highlights the difficulty of identifying minority classes and the importance of evaluating class-specific performance rather than relying on accuracy alone.


## Key Learnings

- Understanding the impact of class imbalance on multiclass classification.
- Comparing oversampling techniques with class-weighted learning.
- Using macro F1-score to evaluate performance across imbalanced classes.
- Applying Optuna for automated hyperparameter optimisation.
- Understanding the importance of stratified cross-validation.
- Translating a trained machine learning model into an interactive application using Streamlit.

## Future Improvements

- Explore additional classification algorithms and ensemble techniques.
- Improve minority-class recall while managing false positives.
- Investigate probability calibration and class-specific decision thresholds.
- Improve input validation and preprocessing consistency between training and deployment.
- Evaluate the model on independent datasets.
- Enhance the Streamlit interface with model interpretation and feature importance visualisations.

## Disclaimer

This project is developed for educational and learning purposes. The dataset and model have not been established as clinically validated. Predictions should not be interpreted as medical advice, cancer diagnoses or treatment recommendations.

---

**Developed as a hands-on exploration of Machine Learning models, imbalanced data classification and optimising performance..**
