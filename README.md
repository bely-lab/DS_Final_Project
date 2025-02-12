# Predicting Mental Health Disorders Using Machine Learning

## Primary Source/Article
**Title:** Predicting Mental Health Disorders Using Machine Learning for Employees in Technical and Non-Technical Companies  
**Authors:** Rahul Katarya, Saurav Maan  
**Year:** 2020  
**Link to Article:** [IEEE Xplore](https://ieeexplore.ieee.org/document/9368923)

## Project Data: 
The dataset used for this project is the Mental Illness (OSMI) 2019 Survey dataset, which includes data on employee mental health status, their work environments, and other relevant features. you can find here: https://www.kaggle.com/osmihelp/datasets
## Project Documentation:
### Project Overview
This project aims to replicate the findings of the 2020 IEEE paper "Predicting Mental Health Disorders Using Machine Learning for Employees in Technical and Non-Technical Companies" by Rahul Katarya and Saurav Maan. The original paper explored the use of machine learning algorithms to predict mental health disorders among employees in both technical and non-technical companies. It used the (OSMI) Survey dataset to train and evaluate six machine learning models.
The primary objectives of this replication project are to:
- Validate the results presented in the paper.
- Gain insights into the practical implementation of machine learning for predicting mental health disorders in the workplace.
- Examine the performance of different machine learning models on the OSMI dataset.
Additionally, a feature importance analysis is conducted to identify which factors are most predictive of mental health disorders in the workplace.
### Folder/Module Structure
project_root/ │ ├── data/ │ ├── raw_data.csv # Raw data from the OSMI Survey │ ├── unprocessed_data.csv # Data with selected features for training │ └── processed_data.csv # Cleaned and processed data for model training │ ├── notebooks/ │ ├── data_distribution_analysis.ipynb # Data distribution visualization and analysis │ ├── model_comparison.ipynb # Model comparison after training │ ├── feature_importance.ipynb # Feature importance analysis │ └── confusion_matrix.ipynb # Confusion matrix for model evaluation │ ├── src/ │ ├── train.py # Script for training machine learning models │ ├── models.py # Contains six different models for prediction (KNN, SVM, etc.) │ ├── main.py # Main script to initiate model training and analysis │ ├── utils.py # Helper functions for loading data and pre-processing │ ├── preprocess.py # Preprocessing functions for cleaning data and handling missing values │ └── logger.py # Logger functions for recording the progress of model training │ ├── test/ │ └── test_functions.py # Unit tests for major functions like data loading, preprocessing, and model evaluation │ └── README.md # Project documentation
### Project Workflow/Pipeline: 
The project follows these steps:
1. Data Preprocessing:
   - Clean and transform raw data.
   - Handle missing values, categorical variables, and normalize age data.
2. Model Evaluation:
   - Evaluate multiple machine learning models using K-fold cross-validation.
   - Measure performance using precision, recall, accuracy, and F1-score.
3. Feature Importance Analysis:
   - Identify the most important features contributing to mental health disorders.
   ![feature_importance image](image.png)
4. Model Selection:
   - Train and evaluate seven models(The six models in the paper and one additional) including KNN, SVM, Random Forest,Decision tree, Naivebayes and XGBoost.
   
For detailed  documentation, refer to the full project documentation in .Results/Documnetation.pdf

## To Run the Project:

Follow the steps below to set up your environment and run the project.
### 1. Install Virtualenv 
Install `virtualenv` globally using pip:
```bash
pip install virtualenv
# Create a virtual environment (specific to this project)
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
# Once activated, (venv) should appear as a prefix in your terminal
# Upgrade pip to the latest version
python -m pip install --upgrade pip

# Install project dependencies 
pip install -e .

# Install development dependencies 
pip install -e .[dev]