import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import warnings
from logger import logger 
warnings.simplefilter(action='ignore', category=FutureWarning)

filepath = "./Data/raw_OSMI_data.csv"  # Loading the original data
try:
    osmi_data = pd.read_csv(filepath)
except Exception as e:
    logger.error(f"Error loading the file: {e}")

# Extracting relevant features based on the paper
selected_features = [
    'Was your employer primarily a tech company/organization?',
    'What is your age?',
    'What is your gender?',
    'Do you have a family history of mental illness?',
    'Have you had a mental health disorder in the past?',
    'Does your employer provide mental health benefits as part of healthcare coverage?',
    'Have you ever discussed your mental health with your employer?',
]

target_variable = 'Do you *currently* have a mental health disorder?'
required_columns = selected_features + [target_variable]

try:
    data_subset = osmi_data[required_columns]
    data_subset.to_csv('./Data/filtered_data.csv', index=False)
    logger.info("Filtered data successfully saved as 'filtered_data.csv'")
except KeyError as e:
    logger.error(f"Missing columns in dataset - {e}")
except Exception as e:
    logger.error(f"An error occurred: {e}")

try:
    data = pd.read_csv('./Data/filtered_data.csv')
    logger.info(f"Dataset shape: {data.shape}")
except Exception as e:
    logger.error(f"Error loading the file: {e}")

# Data Cleaning
try:
    dont_know_variations = ["Don't Know", "I don't know", "i don't know", "don't know"]
    data.replace(dont_know_variations, np.nan, inplace=True)
    
    yes_variations = ["Yes", "yes", "YES", "yEs", "YeS", "Yess", "Yesss"]
    data.replace(yes_variations, "Yes", inplace=True)
    
    no_variations = ["No", "no", "NO", "nO"]
    data.replace(no_variations, "No", inplace=True)
    
    data = data.dropna(subset=['Do you *currently* have a mental health disorder?'])
    
    if "What is your age?" in data.columns:
        age_imputer = SimpleImputer(strategy='mean')
        data["What is your age?"] = age_imputer.fit_transform(data[["What is your age?"]])
    
    imputer = SimpleImputer(strategy='most_frequent')
    data.iloc[:, :] = imputer.fit_transform(data)
except Exception as e:
    logger.error(f"An error occurred: {e}")

try:
    data['Do you *currently* have a mental health disorder?'] = data['Do you *currently* have a mental health disorder?'].replace('Possibly', pd.NA)
    data.dropna(subset=['Do you *currently* have a mental health disorder?'], inplace=True)
except KeyError:
    logger.error("The column 'Do you *currently* have a mental health disorder?' is missing from the dataset.")

columns_with_possibly = selected_features.copy()
for col in columns_with_possibly:
    try:
        data[col] = data[col].replace('Possibly', pd.NA)
        data[col].fillna(data[col].mode()[0], inplace=True)
    except KeyError:
        logger.error(f"The column '{col}' is missing from the dataset.")

# Mapping Gender and Binary Columns 
try:
    if 'What is your gender?' in data.columns:
        data['What is your gender?'] = data['What is your gender?'].replace({'male': 'Male', 'm': 'Male', 'female': 'Female'})
        data['What is your gender?'] = data['What is your gender?'].map({'Male': 1, 'Female': 0})
        data['What is your gender?'].fillna(data['What is your gender?'].mode()[0], inplace=True)
        data['What is your gender?'] = data['What is your gender?'].astype(int)
    
    binary_columns =[col for col in selected_features + [target_variable] if col != "What is your age?"]

    binary_mapping = {'Yes': 1, 'No': 0, True: 1, False: 0}
    
    for col in binary_columns:
        if col in data.columns:
            data[col] = data[col].map(binary_mapping)
            data[col].fillna(data[col].mode()[0], inplace=True)
            data[col] = data[col].astype(int)
except Exception as e:
    logger.error(f"An error occurred: {e}")

# MinMax Scaling for Age
try:
    data['What is your age?'] = pd.to_numeric(data['What is your age?'], errors='coerce')
    scaler = MinMaxScaler()
    data[['What is your age?']] = scaler.fit_transform(data[['What is your age?']])
except Exception as e:
    logger.error(f"An error occurred during age transformation: {e}")

data.to_csv('./Data/final_cleaned_data.csv', index=False)
