import os
import pandas as pd
import sys
from logger import logger  

def load_data():
    """Load preprocessed data from CSV with a dynamic file path."""
    
    if 'ipykernel' in sys.modules:
        filepath = os.path.join(os.path.abspath(os.path.join('..')), 'Data', 'final_cleaned_data.csv')
    else:
        filepath = os.path.join(os.path.abspath(os.path.join('.')), 'Data', 'final_cleaned_data.csv')

    try:
        data = pd.read_csv(filepath)
        logger.info(f"Dataset loaded with shape: {data.shape}")
        
        X = data.drop(columns=['Do you *currently* have a mental health disorder?'])
        y = data['Do you *currently* have a mental health disorder?']
        
        return X, y
    
    except FileNotFoundError:
        logger.error(f"File not found at {filepath}. Check the path.")
        return None, None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None
