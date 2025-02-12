import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from models import get_model
from logger import logger  

def perform_cross_validation(model, X, y, cv=5):
    """
    Perform K-Fold Cross Validation on the given model and return the results.
    """
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    try:
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        return {
            'accuracy': np.mean(cv_results['test_accuracy']),
            'precision': np.mean(cv_results['test_precision']),
            'recall': np.mean(cv_results['test_recall']),
            'f1': np.mean(cv_results['test_f1'])
        }
    except Exception as e:
        logger.error(f"Error during cross-validation: {e}")
        return {'accuracy': None, 'precision': None, 'recall': None, 'f1': None}

def evaluate_models(X, y, model_names=None, output_path='./Results/model_evaluation_results.csv'):
    """
    Evaluates multiple models using cross-validation and saves the results to a CSV file.
    """
    if model_names is None:
        model_names = ['KNN', 'SVM', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'XGBoost']
    
    all_results = []
    
    for model_name in model_names:
        try:
            model = get_model(model_name)  
            if model is None:
                logger.warning(f"Model '{model_name}' could not be initialized.")
                continue
            
            results = perform_cross_validation(model, X, y)
            all_results.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1': results['f1']
            })
        
        except Exception as e:
            logger.error(f"Error evaluating model '{model_name}': {e}")
    
    # Convert results to DataFrame and save
    try:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results have been saved to '{output_path}'.")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")
