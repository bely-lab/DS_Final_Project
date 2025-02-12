from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import numpy as np
from logger import logger  

# Model Definitions
def create_knn_model():
    try:
        return KNeighborsClassifier(leaf_size= 20, metric='euclidean', n_neighbors= 5, weights= 'uniform')
    except Exception as e:
        logger.error(f"Error creating KNN model: {e}")
        raise

def create_svm_model():
    try:
        return SVC(C=10, degree=3, gamma='scale', kernel='rbf', random_state=42)
    except Exception as e:
        logger.error(f"Error creating SVM model: {e}")
        raise

def create_logreg_model():
    try:
        return LogisticRegression(max_iter=300, C=1, class_weight={0: 1, 1: 1.3}, random_state=42, penalty='l2')
    except Exception as e:
        logger.error(f"Error creating Logistic Regression model: {e}")
        raise

def create_decision_tree_model():
    try:
        return DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=10, min_samples_split=2, max_leaf_nodes=20, class_weight={0: 1, 1: 1})
    except Exception as e:
        logger.error(f"Error creating Decision Tree model: {e}")
        raise

def create_random_forest_model():
    try:
        return RandomForestClassifier(random_state=42, class_weight='balanced', max_features='sqrt', min_samples_leaf=4, min_samples_split=2, n_estimators=50)
    except Exception as e:
        logger.error(f"Error creating Random Forest model: {e}")
        raise

def create_naive_bayes_model():
    try:
        return GaussianNB(var_smoothing=np.float64(1.0))
    except Exception as e:
        logger.error(f"Error creating Naive Bayes model: {e}")
        raise

def create_xgboost_model():
    try:
        return XGBClassifier(colsample_bytree= 0.8, gamma= 0, learning_rate= 0.01, max_depth=3, n_estimators= 50, subsample= 0.8, random_state=42)
    except Exception as e:
        logger.error(f"Error creating XGBoost model: {e}")
        raise

# Dictionary to manage models dynamically
models = {
    'KNN': create_knn_model,
    'SVM': create_svm_model,
    'Logistic Regression': create_logreg_model,
    'Decision Tree': create_decision_tree_model,
    'Random Forest': create_random_forest_model,
    'Naive Bayes': create_naive_bayes_model,
    'XGBoost': create_xgboost_model
}

def get_model(model_name):
    try:
        model_func = models.get(model_name)
        if model_func:
            model = model_func()
            return model
        else:
            raise ValueError(f"Model {model_name} not found.")
    except Exception as e:
        logger.error(f"Error in get_model function: {e}")
        raise
