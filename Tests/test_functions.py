import unittest
import sys
import os
import pandas as pd

# Add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from logger import logger
from models import get_model
from Train import perform_cross_validation
from sklearn.datasets import make_classification

class TestFunctions(unittest.TestCase):
    def setUp(self):
        """Setup before each test."""
        logger.info("Setting up the test environment.")
        self.X, self.y = make_classification(n_samples=251, n_features=5, n_classes=2, random_state=42)
    
    def test_get_model(self):
        """Test if the model retrieval function works correctly."""
        try:
            logger.info("Testing model retrieval function.")
            knn_model = get_model('KNN')
            self.assertEqual(knn_model.__class__.__name__, 'KNeighborsClassifier')
            logger.info("KNN model retrieved successfully.")

            svm_model = get_model('SVM')
            self.assertEqual(svm_model.__class__.__name__, 'SVC')
            logger.info("SVM model retrieved successfully.")
            
            # Check for invalid model name
            with self.assertRaises(ValueError):
                get_model('Invalid Model')
            logger.warning("An invalid model name was provided, and an exception was raised.")
        except Exception as e:
            logger.error(f"Error in test_get_model: {e}")
            raise

    def test_perform_cross_validation(self):
        """Test if the cross-validation function returns the correct structure and values."""
        try:
            logger.info("Testing cross-validation function.")
            knn_model = get_model('KNN')
            results = perform_cross_validation(knn_model, self.X, self.y)
            
            # Check if the returned results contain the expected keys
            self.assertIn('accuracy', results)
            self.assertIn('precision', results)
            self.assertIn('recall', results)
            self.assertIn('f1', results)
            logger.info("Cross-validation results contain expected keys.")

            # Check if results are of the right type (float)
            self.assertIsInstance(results['accuracy'], float)
            self.assertIsInstance(results['precision'], float)
            self.assertIsInstance(results['recall'], float)
            self.assertIsInstance(results['f1'], float)
            logger.info("Cross-validation results are of the correct type.")
        except Exception as e:
            logger.error(f"Error in test_perform_cross_validation: {e}")
            raise

    def test_data_loading_and_preprocessing(self):
        """Test if the data is loaded and preprocessed correctly."""
        try:
            logger.info("Testing data loading and preprocessing.")
            data = pd.read_csv('./Data/final_cleaned_data.csv')
            self.assertEqual(data.shape[0], 251)  #
            self.assertTrue('Do you *currently* have a mental health disorder?' in data.columns)
            logger.info("Data loaded and contains the required column.")

            # Check for missing values
            self.assertTrue(data.isnull().sum().sum() == 0)  # After preprocessing, no missing values
            logger.info("No missing values found in the data.")
            
            # Check if age is scaled
            self.assertTrue(data['What is your age?'].min() >= 0 and data['What is your age?'].max() <= 1)
            logger.info("Age data is scaled correctly.")
        except Exception as e:
            logger.error(f"Error in test_data_loading_and_preprocessing: {e}")
            raise

if __name__ == '__main__':
    try:
        logger.info('Starting unit tests.')
        unittest.main()
    except Exception as e:
        logger.error(f"An error occurred while running the tests: {e}")
