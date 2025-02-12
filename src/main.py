from utils import load_data
from Train import evaluate_models
from logger import logger  

def main():
    try:
        # Load the preprocessed data
        X, y = load_data()

        if X is not None and y is not None:
            # Train and evaluate models and save results
            evaluate_models(X, y)
        else:
            logger.error("Error: Failed to load data, skipping model evaluation.")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()
