import sys
import os
import numpy as np
import pandas as pd

# =================================================================
# FIX for ModuleNotFoundError
# =================================================================
# This adds the project's root directory to the Python path.
# Now, any script can import modules like 'logger' or from the 'src' folder.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# =================================================================

from logger import logger
from exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_training_pipeline():
    """
    This function orchestrates the model training pipeline, including data loading,
    transformation, and model training.
    """
    try:
        logger.info("="*50)
        logger.info("STARTING THE TRAINING PIPELINE")
        logger.info("="*50)
        logger.info("Loading data")
        data = {
            'text': [
                'I need help with my bill please', 'My internet is not working at all', 
                'How do I reset my password?', 'Your service is absolutely terrible.',
                'Can you check my account balance?', 'The connection keeps dropping.',
                'Forgot my username, can you help?', 'I want to file a formal complaint.'
            ],
            'category': [
                'billing', 'technical', 'account', 'feedback',
                'billing', 'technical', 'account', 'feedback'
            ]
        }
        df = pd.DataFrame(data)
        target_column = 'category'
        logger.info("Data loaded successfully")

        # --- Data Transformation ---
        logger.info("Starting data transformation process")
        data_transformation = DataTransformation()
        
        # This now correctly passes the target column name
        X_train, y_train, X_test, y_test, _ = data_transformation.initiate_data_transformation(
            df, 
            target_column_name=target_column
        )

        # --- Prepare Arrays for Model Training ---
        logger.info("Preparing training and test arrays")
        
        # Convert sparse matrix from TF-IDF to a dense array
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
        
        # Reshape the 1D target arrays (y_train, y_test) into 2D column vectors
        y_train_reshaped = np.array(y_train).reshape(-1, 1)
        y_test_reshaped = np.array(y_test).reshape(-1, 1)
        
        logger.info(f"Shape of X_train_dense: {X_train_dense.shape}")
        logger.info(f"Shape of y_train_reshaped: {y_train_reshaped.shape}")
        
        # --- PREVENTATIVE MEASURE ---
        # Assert shapes before concatenating to catch any final mismatches.
        assert X_train_dense.shape[0] == y_train_reshaped.shape[0], "CRITICAL: Row count mismatch before combining train arrays!"
        assert X_test_dense.shape[0] == y_test_reshaped.shape[0], "CRITICAL: Row count mismatch before combining test arrays!"

        # Combine features and target into a single array for the model trainer
        train_arr = np.hstack([X_train_dense, y_train_reshaped])
        test_arr = np.hstack([X_test_dense, y_test_reshaped])
        
        logger.info(f"Successfully created combined train array with shape: {train_arr.shape}")

        # --- Model Training ---
        logger.info("Starting model training")
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_array=train_arr, test_array=test_arr)
        
        logger.info("="*50)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*50)

    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_training_pipeline()