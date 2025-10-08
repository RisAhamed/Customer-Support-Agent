import os
import sys
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from root directory
from exception import CustomException
from logger import logger

# Import from src directory
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_training_pipeline():
    """Run the full training pipeline from data transformation to model training."""
    try:
       
        data_path = os.path.join('data', 'processed_tickets.csv')
        
        logger.info("Starting data transformation process")
        data_transformation = DataTransformation()        
        X_train_processed, X_test_processed, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation(data_path)
        
    
        logger.info("Preparing training and test arrays")
    
        if hasattr(X_train_processed, 'toarray'):
            X_train_dense = X_train_processed.toarray()
        else:
            X_train_dense = np.array(X_train_processed)
            
        if hasattr(X_test_processed, 'toarray'):
            X_test_dense = X_test_processed.toarray()
        else:
            X_test_dense = np.array(X_test_processed)
        
        
        logger.info(f"X_train_dense shape: {X_train_dense.shape}")
        logger.info(f"y_train shape: {np.array(y_train).shape}")
        train_arr = np.column_stack((X_train_dense, y_train))
        test_arr = np.column_stack((X_test_dense, y_test))
    
        logger.info(f"Final training array shape: {train_arr.shape}")
        logger.info(f"Final test array shape: {test_arr.shape}")
    
        logger.info("Starting model training process")
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)
        
        logger.info("Training pipeline completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_training_pipeline()