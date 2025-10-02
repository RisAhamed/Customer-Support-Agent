
import os
import sys
import pandas as pd
import dill
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from exception import CustomException
from logger import logger

@dataclass
class ModelTrainerConfig:
    """Configuration class for the model trainer."""
    trained_model_file_path: str = os.path.join("saved_models", "model.pkl")

class ModelTrainer:
    """This class handles the model training and evaluation process."""
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        """
        Trains and evaluates a dictionary of models, returning a report.
        """
        try:
            report = {}
            for model_name, model in models.items():
                
                model.fit(X_train, y_train)
                
                
                y_test_pred = model.predict(X_test)
                
               
                test_model_f1_score = f1_score(y_test, y_test_pred)
                
                report[model_name] = test_model_f1_score
            
            return report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self, train_array, test_array):
        """
        Orchestrates the model training, evaluation, and logging process.
        """
        try:
            logger.info("Starting model training process")
            
        
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
          
            models = {
                "Logistic Regression": LogisticRegression(multi_class='ovr', solver='liblinear'),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
            }
            
            logger.info("Evaluating candidate models")
            model_report = self.evaluate_models(X_train, y_train, X_test, y_test, models)
            
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            logger.info(f"Best model found: {best_model_name} with F1 Score: {best_model_score}")

            
            with mlflow.start_run():
                logger.info("Starting MLflow experiment logging")
                
                
                mlflow.log_param("best_model_name", best_model_name)
                mlflow.log_metric("f1_score_weighted", best_model_score)


                mlflow.sklearn.log_model(best_model, "model")
                logger.info("Model artifact logged to MLflow")

            mlflow.end_run()

            logger.info("Saving the best model to a file")
         
            with open(self.model_trainer_config.trained_model_file_path, "wb") as file_obj:
                dill.dump(best_model, file_obj)

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)