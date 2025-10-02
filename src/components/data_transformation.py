import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from logger import logger
from exception import CustomException
from utils import save_object

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('saved_models', 'preprocessor.pkl')

    def get_data_transformer_object(self):
        """
        This function is responsible for creating the data transformation pipeline
        which includes TF-IDF vectorization for the text data.
        """
        try:
            logger.info("Creating data transformer object")
            
           
            text_pipeline = Pipeline(steps=[
                ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000))
            ])

           
            preprocessor = ColumnTransformer(
                transformers=[
                    ('text_processing', text_pipeline, 'text')
                ],
                remainder='passthrough'  )
            
            logger.info("Data transformer object created successfully")
            return preprocessor

        except Exception as e:
            logger.error("Failed to create data transformer object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, df, target_column_name):
        """
        This method handles the data transformation, including splitting the data,
        applying the preprocessor, and saving the preprocessor object.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            target_column_name (str): The name of the target variable column.
        
        Returns:
            tuple: A tuple containing processed train/test feature arrays (X),
                   train/test target arrays (y), and the path to the saved preprocessor.
        """
        try:
            logger.info("Data transformation initiated")
            
          
            X = df.drop(columns=[target_column_name], axis=1)
            y = df[target_column_name]

            logger.info("Performing train-test split")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       
            assert X_train.shape[0] == y_train.shape[0], "Mismatch in training data samples after split!"
            assert X_test.shape[0] == y_test.shape[0], "Mismatch in testing data samples after split!"
            logger.info("Train-test split completed and shapes verified.")

            preprocessing_obj = self.get_data_transformer_object()

            logger.info("Fitting preprocessing object on training data")
            X_train_processed = preprocessing_obj.fit_transform(X_train)
            
            logger.info("Transforming test data")
            X_test_processed = preprocessing_obj.transform(X_test)

            save_object(
                file_path=self.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logger.info(f"Saving preprocessing object to: {self.preprocessor_obj_file_path}")
            
            logger.info("Data transformation completed successfully")
            
            return (
                X_train_processed,
                y_train,
                X_test_processed,
                y_test,
                self.preprocessor_obj_file_path
            )

        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)