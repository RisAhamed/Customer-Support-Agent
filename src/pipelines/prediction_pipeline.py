import os
import sys
import numpy as np
import dill
import pandas as pd

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from root directory
from logger import logger
from exception import CustomException

# Import from src directory
from src.components.llm_service import LLMService


class PredictionPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join('saved_models', 'preprocessor.pkl')
        self.model_path = os.path.join('saved_models', 'model.pkl')
        self.llm_Service = LLMService()
    
    def predict(self, text:str):
        try:
            logger.info("starting prediction pipeline")

            with open(self.preprocessor_path, "rb") as f:
                preprocessor = dill.load(f)
            with open(self.model_path, "rb") as f:
                model = dill.load(f)

            text_df  =pd.DataFrame([text], columns = ['text'])
            text_transformed = preprocessor.transform(text_df)
            category = model.predict(text_transformed)[0]
            summary = self.llm_Service.get_ticket_summary(text)
            entities  =self.llm_Service.extract_ticket_info(text)
            logger.info("prediction pipeline completed successfully")
            return {
                "category": category,
                "summary": summary,
                "entities": entities
            }
        
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    sample_text = "I need help with my bill please"
    pipeline = PredictionPipeline()
    result = pipeline.predict(sample_text)
    print(result)

