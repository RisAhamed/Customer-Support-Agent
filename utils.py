import os
import sys
import dill

# Make imports work even when utils.py is imported from subdirectories
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exception import CustomException
from logger import logger
def save_object(file_path, obj):
    """
    Save a Python object to a file using dill.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (any): The Python object to be saved.

    Raises:
        CustomException: If there is an error during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def load_object(file_path):
    """
    Load a Python object from a file using dill.
    
    Args:
        file_path (str): The path to the file where the object is stored.
    
    Returns:
        any: The loaded Python object.
        
    Raises:
        CustomException: If there is an error during the loading process.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        return obj

    except Exception as e:
        raise CustomException(e, sys)