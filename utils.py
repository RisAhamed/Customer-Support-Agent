import os
import sys
import dill
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