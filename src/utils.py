import pickle
import dill
import pandas as pd
import numpy as np
import sys
import os
from src.exception import CustomException
def save_obj(file_path:str,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as f:
            dill.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)