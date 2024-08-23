# utils.py
import pickle
import dill
import pandas as pd
import numpy as np
import sys
import os
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
def save_obj(file_path:str,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as f:
            dill.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(
            x_train,
            y_train,
            x_test,
            y_test,
            models,
            params):
    try:
        report={}
        for modelname,model in models.items():
            gs = GridSearchCV(model,params[modelname],cv=5)
            gs.fit(x_train, y_train)
            # now train the actual model file using best params
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            # evaluate
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            train_score=r2_score(y_train,y_train_pred)
            test_score=r2_score(y_test,y_test_pred)
            report[modelname]=(train_score,test_score)
        return report
    except Exception as e:
        raise CustomException(e,sys)
