# model_trainer.py
import os,sys
from src.utils import save_obj
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model

# from catboost import CatBoostRegressor # incompatible with python 3.12 or above and numpy 2.0
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

def train(model,x_train,y_train):
    model.fit(x_train,y_train)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

@dataclass
class ModelTrainer:
    model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("spliting data to x_train,y_train,x_test,y_test")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],train_arr[:,-1],
                test_arr[:,:-1],test_arr[:,-1]
            )
            models={
                "LinearRegression":LinearRegression(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor()
                # ,"CatBoostRegressor":CatBoostRegressor()
            }
            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "KNeighborsRegressor":{
                    'n_neighbors':[3,5,10],
                    'weights': ["uniform","distance"]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostingRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            logging.info("started all models evaluation")
            model_report:dict=evaluate_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params)
            
            # to get best model name and score from the model report:
            sorted_models_names=sorted(model_report,key=lambda key:-model_report[key][1])
            best_model_name=sorted_models_names[0]
            best_test_r2=model_report[best_model_name][1]
            best_model = models[best_model_name]
            sorted_r2_score=[(model_name,model_report[model_name][1]) for model_name in sorted_models_names]
            logging.info(f"all models evaluated result:{sorted_r2_score}")
            if best_test_r2<0.6:
                raise CustomException("no best model fund")
            logging.info(f'found best model and its test r2 score: {best_model_name}-> {best_test_r2}')
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            return (best_model_name, best_test_r2)
        
        except Exception as e:
            raise CustomException(e,sys)
        