# Optuna

import xgboost as xgb
import numpy as np
import pandas as pd
import random
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

target = train['TARGET']
data = train.drop(['TARGET'],axis=1)

def objective(trial,data=data,target=target):
    
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2,random_state=42)
    param = {

        'lambda': trial.suggest_uniform('lambda',0.001,0.1),
        'alpha': trial.suggest_uniform('alpha',0.1,0.2),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3,1.0),
        'subsample': trial.suggest_uniform('subsample', 0.4,0.8),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.05,0.08),
        'n_estimators': trial.suggest_int('n_estimators', 1000,4000),
        'max_depth': trial.suggest_int('max_depth', 3,10),
        'random_state': trial.suggest_int('random_state', 400,1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 10,100),
        'objective': trial.suggest_categorical('objective',['reg:logistic']), 
        'tree_method': trial.suggest_categorical('tree_method',['gpu_hist']),  # 'gpu_hist','hist'       
        'use_label_encoder': trial.suggest_categorical('use_label_encoder',[False])
    }
    model = xgb.XGBClassifier(**param)      
    model.fit(train_x,train_y,eval_set=[(test_x,test_y)],early_stopping_rounds=100,verbose=False)
    preds = model.predict(test_x)
    rmse = mean_squared_error(test_y, preds,squared=False)
    
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=8)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# XGB

Best_trial=study.best_trial.params
target = train.pop("TARGET")

X_fit, X_eval, y_fit, y_eval= train_test_split(
    train, target, test_size=0.15, random_state=1
)

xgtrain = xgb.DMatrix(X_fit, y_fit)
xgtest = xgb.DMatrix(test)

#n_estimators and early_stopping_rounds should be increased
clf = xgb.XGBClassifier(**Best_trial)

# fitting
clf.fit(X_fit, y_fit, early_stopping_rounds=35,  eval_metric="logloss", eval_set=[(X_eval, y_eval)])

y_pred= clf.predict_proba(test,ntree_limit=clf.best_ntree_limit)[:,1]
