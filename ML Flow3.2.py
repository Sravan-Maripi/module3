#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np


# In[19]:


import mlflow
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.model_selection import GridSearchCV


# In[20]:


import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz" 

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[21]:


fetch_housing_data()


# In[22]:


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[23]:


housing = load_housing_data()


# In[24]:


housing.head()


# In[25]:


remote_server_uri = "http://0.0.0.0:5000" # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)


# In[26]:


mlflow.tracking.get_tracking_uri()


# In[27]:


mlflow.set_experiment("Housing Price")


# In[28]:


col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names] # get the column indices

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]


# In[29]:


def split_load_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    housing = pd.read_csv(csv_path)
    
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    
    housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]
            
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
            
    return strat_train_set, strat_test_set



# In[30]:


def pipeline_transform(data):
    
    train = data
    
    housing = train.drop("median_house_value", axis=1)
                
    label = train["median_house_value"].copy()
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    
    housing_num = train.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs),
    ])
    
    housing_prepared = full_pipeline.fit_transform(train)
    
    return housing_prepared, label


# In[31]:


def eval_metrics(actual, pred):
    # compute relevant metrics
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# In[32]:


def train(max_features='auto', n_estimators=100, alpha=0.5, l1_ratio=0.5):
    
    with mlflow.start_run(run_name='PARENT_RUN') as parent_run:
        
        mlflow.log_param("parent", "yes")
        
        with mlflow.start_run(run_name='LOAD DATA', nested=True) as load_data:
            
            mlflow.log_param("child", "yes")
            
            data_path = "datasets/housing/housing.csv"
            
            train_set, test_set = split_load_data()
            
            with mlflow.start_run(run_name='DATA PREPARATION', nested=True) as data_preparation:
                
                mlflow.log_param("child", "yes")
                
                train_prepared, labels = pipeline_transform(train_set)
                
                with mlflow.start_run(run_name='TRAINING', nested=True) as training:
                    
                    mlflow.log_param("child", "yes")                    
                   
                    rf = RandomForestRegressor(random_state=42)
                    train1_prepared,labels1 = pipeline_transform(test_set)
                    rf.fit(train_prepared,labels)
                    predict=rf.predict(train1_prepared)
                    (rmse, mae, r2) = eval_metrics(labels1,predict)
                            
                    print("Random Forest Model (max_features=%s, n_estimators=%f):" % (max_features, n_estimators))
                    print("  RMSE: %s" % rmse)
                    print("  MAE: %s" % mae)
                    print("  R2: %s" % r2)                                                                                   
                    param_grid = [
                        # try 12 (3×4) combinations of hyperparameters
                       {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                       # then try 6 (2×3) combinations with bootstrap set as False
                       {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
                        ]
                    forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
                    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                                                scoring='neg_mean_squared_error',
                                                return_train_score=True)
                    grid_search.fit(train_prepared,labels)
                    final_model = grid_search.best_estimator_
                   
                    final_predictions = final_model.predict(train1_prepared)
                    final_rmse = mean_squared_error(labels1, final_predictions, squared=False)
                    (rmse, mae, r2) = eval_metrics(labels1, final_predictions)
                                                                                                
                        # Print out metrics                                                                                                                                           
                                                                      
                    confidence = 0.95
                    squared_errors = (final_predictions - labels1) ** 2
                    print(grid_search.best_estimator_)
                    print("  RMSE: %s" % rmse)
                    print("  MAE: %s" % mae)
                    print("  R2: %s" % r2)
                
                    print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                                   loc=squared_errors.mean(),
                                                   scale=stats.sem(squared_errors))))                                                        
                            
                            # Log parameter, metrics, and model to MLflow
                    mlflow.log_param(key="max_features", value=max_features)
                    mlflow.log_param(key="n_estimators", value=n_estimators)
                    mlflow.log_metric(key="rmse", value=rmse)
                    mlflow.log_metrics({"mae": mae, "r2": r2})
                    mlflow.log_artifact(data_path)
                    print("Save to: {}".format(mlflow.get_artifact_uri()))                            
                    mlflow.sklearn.log_model(rf, "model")
            


# In[33]:


train()


# In[ ]:





# In[ ]:




