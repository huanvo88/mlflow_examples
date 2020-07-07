import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

import mlflow as mf
from mlflow import log_metric, log_param, log_artifact
import mlflow.lightgbm

#set logging
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

#suppress warnings
import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

#read in the arguments
def parse_args():
    parser = argparse.ArgumentParser(description='LightGBM example')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='learning rate to update step size at each boosting step (default: 0.3)')
    parser.add_argument('--colsample-bytree', type=float, default=1.0,
                        help='subsample ratio of columns when constructing each tree (default: 1.0)')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample ratio of the training instances (default: 1.0)')
    return parser.parse_args()

#the main function
def main():

    # parse command-line arguments
    args = parse_args()

    #read in the dataset
    df = pd.read_csv('train.csv')
    df.drop(['Id'], axis = 1, inplace = True)
    #separate features and target
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]
    logger.info(f'read in the data and drop the Id column, shape is {df.shape}')

    #log the target
    y = np.log(y)
    logger.info('log the target')

    #remove columns with percentage of missing values > 40%
    missing = X.isnull().sum().sort_values(ascending = False)/X.shape[0]*100
    dropped = list(missing[missing > 40].index)
    X.drop(dropped, axis = 1, inplace = True)
    logger.info(f'The features {dropped} have more than 40% of missing values and are dropped')

    #missing features less than 40%
    missing = X.isnull().sum().sort_values(ascending = False)/X.shape[0]*100
    missing_features = list(missing[missing > 0].index)
    logger.info(f'These are the features with less than 40% missing {missing_features}')

    #split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    logger.info(f'Split into train set shape {X_train.shape} and test set shape {X_test.shape}')

    #impute missing features
    for fea in missing_features:
        if X_train[fea].dtypes == 'O':
            most = X_train[fea].value_counts()[0]
            X_train[fea].fillna(most, inplace = True)
            X_test[fea].fillna(most, inplace = True)
            logger.info(f'Fill {fea} by the most popular level {most}')
        else:
            med = X_train[fea].median()
            X_train[fea].fillna(med, inplace = True)
            X_test[fea].fillna(med, inplace = True)
            logger.info(f'Fill {fea} by median {med}')
    logger.info(f'Missing values are filled, the number of missing values is {X_train.isnull().sum().sum()+X_test.isnull().sum().sum()}')

    #one hot encode the cat features
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    #align the train and test, keep only the columns present in both dataframes
    X_train, X_test = X_train.align(X_test, join = 'inner', axis = 1)
    logger.info(f'One hot encode the train set, train shape is {X_train.shape}')
    logger.info(f'One hot encode the test set, test shape is {X_test.shape}')

    #fit lightgbm model
    train_set = lgb.Dataset(X_train, label=y_train)
    test_set  = lgb.Dataset(X_test, label = y_test)

    with mlflow.start_run():

        params = {
                'learning_rate': args.learning_rate,
                'metric': 'rmse',
                'colsample_bytree': args.colsample_bytree,
                'subsample': args.subsample,
                'seed': 42,
                }

        model = lgb.train(params, train_set, num_boost_round=1000,
                                  valid_sets=[test_set], valid_names=['test'],
                          early_stopping_rounds = 100)

        pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        logger.info('fit a lightgbm model')
        logger.info(f'rmse: {rmse}')
        logger.info(f'mae: {mae}')
        logger.info(f'r2: {r2}')

        #log the model
        mf.log_param("learning_rate", args.learning_rate)
        mf.log_param("colsample_bytree", args.colsample_bytree)
        mf.log_param("subsample", args.subsample)
        mf.log_metric('rmse', rmse)
        mf.log_metric('r2', r2)
        mf.log_metric('mae', mae)
        mf.log_artifact('train.csv')
        mf.lightgbm.log_model(model, 'model')


if __name__ == '__main__':
    mf.set_experiment('house_prices')
    main()
