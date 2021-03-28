import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


def train_and_test_one_time(params):

    print('Loading data...')
    # load or create your dataset
    #df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
    #df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')

    train_list = list(range(100000))
    train_data = {'X_Train':  train_list}
    df_train = pd.DataFrame(train_data, columns = ['X_Train'])

    test_list = list(range(110000, 150000))
    test_data = {'X_Test':  test_list}

    df_test = pd.DataFrame(test_data, columns = ['X_Test'])

    X_train = df_train
    X_test = df_test
    y_train = []
    y_test = []

    for i in range(len(train_list)): y_train.append((1000 * train_list[i] + 10))
    y_train = np.array(y_train)

    for j in range(len(test_list)): y_test.append((1000 * test_list[j] + 10))
    y_test = np.array(y_test)


    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    
    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5)

    print('Saving model...')
    # save model to file
    gbm.save_model('model.txt')

    print('Starting predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    print(y_pred)

    mse = mean_squared_error(y_test, y_pred)

    # eval
    print('The rmse of prediction is:', mse ** 0.5)

    return mse


def optimize_lightGBM():

    # specify your configurations as a dict
    params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 2, #31
            'learning_rate': 0.07, #0.05
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            #'reg_sqrt': False
        }

    mse = train_and_test_one_time(params)

    '''
    while(mse > 1):
        params["learning_rate"] -= 0.001
        params["num_leaves"] += 1
        mse = train_and_test_one_time(params)
        print(mse)
    '''


optimize_lightGBM()