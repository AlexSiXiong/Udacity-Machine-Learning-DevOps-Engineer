import joblib
import logging

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


logging.basicConfig(
    filename='./model/log',
    level=logging.INFO,
    filemode='a',
    format='%(name)s - %(levelname)s - %(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S')

cat_features = [
        "age",
        "workclass",
        "sex",
        "marital-status",
        "occupation",
        "race",
        "hours-per-week",
        "salary"
        ]

def split_data(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)
        logging.info('Data split done.')
        return X_train, X_test, y_train, y_test
    except BaseException:
        logging.info('Data split error.')


def train_save_model(X, y):
    folds = 5
    kf = KFold(n_splits=folds, random_state=42, shuffle=True)

    model_arr = []
    roc_arr = []
    i = 1
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = lgb.LGBMClassifier(objective='binary',
                random_state=42,
                metric='binary_logloss',
                scale_pos_weight=0.34)

        clf.fit(X_train, y_train)
        roc_ = roc_auc_score(y_test, clf.predict(X_test))
        model_arr.append(clf)
        roc_arr.append(roc_)
        logging.info('Fold:{}'.format(i))
        logging.info('Training set score: {:.4f}'.format(clf.score(X_train, y_train)))
        logging.info('Test set score: {:.4f}'.format(clf.score(X_test, y_test)))
        logging.info('ROC:{}'.format(roc_))
        i += 1

    max_index = np.argmax(np.array(roc_arr))
    best_clf = model_arr[max_index]

    # save model
    joblib.dump(best_clf, './model/best_clf.pkl')
    logging.info('Model saved.')
    return best_clf

def test_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    logging.info('Model testset accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    logging.info('Model testset roc score: {0:0.4f}'.format(roc_auc_score(y_test,y_pred)))

if __name__ == '__main__':
    df = pd.read_csv('./data/census_eda.csv')
    np_data = entire_data_processing(df)
    
    y = np_data[:, -1]
    X = np_data[:, :-1]

    X_train, X_test, y_train, y_test = split_data(X, y)
    best_clf = train_save_model(X_train, y_train)
    test_model(X_test, y_test, best_clf)
