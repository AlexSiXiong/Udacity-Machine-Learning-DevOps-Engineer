import sys
import os
import joblib
import logging

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

sys.path.append('.')
from  data_processing.process_data import entire_data_processing, process_label, categorize_age,label_encoder, onehot_encoder,label_encoding_attribute


logging.basicConfig(
    filename='./log',
    level=logging.INFO,
    filemode='a',
    format='%(name)s - %(levelname)s - %(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S')

cat_features = [
    'age',
    'sex',
    'race',
    'occupation',
    'workclass',
    'marital-status',
    'salary'
]

def encoder_attributes(df, training=False):
    label_encoding_attribute(df, training=False)
    age_onehot_data = onehot_encoder(df, 'age', training=False)
    
    df.drop('age', axis=1, inplace=True)
    return np.concatenate([age_onehot_data, df.values], axis=1)

def slices(df):
    logging.info('Slices function is called.')
    lgbm_model = joblib.load('./model/best_clf.pkl')
    
    process_label(df)
    categorize_age(df)

    _, test = train_test_split(df, test_size = 0.2, shuffle=True)
    
    # init df storing the result
    all_scores_df = pd.DataFrame(
        columns=[
            "attribute",
            "category",
            "num_samples",
            "accuracy",
            "roc",
        ]
    )
    
    cat_features.remove('salary')
    for attribute in cat_features:
        for category in test[attribute].unique():

            filtered_df = test[test[attribute] == category]
            n_samples = len(filtered_df)

            np_data = encoder_attributes(filtered_df)

            X = np_data[:, :-1]
            y = np_data[:, -1]

            if len(set(y)) == 1:
                accuracy = list(set(y))[0]
                roc_ = -1
            else:    
                y_pred = lgbm_model.predict(X)

                accuracy = int(accuracy_score(y, y_pred) * 10000)/10000
                roc_ = int(roc_auc_score(y,y_pred) * 10000)/10000

            scores_list = [
                    attribute,
                    category,
                    n_samples,
                    accuracy,
                    roc_]

            scores_series = pd.Series(scores_list, index=all_scores_df.columns)
            all_scores_df = all_scores_df.append(scores_series, ignore_index=True)

    all_scores_df.to_csv('./slices/slice_output.csv',index=False)
    
    
if __name__ == '__main__':
    data_file = './data/census_cleaned.csv'
    df = pd.read_csv(data_file)
    slices(df)
