import os 
import logging

import pandas as pd
import numpy as np

from sklearn import preprocessing


logging.basicConfig(
    filename='./log',
    level=logging.INFO,
    filemode='a',
    format='%(name)s - %(levelname)s - %(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S')


def get_age_group(age):
    if age < 18:
        return 'underage'
    elif (age >= 18) & (age < 60):
        return 'adult'
    else:
        return 'elder'


def get_salary_group(salary):
    if salary == '<=50K':
        return 1
    return 0


def process_label(df):
    df['salary'] = df['salary'].apply(get_salary_group)


def categorize_age(df):
    df['age'] = df['age'].apply(get_age_group)


def save_encoder_classes(path, attribute, le):
    if not os.path.exists(path):
        np.save(path, le.classes_)
        logging.info('init and save encoder classes_')


def label_encoder(df, attribute, training):
    le = preprocessing.LabelEncoder()
    
    if training:
        le.fit(list(set(df[attribute])))        
        save_encoder_classes(f'./data/label_encoders/{attribute}.npy', attribute, le)
    else:
        le.classes_ = np.load(f'./data/label_encoders/{attribute}.npy')
    df[attribute] = le.transform(df[attribute])


def label_encoding_attribute(df, training):
    label_encoder(df,'sex', training)
    label_encoder(df,'race', training)
    label_encoder(df,'occupation', training)
    label_encoder(df,'workclass', training)
    label_encoder(df,'marital-status', training)
    

def onehot_encoder(df, attribute, training):
    lb = preprocessing.LabelBinarizer()
    
    if training:
        lb.fit(list(set(df[attribute])))
        save_encoder_classes(f'./data/onehot_encoders/{attribute}.npy', attribute, lb)    
    else:
        lb.classes_ = np.load(f'./data/onehot_encoders/{attribute}.npy')
    return lb.transform(df[attribute])


def entire_data_processing(df, training):
    process_label(df)
    categorize_age(df)
    label_encoding_attribute(df, training)
    age_onehot_data = onehot_encoder(df, 'age', training)
    df.drop('age', axis=1, inplace=True)
    return np.concatenate([age_onehot_data, df.values], axis=1)

