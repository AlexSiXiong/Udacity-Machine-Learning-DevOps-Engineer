"""
This module performs unit tests to churn_library.py

Author: Alex
Date: 31/08/2021
"""
import logging
import math
import os
import time

from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models

log_file = f"./logs/churn_library_{time.strftime('%b_%d_%Y_%H_%M_%S')}.log"
if not os.path.exists('./logs/'):
    os.mkdir('./logs/')

if os.path.isfile(log_file):
    print('Log file exists, ready to test.')
if not os.path.isfile(log_file):
    f = open(log_file, mode="w", encoding="utf-8")
    print('Log file does not exists. Created SUCCESS.')
    f.close()

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data_func):
    """
    test import_data() function

    Input:
            import_data_func: function under test
    Output:
            None
    """
    try:
        data_frame = import_data_func("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda_func, test_data):
    """
    test perform_eda() function

    Input:
            perform_eda_func: function under test
            test_data: DataFrame, the data for testing
    Output:
            None
    """
    # execute the func
    perform_eda_func(test_data)

    attributes = ['Churn', 'Customer_Age', 'Marital_Status', 'Total_Trans_Ct', 'heatmap']
    attributes = [i.lower() for i in attributes]

    for i in attributes:
        try:
            assert os.path.isfile(r"./images/eda/{}.png".format(i))
            logging.info("Testing perform_eda: %s is created: SUCCESS", i)
        except AssertionError as err:
            logging.error("Testing perform_eda function: CANNOT find %s", i)
            raise err
    logging.info("Testing perform_eda function: SUCCESS")


def test_encoder_helper(encoder_helper_func, test_data):
    """
    test encoder_helper() function

    Input:
            encoder_helper_func: function under test
            test_data: DataFrame, the data for testing
    Output:
            None
    """
    encoder_helper_func(test_data)
    response = ['Gender_Churn',
                'Education_Level_Churn',
                'Marital_Status_Churn',
                'Income_Category_Churn',
                'Card_Category_Churn']
    for i in response:
        try:
            assert i in test_data.columns
            logging.info("Testing encoder_helper: Encoding for %s is SUCCESSFUL", i)
        except AssertionError as err:
            logging.error("Testing encoder_helper: Encoding for %s FAILED", i)
            raise err
    logging.info("Testing encoder_helper function: SUCCESS")


def test_perform_feature_engineering(perform_feature_engineering_func, test_data):
    """
    test perform_feature_engineering() function

    Input:
            perform_feature_engineering_func: function under test
            test_data: DataFrame, the data for testing
    Output:
            an array that stores the split train and test dataset
    """
    x_train, x_test, y_train, y_test = perform_feature_engineering_func(test_data)

    try:
        assert 'Churn' in test_data.columns
        logging.info("Testing perform_feature_engineering: label column created SUCCESSFUL")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: label column created FAILED")
        raise err

    try:
        assert list(test_data['Churn'].unique()) == [0, 1]
        logging.info("Testing perform_feature_engineering: 'Churn' is binary")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: FAILED. 'Churn' is not binary")
        raise err

    try:
        assert math.floor(test_data.shape[0] * 0.7) == x_train.shape[0]
        assert test_data.shape[0] - math.floor(test_data.shape[0] * 0.7) == len(y_test)
        logging.info("Testing perform_feature_engineering: data split SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: shape of split data does not match FAILED")
        raise err
    return [x_train, x_test, y_train, y_test]


def test_train_models(train_models_func, test_data):
    """
    test train_models() function

    Input:
            train_models_func: function under test
            test_data: split train and test dataset
    Output:
            None
    """
    model_dict = train_models_func(test_data)
    try:
        assert os.path.isfile(r'./models/rfc_model.pkl')
        assert os.path.isfile(r'./models/logistic_model.pkl')
        assert len(model_dict) == 2
        logging.info("Testing train_models: Model default training SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: Model default training FAILED")
        raise err


def run_all_tests():
    """
        Runs all the tests written above
    """
    test_import(import_data)
    data_for_test = import_data(r'./data/bank_data2.csv')
    test_eda(perform_eda, data_for_test)
    test_encoder_helper(encoder_helper, data_for_test)
    data_for_test2 = test_perform_feature_engineering(perform_feature_engineering, data_for_test)
    test_train_models(train_models, data_for_test2)


if __name__ == "__main__":
    run_all_tests()
