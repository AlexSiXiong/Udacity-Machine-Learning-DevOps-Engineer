"""
This program aims for analysis a bank's customer data.
Logistic Regression ans Random Forest were applied in analysis the data.

Author: Alex
Date: 31/08/2021
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

sns.set()
FIG_SIZE = (16, 8)


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df_chunks = pd.read_csv(pth, chunksize=1000)

    res = []
    for chunk in df_chunks:
        res.append(chunk)
    return pd.concat(res)


def perform_eda(data_frame):
    """
    perform eda on data_frame and save figures to images folder

    input:
            data_frame: pandas dataframe

    output:
            None
    """
    path = './images/eda/'
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    # ---------- check the distributions of 'Churn' and 'Customer_age' -----------
    for attribute in ['Churn', 'Customer_Age']:
        fig = plt.figure(figsize=FIG_SIZE)
        data_frame[attribute].hist()

        fig.text(0.5, 0.04, attribute, ha='center')
        fig.text(0.04, 0.5, 'Amount', va='center', rotation='vertical')
        plt.savefig(path + attribute.lower() + '.png')
        plt.close()

    # ---------- check distribution of Martital_Status ----------
    attribute = 'Marital_Status'
    plt.figure(figsize=FIG_SIZE)
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(path + attribute.lower() + '.png')
    plt.close()

    # ---------- check distribution of total transaction ----------
    attribute = 'Total_Trans_Ct'
    plt.figure(figsize=FIG_SIZE)
    sns.histplot(data_frame[attribute], kde=True, stat="density", linewidth=0)
    plt.savefig(path + attribute.lower() + '.png')
    plt.close()

    # ---------- check heatmap of all attributes ----------
    plt.figure(figsize=(30, 15))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(path + 'heatmap.png')
    plt.close()


def encoder_helper(data_frame, response=None):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response:
            string of response name
            [optional argument that could be used for naming variables or index y column]

    - ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    - cat_columns

    output:
            None
            (df: pandas dataframe with new columns for)
    """
    if response is None:
        response = ['Gender',
                    'Education_Level',
                    'Marital_Status',
                    'Income_Category',
                    'Card_Category']
    for category in response:
        category_lst = []
        attribute_groups = data_frame.groupby(category).mean()['Churn']
        for val in data_frame[category]:
            category_lst.append(attribute_groups.loc[val])
        data_frame[category + '_Churn'] = category_lst
    """
    Another Solution:
    
    for col in category_lst:
        new_col = col + suffix
        temp[new_col] = temp.groupby(col)["Churn"].transform("mean")
    
    """


def perform_feature_engineering(data_frame):
    """
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or
              index y column]

    output:
              X_train: x training data
              X_test: x testing data
              y_train: y training data
              y_test: y testing data
    """

    label_data = data_frame['Churn']
    train_data = extract_training_features(data_frame)

    x_train, x_test, y_train, y_test = train_test_split(train_data, label_data, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def feature_importance_plot(model, data_frame, output_pth='./images/results/feature_importance.png'):
    """
    creates and stores the feature importance in pth

    input:
            model: model object has an API - feature_importances_
            data_frame: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importance
    importance = model.feature_importances_
    # Sort feature importance in descending order
    indices = np.argsort(importance)[::-1]

    # Rearrange feature names so they match the sorted feature importance
    x_data = extract_training_features(data_frame)
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=FIG_SIZE)

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importance[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(data, models=None):
    """
    train, store model results: images + scores, and store models

    input:
            data: an array that contains data below
            1) x_train: x training data
            2) x_test: x testing data
            3) y_train: y training data
            4) y_test: y testing data
    output:
            model_dict: an dictionary storing the model names and the corresponding models
            key - model names - string
            value - model
    """
    if models is None:
        models = ['lrc', 'rfc']

    x_train, x_test, y_train, y_test = data

    model_dict = dict()
    cv_rfc, lrc = init_models()
    if 'lrc' in models:
        lrc.fit(x_train, y_train)
        store_models(lrc, './models/logistic_model.pkl')
        model_dict['lrc'] = lrc

        y_train_predictions_lr = lrc.predict(x_train)
        y_test_predictions_lr = lrc.predict(x_test)
        data = [y_train, y_test, y_train_predictions_lr, y_test_predictions_lr]
        generate_model_score_report('Logistic_Regression', data)

    if 'rfc' in models:
        cv_rfc.fit(x_train, y_train)
        store_models(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        model_dict['rfc'] = cv_rfc.best_estimator_

        y_train_predictions_rf = cv_rfc.best_estimator_.predict(x_train)
        y_test_predictions_rf = cv_rfc.best_estimator_.predict(x_test)
        data = [y_train, y_test, y_train_predictions_rf, y_test_predictions_rf]
        generate_model_score_report('Random_Forest', data)

    # ---------- save roc graphs ----------
    save_roc_curves(model_dict, x_test, y_test)

    return model_dict


def init_models():
    """
    model initiation
    input:
              None
    output:
              initiated cv_lrc and rfc models
    """

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # cv_rfc
    rfc = RandomForestClassifier(random_state=42)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # lrc
    lrc = LogisticRegression()
    return cv_rfc, lrc


def store_models(model_name, pth):
    """
    store models

    input:
            model
    output:
            None
    """
    joblib.dump(model_name, pth)


def generate_model_score_report(model_name, data, output_pth='./images/results/Report_'):
    """
    input:
            model_name: the name of the model obtained
            y_train: train data in test set
            y_test: test data in test set
            train_pred: the prediction result using the train data
            test_pred: the prediction result using the test data
            output_pth: path that saves the output report images
    output:
            None
    """
    y_train, y_test, train_pred, test_pred = data

    font_size = {'fontsize': 10}
    f_style = 'monospace'

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    ax.text(0.01, 1.25, str(model_name + ' Train'), font_size, fontproperties=f_style)
    ax.text(0.01, 0.05, str(classification_report(y_test, test_pred)), font_size, fontproperties=f_style)
    ax.text(0.01, 0.6, str(model_name + ' Test'), font_size, fontproperties=f_style)
    ax.text(0.01, 0.7, str(classification_report(y_train, train_pred)), font_size, fontproperties=f_style)

    ax.axis('off')
    fig.savefig(output_pth + model_name + '_results.png', dpi=300, bbox_inches="tight")


def save_roc_curves(model_dict, x_test, y_test, path='./images/results/roc_curve_result.png'):
    """
    Generates and saves ROC Curves for the models on the test datasets

    Input:
            model_dict: an dictionary storing the model names and the corresponding models
            key - model names - string
            value - model

            x_test: pandas series; training data for the testing set
            y_test: pandas series; target data for the testing set
    Output:
            None
    """

    plt.cla()
    plt.figure(figsize=(15, 8))

    for i in model_dict:
        if 'lrc' in i:
            lrc_model = model_dict[i]
            plot_roc_curve(lrc_model, x_test, y_test)
        if 'rfc' in i:
            rfc_model = model_dict[i]
            axis = plt.gca()
            plot_roc_curve(rfc_model, x_test, y_test, ax=axis, alpha=0.8)

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def extract_training_features(data_frame, response=None):
    """
    Extract the features that will be used for model training

    Input:
            data_frame: the dataframe that stores the data
            response: an array that contains all the features
    Output:
            data_frame[response]: the dataframe with features required
    """
    if response is None:
        response = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                    'Total_Relationship_Count', 'Months_Inactive_12_mon',
                    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                    'Income_Category_Churn', 'Card_Category_Churn']
    return data_frame[response]


if __name__ == '__main__':
    DF_DATA_FRAME = import_data('./data/bank_data2.csv')
    perform_eda(DF_DATA_FRAME)
    encoder_helper(DF_DATA_FRAME)
    X_TRAIN1, X_TEST1, Y_TRAIN1, Y_TEST1 = perform_feature_engineering(DF_DATA_FRAME)
    RES_DICT = train_models([X_TRAIN1, X_TEST1, Y_TRAIN1, Y_TEST1], ['rfc'])
    feature_importance_plot(RES_DICT['rfc'], DF_DATA_FRAME)
