import os
import joblib
from fastapi import FastAPI
from typing import Literal
from pandas import DataFrame
import numpy as np
import uvicorn
from pydantic import BaseModel

from  DevOps_Proj3_v1.data_processing.process_data import entire_data_processing, process_label, categorize_age,label_encoder, onehot_encoder,label_encoding_attribute

# Set up DVC on Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")



# Create app
app = FastAPI()

# POST Input Schema
class ModelInput(BaseModel):
    age: int
    workclass: Literal['State-gov',
                       'Self-emp-not-inc',
                       'Private',
                       'Federal-gov',
                       'Local-gov',
                       'Self-emp-inc',
                       'Without-pay']
    fnlgt: int
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    education_num: int
    marital_status: Literal["Never-married",
                            "Married-civ-spouse",
                            "Divorced",
                            "Married-spouse-absent",
                            "Separated",
                            "Married-AF-spouse",
                            "Widowed"]
    occupation: Literal["Tech-support",
                        "Craft-repair",
                        "Other-service",
                        "Sales",
                        "Exec-managerial",
                        "Prof-specialty",
                        "Handlers-cleaners",
                        "Machine-op-inspct",
                        "Adm-clerical",
                        "Farming-fishing",
                        "Transport-moving",
                        "Priv-house-serv",
                        "Protective-serv",
                        "Armed-Forces"]
    relationship: Literal["Wife", "Own-child", "Husband",
                          "Not-in-family", "Other-relative", "Unmarried"]
    race: Literal["White", "Asian-Pac-Islander",
                  "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Female", "Male"]
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']

    class Config:
        schema_extra = {
            "example": {
                "age": 27,
                "workclass": 'State-gov',
                "fnlgt": 77516,
                "education": 'Bachelors',
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Tech-support",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Female",
                "capital_gain": 2000,
                "capital_loss": 0,
                "hours_per_week": 35,
                "native_country": 'United-States'
            }
        }


def data_folder_path(subfolder, file):
    """[summary]
    Args:
        subfolder ([str]): [subfolder name]
        file ([str]): [file name]
    Returns:
        [str]: [file complete path]
    """
    root = os.path.dirname(os.getcwd())
    return os.path.join(root, "Udacity-Machine-Learning-DevOps-Engineer/DevOps_Proj3_v1", subfolder, file)

# Load artifacts
lgbm_model = joblib.load(data_folder_path('model','best_clf.pkl'))


# Root path
@app.get("/")
async def root():
    return {
        "Hi": "This app predicts wether income exceeds $50K/yr through user input info"}

# Prediction path
@app.post("/inference")
async def inference(input: ModelInput):

    input_data = np.array([[
        input.age,
        input.workclass,
        input.fnlgt,
        input.education,
        input.education_num,
        input.marital_status,
        input.occupation,
        input.relationship,
        input.race,
        input.sex,
        input.capital_gain,
        input.capital_loss,
        input.hours_per_week,
        input.native_country]])

    original_cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours-per-week",
        "native-country"]

    input_df = DataFrame(data=input_data, columns=original_cols)
    
    cat_features = ["age",
        "workclass",
        "sex",
        "marital-status",
        "occupation",
        "race",
        "hours-per-week"]
    
    # process data
    df = input_df[cat_features]
    label_encoding_attribute(df, training=False)
    age_onehot_data = onehot_encoder(df, 'age', training=False)
    df.drop('age', axis=1, inplace=True)

    x = np.concatenate([age_onehot_data, df.values], axis=1)  
    y = lgbm_model.predict(x)

    pred = ""
    if y == 0:
        pred = ">=50K"
    else:
        pred = "<=50K"

    return {"prediction": pred}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
