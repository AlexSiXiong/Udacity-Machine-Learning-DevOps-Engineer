"""Basic data cleaning"""

import logging 
import argparse
import pandas as pd


def load_data(path):
    """
    Read csv data into dataframe

    input:
        path - the file path

    output:
        df - the dataframe
    """
    try:
        df = pd.read_csv(path)
        logging.info('SUCCESS: data loaded success.')
        return df
    except BaseException:
        logging.info('ERROR: data loaded failed.')


def clean_data(df):
    """
    Remove the space in the str
    drop nan value - which removed 2000+ records
    remove some unused columns

    input:
        df - the census dataframe data
    
    output:
        df - the cleaned data
    """
    try:
        df.columns = df.columns.str.strip()

        df['salary'] = df['salary'].str.lstrip()
        df.drop("fnlgt", axis="columns", inplace=True)
        df.drop("education-num", axis="columns", inplace=True)
        df.drop("capital-gain", axis="columns", inplace=True)
        df.drop("capital-loss", axis="columns", inplace=True)
        
        df.drop_duplicates(ignore_index=True, inplace=True)
        df.replace({'?': None}, inplace=True)
        df.dropna(inplace=True)

        df.to_csv('census_cleaned.csv')
        logging.info('SUCCESS: cleaning data saved in file census_cleaned.csv')
        return df
    except BaseException:
        logging.info('FAIL: data cleaning failed.')


def make_args():
    """
    Create an ArgumentParser.
    path - path to an image file
    """
    
    parser = argparse.ArgumentParser(
        description="Load the data."
    )
    
    parser.add_argument(
        "--path",
        type=str,
        help="The path for the census data to be load.",
        default='./census.csv'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    df = load_data(args.path)
    clean_data(df)
