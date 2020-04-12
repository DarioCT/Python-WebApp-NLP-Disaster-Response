"""
Preprocessing of Data

Sample Script Execution:
    >>> python etl.py data/disaster_messages.csv data/disaster_categories.csv data/database.db

Arguments Description:
    1) File path to CSV file containing messages (e.g. data/disaster_messages.csv)
    2) File path to CSV file containing categories (e.g. data/disaster_categories.csv)
    3) File path to SQLite destination database (e.g. data/database.db)
"""

# Import all the relevant libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Messages Data with Categories

    Arguments:
        messages_filepath : str --> Path to the CSV file containing messages
        categories_filepath : str --> Path to the CSV file containing categories
    Output:
        df : dataframe --> Combined data containing messages and categories
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    Data cleaning

    Arguments:
        df : dataframe --> Combined data containing messages and categories
    Outputs:
        df : dataframe --> Combined data containing messages and categories with categories cleaned up
    """

    # Split the categories
    categories = df["categories"].str.split(";", expand = True)

    # Set categories column names
    categories.columns = categories.iloc[0,:].str.replace("-0","").str.replace("-1","").values

    for col in categories.columns:
        categories[col] = categories[col].str[-1].astype(float)

    # Change the label 2 to 1 in category related as our algorithm expects binary catergories
    categories["related"][categories["related"] == 2] = 1

    df = pd.concat([df, categories], axis = 1).drop(["categories"], axis = 1)

    # Remove duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    """
    Save data to a SQLite Database

    Arguments:
        df : dataframe --> cleaned data
        database_filename : str --> path for SQLite databse
    """

    engine = create_engine('sqlite:///'+ database_filename)
    table_name = 'DRTable'
    df.to_sql(table_name, engine, index=False)

def main():
    """
    The main function which will kick off the data processing functions:
        1) Load Messages Data with Categories
        2) Clean Data
        3) Save Data to SQLite Database
    """


    # Execute the ETL pipeline if the count of arguments is matching to 4
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] # Extract the parameters in relevant variable

        print('Loading messages data from {} ...\nLoading categories data from {} ...'
              .format(messages_filepath, categories_filepath))

        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data ...')
        df = clean_data(df)

        print('Saving data to SQLite database ...')
        save_data(df, database_filepath)

        print('Done!')

    else: # Print the help message so that user can execute the script with correct parameters
        print("Please provide the arguments correctly: \nSample Script Execution:\n\
        >>> python etl.py disaster_messages.csv disaster_categories.csv database.db \n\
        Arguments Description: \n\
        1) File path to CSV file containing messages (e.g. data/disaster_messages.csv)\n\
        2) File path to CSV file containing categories (e.g. data/disaster_categories.csv)\n\
        3) File path to SQLite destination database (e.g. data/database.db)")

if __name__ == '__main__':
    main()
