"""
Classifier Trainer
This script trains the machine learning model and saves as a pickle file

Sample Script Execution:
>>> python train_classifier.py data/database.db model/classifier.pkl

Arguments:
    file path --> of SQLite destination database (e.g. database.db)
    file path --> filename where to pickle trained ML model (e.g. model.pkl)
"""

# import libraries
import os
import re
import sys
import pickle
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report

from xgboost import XGBClassifier

import warnings
warnings.simplefilter('ignore')


def load_data_from_db(database_filepath):
    """
    Load data from the SQLite database

    Arguments:
        database_filepath -> path to SQLite destination database (e.g. database.db)
    Output:
        X : dataframe  --> a dataframe containing features
        Y : dataframe  --> a dataframe containing labels
        categories --> list of categories
    """

    engine = create_engine('sqlite:///' + database_filepath)
    table_name = 'DRTable'
    df = pd.read_sql_table(table_name,engine)

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)


    categories = Y.columns # This will be used for visualization purpose
    return X, Y, categories


def tokenize(text):
    """Normalize, tokenize and lemmatize input text string

    Arguments:
        text: string  --> input text message

    Returns:
        wordsFiltered: list of strings
    """

    wordnet_lemmatizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))

    # Clean the text
    text = re.sub(pattern = '[^a-zA-Z0-9]', repl = ' ', string = text.lower())

    # Extract the word tokens from the provided text
    words = word_tokenize(text)

    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    wordsFiltered = [wordnet_lemmatizer.lemmatize(w) for w in words if w not in stopWords]

    return wordsFiltered


def build_pipeline():
    """
    Builds the ML Pipeline

    Output:
        pipeline : sklearn object --> A Sklearn ML Pipeline that process text messages and apply a classifier.

    """

    pipeline = Pipeline([
        ('vectorize', CountVectorizer(tokenizer = tokenize, max_features = 3000)),
        ('tfidf', TfidfTransformer()),
        ('model', MultiOutputClassifier(RandomForestClassifier(n_estimators = 100, class_weight = 'balanced', min_samples_split = 40)))
    ])

    return pipeline


def single_performance_metric(actual, predicted):
    """Weighted F1_score

    Arguments:
        actual: numpy array --> Array containing actual labels.
        predicted: numpy array --> Array containing predicted labels.


    Returns:
        f1_score: float --> Weighted F1_score.
    """
    return f1_score(actual, predicted, average = 'weighted')


def evaluate_pipeline(pipeline, X_test, Y_test, category_names):
    """
    Evaluate Model function

    This function applies a ML pipeline to a test set and prints out the model f1_score

    Arguments:
        pipeline -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    Y_pred = pipeline.predict(X_test)

    f1_score = single_performance_metric(Y_test,Y_pred)

    print('F1_score {0:.2f}% ...'.format(f1_score*100))


def main():
    """
    Train Classifier Main function that does the following:

        1) Extract data from SQLite database
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle file

    """
    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        print('Loading data from {} ...'.format(database_filepath))
        X, Y, categories = load_data_from_db(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

        print('Building the ML pipeline ...')
        pipeline = build_pipeline()

        print('Training the ML pipeline ...')
        pipeline.fit(X_train, Y_train)

        print('Evaluating ML model ...')
        evaluate_pipeline(pipeline, X_test, Y_test, categories)

        print('Saving ML model to {} ...'.format(pickle_filepath))
        pickle.dump(pipeline, open(pickle_filepath, 'wb'))

        print('Done!')

    else:
         print("Please provide the arguments correctly: \nSample Script Execution:\n\
         > python train_classifier.py ../data/disaster_response_db.db classifier.pkl \n\
         Arguments Description: \n\
         1) Path to SQLite destination database (e.g. disaster_response_db.db)\n\
         2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl")


if __name__ == '__main__':
    main()
