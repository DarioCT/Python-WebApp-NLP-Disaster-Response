import json
import plotly
import pandas as pd

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


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

# load data
engine = create_engine('sqlite:///../data/database.db')
df = pd.read_sql_table('DRTable', engine)

# load model
model = joblib.load("../model/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_names = df.iloc[:,4:].columns
    category_boolean = (df.iloc[:,4:] != 0).sum().values


    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean,
                    marker_color='rgb(0,167,0)'


                )
            ],

            'layout': {
                #'width':800,
                'height':500,
                'title': '<b>Distribution of Message Categories<b>',
                'font': {
                        'family': 'Roboto'
                        },
                'hover_data' : ["x", "y"],
                'yaxis': {
                    'title': "Examples per Category"
                },
                'xaxis': {
                    'title': "",
                    'tickangle': 270,
                    'orientation' : "v",
                    'tickfont': {
                            'size': 10
                            }
                }
            }
        },

        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
