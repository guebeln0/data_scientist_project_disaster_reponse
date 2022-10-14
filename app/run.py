import json
import plotly
import pandas as pd
import numpy as np
import string
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from plotly import utils
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text, stop_words=set(stopwords.words('english'))):
    ''' Remove punctuation, strip unncessary spaces, lematize,
    remove stopwords and tokenize text

    Args:
        text: str
    returns:
        clean_tokens: list of tokens
    '''
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[0-9]+', '', text)
    tokens = word_tokenize(text)

    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in filtered_tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract features for seond visual
    features = df.iloc[:,3:].sum().sort_values(ascending=True)
    features_counts = features.values
    features_names = features.index

    # creates heatmap of correlations
    heatmap = df.iloc[:,3:].corr()

    # create a mask to keep only value of the low triangle in correlation matrix
    mask = np.triu(np.ones_like(heatmap, dtype=bool))
    heatmap_LT = heatmap.mask(mask)

    # Create 3 visualisation charts
    graphs = [
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
        },
        {
            'data': [
                Bar(
                    x=features_counts,
                    y=features_names,
                    orientation='h',
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'height': 700,
                'margin': {
                    'pad': 5
                },
                'yaxis': {
                    'title': "Categories",
                    'dtick': 1
                },
                'xaxis': {
                    'title': "Count"
                },
            }
        },
        {
            'data': [
                Heatmap(
                    x=features_names,
                    y=features_names,
                    z=heatmap_LT
                )
            ],

            'layout': {
                'title': 'Correlation of features in training dataset',
                'height': 800,
                'width': 800,
                'xaxis_showgrid': False,
                'yaxis_showgrid': False,
    '           yaxis_autorange': 'reversed'
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=utils.PlotlyJSONEncoder)

    print(graphJSON)
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
