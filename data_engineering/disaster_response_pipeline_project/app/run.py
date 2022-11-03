# load libraries
import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request
import joblib
from sqlalchemy import create_engine
from train_classifier import FirstWordIsVerb, tokenize
from graphs import return_training_figures
from plotly.graph_objs import Bar


# initialise Flask app
app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    define the app action and html response
    when a request suffix ends with "/index"
    """
    # create visuals
    graphs = return_training_figures()

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    define the app action and html response
    when a request suffix ends with "/go"
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # make model prediction based on input query
    probs = model.predict_proba([query])

    # create list of classification probabilities
    classification_probabilities = []
    for p in probs:
        classification_probabilities.append(p[0][1])

    # replace underscore with space
    classification_names = [name.replace('_', ' ') for name in df.columns[4:]]

    # create graph for classification probabilities
    graph_probs = [
        Bar(
            x=classification_names,
            y=classification_probabilities,
        )
    ]

    layout_probs = dict(title='Message Classification Probability',
                        yaxis=dict(title='Probability'))

    # create figures list and append graphs
    figures = []
    figures.append(dict(data=graph_probs, layout=layout_probs))

    # encode probability plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    # render go.html with plotly graphs
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        ids=ids,
        graphJSON=graphJSON
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
