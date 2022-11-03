# imports
import pandas as pd
from sqlalchemy import create_engine
from plotly.graph_objs import Bar


def return_training_figures():
    """
    This function creates figure objects to use in the homepage of
    the disaster response application.
    """

    # load data
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('disaster_messages', engine)

    # create graph one
    graph_one = []
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_one.append(
        Bar(
            x=genre_names,
            y=genre_counts,
        )
    )

    layout_one = dict(title='Distribution of Message Genres',
                      yaxis=dict(title='Count'))

    # create graph two
    graph_two = []
    category_counts = df.iloc[:, 4:].sum()
    category_names = list(category_counts.index)
    category_names = [name.replace('_', ' ') for name in category_names]

    graph_two.append(
        Bar(
            x=category_names,
            y=category_counts,
        )
    )

    layout_two = dict(title='Distribution of Message Classifications',
                      yaxis=dict(title='Count'))

    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))

    return figures
