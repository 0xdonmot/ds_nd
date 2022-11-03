# load libraries
import sys
import pandas as pd
import re
from sqlalchemy import create_engine

# load nltk libraries
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

# load sklearn text transformation libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier

# load XGBoost Classifier
from xgboost import XGBClassifier

# load joblib to export model
import joblib


def load_data(database_filepath):
    """
    This function loads data from the specified database path.
    It returns a vector of message text features, a matrix of
    classification levels, and the name of the categories. 
    """
    # read in data from sql database
    sqlpath = 'sqlite:///' + database_filepath
    engine = create_engine(sqlpath)
    df = pd.read_sql('SELECT * FROM disaster_messages', engine)

    # convert features and target to numpy array
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns

    return X, Y, category_names


def tokenize(text):
    """
    Function to tokenize text input.
    """
    # remove punctuation
    pattern = '[^A-Za-z0-9]'
    text = re.sub(pattern, ' ', text)

    # convert to lowercase
    text = text.lower().strip()

    # tokenize
    words = word_tokenize(text)

    # remove stopwords (common words that don't add much meaning)
    stop_words = stopwords.words('english')
    words = [word for word in words if word not in stop_words]

    # lemmatize nouns (convert words to their roots)
    words = [WordNetLemmatizer().lemmatize(word, pos='n') for word in words]

    # lemmatize verbs
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    # stem words (reduce words to their stem)
    words = [PorterStemmer().stem(word) for word in words]

    return words


# create custom class
class FirstWordIsVerb(BaseEstimator, TransformerMixin):
    """
    This class is an estimator object that outputs a boolean
    indicating if the first word is a verb or not.
    """
    # tokenize text and output True if first word is a verb else False
    def starting_verb(self, text):
        pos_tags = pos_tag(tokenize(text))
        if len(pos_tags) == 0:
            return False
        first_word, first_tag = pos_tags[0]
        if first_tag in ['VB', 'VBP']:
            return True
        return False

    # estimator must have fit method
    def fit(self, X, y=None):
        return self

    # tranformer must have transformer method
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Function for fitting the extreme gradient boosting model
    """

    # create pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
                ])),
            ('first_word_is_verb', FirstWordIsVerb())
                ])),
        ('clf', MultiOutputClassifier(XGBClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function makes a prediction on the test data, then
    iterates through the category names and displays classification
    reports
    """
    Y_pred = model.predict(X_test)
    number_of_columns = len(category_names)
    for i in range(number_of_columns):
        print(category_names[i])
        print(classification_report(Y_test[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        # grab path arguments from system input
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
