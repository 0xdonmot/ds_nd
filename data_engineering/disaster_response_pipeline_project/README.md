# Disaster Response Pipeline Project

### Table of Contents

1. [Instructions](#instructions)
2. [Project Summary](#summary)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python app/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python app/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


explanation of files in repo and summary

### Project Summary<a name="summary"></a>
For this project, I developed a natural language processing model which takes in user queries and classifies the messages. There are 36 different types of categories, and messages can be classified as belonging to more than one category.

An XGBoost classifier was used to build the model, using average accuracy across all the categories as the measurement metric.

Across the 36 different categories, the number of positive classifications in the training data varies widely. For example, 77% of messages were classified as related, whilst less than 1% of messages were classified as tools. With this in mind, two immediate improvements could be made to this model.
- training a different classifier for each category.
- adjusting the models in categories with low classification rates to target a high recall and a low type II error rate.

### File Descriptions<a name="files"></a>
There is are various files available in this repo:

- app
    - templates: contains two html files used to render the web app.
    - graphs.py: a script to produce graphs for the web app.
    - process_data.py: a script which loads csv data, cleans it and stores it in a SQL database.
    - train_classifier.py: a script which loads the cleaned data from the SQL database and trains it using the xgboost classifier.
    - run.py: a script which runs the flask web app.
- models
    - classifier.pkl: a pickle file containing the trained xgboost classifier.
- data
    - DisasterResponse.db: a sql database file containing the cleaned data.
    - disaster_messages.csv: a csv file containing the disaster messages data.
    - disaster_categories.csv: a csv file containing information on the disaster classifications.
- ml_pipeline_preparation: an exploration file used for training and optimising the machine learning algorithms.

### Licensing, Authors, Acknowledgements<a name="licensing"></a>
Credit goes to the Figure Eight / Appen for providing the data.