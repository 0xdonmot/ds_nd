# Sparkify Project

### Table of Contents

1. [Instructions](#instructions)
2. [Project Summary](#summary)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

### Instructions:
1. There are a couple of extra libraries needed to run the code here beyond the Anaconda distribution of Python:
- pyspark

### Project Summary<a name="summary"></a>
For this project, I analysed user data from a fictional music streaming company, Sparkify, to try to identify users that are likely to cancel (churn) their subscriptions.

The data contained historic actions of Sparkify's users, including their 'cancellation' actions. Therefore, the challenge was to identify the behaviour or characteristics of such users that would indicate them as a potential churn candidate.

After exploring the data, I decided to use the following features for modelling.

_User Features_
- Gender
- Level
- Registration Length

_Page Actions (total count of page actions)_
- Thumbs down
- Thumbs up
- Roll adverts
- Add to playlist
- Add friend
- Downgrade
- Next song
- Home
- Help
- Logout

I selected a Random Foreast Classifier, after testing various models on a sample of the full dataset. Subsequently, I optimised the Classifier using cross validation.

### File Descriptions<a name="files"></a>
There are two files available in this repo:

- Analysis file: sparkify_analysis.ipynb
- Data file: medium-sparkify-event-data.json

### Licensing, Authors, Acknowledgements<a name="licensing"></a>
Credit goes to the Udacity for providing the data.