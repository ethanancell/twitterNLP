# twitterNLP
## Overview
This is a repository for a faculty-mentored undergraduate research project in Spring 2020 covering comparative methods for natural language processing of tweets in reference to a real-life disaster or not.

The Kaggle competition link is found at https://www.kaggle.com/c/nlp-getting-started

A rough timeline I followed for the project is the following:
- Get familiarity with some starter methods of natural language processing, including deep-learning and the use of deep-learning libraries
- Apply some of these "starter" methods to the dataset to obtain a model that can output predictions
- Apply more "cutting-edge" methods to this dataset by researching bidirectional models such as BERT
- Explore the benefits and drawbacks of different methods compared to others

## File Guide
Here is a guide to each of the files that is present inside this Github repository:

#### Root level

#### model_scripts
* randomforest.py - This is a script that random forests on Spacy text vectorizations for the model. This script allows for the option to either output on validation data from the training data set, or output on the test dataset from the competition. If "run_validation" is set to false, then the script will create an extra CSV file that includes an extra column as the results of the prediction from the random forest model.
* svm.py - Extremely similar to the Random Forest classifier. This also allows for the option of predicting on a validation set or on the test dataset.

#### data
* train.csv - Labeled training data for the tweets that contained a reference to a real life disaster or not.
* test.csv - Unlabeled test data for the tweets. A submission for the Kaggle competition consists of creating the labels for the data, and then submitting a labeled version of test.csv onto Kaggle's website.
