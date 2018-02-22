#!/usr/bin/python

import sys
import pickle
import logging

import numpy as np
import pandas as pd

sys.path.append("/home/g_leonidovich/Documents/ud120-projects/tools/")
sys.path.append("/home/g_leonidovich/Documents/ud120-projects/final_project/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


class EstimatorSelectionHelper:
    """
    This is a helper class for running parameter grid search
    across different classification or regression models.

    Borrowed (and adapted) from:
    http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/
    """

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv, n_jobs=1, verbose=1, scoring=None, refit=True):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self):
        df = pd.concat([pd.DataFrame([model, results.best_score_, results.best_params_],
                columns = ['model',
                           'best_score',
                           'best_parameters']) for model, results in self.grid_searches.iteritems()],
                       ignore_index=True)

        return df


def select_features_by_nan(data, features, percentage=0.8):
    """
    Reduces the list of Enron dataset features based on
    number of rows that have the value of "NaN"

    :param features: a list of features to select from
    :param percentage: percentage threshold (of NaN rows) for selecting features
    :return: a list of features that meet the threshold criterion
    """

    df = pd.DataFrame(data)
    df = df.transpose()
    df.index.name = "name"
    df.reset_index(inplace=True)
    df = df[features]

    nan_counts = []

    # loop through each feature in the dataset
    for c in df:
        # create a Series with a count of each unique value in a column
        counts = df[c].value_counts()
        # store the number of NaNs for each column
        if 'NaN' in counts.keys():
            nan_counts.append(counts['NaN'])
        else:
            nan_counts.append(0)

    # create a DataFrame of features and their counts
    nan_counts_df = pd.DataFrame({'columns': df.columns,
                                  'NaN_counts': nan_counts},
                                 columns=['columns', 'NaN_counts'])

    # store the percentage of NaN for each feature
    nan_counts_df['percent'] = nan_counts_df['NaN_counts'] / 145

    # return a list containing the names of features
    # with the acceptable amount of NaNs
    return [r['columns'] for i, r in nan_counts_df.iterrows() if r['percent'] <= percentage]


PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"


def test_classifier(clf, dataset, feature_list, folds=1000):
    # taken from the main tester file to be able to use logging for output

    data = featureFormat(dataset, feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
        return [
            clf,
            PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision=5),
            RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives,
                                           true_negatives)
            ]
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


def main():
    # initiate a logger for later model comparisons
    logging.basicConfig(
        format='%(message)s',
        filename='output.log',
        level=logging.DEBUG
    )
    logger = logging.getLogger('model_output')

    # start with all features except for email addresses
    all_features = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                    'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                    'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                    'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                    'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

    with open("/home/g_leonidovich/Documents/ud120-projects/final_project/final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    # remove the TOTAL row outlier
    del data_dict["TOTAL"]

    # select features by the number of NaNs
    features_list = select_features_by_nan(data_dict, all_features, 0.8)
    logger.info("NaN_percentage: %f" % 0.8)
    logger.info("Features_by_NaN:\n%s" % str(features_list))

    data = pd.DataFrame(data_dict)
    data = data.transpose()

    # functions to be applied to the DataFrame
    def email_fractions_from_poi(row):
        if row['to_messages'] == "NaN":
            return 0
        else:
            return pd.to_numeric(row['from_poi_to_this_person']) / pd.to_numeric(row['to_messages'])

    def email_fractions_to_poi(row):
        if row['to_messages'] == "NaN":
            return 0
        else:
            return pd.to_numeric(row['from_this_person_to_poi']) / pd.to_numeric(row['to_messages'])

    # create features for the fractions of emails to and from POIs
    data['fraction_from_poi'] = data.apply(email_fractions_from_poi, axis=1)
    data['fraction_to_poi'] = data.apply(email_fractions_to_poi, axis=1)

    # append the created features to the features list
    # features_list.append('fraction_from_poi')
    # features_list.append('fraction_to_poi')

    # test without original features
    # features_list.remove('from_poi_to_this_person')
    # features_list.remove('to_messages')
    # features_list.remove('from_this_person_to_poi')
    # features_list.remove('from_messages')

    data = data.transpose()
    data_dict = data.to_dict()
    my_dataset = data_dict

    data = featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                                test_size=0.3, random_state=42)

    logger.info("Features_Used: %s\n" % str(features_list))

    # scale features before training
    # scaler = StandardScaler()
    # scaler.fit(features_train)

    # scaled_features_train = scaler.transform(features_train)
    # scaled_features_test = scaler.transform(features_test)
    # logger.info("scaled")
    # logger.info("non-scaled")

    # tune the parameters of several classifiers at once using the helper Class
    # models = {
    #     'k-NN': KNeighborsClassifier(),
    #     'DecisionTree': DecisionTreeClassifier(),
    #     'RandomForest': RandomForestClassifier(),
    #     'LogisticReg': LogisticRegression(),
    #     'MLP': MLPClassifier()
    # }

    # params = {
    #     'k-NN': {
    #         'n_neighbors': [2, 3, 4, 5, 10],
    #         'weights': ['uniform', 'distance'],
    #         'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    #         'leaf_size': [5, 10, 20, 30, 50],
    #         'p': [1, 2]
    #     },
    #     'DecisionTree': {
    #         'criterion': ['gini', 'entropy'],
    #         'splitter': ['best', 'random'],
    #         'max_depth': [None, 5, 10, 15, 30]
    #     },
    #     'RandomForest': {
    #         'n_estimators': [5, 10, 20, 30, 50, 100],
    #         'criterion': ['gini', 'entropy'],
    #         'max_depth': [None, 5, 10, 15, 30]
    #     },
    #     'LogisticReg': {
    #         'C': np.logspace(-2, 10, 10),
    #         'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    #     },
    #     'MLP': {
    #         'activation': ['identity', 'logistic', 'tanh', 'relu'],
    #         'solver': ['lbfgs', 'sgd', 'adam'],
    #         'alpha': np.logspace(-9, 3, 10),
    #         'learning_rate': ['constant', 'invscaling', 'adaptive'],
    #         'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    #         'max_iter': [50, 100, 200, 500]
    #     }
    # }

    # helper = EstimatorSelectionHelper(models, params)
    # helper.fit(features_train, labels_train, cv=None, scoring='f1', n_jobs=-1)
    # helper.fit(scaled_features_train, labels_train, cv=None, scoring='f1', n_jobs=-1)

    # log the classification report for each tuned algorithm
    # for algorithm, gsc in helper.grid_searches.iteritems():
    #     logger.info("%s :" % algorithm)
    #     logger.info("best_params: %s" % str(gsc.best_params_))
    #     # prediction = gsc.predict(features_test)
    #     prediction = gsc.predict(scaled_features_test)
    #     report = classification_report(labels_test, prediction)
    #     logger.info("\n%s" % report)

    # test the final model and log the report
    clf = make_pipeline(StandardScaler(),
                        KNeighborsClassifier(n_neighbors=2,
                                             weights='distance',
                                             leaf_size=5,
                                             algorithm='ball_tree',
                                             p=1))

    for line in test_classifier(clf, my_dataset, features_list):
        logger.info(line)

    dump_classifier_and_data(clf, my_dataset, features_list)
    return


if __name__ == "__main__":
    main()
