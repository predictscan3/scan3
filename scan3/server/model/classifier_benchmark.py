import numpy as np
import sys
from scipy.stats import mode
import pandas as pd; pd.set_option('display.expand_frame_repr', False)
from time import time
import matplotlib.pyplot as plt
import logging

from classification_assess import get_performance
from keras.utils import np_utils
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

class RandomPredictor:
    """
    Given the test set, provides random predictions that follow a uniform
    distribution.
    """

    def __init__(self):

        self.unique_labels = None
        self.freqs = None

    def fit(self, train_data, train_labels):
        """
        Based on the training labels, learn the unique class labels, but not the frequencies.
        It accepts train_data (X_train) but ignore is completely
        :param train_labels: (np.array) training set labels
        :return: None
        """
        self.unique_labels = np.unique(train_labels)
        self.freqs = np.ones(len(self.unique_labels)) / len(self.unique_labels)

    def predict(self, test_set):
        """

        :param test_set: (np.array)  test_set inputs
        :return: pred_labels: (np.array) prediction labels
        """
        if self.freqs is not None:
            pred_labels = np.random.choice(self.unique_labels, len(test_set),
                                           p=self.freqs)
        else:
            raise Exception("Prior predictor has not been fitted. Class label priors are unknown.")

        return pred_labels

class PriorPredictor:
    """
    Given the test set, provides random predictions that follow the training set
    class label distribution.
    """

    def __init__(self):

        self.unique_labels = None
        self.freqs = None

    def fit(self, train_data, train_labels):
        """
        Based on the training set, learn the unique class labels and their frequencies.
        It accepts train_data (X_train) but ignore is completely
        :param train_labels: (np.array) training set labels
        :return: None
        """
        if len(train_labels.shape) ==1: # if a 1-d label array
            self.unique_labels, counts = np.unique(train_labels, return_counts= True)
            self.freqs = counts / np.sum(counts)
        else: # if an N-d label array
            self.freqs = train_labels.sum(axis=0) / len(train_labels)
            self.unique_labels = np.array(list(range(train_labels.shape[1]))) # label-encoded

    def predict(self, test_set):
        """

        :param test_set: (np.array)  test_set inputs
        :return: pred_labels: (np.array) prediction labels
        """
        if self.freqs is not None:
            pred_labels = np.random.choice(self.unique_labels, len(test_set),
                                           p=self.freqs)
        else:
            raise Exception("Prior predictor has not been fitted. Class label priors are unknown.")

        return pred_labels

CLASSIFIER_LIST_FULL = [(RandomPredictor(), 'Random Predictor'),
                        (PriorPredictor(), 'Prior Predictor'),
                        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                        (Perceptron(n_iter=50), "Perceptron"),
                        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                        (KNeighborsClassifier(n_neighbors=10), "kNN"),
                        (RandomForestClassifier(n_estimators=100), "Random Forest"),
                        (SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"), 'Elastic Net'),
                        (LinearSVC(penalty='l1', dual=False, tol=1e-3), 'SVM L1'),
                        (LinearSVC(penalty='l2', dual=False, tol=1e-3), 'SVM L2'),
                        (NearestCentroid(), 'Nearest Centroid'),
                        (Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),
                                   ('classification', LinearSVC(penalty="l2"))]),'Lin SVC with L1 feat selection')]


CLASSIFIER_LIST_NLP = [(RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                        (Perceptron(n_iter=50), "Perceptron"),
                        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                        (KNeighborsClassifier(n_neighbors=10), "kNN"),
                        (RandomForestClassifier(n_estimators=100), "Random Forest"),
                        (SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"), 'Elastic Net'),
                        (LinearSVC(penalty='l1', dual=False, tol=1e-3), 'SVM L1'),
                        (LinearSVC(penalty='l2', dual=False, tol=1e-3), 'SVM L2'),
                        (NearestCentroid(), 'Nearest Centroid'),
                        (MultinomialNB(alpha=.01), 'Sparse Multinomial NB'),
                        (BernoulliNB(alpha=.01), 'Sparse Bernoulli NB'),
                        (Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),
                                   ('classification', LinearSVC(penalty="l2"))]),'Lin SVC with L1 feat selection')]

PERFORMANCE_COLS = ['accuracy', 'kappa', 'precision', 'recall',  'prior', 'AUPRC', 'AUC']
# Benchmark Legacy Parameters
PRINT_REPORT = True
PRINT_CM = True
USE_HASHING = False
PRINT_TOP10 = False


class Ensembler():

    def __init__(self, classifier_list):

        self.clf_list = classifier_list

    def fit(self, Xtrain, ytrain):
        """
        Ensembler is designed to use already fitted classifiers, so this method is a stub.
        :return:
        """
        pass

    def predict(self, Xtest):
        """
        Predict with all classifiers in the ensemble, for the final prediction via
        majority-voting.
        :param Xtest:
        :return:
        """
        preds = np.stack([clsf.predict(Xtest) for clsf in self.clf_list], axis=1)

        # Majority Voting
        ensemble_preds = np.squeeze( np.apply_along_axis(lambda x: mode(x)[0], 1, preds))

        return ensemble_preds


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# #############################################################################
def plot_results(perf_df, results):

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, perf, training_time, test_time = results

    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, perf_df.accuracy.values, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
             color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, perf_df.description.values):
        plt.text(-.3, i, c)

    plt.show()


def benchmark(clf, name, label_priors, Xtrain, ytrain, Xtest, ytest, feature_names, class_labels):
    """

    :param clf:
    :param name:
    :param label_priors:
    :param Xtrain:
    :param ytrain:
    :param Xtest:
    :param ytest:
    :param feature_names:
    :param class_labels:
    :return:
    """
    if class_labels.dtype.name != 'str':
        class_labels = class_labels.astype(str)

    logging.debug(clf)
    t0 = time()
    clf.fit(Xtrain, ytrain)
    train_time = time() - t0

    t0 = time()
    pred = clf.predict(Xtest)
    test_time = time() - t0

    perf = get_performance(posterior_probs=pd.DataFrame(np.stack([1-pred, pred], axis=1), columns=[1,0]),
                           true_labels=ytest,
                           labels=[1,0], perf_columns=PERFORMANCE_COLS,
                           label_priors=pd.Series(label_priors))
    logging.debug(perf)

    if hasattr(clf, 'coef_'):
        logging.debug("dimensionality: %d" % clf.coef_.shape[1])
        logging.debug("density: %f" % density(clf.coef_))

        if PRINT_TOP10 and feature_names is not None:
            logging.debug("top 10 keywords per class:")
            for i, label in enumerate(class_labels):
                top10 = np.argsort(clf.coef_[i])[-10:]
                logging.debug(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        logging.debug('')

    if PRINT_REPORT:
        logging.debug("Classification report:")
        logging.debug(metrics.classification_report(ytest, pred,
                                            target_names=class_labels))

    if PRINT_CM:
        logging.debug("Confusion matrix:")
        logging.debug(metrics.confusion_matrix(ytest, pred))

    logging.debug('')
    clf_descr = str(clf).split('(')[0]
    perf['classifier_name'] = name

    return {"description": clf_descr, 'classifier': clf, "performance": perf,
            "train_time": train_time, "test_time": test_time}


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

