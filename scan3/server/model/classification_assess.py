# Script that contains functions for performance assessment of classification
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import label_binarize
from sklearn.utils import column_or_1d


def _check_binary_probabilistic_predictions(y_true, y_prob):
    """Check that y_true is binary and y_prob contains valid probabilities"""
    assert len(y_true) == len(y_prob)

    labels = np.unique(y_true)

    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. "
                         "Provided labels %s." % labels)

    if y_prob.max() > 1:
        raise ValueError("y_prob contains values greater than 1.")

    if y_prob.min() < 0:
        raise ValueError("y_prob contains values less than 0.")

    return label_binarize(y_true, labels)[:, 0]


def calibration_curve(y_true, y_prob, normalize=False, n_bins=5):
    """Compute true and predicted probabilities for a calibration curve.
    Read more in the :ref:`User Guide <calibration>`.
    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    normalize : bool, optional, default=False
        Whether y_prob needs to be normalized into the bin [0, 1], i.e. is not
        a proper probability. If True, the smallest value in y_prob is mapped
        onto 0 and the largest one onto 1.
    n_bins : int
        Number of bins. A bigger number requires more data.
    Returns
    -------
    prob_true : array, shape (n_bins,)
        The true probability in each bin (fraction of positives).
    prob_pred : array, shape (n_bins,)
        The mean predicted probability in each bin.
    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    """

    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is "
                         "set to False.")

    y_true = _check_binary_probabilistic_predictions(y_true, y_prob)

    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

    return [prob_true, prob_pred]


def get_performance(posterior_probs, true_labels, labels, label_priors,
                    verbose=False, threshold=None, perf_columns=None):
    '''
    Given the true labels and posterior probabilites for multi-label (#labels > 2) classification,
    return the performance metrics.
    Makes use of: http://scikit-learn.org/stable/modules/classes.html#classification-metrics
    Parameters:
    -------------------------------------------------------
    posterior_probs   - (pd.DataFrame) contains the posterior probabilities of the classifier
                         where each row is an observation and columns are features
    true_labels       - (pd.Series) true class labels for each observation
    labels            - (list) class label names of the output variable
    label_priors      - (pd.Series) the prior distribution of class labels, used by BIR calculation
    threshold         - (float) classification threshold, disregard predictions where max posterior
                        probability of the modal class are less then threshold.
    perf_columns      - (list) if None, equals to ['accuracy', 'avg_BIR', 'kappa']
                        otherwise the list of performance metric names as strings.
    '''

    performance = dict()  # dictionary of different performance metric names and values
    if perf_columns is None:
        perf_columns = ['accuracy', 'avg_BIR', 'kappa']
    else:
        perf_columns = perf_columns

    if not isinstance(posterior_probs, pd.DataFrame):
        raise TypeError('The posterior probs need to be Pandas DataFrame.')

    if sum(posterior_probs.values == 0).sum():  # are there any 0 posterior probs if so, treat them with epsilon
        epsilon = 5e-3
        posterior_probs = np.clip(posterior_probs, a_min=epsilon, a_max=epsilon)

    if verbose:
        print("......Calculating prediction performance for test set.")

    pred_probs = posterior_probs[labels].astype(np.float32)  # all floats since we will have to take log in calc_BIR
    if isinstance(true_labels, np.ndarray):
        true_labels = pd.Series(true_labels)  # convert true labels to Series if necessary
    elif isinstance(true_labels, str):
        true_labels = pd.Series(np.array([true_labels]))  # if there is a single true_label, convert to Series

    if threshold is not None:
        pred_probs, true_labels = threshold_decision(true_labels, pred_probs, threshold)

    # Simple Maximum A-posteriori (MAP) estimate
    pred_labels = pred_probs.idxmax(axis=1)

    # Calculate the confusion matrix
    confusion_mat = pd.DataFrame(metrics.confusion_matrix(true_labels.values, pred_labels.values, labels=labels),
                                 index=['True_' + str(t) for t in labels],
                                 columns=['Pred_' + str(p) for p in labels])

    # Calculate performance metrics if they are specified in perf_columns
    if 'accuracy' in perf_columns:
        performance['accuracy'] = sum(np.diag(confusion_mat.values)) / float(confusion_mat.sum().sum())
    if 'precision' in perf_columns:
        performance['precision'] = metrics.precision_score(true_labels.values,
                                                           pred_labels.values, labels=labels, average=None)
    if 'recall' in perf_columns:
        performance['recall'] = metrics.recall_score(true_labels.values, pred_labels.values,
                                                     labels=labels, average=None)
    if 'AUPRC' in perf_columns:  # area under the precision-recall curve
        if len(labels) != 2:
            raise ValueError('Cannot calculate AUPRC for %i class problems.' % (len(labels)))
        # http://scikit-learn.org/0.15/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
        performance['AUPRC'] = metrics.average_precision_score(true_labels.values, pred_labels.values)

    if 'kappa' in perf_columns:
        # http://stats.stackexchange.com/questions/82162/kappa-statistic-in-plain-english
        exp_accuracy = sum(
            confusion_mat.sum(axis=0).values * confusion_mat.sum(axis=1).values / confusion_mat.sum().sum()) \
                       / float(confusion_mat.sum().sum())
        performance['kappa'] = (performance['accuracy'] - exp_accuracy) / (1 - exp_accuracy)
    if 'prior' in perf_columns:
        performance['prior'] = label_priors.values
    if 'AUC' in perf_columns:
        if len(labels) != 2:
            raise ValueError('Cannot calculate AUROC for %i class problems.' % (len(labels)))
        # plot ROC curve, based on class label with index 1 (hard-coded for now)
        fpr, tpr, _ = metrics.roc_curve(true_labels, posterior_probs[1])
        performance['AUC'] = metrics.auc(fpr, tpr)
        # visualize.plot_ROC(fpr, tpr, performance['AUC'], labels[1])

    performance = pd.Series([performance[col] for col in perf_columns],
                            index=perf_columns)

    return performance.apply(lambda x: np.round(x, 3))


def threshold_decision(true_labels, pred_probs, threshold):
    '''
    Given the posterior probabilities, true_labels and a decision threshold, return a subset
    of the posterior probabilities and true_labels where the maximum posterior prob is >= threshold.
    In other words, we elect not to make a decision for cases where max(pred_probs) is lower than
    the threshold probability.
    Parameters:
    true_labels - the true labels (numpy array)
    pred_probs  - the posterior probabilities for all labels (Pandas Series)
    threshold   - posterior probability threshold of the modal class for choosing to predict
    '''
    accepted_idx = pred_probs.max(axis=1) > threshold
    filtered_pred_probs = pred_probs.loc[accepted_idx.values, :]
    filtered_true_labels = true_labels[accepted_idx.values]

    return filtered_pred_probs, filtered_true_labels


def calc_BIR_vectorised(true_labels, pred_probs, priors, labels):
    '''
    Given the true labels and the posterior probabilities, calculates the Bayesian Information
    Reward (BIR) for the entire dataset.
    Parameters:
    true_labels - the true labels (pandas Series)
    pred_probs  - the posterior probabilities for all labels (Pandas DataFrame)
    priors      - prior probabilities of labels
    labels      - the label names as a list
    '''

    # treat any zeros in priors by adding a small value, hardcoded here as 5e-2
    # for motif prediction, priors are prearranged to never be zero, however for direction prediction
    # it is not the case. So this check belwo is exclusively for direction prediction cases where if the train_size
    # is very samll (e.g. 1 day), a direction label may not have been encountered and prior (of e.g. 'flat') is equal to 0.
    smooth_coeff = 5e-2
    # calc_indiv_BIR_vectorised() does not like df's with date_time index, so reset just in case.
    pred_probs.reset_index(inplace=True, drop=True)

    if any(priors == 0):
        priors = (priors + smooth_coeff) / (priors + smooth_coeff).sum()

    BIRs = np.array([np.nan for _i in range(len(true_labels))])
    for j in range(len(true_labels)):
        BIRs[j] = calc_indiv_BIR_vectorised(pred_probs.iloc[j], true_labels.iloc[j], priors, labels)
    # Return the average BIR across all test cases. The average is calculated to normalise for test set size
    # so I can compare the BIR of a test set of size 5 with one with a size of 10.
    BIR_avg = BIRs.sum() / len(BIRs)

    return BIR_avg


def calc_indiv_BIR_vectorised(row, true_label, priors, labels):
    '''
    Return the mean BIR for the given case. Calculated as per equation 3
    in http://www.csse.monash.edu.au/~korb/pubs/ai02.pdf
    Parameters:
    row        - the posterior probs and true label for the current case (row) in the dataset
    true_label -
    priors     - padnas Dataframe that contains the prior distributions of all labels
    labels     - the label names as a list
    '''
    # Correct classification:
    correct_reward = 1 - np.log(row[true_label]) / np.log(priors.loc[true_label])
    # Incorrect classification:
    incorrect_reward = (len(labels) - 1) - (np.log(1 - row.loc[row.index != true_label]) \
                                            / np.log(1 - priors.loc[priors.index != true_label])).sum()
    result = (correct_reward + incorrect_reward) / len(labels)

    if np.isinf(result):
        raise ArithmeticError('np.inf encounteed.')

    return result


def calc_BIR_old(true_labels, pred_probs, priors, labels):
    '''
    Given the true labels and the posterior probabilities, calculates the Bayesian Information
    Reward (BIR) for the entire dataset.
    Parameters:
    true_labels - the true labels (numpy array)
    pred_probs  - the posterior probabilities for all labels (Pandas Series)
    priors      - prior probabilities of labels
    labels      - the label names as a list
    '''
    pred_probs.loc[:, 'True'] = true_labels.values
    BIRs = pred_probs.apply(lambda row: calc_indiv_BIR_old(row, priors, labels), axis=1)
    # Return the average BIR across all test cases. The average is calculated to normalise for test set size
    # so I can compare the BIR of a test set of size 5 with one with a size of 10.
    BIR_avg = BIRs.sum() / len(BIRs)

    return BIR_avg


def calc_indiv_BIR_old(row, priors, labels):
    '''
    Return the mean BIR for the given case. Calculated as per equation 3
    in http://www.csse.monash.edu.au/~korb/pubs/ai02.pdf
    Parameters:
    row   - the posterior probs and true label for the current case (row) in the dataset
    priors - the prior distributions of all labels
    labels      - the label names as a list
    '''
    BIR_vector = map(lambda idx: 1 - np.log(row[idx]) / np.log(priors.loc[idx]) if idx == row['True']
    else 1 - np.log(1 - row[idx]) / np.log(1 - priors.loc[idx]),
                     labels)

    return sum(BIR_vector) / len(BIR_vector)


##### GENERIC UTILITY FUCNTIONS #######
def get_ts_true_directions(ts, predict_freq, predict_horizon, granularity):
    '''
    Given ts that contains the time series data, the function returns true direction difference labels by
    1. resampling the pandas Series by predict_freq in ticks, 2. getting the deltas of the resampled Series
    by predict_horizon and 3. finally mapping to Up, Down or Flat based on the sign of differences.
    Returns a numpy array of labels.
    Parameters:
    ----------------------
    ts              - a pd Series that contains the raw ts from which we infer the directions by
                      shifting the ts by predict_horizon
    predict_freq    - make a prediction every this many seconds
    predict_horizon - the prediction horizon in seconds which will be used to calculate
    granularity     - the granularity of the input ts, used to calculate shift_ticks along with predict_horizon
    '''
    assert (predict_horizon >= predict_freq), 'For now Pred horizon has to be greater than or equal to freq'
    resample_ticks = predict_freq / granularity  # downsample by this number of ticks.
    resampled_ts = ts.iloc[::resample_ticks]  # resample to only take every (shift_ticks)'th row in df.
    diff_ticks = predict_horizon / predict_freq
    assert float(diff_ticks).is_integer(), 'predict_horizon/predict_freq needs to be a whole number'
    diffs = resampled_ts.diff(periods=diff_ticks)  # take diff over rows by predict_horizon ticks
    diff_labels = diffs.apply(get_direction_label)
    diff_labels.name = 'labels'  # make the name of the Series labels to be used while inputting to assess.get_performance()

    return diff_labels


def get_direction_label(row):
    ''' Simple function that returns string direction label based on the sign of the input.'''
    if row == 0:
        label = 'Flat'
    elif row > 0:
        label = 'Up'
    elif row < 0:
        label = 'Down'
    else:
        label = np.nan

    return label


##### ASSESSMENT PLOTS #############
def plot_decision_regions(X, y, clf,
                          ax=None,
                          X_highlight=None,
                          res=0.02, legend=1,
                          hide_spines=True,
                          markers='s^oxv<>',
                          colors='red,blue,limegreen,gray,cyan'):
    # http://stackoverflow.com/questions/22294241/plotting-a-decision-boundary-separating-2-classes-using-matplotlibs-pyplot?lq=1
    """Plot decision regions of a classifier.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Feature Matrix.
    y : array-like, shape = [n_samples]
        True class labels.
    clf : Classifier object.
        Must have a .predict method.
    ax : matplotlib.axes.Axes (default: None)
        An existing matplotlib Axes. Creates
        one if ax=None.
    X_highlight : array-like, shape = [n_samples, n_features] (default: None)
        An array with data points that are used to highlight samples in `X`.
    res : float (default: 0.02)
        Grid width. Lower values increase the resolution but
        slow down the plotting.
    hide_spines : bool (default: True)
        Hide axis spines if True.
    legend : int (default: 1)
        Integer to specify the legend location.
        No legend if legend is 0.
    markers : list
        Scatterplot markers.
    colors : str (default 'red,blue,limegreen,gray,cyan')
        Comma separated list of colors.

    Returns
    ---------
    ax : matplotlib.axes.Axes object

    """
    # check if data is numpy array
    for a in (X, y):
        if not isinstance(a, np.ndarray):
            raise ValueError('%s must be a NumPy array.' % a.__name__)

    if ax is None:
        ax = plt.gca()

    if not y.dtype == int:
        y = y.astype(int)

    # check if test data is provided
    plot_testdata = True
    if not isinstance(X_highlight, np.ndarray):
        if X_highlight is not None:
            raise ValueError('X_test must be a NumPy array or None')
        else:
            plot_testdata = False

    if len(X.shape) == 2 and X.shape[1] > 1:
        dim = '2d'
    else:
        dim = '1d'

    marker_gen = cycle(list(markers))

    # make color map
    n_classes = len(np.unique(y))
    colors = colors.split(',')
    cmap = ListedColormap(colors[:n_classes])

    # plot the decision surface
    if dim == '2d':
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    else:
        y_min, y_max = -1, 1

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))

    if dim == '2d':
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
    else:
        y_min, y_max = -1, 1
        Z = clf.predict(np.array([xx.ravel()]).T)

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    ax.axis(xmin=xx.min(), xmax=xx.max(), y_min=yy.min(), y_max=yy.max())

    # plot class samples

    for c in np.unique(y):
        if dim == '2d':
            y_data = X[y == c, 1]
        else:
            y_data = [0 for i in X[y == c]]

        ax.scatter(x=X[y == c, 0],
                   y=y_data,
                   alpha=0.8,
                   c=cmap(c),
                   marker=next(marker_gen),
                   label=c)

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if not dim == '2d':
        ax.axes.get_yaxis().set_ticks([])

    if legend:
        legend = plt.legend(loc=legend,
                            fancybox=True,
                            framealpha=0.3,
                            scatterpoints=1,
                            handletextpad=-0.25,
                            borderaxespad=0.9)

        ax.add_artist(legend)

    if plot_testdata:
        if dim == '2d':
            ax.scatter(X_highlight[:, 0],
                       X_highlight[:, 1],
                       c='',
                       alpha=1.0,
                       linewidth=1,
                       marker='o',
                       s=80)
        else:
            ax.scatter(X_highlight,
                       [0 for i in X_highlight],
                       c='',
                       alpha=1.0,
                       linewidth=1,
                       marker='o',
                       s=80)

    return ax


def plot_probability_heatmap(X, y, clf):
    '''
    http://stackoverflow.com/questions/28256058/plotting-decision-boundary-of-logistic-regression
    Parameters:
    -----------------------------------------------------------
    X    - predictor variables Pandas dataframe
    y    - outcome variable indices Pandas series
    clf  - the trained scikitlearn classifier
    '''
    # make a continuous grid of values and evaluate the probability of each (x1, x2) point in the grid
    xx, yy = np.mgrid[X.iloc[:, 0].min():X.iloc[:, 0].max():.1, X.iloc[:, 1].min():X.iloc[:, 1].max():.1]
    grid = np.c_[xx.ravel(), yy.ravel()]
    # plot the probability grid as a contour map and additionally show the test set samples on top of it
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(X.iloc[100:, 0], X.iloc[100:, 1], c=y.iloc[100:], s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(X.iloc[:, 0].min(), X.iloc[:, 0].max()), ylim=(X.iloc[:, 1].min(), X.iloc[:, 1].max()),
           xlabel="$X_1$", ylabel="$X_2$")

    plt.show()