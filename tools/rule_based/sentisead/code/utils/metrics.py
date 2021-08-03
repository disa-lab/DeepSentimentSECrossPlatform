'''
Created on Mar 25, 2014

@author: gias
'''
from __future__ import division

from sklearn.metrics import matthews_corrcoef, roc_curve,auc
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from six import string_types
from six.moves import xrange as range
from sklearn.metrics import confusion_matrix as sk_cm, f1_score, SCORERS

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from ml_metrics.quadratic_weighted_kappa import *

# from https://github.com/EducationalTestingService/skll/blob/5ea61b8dfc23570e661468457a262b6c2242daa9/skll/metrics.py
def kappa(y_true, y_pred, weights=None, allow_off_by_one=False):
    """
    for this: convert labels to numeric range
    for sentiment detection convert p to +1, n to -1 and o to 0. Then it will calculate the weighted kappa 
    for Nicole's MSR paper, the weights value would be 'linear'
    
    Calculates the kappa inter-rater agreement between two the gold standard
    and the predicted ratings. Potential values range from -1 (representing
    complete disagreement) to 1 (representing complete agreement).  A kappa
    value of 0 is expected if all agreement is due to chance.
    In the course of calculating kappa, all items in `y_true` and `y_pred` will
    first be converted to floats and then rounded to integers.
    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.
    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.
    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float
    :param weights: Specifies the weight matrix for the calculation.
                    Options are:
                        -  None = unweighted-kappa
                        -  'quadratic' = quadratic-weighted kappa
                        -  'linear' = linear-weighted kappa
                        -  two-dimensional numpy array = a custom matrix of
                           weights. Each weight corresponds to the
                           :math:`w_{ij}` values in the wikipedia description
                           of how to calculate weighted Cohen's kappa.
    :type weights: str or numpy array
    :param allow_off_by_one: If true, ratings that are off by one are counted as
                             equal, and all other differences are reduced by
                             one. For example, 1 and 2 will be considered to be
                             equal, whereas 1 and 3 will have a difference of 1
                             for when building the weights matrix.
    :type allow_off_by_one: bool
    """
    #logger = logging.getLogger(__name__)

    # Ensure that the lists are both the same length
    assert(len(y_true) == len(y_pred))

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    # Note: numpy and python 3.3 use bankers' rounding.
    try:
        y_true = [int(np.round(float(y))) for y in y_true]
        y_pred = [int(np.round(float(y))) for y in y_pred]
    except ValueError as e:
        print("For kappa, the labels should be integers or strings "
                     "that can be converted to ints (E.g., '4.0' or '3').")
        raise e

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred,
                                labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, string_types):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ''
    if weights is None:
        weights = np.empty((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            diff = abs(i - j)
            if allow_off_by_one and diff:
                diff -= 1
            if wt_scheme == 'linear':
                weights[i, j] = diff
            elif wt_scheme == 'quadratic':
                weights[i, j] = diff ** 2
            elif not wt_scheme:  # unweighted
                weights[i, j] = bool(diff)
            else:
                raise ValueError('Invalid weight scheme specified for '
                                    'kappa: {}'.format(wt_scheme))

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[: num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[: num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(weights):
        k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))

    return k

def kendall_tau(y_true, y_pred):
    """
    Calculate Kendall's tau between ``y_true`` and ``y_pred``.
    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float
    :returns: Kendall's tau if well-defined, else 0
    """
    ret_score = kendalltau(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0


def spearman(y_true, y_pred):
    """
    Calculate Spearman's rank correlation coefficient between ``y_true`` and
    ``y_pred``.
    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float
    :returns: Spearman's rank correlation coefficient if well-defined, else 0
    """
    ret_score = spearmanr(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0


def pearson(y_true, y_pred):
    """
    Calculate Pearson product-moment correlation coefficient between ``y_true``
    and ``y_pred``.
    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float
    :returns: Pearson product-moment correlation coefficient if well-defined,
              else 0
    """
    ret_score = pearsonr(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0


def f1_score_least_frequent(y_true, y_pred):
    """
    Calculate the F1 score of the least frequent label/class in ``y_true`` for
    ``y_pred``.
    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float
    :returns: F1 score of the least frequent label
    """
    least_frequent = np.bincount(y_true).argmin()
    return f1_score(y_true, y_pred, average=None)[least_frequent]

def mcc(expecteds, gots):
    sample_weight = None
    mcc = matthews_corrcoef(expecteds, gots, sample_weight)
    return mcc 

def compute_auc(expecteds,gots, pos_label = 1):
    fpr, tpr, thresholds = roc_curve(expecteds, gots, pos_label)
    return auc(fpr, tpr)

def jaccard(set1, set2):
    union = set1.union(set2)
    if len(union) == 0: return 0
    intersection = set1.intersection(set2)
    
    sim = len(intersection)/len(union)
    return sim


def precision(tp, fp):
    denom = tp + fp
    
    if denom != 0:
        return tp/denom
    else:
        return 0

def recall(tp, fn):
    denom = tp + fn
    
    if denom != 0:
        return tp/(denom)
    else:
        return 0

def balanced_accuracy(tp, fp, tn, fn):
    denom_p = tp + fp
    denom_n = tn + fn
    if denom_p == 0 and denom_n == 0: return 0
    cost = 0.5
    if denom_p != 0:
        bp = cost*(tp/denom_p)
    else:
        bp = 0
    if denom_n != 0:
        bn = cost*(tn/denom_n)
    else:
        bn = 0
    
    
    acc = bp + bn
    return acc

def balanced_accuracy_alt(tp, fp, tn, fn):
    # sensitivity = tp/ (tp + fn)
    # specificity = tn / (tn + fp) 
    
    denom_sensitivity = tp + fn
    denom_specificity = tn + fp
    
    if denom_sensitivity == 0 and denom_specificity == 0: return 0
    cost = 0.5
    if denom_sensitivity != 0:
        bp = cost * ( tp / denom_sensitivity )
    else:
        bp = 0
    if denom_specificity != 0:
        bn = cost * ( tn / denom_specificity )
    else:
        bn = 0
    
    acc = bp + bn
    return acc


def f1_score(precision, recall):
    denom = precision + recall
    if denom == 0: return 0
    else:
        return 2*precision*recall/(denom)

def accuracy(list_expected, list_result):
    right = 0
    wrong = 0
    for i, expected in enumerate(list_expected):
        if expected != list_result[i]:
            wrong += 1
        else:
            right += 1
    denom = right + wrong
    if denom != 0:
        acc = right/denom
    else:
        acc = 0
    return acc

def accuracy_alt(tp, fp, tn, fn):
    r = tp + tn
    wrong = fp + fn
    total = r + wrong
    #print "right = %i, total = %i, tp = %i, tn = %i, fp = %i, fn = %i"%(r, total, tp, tn, fp, fn) 
    if total != 0:
        return r / total 
    else:
        return 0

## expected and got  = list of 1 and 0
def compute_performance( expected, got):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i, e in enumerate(expected):
        g = got[i]
        if e == g: 
            if e == 1: tp += 1
            else: tn += 1
        else:
            if e == 1: fn += 1
            else: fp += 1
    acc = accuracy_alt(tp, fp, tn, fn)
    pre = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(pre, rec)
    return acc, pre, rec, f1

def compute_performance_alt( tp, tn, fp, fn):
    acc = accuracy_alt(tp, fp, tn, fn)
    pre = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(pre, rec)
    return acc, pre, rec, f1

def convertRatingsForKappa(ratings): # p = 1, n = -1, o = 0
    nr = []
    for r in ratings:
        if r == 'p':
            nr.append(1)
        elif r == 'n': 
            nr.append(-1)
        else:
            nr.append(0)
    return nr

def computePercentAgreement(t1, t2, df):
    r1Isnull = pd.isnull(df[t1])
    r2Isnull = pd.isnull(df[t2])
    r1s = []
    r2s = []
    for index, row in df.iterrows():
        if r1Isnull[index]: continue
        if r2Isnull[index]: continue
        v1 = row[t1]
        v2 = row[t2]
        r1s.append(v1)
        r2s.append(v2)   
    perfects = 0
    dSeveres = 0
    dMilds = 0
    totals = 0
    nR1s = convertRatingsForKappa(r1s)
    nR2s = convertRatingsForKappa(r2s)
    for i, r1 in enumerate(nR1s):
        r2 = nR2s[i]
        diff = abs(r1-r2)
        if diff > 1:
            dSeveres += 1
        elif diff == 1:
            dMilds += 1
        else:
            perfects += 1
        totals += 1
    rPerfects = perfects/totals
    rSeveres = dSeveres/totals
    rMilds = dMilds/totals
    return rPerfects, rSeveres, rMilds
def computedWeightedKappaFromDF(expCol, gotCol, df):
    r1Isnull = pd.isnull(df[expCol])
    r2Isnull = pd.isnull(df[gotCol])
    r1s = []
    r2s = []
    for index, row in df.iterrows():
        if r1Isnull[index]: continue
        if r2Isnull[index]: continue
        v1 = row[expCol]
        v2 = row[gotCol]
        r1s.append(v1)
        r2s.append(v2)   
    k = computeWeightedKappa(r1s, r2s)
    return k

def computeWeightedKappa(exps, gots):
    nExps = convertRatingsForKappa(exps)
    nGots = convertRatingsForKappa(gots)
    k = linear_weighted_kappa(nExps, nGots)
    return k

class PerformanceMultiClass(object):
    
    def __init__(self, yTrue, yPred, labels = ['p', 'n', 'o']): # p = positive, n = negative, o = neutral
        #http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        # rows pred, cols truth in cm
        self.cm = sk_cm(yTrue, yPred, labels = labels)
        self.labels = set(labels)
        self.trueLabels = set(yTrue)
        askedVsTrueLabels = self.trueLabels.intersection(self.labels)
        if len(askedVsTrueLabels) != len(self.labels):
            print "Mismatch number of labels between Truth and Provided Classlist. True = %i. No of matched = %i (Out of %i in Asked)"%(len(self.trueLabels), len(askedVsTrueLabels), len((self.labels)))
        
    # from: https://www.python-course.eu/confusion_matrix.php
    # check also: https://stats.stackexchange.com/questions/91044/how-to-calculate-precision-and-recall-in-a-3-x-3-confusion-matrix/91046#91046
    # good explanation: http://text-analytics101.rxnlp.com/2014/10/computing-precision-and-recall-for.html
    # quora answer: https://www.quora.com/How-do-I-compute-precision-and-recall-for-a-multi-class-classification-problem-Can-I-only-compute-accuracy
    # stack overflow answer: https://stackoverflow.com/questions/45603956/class-wise-precision-and-recall-for-multi-class-classification-in-tensorflow?rq=1    
    def getLabelPosition(self, label):
        labelPosition = -1
        for i, l in enumerate(self.labels):
            if l == label:
                labelPosition = i 
        if labelPosition == -1: 
            print "label '", label, "' not found in the provided classlist!"
            return None  
        else:
            return labelPosition
    def precision(self, label):
        print("===> precision method is called with label: %s" %label)
        label = self.getLabelPosition(label)
        if label is None: 
            print("====> label is none returning")
            return
        print("CM is %s" % self.cm)
        col = self.cm[:, label]
        print("label: %s and col: %s" % (label, col))
        total = col.sum()
        print("+===== Total %s col: %s" % (total, col))
        if total == 0: return np.nan
        else:
            return self.cm[label, label] / col.sum()

    def _precision(self, labelPosition):
        col = self.cm[:, labelPosition]
        total = col.sum()
        if total == 0: return np.nan
        else:
            return self.cm[labelPosition, labelPosition] / col.sum()
    def recall(self, label):
        label = self.getLabelPosition(label)
        if label is None: return
        if label > (len(self.labels) - 1) or label < 0 :
            print "label '", label, "' not found in the provided classlist!"
            return        
        row = self.cm[label, :]
        total = row.sum()
        if total == 0: return np.nan
        else:
            return self.cm[label, label] / row.sum()
    def _recall(self, labelPosition):
        row = self.cm[labelPosition, :]
        total = row.sum()
        if total == 0: return np.nan
        else:
            return self.cm[labelPosition, labelPosition] / row.sum()

    def precision_macro_average(self):
        rows, columns = self.cm.shape
        sum_of_precisions = 0
        for labelPosition in range(rows):
            #if label not in 
            pr = self._precision(labelPosition)
            if np.isnan(pr): continue
            sum_of_precisions += pr
        return sum_of_precisions / rows    

    def recall_macro_average(self):
        rows, columns = self.cm.shape
        sum_of_recalls = 0
        for labelPosition in range(columns):
            recall = self._recall(labelPosition)
            if np.isnan(recall): continue
            sum_of_recalls += recall
        return sum_of_recalls / columns        
    
    def f1_macro_average(self):
        precision = self.precision_macro_average()
        recall = self.recall_macro_average()
        f1 = f1_score(precision, recall)
        return f1
    
    def compute_micro_average(self):
        #print self.cm
        rows, columns = self.cm.shape
        tp = 0
        fp = 0
        fn = 0
        for label in range(rows):
            tp += self.cm[label, label]
            for k in range(columns):
                if k == label: continue
                #print "fp ", k, self.cm[label, k]
                fp += self.cm[label, k]
            for k in range(rows):
                if k == label: continue
                #print "fn ", k, self.cm[k, label]
                fn += self.cm[k, label]
        p_denom = tp + fp
        #print tp, fp, fn 
        if p_denom != 0:
            precision = tp / p_denom
        else:
            precision = None
        r_denom = tp + fn 
        if r_denom != 0:
            recall = tp / r_denom
        else:
            recall = None
        f1 = f1_score(precision, recall)
        return precision, recall, f1    

# From: https://www.youtube.com/watch?v=FAr2GmWNbT0    
class PerformanceMultiClassYouTube(object):
    
    def __init__(self, yTrue, yPred, labels = ['p', 'n', 'o']): # p = positive, n = negative, o = neutral
        #http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        # rows pred, cols truth in cm
        self.cm = sk_cm(yTrue, yPred, labels = labels)
        self.labels = set(labels)
        self.trueLabels = set(yTrue)
        askedVsTrueLabels = self.trueLabels.intersection(self.labels)
        if len(askedVsTrueLabels) != len(self.labels):
            print "Mismatch number of labels between Truth and Provided Classlist. True = %i. No of matched = %i (Out of %i in Asked)"%(len(self.trueLabels), len(askedVsTrueLabels), len((self.labels)))
        
    # from: https://www.python-course.eu/confusion_matrix.php
    # check also: https://stats.stackexchange.com/questions/91044/how-to-calculate-precision-and-recall-in-a-3-x-3-confusion-matrix/91046#91046
    # good explanation: http://text-analytics101.rxnlp.com/2014/10/computing-precision-and-recall-for.html
    # quora answer: https://www.quora.com/How-do-I-compute-precision-and-recall-for-a-multi-class-classification-problem-Can-I-only-compute-accuracy
    # stack overflow answer: https://stackoverflow.com/questions/45603956/class-wise-precision-and-recall-for-multi-class-classification-in-tensorflow?rq=1    
    def _precision(self, tp, fp):
        denom = tp + fp
        
        if denom != 0:
            return tp/denom
        else:
            return np.nan

    def _recall(self, tp, fn):
        denom = tp + fn
        
        if denom != 0:
            return tp/(denom)
        else:
            return np.nan

    def _f1_score(self, p, r):
        denom = p + r
        if denom == 0: return np.nan
        else:
            return 2*p*r/(denom)
    
    def precision(self, label):
        if label > (len(self.labels) - 1) or label < 0 :
            print "label '", label, "' not found in the provided classlist!"
            return
        tp = self.cm[label,label]
        fp = sum(self.cm[:,label]) - tp
        fn = sum(self.cm[label,:]) - tp
        tn = sum(sum(self.cm)) - (sum(self.cm[:,label]) + sum(self.cm[label,:]))
        return self._precision(tp, fp)


    def recall(self, label):
        if label > (len(self.labels) - 1) or label < 0 :
            print "label '", label, "' not found in the provided classlist!"
            return        
        tp = self.cm[label,label]
        fp = sum(self.cm[:,label]) - tp
        fn = sum(self.cm[label,:]) - tp
        tn = sum(sum(self.cm)) - (sum(self.cm[:,label]) + sum(self.cm[label,:]))
        return self._recall(tp, fn)
    
    def precision_macro_average(self):
        rows, columns = self.cm.shape
        sum_of_precisions = 0
        for label in range(rows):
            #if label not in 
            pr = self.precision(label)
            if np.isnan(pr): continue
            sum_of_precisions += pr
        return sum_of_precisions / rows    

    def recall_macro_average(self):
        rows, columns = self.cm.shape
        sum_of_recalls = 0
        for label in range(columns):
            rec = self.recall(label)
            if np.isnan(rec): continue
            sum_of_recalls += rec
        return sum_of_recalls / columns        
    
    def f1_macro_average(self):
        pr = self.precision_macro_average()
        rec = self.recall_macro_average()
        f1 = self._f1_score(pr, rec)
        return f1
    
    def compute_micro_average(self):
        #print self.cm
        rows, columns = self.cm.shape
        tps = 0
        fps = 0
        fns = 0
        tns = 0
        for label in range(rows):
            tp = self.cm[label,label]
            tps += tp
            fp = sum(self.cm[:,label]) - tp
            fps += fp
            fn = sum(self.cm[label,:]) - tp
            fns += fn
            tn = sum(sum(self.cm)) - (sum(self.cm[:,label]) + sum(self.cm[label,:]))
            tns += tn        
        precision = self._precision(tps, fps)
        recall = self._recall(tps, fns)
        f1 = self._f1_score(precision, recall)

        return precision, recall, f1    