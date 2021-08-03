'''
Created on Apr 6, 2019

@author: Gias
'''

import os
import re
import pandas as pd 
import nltk
from nltk.stem.snowball import SnowballStemmer
from imblearn.over_sampling import SMOTE
from statistics import mean
import cPickle as pickle
import numpy as np
import argparse
import csv
from django.conf import settings
import utils.fileutils as fileutils
from utils import nlputils
import scipy as sp
from scipy.sparse import coo_matrix, hstack
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import utils.metrics as metrics
import sentiplus.DiversityMetrics as dm
from nltk.stem.snowball import SnowballStemmer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
import math
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
from sentiplus.Hybrid import Utils
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from multiprocessing import Process

stopWords = set(stopwords.words('english'))
stemmer =SnowballStemmer("english")

rules = {
    'VB_FEAT_POL->(ADBMOD_POS|ADBMOD_NEG)*',
    'ADBMOD*->(VB_FEAT_POS|VB_FEAT_NEG)',
    'ADBMOD*->(VB_FEAT_POS|VB_FEAT_NEG)',
    'AMOD->COMPOUND->DOBJ'
    }

vocabulary = {
    'VB_FEAT_POL': ['work'],
    'VB_FEAT_POS': ['work'],
    'ADBMOD_POS': ['ok', 'great', 'POSITIVE_POLAR_WORDS'],
    'ADBMOD_NEG': ['crap', 'NEGATIVE_POLAR_WORDS'],
    'ADBMOD': ['just'],
    
}

def matchPattern(textDeps, pattern):
    pattern = "VB_FEAT -> ADBMOD_POL"
    vocab = {}
    vocab["VB_FEAT"] = ["work"]
    vocab["ADBMOD_POL"] = ["fine"]
    for dep in textDeps:
        tag = dep["tag"]
        if tag == pattern[0]:
            word = dep["lemma"]
            if word in vocab[pattern[0]]:
                start = True
                continue
        if start == True:
            if tag == pattern[1]:
                if word in vocab[pattern[1]]:
                    found = True
