'''
Created on Mar 23, 2019

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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import utils.metrics as metrics
import sentiplus.DiversityMetrics as dm
from nltk.stem.snowball import SnowballStemmer
from imblearn.over_sampling import SMOTE
import math
from nltk.tokenize import sent_tokenize, word_tokenize

MAPVALS = {
            'p': 1,
            'n': -1,
            'o': 0,
            'E': 1,
            'R': 2,
            'D': 3,
            'N': 4,
            }

def replace_all(text, dic):
    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text






#logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s %(levelname)s %(message)s')


emodict=[]
contractions_dict=[]


rootdir = settings.PAPER_DIR["sentiplus"]
dirAdaptive = os.path.join(rootdir, "adaptive")
dirHybrid = os.path.join(rootdir, "Hybrid")
#dir = os.path.join(rootdir, "ModelSentiCR")
#dirDictionary = os.path.join(dirModel, "Dictionary")
#dirDictionaryOriginal = os.path.join(dirDictionary, "Original")
#dirTrainedModels = os.path.join(dirModel, "TrainedModels")
#dirTrainedModelsOriginal = os.path.join(dirTrainedModels, "Original")

dirSentimentLexicons = os.path.join(rootdir, "senti_lexicons")

infileContractions = os.path.join(dirSentimentLexicons, "Contractions.txt")
infileEmotionLookup = os.path.join(dirSentimentLexicons, "EmoticonLookupTable.txt")

# Read in the words with sentiment from the dictionary
with open(infileContractions,"r") as contractions,\
     open(infileEmotionLookup,"r") as emotable:
    contractions_reader=csv.reader(contractions, delimiter='\t')
    emoticon_reader=csv.reader(emotable,delimiter='\t')

    #Hash words from dictionary with their values
    contractions_dict = {rows[0]:rows[1] for rows in contractions_reader}
    emodict={rows[0]:rows[1] for rows in emoticon_reader}

    contractions.close()
    emotable.close()



grammar= r"""
NegP: {<VERB>?<ADV>+<VERB|ADJ>?<PRT|ADV><VERB>}
{<VERB>?<ADV>+<VERB|ADJ>*<ADP|DET>?<ADJ>?<NOUN>?<ADV>?}

"""
chunk_parser = nltk.RegexpParser(grammar)


contractions_regex = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
     def replace(match):
         return contractions_dict[match.group(0)]
     return contractions_regex.sub(replace, s.lower())


url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def remove_url(s):
    return url_regex.sub(" ",s)

negation_words =['not', 'never', 'none', 'nobody', 'nowhere', 'neither', 'barely', 'hardly',
                     'nothing', 'rarely', 'seldom', 'despite' ]

emoticon_words=['PositiveSentiment','NegativeSentiment']

def negated(input_words):
    """
    Determine if input contains negation words
    """
    neg_words = []
    neg_words.extend(negation_words)
    for word in neg_words:
        if word in input_words:
            return True
    return False

def prepend_not(word):
    if word in emoticon_words:
        return word
    elif word in negation_words:
        return word
    return "NOT_"+word

def handle_negation(comments):
    sentences = nltk.sent_tokenize(comments)
    modified_st=[]
    for st in sentences:
        allwords = nltk.word_tokenize(st)
        modified_words=[]
        if negated(allwords):
            part_of_speech = nltk.tag.pos_tag(allwords,tagset='universal')
            chunked = chunk_parser.parse(part_of_speech)
            #print("---------------------------")
            #print(st)
            for n in chunked:
                if isinstance(n, nltk.tree.Tree):
                    words = [pair[0] for pair in n.leaves()]
                    #print(words)

                    if n.label() == 'NegP' and negated(words):
                        for i, (word, pos) in enumerate(n.leaves()):
                            if (pos=="ADV" or pos=="ADJ" or pos=="VERB") and (word!="not"):
                                modified_words.append(prepend_not(word))
                            else:
                                modified_words.append(word)
                    else:
                         modified_words.extend(words)
                else:
                    modified_words.append(n[0])
            newst =' '.join(modified_words)
            #print(newst)
            modified_st.append(newst)
        else:
            modified_st.append(st)
    return ". ".join(modified_st)

def preprocess_text(text):
    if text is None: return []
    #comments = text.encode('ascii', 'ignore')
    text = text.lower()
    text = text.replace("\\", "")
    comments = text
    comments = expand_contractions(comments)
    comments = remove_url(comments)
    comments = replace_all(comments, emodict)
    #comments = handle_negation(comments)
    #comments = detectSentimentDominentClause(comments)
    return  comments

def computePerformanceOverallOfLearner(learnerCol, infile, filenames, truthCol="ManualLabel"):
        #infile = os.path.join(dirConsolidated, "ResultsConsolidated_"+algo+".xls")
        df = fileutils.readExcel(infile, "Sheet1", encoding="ISO-8859-1")
        exps = []
        gots = []
        labels = set()
        for index, row in df.iterrows():
            fname = row["File"]
            fname = fname.split("_")[0]
            if fname not in filenames:
                #print fname, " not in filenmaes"
                #return
                continue
            else:
                exp = row[truthCol]
                got = row[learnerCol]
                labels.add(exp)
                exps.append(exp)
                gots.append(got)
        computer = metrics.PerformanceMultiClass(exps, gots, labels = list(labels))
        for label in labels:
            pr = computer.precision(label)
            re = computer.recall(label)
            f1 = 2*pr*re/(pr+re)
            print "Label = %s. Precision = %.3f. Recall = %.3f. F1 = %.3f"%(label, pr, re, f1)
        f1_macro = computer.f1_macro_average()
        pr_macro = computer.precision_macro_average()
        rec_macro = computer.recall_macro_average()
        f1_micro, _, _ = computer.compute_micro_average()
        print "F1 Macro = %.3f. Micro = %.3f"%(f1_macro, f1_micro)
        print "Macro Precision = %.3f. Recall = %.3f"%(pr_macro, rec_macro)
        print "-------------------------------"


def computePerformancOfLearner(learnerCol, infile, filenames, truthCol="ManualLabel"):
        #infile = os.path.join(dirConsolidated, "ResultsConsolidated_"+algo+".xls")
        df = fileutils.readExcel(infile, "Sheet1", encoding="ISO-8859-1")
        
        
        exps = dict()
        gots = dict()
        labels = dict()
        for index, row in df.iterrows():
            fname = row["File"]
            fname = fname.split("_")[0]
            if fname not in filenames:
                #print fname, " not in filenmaes"
                continue
            else:
                if fname not in exps:
                    exps[fname] = []
                    gots[fname] = []
                    labels[fname] = set()
            exp = row[truthCol]
            got = row[learnerCol]
            labels[fname].add(exp)
            exps[fname].append(exp)
            gots[fname].append(got)
        for fname in filenames:
            computer = metrics.PerformanceMultiClass(exps[fname], gots[fname], labels = list(labels[fname]))
            for label in labels[fname]:
                pr = computer.precision(label)
                re = computer.recall(label)
                f1 = 2*pr*re/(pr+re)
                print "File %s. Label = %s. Precision = %.3f. Recall = %.3f. F1 = %.3f"%(fname, label, pr, re, f1)
            f1_macro = computer.f1_macro_average()
            f1_micro, _, _ = computer.compute_micro_average()
            print "File = %s. F1 Macro = %.3f. Micro = %.3f"%(fname, f1_macro, f1_micro)
            print "-------------------------------"