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
from sentiplus.Hybrid import Utils
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))
stemmer =SnowballStemmer("english")

mystop_words=[
'i', 'me', 'my', 'myself', 'we', 'our',  'ourselves', 'you', 'your',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'themselves',
 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
'and',  'if', 'or', 'as', 'until',  'of', 'at', 'by',  'between', 'into',
'through', 'during', 'to', 'from', 'in', 'out', 'on', 'off', 'then', 'once', 'here',
 'there',  'all', 'any', 'both', 'each', 'few', 'more',
 'other', 'some', 'such',  'than', 'too', 'very', 's', 't', 'can', 'will',  'don', 'should', 'now'
# keywords
 'while', 'case', 'switch','def', 'abstract','byte','continue','native','private','synchronized',
 'if', 'do', 'include', 'each', 'than', 'finally', 'class', 'double', 'float', 'int','else','instanceof',
 'long', 'super', 'import', 'short', 'default', 'catch', 'try', 'new', 'final', 'extends', 'implements',
 'public', 'protected', 'static', 'this', 'return', 'char', 'const', 'break', 'boolean', 'bool', 'package',
 'byte', 'assert', 'raise', 'global', 'with', 'or', 'yield', 'in', 'out', 'except', 'and', 'enum', 'signed',
 'void', 'virtual', 'union', 'goto', 'var', 'function', 'require', 'print', 'echo', 'foreach', 'elseif', 'namespace',
 'delegate', 'event', 'override', 'struct', 'readonly', 'explicit', 'interface', 'get', 'set','elif','for',
 'throw','throws','lambda','endfor','endforeach','endif','endwhile','clone'
]
for w in mystop_words:
    stopWords.add(w)
stopWords = list(stopWords)

def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens)
    return stems
class SentimentData_SentiCRCustomized:
    def __init__(self, text,rating):
        self.text = text
        self.rating =rating

class SentiCRCustomized:
    def __init__(self, infileTraining, infileModel, featCols, training=True, encoding = 'ISO-8859-1',
                 infileSheetName = "Sheet1", infileSentCol = "Sentence", infileRatingCol = "ManualLabel_HotEncoded",
                 algo="GBT"):
        self.additionalCols = featCols
        self.algo = algo
        #self.indir = "/home/gias/dev/opinion/papers/opinionvalue/SentiCR"
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, sublinear_tf=True, max_df=0.5,
                                     stop_words=stopWords, min_df=3)
        #modelFile = infileTraining.split('_Train')[0]+"_"+algo+".pkl"
        self.modelFile = infileModel #os.path.join(dirTrainedModelsOriginal, modelFile) #os.path.join(self.indir, "crpolar.pkl")
        self.trainingFile = infileTraining#os.path.join(dirTrainedModelsOriginal, infileTraining)
        self.encoding = encoding
        print ("Algo = ", algo)
        if training == True:
            print("Training ....")
            self.training_data=self.read_data_from_oracle_pd(infileSheetName, infileSentCol, infileRatingCol)
            self.model = self.create_model_from_training_data()
            print("saving model ", self.modelFile)
            with open(self.modelFile, 'wb') as f:
                pickle.dump(self.model, f)
        else:
            with open(self.modelFile, 'rb') as f:
                self.model = pickle.load(f)
                training_comments=[]
                self.training_data=self.read_data_from_oracle_pd(infileSheetName, infileSentCol, infileRatingCol)
                for sentidata in self.training_data:
                    comments = Utils.preprocess_text(sentidata.text)
                    training_comments.append(comments)
                self.vectorizer.fit_transform(training_comments).toarray()
        #self.model = self.create_model_from_training_data()
        # discard stopwords, apply stemming, and discard words present in less than 3 comments

    def get_classifier(self):
        algo=self.algo

        if algo=="GBT":
            return GradientBoostingClassifier(learning_rate=0.1, n_estimators=500,max_depth=10, min_samples_split=100, 
                                         min_samples_leaf=20, subsample=0.85, random_state=10)
        if algo=="GBTSentiCR":
            return GradientBoostingClassifier()
        elif algo=="RF":
            return  RandomForestClassifier()
        elif algo=="ADB":
            return AdaBoostClassifier()
        elif algo =="DT":
            return  DecisionTreeClassifier()
        elif algo=="NB":
            return  BernoulliNB()
        elif algo=="SGD":
            return  SGDClassifier()
        elif algo=="SVC":
            return LinearSVC()
        elif algo=="MLPC":
            return MLPClassifier(activation='logistic',  batch_size='auto',
            early_stopping=True, hidden_layer_sizes=(100,), learning_rate='adaptive',
            learning_rate_init=0.1, max_iter=5000, random_state=1,
            solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
            warm_start=False)
        return 0

    def create_model_from_training_data(self):
        training_comments=[]
        training_ratings=[]
        print("Training classifier model..")
        for sentidata in self.training_data:
            comments = Utils.preprocess_text(sentidata.text)
            training_comments.append(comments)
            training_ratings.append(sentidata.rating)
        X = hstack((self.vectorizer.fit_transform(training_comments),
                      self.train_df[self.additionalCols].values),
                      format='csr')
        #X_train = self.vectorizer.fit_transform(training_comments).toarray()
        X_train = X.toarray()
        Y_train = np.array(training_ratings)

        #Apply SMOTE to improve ratio of the minority class
        smote_model = SMOTE(ratio=0.5, random_state=None, k=None, k_neighbors=10, m=None, m_neighbors=10, out_step=.0001,
                   kind='regular', svm_estimator=None, n_jobs=1)
        model=self.get_classifier()
        try:
            X_resampled, Y_resampled=smote_model.fit_sample(X_train, Y_train)
            model.fit(X_resampled, Y_resampled)
        except:
            model.fit(X_train, Y_train)

        
        
        #model.fit(X_train, Y_train)

        return model

    def read_data_from_oracle_pd(self, sheetName="Sheet1", sentCol="Sentence", ratingCol ="ManualLabel_HotEncoded"):
        print("Reading data from oracle..")
        oracle_data=[]
        if self.trainingFile.endswith(".csv") == False:
            self.train_df = fileutils.readExcel(self.trainingFile, sheetName, encoding = self.encoding)
        else:
            self.train_df = pd.read_csv(self.trainingFile, encoding = self.encoding)
        for index, row in self.train_df.iterrows():
            text = row[sentCol]
            rating = row[ratingCol]
            comments = SentimentData_SentiCRCustomized(text, rating)
            oracle_data.append(comments)
        return  oracle_data

    def get_sentiment_polarity(self,text, additionalColVals):
        comment=Utils.preprocess_text(text)
        #print (text)
        #print (comment)
        #print ("-----------")
        feature_vector= hstack((self.vectorizer.transform([comment]),
                      additionalColVals),
                      format='csr')
        feature_vector = feature_vector.toarray()
        #feature_vector=self.vectorizer.transform([comment]).toarray()
        sentiment_class=self.model.predict(feature_vector)
        return sentiment_class
