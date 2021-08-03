'''
Created on Jul 5, 2019

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
import nltk.data
from nltk.corpus import stopwords
from gensim.models import word2vec

stopWords = set(stopwords.words('english'))


# This function converts a text to a sequence of words.
def review_wordlist(review, remove_stopwords=False):
    # 1. Removing html tags
    review_text = review #BeautifulSoup(review).get_text()
    # 2. Removing non-letter.
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    # 3. Converting to lower case and splitting
    words = review_text.lower().split()
    # 4. Optionally remove stopwords
    if remove_stopwords:
        words = [w for w in words if not w in stopWords]
    
    return(words)

# This function splits a review into sentences
def review_sentences(review, remove_stopwords=False):
    #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # 1. Using nltk tokenizer
    #raw_sentences = tokenizer.tokenize(review.strip())
    raw_sentences = sent_tokenize(review)
    sentences = []
    # 2. Loop for each sentence
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_wordlist(raw_sentence,\
                                            remove_stopwords))

    # This returns the list of lists
    return sentences

# Function to average all word vectors in a paragraph
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    # Dividing the result by number of words to get average
    #print featureVec
    if nwords is None or nwords == 0: 
        nwords = 1
    featureVec = np.divide(featureVec, nwords)
    return featureVec

# Function for calculating the average feature vector
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        # Printing a status message every 1000th review
        #if counter%1000 == 0:
        #    print("Review %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1
        
    return reviewFeatureVecs

def get_classifier(algo):
        if algo=="GBT":
            return GradientBoostingClassifier(learning_rate=0.1, n_estimators=500,max_depth=10, min_samples_split=100, 
                                         min_samples_leaf=20, subsample=0.85, random_state=10)
        if algo=="GBTSentiCR":
            return GradientBoostingClassifier()
        elif algo=="RF":
            return  RandomForestClassifier( n_estimators=100)
        elif algo=="ADB":
            return AdaBoostClassifier()
        elif algo =="DT":
            return  DecisionTreeClassifier()
        elif algo=="NB":
            return  BernoulliNB()
        elif algo=="SGD":
            return  SGDClassifier()
        elif algo=="SVC":
            return LinearSVC(C=1.0, loss = "hinge", max_iter=1000, penalty="l2")
            #return LinearSVC()
        elif algo == "SGD":
            return SGDClassifier(alpha=.0001, n_iter=2000, 
                                          epsilon=0.5, loss='log',penalty="l2", 
                                          power_t=0.5, warm_start=False, shuffle=True),
        elif algo == "LogisticRegression":
            return LogisticRegression()
        elif algo=="MLPC":
            return MLPClassifier(activation='logistic',  batch_size='auto',
            early_stopping=True, hidden_layer_sizes=(100,), learning_rate='adaptive',
            learning_rate_init=0.1, max_iter=5000, random_state=1,
            solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
            warm_start=False)
        return 0



class CommonUtils(object):
    
    def __init__(self):
        pass
            
    def prepareTrainTestFiles(self, infile, outdir):
        
        df = fileutils.readExcel(infile, "Sheet1", encoding="ISO-8859-1")
        folds = dict()

        for index, row in df.iterrows():
            uid = row["UID"]
            fname = row["File"].split("_")[0]
            if fname not in folds:
                folds[fname] = dict()
            fold = row["File"].split("_")[-1]
            fold = int(fold)
            if fold not in folds[fname]:
                folds[fname][fold] = []
            folds[fname][fold].append(row)
        for fname in folds:
            for i in range(10):
                test = i 
                outfileTest = fname+"_Test_"+str(test)+".csv"
                outfileTest = os.path.join(outdir, outfileTest)
                testData = folds[fname][test]
                records = []
                for row in testData:
                    records.append(row)
                df = pd.DataFrame.from_dict(records)
                df.to_csv(outfileTest, encoding="ISO-8859-1", index=False)
                outfileTrain = fname+"_Train_"+str(test)+".csv"
                outfileTrain = os.path.join(outdir, outfileTrain)
                records = []
                for j in range(10):
                    if i == j: continue
                    trainData = folds[fname][j]
                    for row in trainData:
                        records.append(row)
                df = pd.DataFrame.from_dict(records)
                df.to_csv(outfileTrain, encoding="ISO-8859-1", index=False)

    def run(self, filename, fold, dirResult, outdir, featCols, algoName, w2vModel="Word2Vec"):
        print (filename, fold)
        infile = filename+"_Train_"+str(fold)+".csv"
        infile = os.path.join(outdir, infile)
        outfileModel = filename+"_Train_"+str(fold)+"_model.pkl"
        outfileModel = os.path.join(dirResult, outfileModel)
        
        outfileW2VModel = filename+"_"+w2vModel+"_"+str(fold)+"_model.pkl"
        if w2vModel == "Word2Vec":
            senticr = SupervisedDetectorWord2Vec(infile, outfileModel, outfileW2VModel, featCols, algoName)
        infileTest = filename+"_Test_"+str(fold)+".csv"
        infileTest = os.path.join(outdir, infileTest)
        dfTest = pd.read_csv(infileTest, encoding="ISO-8859-1")
        results = []
        for index, row in dfTest.iterrows():
            text = row["Sentence"]
            additionalColVals = []
            for col in featCols:
                additionalColVals.append(row[col])
            label = senticr.get_sentiment_polarity(text, additionalColVals)[0]
            if label == 1:
                label = "p"
            elif label == -1:
                label = "n"
            elif label == 0:
                label = "o"
            else:
                label = "WTF"
            results.append(label)
        dfTest["DSOW2V"] = (pd.Series(results)).values
        outfileResults = filename+"_Test_"+str(fold)+"_Results.csv"
        outfileResults = os.path.join(dirResult, outfileResults)
        dfTest.to_csv(outfileResults, index = False, encoding="ISO-8859-1")
    def trainTestSentiCRCustomized(self, algoName, outdir, featCols, filenames, parallelized=False):
        dirResult = os.path.join(outdir, "DSOW2V_"+algoName)
        fileutils.make_dir(dirResult)
        #outdir = os.path.join(rootdir, "results_senticr")
        folds = 10
        for filename in filenames:
            ps = []
            for i in range(folds):
                p = Process(target=self.run, args = (filename, i, dirResult, outdir, featCols, algoName))
                ps.append(p)
            if parallelized == True:
                for p in ps:
                    p.start()
                for p in ps:
                    p.join()
            else:    
                for p in ps:
                    p.start()
                    p.join()

    def consolidateResults(self, algo, outdir, filenames, dirConsolidated):
        #algos = ["RF", "ADB", "GBT"]
        records = []
        dirResults = os.path.join(outdir, "DSOW2V_"+algo)
        folds = 10
        for filename in filenames:
            for i in range(folds):
                fid = filename + "_Test_"+str(i)
                infile = fid+"_Results.csv"
                infile = os.path.join(dirResults, infile)
                df = pd.read_csv(infile, encoding="ISO-8859-1")
                for index, row in df.iterrows():
                    records.append(row)
                        
        # now append to existing consolidated        
        outfile = os.path.join(dirConsolidated, "ResultsConsolidated_"+algo+".xls")
        df = pd.DataFrame.from_dict(records)
        fileutils.writeExcel(outfile, "Sheet1", df)   

    def computePerformanceOverallOfLearner(self, algo, learnerCol, dirConsolidated, filenames):
        infile = os.path.join(dirConsolidated, "ResultsConsolidated_"+algo+".xls")
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
                exp = row["ManualLabel"]
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


    def computePerformancOfLearner(self, algo, learnerCol, dirConsolidated, filenames):
        infile = os.path.join(dirConsolidated, "ResultsConsolidated_"+algo+".xls")
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
            exp = row["ManualLabel"]
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
                print "File %s. Label = %s. Precision = %.2f. Recall = %.2f. F1 = %.2f"%(fname, label, pr, re, f1)
            f1_macro = computer.f1_macro_average()
            f1_micro, _, _ = computer.compute_micro_average()
            print "File = %s. F1 Macro = %.2f. Micro = %.2f"%(fname, f1_macro, f1_micro)
            print "-------------------------------"



class DSOW2V(object):
    
    def __init__(self, rootdir):
        
        self.basedir = os.path.join(rootdir, "Hybrid")
        self.featCols =  [
                                'DsoLabelFullText_HotEncoded',
                                'Pscore_FullText',
                                'Nscore_FullText',
                                #'DsoLabelFirstWord_HotEncoded',
                                #'DsoLabelLastWord_HotEncoded',
                                #"Pscore_FirstPolar",
                                #"Pscore_LastPolar",
                                #"Nscore_FirstPolar",
                                #"Nscore_LastPolar"
                                ]
        
        self.indir = self.basedir
        self.infile = os.path.join(self.indir, "ConsolidatedWithBestTrainedClfsFromThreeSETools.xls")

        self.filenames = ["DatasetLinJIRA", "BenchmarkUddinSO", "DatasetLinAppReviews", 
                     "DatasetLinSO", "DatasetSenti4SDSO", "OrtuJIRA"]
        
        self.outdir = os.path.join(self.basedir, "DSOW2V")
        fileutils.make_dir(self.outdir)
        self.dirConsolidated = os.path.join(self.outdir, "consolidated")
        fileutils.make_dir(self.dirConsolidated)
        self.utils = CommonUtils()

    def prepareTrainTestFiles(self):
        self.utils.prepareTrainTestFiles(self.infile, self.outdir)
        
    def trainTestSupervisedDetector(self, algoName, parallelized=False):
        self.utils.trainTestSentiCRCustomized(algoName, self.outdir, self.featCols, self.filenames, parallelized)

    def consolidateResults(self, algo):
        self.utils.consolidateResults(algo, self.outdir, self.filenames, self.dirConsolidated)

    def computePerformanceOverallOfLearner(self, algo, learnerCol):
        self.utils.computePerformanceOverallOfLearner(algo, learnerCol, self.dirConsolidated, self.filenames)
            
    def computePerformancOfLearner(self, algo, learnerCol):
        self.utils.computePerformancOfLearner(algo, learnerCol, self.dirConsolidated, self.filenames)

    def pipeline(self, algo, parallelized=False):
        learnerCol = "DSOW2V"
        algoSpec = get_classifier(algo)
        self.trainTestSupervisedDetector(algo, parallelized)
        self.consolidateResults(algo)

        print algoSpec 
        print "-"*80
        
        print "Overall Performance"
        self.computePerformanceOverallOfLearner(algo, learnerCol)
        print "-"*80
        print "By File Performance"
        print "-"*80
        self.computePerformancOfLearner(algo, learnerCol)
        


def preprocess_text(text):
   
    if text is None: return []
    #comments = text.encode('ascii', 'ignore')
    text = text.lower()
    text = text.replace("\\", "")
    comments = text
    #comments = Utils.expand_contractions(comments)
    comments = Utils.remove_url(comments)
    comments = Utils.replace_all(comments, Utils.emodict)
    #comments = tweet_cleaner_updated(comments)
    #comments = Utils.handle_negation(comments)
    #comments = detectSentimentDominentClause(comments)    return  comments
    #print comments
    return comments 
class SentimentData:
    def __init__(self, text,rating):
        self.text = text
        self.rating =rating


class SupervisedDetectorWord2Vec(object):
    
    def __init__(self, infile, outfileModel, outfileW2VModel, featCols, algo, training=True,
                 infileSheetName = "Sheet1", infileSentCol = "Sentence", infileRatingCol = "ManualLabel_HotEncoded"):
        self.trainingFile = infile
        self.num_features = 300
        self.context = 10        # Context window size
        self.min_word_count = 40 # Minimum word count
        self.num_workers = 4     # Number of parallel threads
        self.downsampling = 1e-3 # (0.001) Downsample setting for frequent words
        #self.clf = get_classifier(algo)
        self.outfileW2VModel = outfileW2VModel
        self.modelFile = outfileModel
        self.additionalCols = featCols
        self.algo = algo
        if training:
            self.training_data=self.read_data_from_oracle_pd(self.trainingFile, infileSheetName, infileSentCol, infileRatingCol)
            self.model = self.create_model_from_training_data()
            print("saving model ", self.modelFile)
            with open(self.modelFile, 'wb') as f:
                pickle.dump(self.model, f)
        else:
            with open(self.modelFile, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.outfileW2VModel, 'rb') as f:
                self.outfileW2VModel = pickle.load(f)
            

    def trainWord2VecModel(self, sentences):
        # Creating the model and setting values for the various parameters
        
        # Initializing the train model
        
        print("Training w2v model....")
        self.w2vModel = word2vec.Word2Vec(sentences,\
                                  workers=self.num_workers,\
                                  size=self.num_features,\
                                  min_count=self.min_word_count,\
                                  window=self.context,
                                  sample=self.downsampling)
        
        # To make the model memory efficient
        self.w2vModel.init_sims(replace=True)
        
        # Saving the model for later use. Can be loaded using Word2Vec.load()
        #model_name = #fname+"_"+str(self.num_features)+"features_"+self.min_word_count+"minwords_"+self.context+"context"
        #model_name = os.path.join(self.outdir, model_name)
        self.w2vModel.save(self.outfileW2VModel)

    def read_data_from_oracle_pd(self, trainingFile, sheetName="Sheet1", sentCol="Sentence", ratingCol ="ManualLabel_HotEncoded"):
        print("Reading data from oracle..")
        oracle_data=[]
        if self.trainingFile.endswith(".csv") == False:
            self.train_df = fileutils.readExcel(trainingFile, sheetName, encoding = "ISO-8859-1")
        else:
            self.train_df = pd.read_csv(self.trainingFile, encoding = "ISO-8859-1")
        for index, row in self.train_df.iterrows():
            text = row[sentCol]
            rating = row[ratingCol]
            comments = SentimentData(text, rating)
            oracle_data.append(comments)
        return  oracle_data

    
    def create_model_from_training_data(self):
        
        training_comments=[]
        training_ratings=[]
        print("Training classifier model..")
        for sentidata in self.training_data:
            comments = preprocess_text(sentidata.text)
            training_comments.append(comments)
            training_ratings.append(sentidata.rating)
        sentences = []
        for comment in training_comments:
            #print comment
            sentences += review_sentences(comment, remove_stopwords=False)
        #print "sentences = ", sentences
        self.trainWord2VecModel(sentences)
        trainComments = []
        for comment in training_comments:
            trainComments.append(review_wordlist(comment, remove_stopwords=False))
        trainDataVecs = getAvgFeatureVecs(trainComments, self.w2vModel, self.num_features)
        X = trainDataVecs
        #hstack((trainDataVecs,                      self.train_df[self.additionalCols].values),                      format='csr')
        #X_train = self.vectorizer.fit_transform(training_comments).toarray()
        X_train = X#X.toarray()
        Y_train =training_ratings #np.array(training_ratings)

        #Apply SMOTE to improve ratio of the minority class
        #smote_model = SMOTE(sampling_strategy=0.5, random_state=None, k_neighbors=10, m_neighbors=10, out_step=.0001, kind='regular', svm_estimator=None, n_jobs=1)
        
        smote_model = SVMSMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=object, 
                               m_neighbors=object, out_step=.0001,n_jobs=1, svm_estimator=None)
        #smote_model = SMOTE(sampling_strategy=0.5)
        
        model= get_classifier(self.algo)
        #model.fit(X_train, Y_train)
        try:
            X_resampled, Y_resampled=smote_model.fit_sample(X_train, Y_train)
            model.fit(X_resampled, Y_resampled)
        except:
            model.fit(X_train, Y_train)

        
        
        #model.fit(X_train, Y_train)

        return model
    
    def get_sentiment_polarity(self,text, additionalColVals):
        comment= preprocess_text(text)
        words = review_wordlist(comment, remove_stopwords=False)
        testDataVecs = getAvgFeatureVecs(words, self.w2vModel, self.num_features)
        feature_vector= testDataVecs
        #hstack((testDataVecs,
        #              additionalColVals),
        #              format='csr')
        #feature_vector = feature_vector.toarray()
        try:
            sentiment_class=self.model.predict(feature_vector)
        except:
            sentiment_class = ['o']
        return sentiment_class
    
    