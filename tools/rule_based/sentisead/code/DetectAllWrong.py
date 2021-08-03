'''
Created on Mar 22, 2019

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

def replace_all(text, dic):
    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

stemmer =SnowballStemmer("english")

def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens)
    return stems

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

def preprocess_text(text):
    if text is None: return []
    #comments = text.encode('ascii', 'ignore')
    text = text.lower()
    text = text.replace("\\", "")
    comments = text
    #comments = detectSentimentDominentClause(comments)
    return  comments


class Utils(object):
    
    def __init__(self):
        pass
            
    def prepareTrainTestFiles7030(self, infile, outdir):
        
        df = fileutils.readExcel(infile, "Sheet1", encoding="ISO-8859-1")
        folds = dict()

        for index, row in df.iterrows():
            uid = row["UID"]
            fname = row["File"].split("_")[0]
            if fname not in folds:
                folds[fname] = dict()
            fold = row["File"].split("_")[-1]
            if fold not in folds[fname]:
                folds[fname][fold] = []
            folds[fname][fold].append(row)
        for fname in folds:
            trainData = folds[fname]["Train"]
            outfileTrain = fname+"_Train.xls"
            outfileTrain = os.path.join(outdir, outfileTrain)
            dfTrain = pd.DataFrame.from_dict(trainData)
            dfTrain.to_excel(outfileTrain, index=False, encoding="ISO-8859-1")

            testData = folds[fname]["Test"]
            outfileTest = fname+"_Test.xls"
            outfileTest = os.path.join(outdir, outfileTest)
            dfTest = pd.DataFrame.from_dict(testData)
            dfTest.to_excel(outfileTest, index=False, encoding="ISO-8859-1")

    def trainTestClassifier7030(self, algoName, outdir, featCols, filenames):
        
        dirResult = os.path.join(outdir, "DetectotAllWrong_"+algoName)
        fileutils.make_dir(dirResult)
        for filename in filenames:
            infile = filename+"_Train.xls"
            infile = os.path.join(outdir, infile)
            outfileModel = filename+"_Train_model.pkl"
            outfileModel = os.path.join(dirResult, outfileModel)
            clf = Classifier(infile, outfileModel, featCols, algo=algoName)
            infileTest = filename+"_Test.xls"
            infileTest = os.path.join(outdir, infileTest)
            dfTest = pd.read_excel(infileTest, sheet_name="Sheet1") #pd.read_csv(infileTest, encoding="ISO-8859-1")
            results = []
            for index, row in dfTest.iterrows():
                text = row["Sentence"]
                additionalColVals = []
                for col in featCols:
                    additionalColVals.append(row[col])
                label = clf.get_label(text, additionalColVals)[0]
                results.append(label)
            dfTest["DetectorAllWrong"] = (pd.Series(results)).values
            outfileResults = filename+"_Test_Results.xls"
            outfileResults = os.path.join(dirResult, outfileResults)
            dfTest.to_excel(outfileResults, index = False, encoding="ISO-8859-1")
                
    def consolidateResults7030(self, algo, outdir, filenames, dirConsolidated):
        #algos = ["RF", "ADB", "GBT"]
        records = []
        dirResults = os.path.join(outdir, "DetectotAllWrong_"+algo)
        for filename in filenames:
            fid = filename + "_Test"
            infile = fid+"_Results.xls"
            infile = os.path.join(dirResults, infile)
            df = pd.read_excel(infile, encoding="ISO-8859-1", sheet_name="Sheet1")
            for index, row in df.iterrows():
                records.append(row)
                        
        # now append to existing consolidated        
        outfile = os.path.join(dirConsolidated, "ResultsConsolidated_"+algo+".xls")
        df = pd.DataFrame.from_dict(records)
        fileutils.writeExcel(outfile, "Sheet1", df)   

    
    def computePerformanceOverallOfLearner(self, algo, dirConsolidated, filenames):
        learnerCol = "DetectorAllWrong"
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
                exp = row["AllWrong"]
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


    def computePerformancOfLearner(self, algo, dirConsolidated, filenames):
        learnerCol = "DetectorAllWrong"
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
            exp = row["AllWrong"]
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


class DetectorAllWrong(object):
    
    def __init__(self):
        self.basedir = r"C:\dev\opinion\papers\sentiplus\sentisead2"
        self.featCols =  [
                                'DsoLabelFullText_HotEncoded',
                                #'Pscore_FullTest',
                                #'Nscore_FullText',
                                'Senti4SD_HotEncoded', 
                                'SentiCR_HotEncoded', 
                                'SentistrengthSE_HotEncoded',
                                #"ShannonPolarOverall",
                                #"ShannonPolarPositive",
                                #"ShannonPolarNegative",
                                #"ShannonVerb",
                                #"ShannonAdjective"
                                ]
        
        self.indir = self.basedir
        self.infile = os.path.join(self.indir, "Consolidated.xls")

        self.filenames = ["BenchmarkUddinSO", "DatasetLinAppReviews", "DatasetLinJIRA", 
                     "DatasetLinSO", 
                     "DatasetSenti4SDSO"]
        
        self.outdir = os.path.join(self.basedir, "AllWrong")
        fileutils.make_dir(self.outdir)
        self.dirConsolidated = os.path.join(self.outdir, "consolidated")
        fileutils.make_dir(self.dirConsolidated)
        self.utils = Utils()

    def prepareTrainTestFiles(self):
        self.utils.prepareTrainTestFiles7030(self.infile, self.outdir)
        
    def trainTestClassifier(self, algoName):
        self.utils.trainTestClassifier7030(algoName, self.outdir, self.featCols, self.filenames)

    def consolidateResults(self, algo):
        self.utils.consolidateResults7030(algo, self.outdir, self.filenames, self.dirConsolidated)

    def computePerformanceOverallOfLearner(self, algo):
        self.utils.computePerformanceOverallOfLearner(algo, self.dirConsolidated, self.filenames)
            
    def computePerformancOfLearner(self, algo):
        self.utils.computePerformancOfLearner(algo, self.dirConsolidated, self.filenames)


class InputData:
    def __init__(self, text,rating):
        self.text = text
        self.rating =rating

class Classifier:
    def __init__(self, infileTraining, infileModel, featCols, training=True, encoding = 'ISO-8859-1',
                 infileSheetName = "Sheet1", infileSentCol = "Sentence", infileRatingCol = "AllWrong",
                 algo="GBT"):
        self.additionalCols = featCols
        self.algo = algo
        #self.indir = "/home/gias/dev/opinion/papers/opinionvalue/SentiCR"
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, sublinear_tf=True, max_df=0.5,
                                     stop_words=mystop_words, min_df=3)
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
                    comments = preprocess_text(sentidata.text)
                    training_comments.append(comments)
                self.vectorizer.fit_transform(training_comments).toarray()
        #self.model = self.create_model_from_training_data()
        # discard stopwords, apply stemming, and discard words present in less than 3 comments

    def get_classifier(self):
        algo=self.algo

        if algo=="GBT":
            return GradientBoostingClassifier(learning_rate=0.1, n_estimators=500,max_depth=10, min_samples_split=100, 
                                         min_samples_leaf=20, subsample=0.85, random_state=10)
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
            comments = preprocess_text(sentidata.text)
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

    def read_data_from_oracle_pd(self, sheetName="Sheet1", sentCol="Sentence", ratingCol ="AllWrong_HotEncoded"):
        print("Reading data from oracle..")
        oracle_data=[]
        if self.trainingFile.endswith(".csv") == False:
            self.train_df = fileutils.readExcel(self.trainingFile, sheetName, encoding = self.encoding)
        else:
            self.train_df = pd.read_csv(self.trainingFile, encoding = self.encoding)
        for index, row in self.train_df.iterrows():
            text = row[sentCol]
            rating = row[ratingCol]
            comments = InputData(text, rating)
            oracle_data.append(comments)
        return  oracle_data

    def get_label(self,text, additionalColVals):
        comment=preprocess_text(text)
        feature_vector= hstack((self.vectorizer.transform([comment]),
                      additionalColVals),
                      format='csr')
        feature_vector = feature_vector.toarray()
        label = self.model.predict(feature_vector)
        return label
