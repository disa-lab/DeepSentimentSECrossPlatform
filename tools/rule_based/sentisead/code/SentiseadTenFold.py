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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso
from sklearn import svm
import utils.metrics as metrics
import sentiplus.DiversityMetrics as dm
from nltk.stem.snowball import SnowballStemmer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
import math
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
# from sentiplus.Hybrid import Utils
import Utils

from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from multiprocessing import Process
from xgboost import XGBClassifier


stopWords = set(stopwords.words('english'))
stemmer =SnowballStemmer("english")
from multiprocessing import Process
#rootdir = r"C:\dev\opinion\papers\sentiplus"

def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text)
    #stems = stem_tokens(tokens)
    stems = tokens
    return stems

def preprocess_text(text, uid=0):
    if text is None: return []
    #comments = text.encode('ascii', 'ignore')
    # try:
    #     text2 = text.lower()
    # except:
    #     # print("===== Exception ===== UID: %d " + str(uid))
    #     # print(uid)
    #     # print(text)
    
    try:
        text = text.lower()
    except:
        text = "hi"
    text = text.replace("\\", "")
    comments = text
    comments = Utils.expand_contractions(comments)
    comments = Utils.remove_url(comments)
    comments = Utils.replace_all(comments, Utils.emodict)
    #comments = tweet_cleaner_updated(comments)
    #comments = Utils.handle_negation(comments)
    #comments = detectSentimentDominentClause(comments)    return  comments
    return comments 

def hotEncodeDSOSEValues(infile, outfile):
    df = pd.read_excel(infile, sheet_name="Sheet1")
    encodes = []
    for index, row in df.iterrows():
        label = row["DSOSE"]
        if label == "p":
            label = 1
        elif label == "n":
            label = -1
        else:
            label = 0
        encodes.append(label)
    df["DSOSE_HotEncoded"] = (pd.Series(encodes)).values 
    df.to_excel(outfile, sheet_name="Sheet1", encoding="ISO-8859-1") 

def hotEncodePOMEValues(infile, outfile):
    df = pd.read_excel(infile, sheet_name="Sheet1")
    encodes = []
    for index, row in df.iterrows():
        label = row["POME"]
        if label == "p":
            label = 1
        elif label == "n":
            label = -1
        else:
            label = 0
        encodes.append(label)
    df["POME_HotEncoded"] = (pd.Series(encodes)).values 
    df.to_excel(outfile, sheet_name="Sheet1", encoding="ISO-8859-1") 

def hotEncodeEnsembleRFValues(infile, outfile):
    df = pd.read_excel(infile, sheet_name="Sheet1")
    encodes = []
    for index, row in df.iterrows():
        label = row["Ensemble_RF"]
        if label == "p":
            label = 1
        elif label == "n":
            label = -1
        else:
            label = 0
        encodes.append(label)
    df["Ensemble_RF_HotEncoded"] = (pd.Series(encodes)).values 
    df.to_excel(outfile, sheet_name="Sheet1", encoding="ISO-8859-1") 


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

    def run(self, filename, fold, dirResult, outdir, featCols, algoName, ngram):
        print (filename, fold)
        infile = filename+"_Train_"+str(fold)+".csv"
        infile = os.path.join(outdir, infile)
        outfileModel = filename+"_Train_"+str(fold)+"_model.pkl"
        outfileModel = os.path.join(dirResult, outfileModel)
        senticr = SupervisedDetector(infile, outfileModel, featCols, algo=algoName, ngram_range=ngram)
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
            #print label
            if algoName == "LASSO":
                if label >= 0.5:
                    label = "p"
                elif label <= -0.5:
                    label = "n"
                else:
                    label = "o"
            else:
                if label == 1:
                    label = "p"
                elif label == -1:
                    label = "n"
                elif label == 0:
                    label = "o"
                else:
                    label = "WTF"
            results.append(label)
        dfTest["Sentisead"] = (pd.Series(results)).values
        outfileResults = filename+"_Test_"+str(fold)+"_Results.csv"
        outfileResults = os.path.join(dirResult, outfileResults)
        dfTest.to_csv(outfileResults, index = False, encoding="ISO-8859-1")
    def runNoBoW(self, filename, fold, dirResult, outdir, featCols, algoName, ngram):
        print (filename, fold)
        infile = filename+"_Train_"+str(fold)+".csv"
        infile = os.path.join(outdir, infile)
        outfileModel = filename+"_Train_"+str(fold)+"_model.pkl"
        outfileModel = os.path.join(dirResult, outfileModel)
        senticr = SupervisedDetectorNoBoW(infile, outfileModel, featCols, algo=algoName, ngram_range=ngram)
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
            #print label
            if algoName == "LASSO":
                if label >= 0.5:
                    label = "p"
                elif label <= -0.5:
                    label = "n"
                else:
                    label = "o"
            else:
                if label == 1:
                    label = "p"
                elif label == -1:
                    label = "n"
                elif label == 0:
                    label = "o"
                else:
                    label = "WTF"
            results.append(label)
        dfTest["Sentisead_NoBoW"] = (pd.Series(results)).values
        outfileResults = filename+"_Test_"+str(fold)+"_Results.csv"
        outfileResults = os.path.join(dirResult, outfileResults)
        dfTest.to_csv(outfileResults, index = False, encoding="ISO-8859-1")

    
    def trainTestSentiCRCustomized(self, algoName, ngram, outdir, featCols, filenames, parallelized=False):
        dirResult = os.path.join(outdir, "Sentisead_"+algoName)
        fileutils.make_dir(dirResult)
        print("method trainTestSentiCRCustomized output dir is created at: " + dirResult)
        #outdir = os.path.join(rootdir, "results_senticr")
        folds = 10
        for filename in filenames:
            ps = []
            for i in range(folds):
                p = Process(target=self.run, args = (filename, i, dirResult, outdir, featCols, algoName, ngram))
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

    def trainTestSentiCRCustomizedNoBoW(self, algoName, ngram, outdir, featCols, filenames, parallelized=False):
        dirResult = os.path.join(outdir, "Sentisead_NoBoW_"+algoName)
        fileutils.make_dir(dirResult)
        #outdir = os.path.join(rootdir, "results_senticr")
        folds = 10
        for filename in filenames:
            ps = []
            for i in range(folds):
                p = Process(target=self.runNoBoW, args = (filename, i, dirResult, outdir, featCols, algoName, ngram))
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
        dirResults = os.path.join(outdir, "Sentisead_"+algo)
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
        print("Consolidated result has been writen to " + outfile)     

    # def consolidateResultsNoBoW(self, algo, outdir, filenames, dirConsolidated):
    #     #algos = ["RF", "ADB", "GBT"]
    #     records = []
    #     dirResults = os.path.join(outdir, "Sentisead_NoBoW_"+algo)
    #     folds = 10
    #     for filename in filenames:
    #         for i in range(folds):
    #             fid = filename + "_Test_"+str(i)
    #             infile = fid+"_Results.csv"
    #             infile = os.path.join(dirResults, infile)
    #             df = pd.read_csv(infile, encoding="ISO-8859-1")
    #             for index, row in df.iterrows():
    #                 records.append(row)
                        
    #     # now append to existing consolidated        
    #     outfile = os.path.join(dirConsolidated, "ResultsConsolidated_NoBoW_"+algo+".xls")
    #     df = pd.DataFrame.from_dict(records)
    #     fileutils.writeExcel(outfile, "Sheet1", df) 

    def computePerformanceOverallOfLearner(self, algo, learnerCol, dirConsolidated, filenames):
        infile = os.path.join(dirConsolidated, "ResultsConsolidated_"+algo+".xls")
        self.computePerformanceOverallOfLearner(infile, learnerCol, filenames)


    def computePerformanceOverallOfLearner(self, infile, learnerCol, filenames):
        # infile = os.path.join(dirConsolidated, "ResultsConsolidated_"+algo+".xls")
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

    # def computePerformanceOverallOfLearnerNoBoW(self, algo, learnerCol, dirConsolidated, filenames):
    #     infile = os.path.join(dirConsolidated, "ResultsConsolidated_NoBoW_"+algo+".xls")
    #     df = fileutils.readExcel(infile, "Sheet1", encoding="ISO-8859-1")
    #     exps = []
    #     gots = []
    #     labels = set()
    #     for index, row in df.iterrows():
    #         fname = row["File"]
    #         fname = fname.split("_")[0]
    #         if fname not in filenames:
    #             #print fname, " not in filenmaes"
    #             #return
    #             continue
    #         else:
    #             exp = row["ManualLabel"]
    #             got = row[learnerCol]
    #             labels.add(exp)
    #             exps.append(exp)
    #             gots.append(got)
    #     computer = metrics.PerformanceMultiClass(exps, gots, labels = list(labels))
    #     for label in labels:
    #         pr = computer.precision(label)
    #         re = computer.recall(label)
    #         f1 = 2*pr*re/(pr+re)
    #         print "Label = %s. Precision = %.3f. Recall = %.3f. F1 = %.3f"%(label, pr, re, f1)
    #     f1_macro = computer.f1_macro_average()
    #     pr_macro = computer.precision_macro_average()
    #     rec_macro = computer.recall_macro_average()
    #     f1_micro, _, _ = computer.compute_micro_average()
    #     print "F1 Macro = %.3f. Micro = %.3f"%(f1_macro, f1_micro)
    #     print "Macro Precision = %.3f. Recall = %.3f"%(pr_macro, rec_macro)
    #     print "-------------------------------"
    
    
    # def computePerformancOfLearner(self, algo, learnerCol, dirConsolidated, filenames):
    #     infile = os.path.join(dirConsolidated, "ResultsConsolidated_"+algo+".xls")
    #     computePerformancOfLearner(infile, learnerCol, filenames)

    def computePerformancOfLearner(self, infile, learnerCol, filenames):
        # infile = os.path.join(dirConsolidated, "ResultsConsolidated_"+algo+".xls")
        print("Alamin edited")
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
            # print(computer)
            print(exps[fname][:50])
            print(gots[fname][:50])

            res = {"p": [0, 0, 0], "o": [0, 0, 0], "n": [0, 0, 0]}
            for label in labels[fname]:
                print("Going to call P R F1 for lable %s" % label)
                pr = computer.precision(label)
                re = computer.recall(label)
                f1 = 2*pr*re/(pr+re)
                print "File %s. Label = %s. Precision = %.2f. Recall = %.2f. F1 = %.2f"%(fname, label, pr, re, f1)
                res[label] = [pr, re, f1]
            f1_macro = computer.f1_macro_average()
            f1_micro, _, _ = computer.compute_micro_average()
            print "File = %s. F1 Macro = %.2f. Micro = %.2f"%(fname, f1_macro, f1_micro)
            print("For latex table: F1 macro micro followed by positive=>Neutral=>Negative Precision Recall F1")
            print("File = %s & %.2f & %.2f" %(fname, f1_macro, f1_micro)), 
            print("%.2f & %.2f & %.2f" % (res['p'][0], res['p'][1], res['p'][2])),           
            print("%.2f & %.2f & %.2f" % (res['o'][0], res['o'][1], res['o'][2])),            
            print("%.2f & %.2f & %.2f" % (res['n'][0], res['n'][1], res['n'][2]))
            print "-------------------------------"



    # def computePerformancOfLearner_alamin(self, infile, learnerCol, filenames):
    #     print("Alamin performance method")
    #     # infile = os.path.join(dirConsolidated, "ResultsConsolidated_"+algo+".xls")
    #     df = fileutils.readExcel(infile, "Sheet1", encoding="ISO-8859-1")
        
        
    #     exps = dict()
    #     gots = dict()
    #     labels = dict()
    #     for index, row in df.iterrows():
    #         fname = row["File"]
    #         fname = fname.split("_")[0]
    #         if fname not in filenames:
    #             #print fname, " not in filenmaes"
    #             continue
    #         else:
    #             if fname not in exps:
    #                 exps[fname] = []
    #                 gots[fname] = []
    #                 labels[fname] = set()
    #         exp = row["ManualLabel"]
    #         got = row[learnerCol]
    #         labels[fname].add(exp)
    #         exps[fname].append(exp)
    #         gots[fname].append(got)
    #     for fname in filenames:
    #         computer = metrics.PerformanceMultiClass(exps[fname], gots[fname], labels = list(labels[fname]))
    #         for label in labels[fname]:
    #             pr = computer.precision(label)
    #             re = computer.recall(label)
    #             f1 = 2*pr*re/(pr+re)
    #             print "File %s. Label = %s. Precision = %.2f. Recall = %.2f. F1 = %.2f"%(fname, label, pr, re, f1)
    #         f1_macro = computer.f1_macro_average()
    #         f1_micro, _, _ = computer.compute_micro_average()
    #         print "File = %s. F1 Macro = %.2f. Micro = %.2f"%(fname, f1_macro, f1_micro)
    #         print "-------------------------------"


    # def computePerformancOfLearnerNoBoW(self, algo, learnerCol, dirConsolidated, filenames):
    #     infile = os.path.join(dirConsolidated, "ResultsConsolidated_NoBoW_"+algo+".xls")
    #     df = fileutils.readExcel(infile, "Sheet1", encoding="ISO-8859-1")
        
        
    #     exps = dict()
    #     gots = dict()
    #     labels = dict()
    #     for index, row in df.iterrows():
    #         fname = row["File"]
    #         fname = fname.split("_")[0]
    #         if fname not in filenames:
    #             #print fname, " not in filenmaes"
    #             continue
    #         else:
    #             if fname not in exps:
    #                 exps[fname] = []
    #                 gots[fname] = []
    #                 labels[fname] = set()
    #         exp = row["ManualLabel"]
    #         got = row[learnerCol]
    #         labels[fname].add(exp)
    #         exps[fname].append(exp)
    #         gots[fname].append(got)
    #     for fname in filenames:
    #         computer = metrics.PerformanceMultiClass(exps[fname], gots[fname], labels = list(labels[fname]))
    #         for label in labels[fname]:
    #             pr = computer.precision(label)
    #             re = computer.recall(label)
    #             f1 = 2*pr*re/(pr+re)
    #             print "File %s. Label = %s. Precision = %.2f. Recall = %.2f. F1 = %.2f"%(fname, label, pr, re, f1)
    #         f1_macro = computer.f1_macro_average()
    #         f1_micro, _, _ = computer.compute_micro_average()
    #         print "File = %s. F1 Macro = %.2f. Micro = %.2f"%(fname, f1_macro, f1_micro)
    #         print "-------------------------------"
    

class Sentisead(object):
    
    def __init__(self, basedir, infile, outdir=None):

        self.basedir = basedir
        self.codedir = os.path.join(self.basedir, "code")
        print("BERT has been added to the feature list")
        self.featCols =  [
                                #"Ensemble_RF_HotEncoded",
                                #'Adaptive_POLAR_ADB_HotEncoded', 
                                #'Adaptive_POLAR_RF_HotEncoded',
                                #'Adaptive_POLAR_GBT_HotEncoded',
                                #'DSOSE_HotEncoded',
                                'DsoLabelFullText_HotEncoded', # opiner
                                #'DsoLabelFullTextW2V_HotEncoded',
                                #'Pscore_FullText',
                                #'Nscore_FullText',
                                #'NscoreW2V_FullText',
                                #'PscoreW2V_FullText',
                                #'DsoLabelFirstWord_HotEncoded',
                                #'DsoLabelLastWord_HotEncoded',                                
                                'Senti4SD_HotEncoded', 
                                'SentiCR_HotEncoded', 
                                'SentistrengthSE_HotEncoded',
                                #'Majority_HotEncoded',
                                'POME_HotEncoded',
                                
                                # This 5 was on by sir's code but is not important. So commenting out 5 line
                                # "ShannonPolarOverall",
                                # "ShannonPolarPositive",
                                # "ShannonPolarNegative",
                                # "ShannonVerb",
                                # "ShannonAdjective",


                                #"ShannonPolarOverallBin",
                                #"ShannonPolarPositiveBin",
                                #"ShannonPolarNegativeBin",
                                #"ShannonVerbBin",
                                #"ShannonAdjectiveBin"

                                # ########### Added new tools ###########


                                # added by alamin with bert
                                'BERT4SentiSE_HotEncoded',

                                # added by alamin sentimoji
                                'sentimoji_HotEncoded'

                                ]
        
        self.indir = os.path.join(self.basedir, "data")
        # self.infile = os.path.join(self.indir, infile_name)
        self.infile = infile

        self.filenames = ["DatasetLinJIRA", "BenchmarkUddinSO", "DatasetLinAppReviews", 
                     "DatasetLinSO", "DatasetSenti4SDSO", "OrtuJIRA"]
        self.filenames = [ "BenchmarkUddinSO"]
        
        if(outdir is None):
            self.outdir = os.path.join(self.basedir, "output", "out")
        else:
            self.outdir = outdir 
        fileutils.make_dir(self.outdir)
        self.dirConsolidated = os.path.join(self.outdir, "consolidated")
        fileutils.make_dir(self.dirConsolidated)
        self.utils = CommonUtils()
        print("Sentisead constructed")
        print("infile: %s" %(self.infile))

    # def setInfile(self, filename):
    #     self.infile = os.path.join(self.indir, filename)
    #     print("infile is %s" % self.infile)

    def prepareTrainTestFiles(self):
        self.utils.prepareTrainTestFiles(self.infile, self.outdir)
        
    def trainTestSupervisedDetector(self, algoName, parallelized=False, ngram=(1,1)):
        self.utils.trainTestSentiCRCustomized(algoName, ngram, self.outdir, self.featCols, self.filenames, parallelized)

    def consolidateResults(self, algo):
        self.utils.consolidateResults(algo, self.outdir, self.filenames, self.dirConsolidated)

    def computePerformanceOverallOfLearner(self, algo, learnerCol):
        self.utils.computePerformanceOverallOfLearner(algo, learnerCol, self.dirConsolidated, self.filenames)

    def computePerformanceOverallOfLearner(self, infile, learnerCol):
        self.utils.computePerformanceOverallOfLearner(infile=infile, learnerCol=learnerCol, filenames=self.filenames)
           
    
    # def computePerformancOfLearner(self, algo, learnerCol):
    #     infile = os.path.join(dirConsolidated, "ResultsConsolidated_"+algo+".xls")
    #     # self.utils.computePerformancOfLearner_alamin(infile=self.infile, learnerCol="Senti4SD", filenames= self.filenames)
    #     # self.utils.computePerformancOfLearner(algo=algo, learnerCol=learnerCol,  dirConsolidated = self.dirConsolidated, filenames=self.filenames)
    #     self.utils.computePerformancOfLearner(self, infile=infile, learnerCol=learnerCol, filenames=self.filenames)
    
    def computePerformancOfLearner(self, infile, learnerCol):
        print(infile)
        self.utils.computePerformancOfLearner(infile=infile, learnerCol=learnerCol, filenames=self.filenames)
    
    # def computePerformancOfLearner(self, learnerCol):
    #     self.utils.computePerformancOfLearner(algo, learnerCol, self.dirConsolidated, self.filenames)
    #     # print("Disa added")
    #     # self.utils.computePerformancOfLearner_alamin(infile=self.infile, learnerCol=learnerCol, filenames= self.filenames)
        
    def pipeline(self, algo, ngram, parallelized=False):
        if ngram == 1:
            ngram = (1,1)
        elif ngram == 2:
            ngram = (1,2)
        elif ngram == 3:
            ngram = (1,3)
        else:
            print "ngram out of accepted range. using unigram!"
            ngram = (1,1)
        learnerCol = "Sentisead"
        algoSpec = get_classifier(algo)
        print("About to create train test files")
        self.prepareTrainTestFiles()
        print("About to call getVectorizer")
        vect = getVectorizer(max_df=0.5, min_df=3, ngram_range=ngram)
        print("About to call trainTestSupervisedDetector")
        self.trainTestSupervisedDetector(algo, parallelized, ngram)
        print("About to call consolidateResults")
        self.consolidateResults(algo)
        print("consolidated results methods has been called")

        print algoSpec 
        print "-"*80
        print "Vectorizer"
        print vect 
        print "-"*80
        
        # print "Overall Performance"
        # self.computePerformanceOverallOfLearner(algo=algo, learnerCol=learnerCol)
        # print "-"*80
        # print "By File Performance"
        # print "-"*80
        # self.computePerformancOfLearner(algo=algo, learnerCol=learnerCol)

    # def alamin_pipeline(self, algo, ngram, parallelized=False):
    #     print("Alamin changed pipeline method not the original one")
    #     if ngram == 1:
    #         ngram = (1,1)
    #     elif ngram == 2:
    #         ngram = (1,2)
    #     elif ngram == 3:
    #         ngram = (1,3)
    #     else:
    #         print "ngram out of accepted range. using unigram!"
    #         ngram = (1,1)
    #     learnerCol = "Sentisead"
    #     algoSpec = get_classifier(algo)
    #     # self.prepareTrainTestFiles()
    #     # vect = getVectorizer(max_df=0.5, min_df=3, ngram_range=ngram)
    #     # self.trainTestSupervisedDetector(algo, parallelized, ngram)
    #     # self.consolidateResults(algo)

    #     print algoSpec 
    #     print "-"*80
    #     # print "Vectorizer"
    #     # print vect 
    #     # print "-"*80
        
    #     print "Overall Performance"
    #     self.computePerformanceOverallOfLearner(algo, learnerCol)
    #     print "-"*80
    #     print "By File Performance"
    #     print "-"*80
    #     self.computePerformancOfLearner(algo, learnerCol)

# class SentiseadNoBoW(object):
    
#     def __init__(self, rootdir):
        
#         # self.basedir = os.path.join(rootdir, "Hybrid")
#         self.basedir = os.path.join(rootdir, "code")
#         self.featCols =  [
#                                 #"Ensemble_RF_HotEncoded",
#                                 #'Adaptive_POLAR_ADB_HotEncoded', 
#                                 #'Adaptive_POLAR_RF_HotEncoded',
#                                 #'Adaptive_POLAR_GBT_HotEncoded',
#                                 #'DSOSE_HotEncoded',
#                                 'DsoLabelFullText_HotEncoded',
#                                 #'DsoLabelFullTextW2V_HotEncoded',
#                                 #'Pscore_FullText',
#                                 #'Nscore_FullText',
#                                 #'NscoreW2V_FullText',
#                                 #'PscoreW2V_FullText',
#                                 #'DsoLabelFirstWord_HotEncoded',
#                                 #'DsoLabelLastWord_HotEncoded',                                
#                                 'Senti4SD_HotEncoded', 
#                                 'SentiCR_HotEncoded', 
#                                 'SentistrengthSE_HotEncoded',
#                                 #'Majority_HotEncoded',
#                                 'POME_HotEncoded',
#                                 "ShannonPolarOverall",
#                                 "ShannonPolarPositive",
#                                 "ShannonPolarNegative",
#                                 "ShannonVerb",
#                                 "ShannonAdjective",
#                                 #"ShannonPolarOverallBin",
#                                 #"ShannonPolarPositiveBin",
#                                 #"ShannonPolarNegativeBin",
#                                 #"ShannonVerbBin",
#                                 #"ShannonAdjectiveBin"

#                                 ]
        
#         # self.indir = self.basedir
#         self.indir = os.pardir.join(rootdir, "data")
#         self.infile = os.path.join(self.indir, "ResultsConsolidatedWithEnsembleAssessment.xls")

#         self.filenames = ["DatasetLinJIRA", "BenchmarkUddinSO", "DatasetLinAppReviews", 
#                      "DatasetLinSO", "DatasetSenti4SDSO", "OrtuJIRA"]
        
#         self.outdir = os.path.join(rootdir, "output")
#         fileutils.make_dir(self.outdir)
#         self.dirConsolidated = os.path.join(self.outdir, "consolidated")
#         fileutils.make_dir(self.dirConsolidated)
#         self.utils = CommonUtils()

#     def prepareTrainTestFiles(self):
#         self.utils.prepareTrainTestFiles(self.infile, self.outdir)
        
#     def trainTestSupervisedDetector(self, algoName, parallelized=False, ngram=(1,1)):
#         self.utils.trainTestSentiCRCustomizedNoBoW(algoName, ngram, self.outdir, self.featCols, self.filenames, parallelized)

#     def consolidateResults(self, algo):
#         self.utils.consolidateResultsNoBoW(algo, self.outdir, self.filenames, self.dirConsolidated)

#     def computePerformanceOverallOfLearner(self, algo, learnerCol):
#         self.utils.computePerformanceOverallOfLearnerNoBoW(algo, learnerCol, self.dirConsolidated, self.filenames)
            
#     def computePerformancOfLearner(self, algo, learnerCol):
#         self.utils.computePerformancOfLearnerNoBoW(algo, learnerCol, self.dirConsolidated, self.filenames)
        

#     def pipeline(self, algo, ngram, parallelized=False):
#         if ngram == 1:
#             ngram = (1,1)
#         elif ngram == 2:
#             ngram = (1,2)
#         elif ngram == 3:
#             ngram = (1,3)
#         else:
#             print "ngram out of accepted range. using unigram!"
#             ngram = (1,1)
#         learnerCol = "Sentisead_NoBoW"
#         algoSpec = get_classifier(algo)
#         self.prepareTrainTestFiles()
#         vect = getVectorizer(max_df=0.5, min_df=3, ngram_range=ngram)
#         self.trainTestSupervisedDetector(algo, parallelized, ngram)
#         self.consolidateResults(algo)

#         print algoSpec 
#         print "-"*80
#         print "Vectorizer"
#         print vect 
#         print "-"*80
        
#         print "Overall Performance"
#         self.computePerformanceOverallOfLearner(algo, learnerCol)
#         print "-"*80
#         print "By File Performance"
#         print "-"*80
#         self.computePerformancOfLearner(algo, learnerCol)


class SentimentData:
    def __init__(self, text, rating, uid = 0):
        self.text = text
        self.rating = rating
        self.uid = uid

def get_classifier(algo):
        if algo=="GBT":
            return GradientBoostingClassifier(learning_rate=0.1, n_estimators=500,max_depth=10, min_samples_split=100, 
                                         min_samples_leaf=20, subsample=0.85, random_state=10)
        if algo=="GBTSentiCR":
            return GradientBoostingClassifier()
        elif algo=="RF":
            return  RandomForestClassifier( n_estimators=150)
        elif algo == "xgb":
            return XGBClassifier()
        elif algo=="ADB":
            return AdaBoostClassifier()
        elif algo =="DT":
            return  DecisionTreeClassifier()
        elif algo=="NB":
            return  BernoulliNB()
        elif algo=="SGD":
            return  SGDClassifier()
        elif algo == "LASSO":
            return Lasso(alpha=0.1)
        elif algo=="SVC":
            return LinearSVC(C=1.0, loss = "hinge", max_iter=100000, penalty="l2")
        elif algo == "SVM":
            return svm.SVC(gamma='scale', decision_function_shape='ovo')
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

def getVectorizer(max_df, min_df, ngram_range):
    vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, sublinear_tf=True, max_df=max_df, 
                                 #stop_words=stopWords, 
                                 min_df=min_df, ngram_range=ngram_range)

#    vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, sublinear_tf=True, max_df=max_df, min_df=min_df, ngram_range=ngram_range)
    
    return vectorizer
class SupervisedDetector:
    def __init__(self, infileTraining, infileModel, featCols, training=True, encoding = 'ISO-8859-1',
                 infileSheetName = "Sheet1", infileSentCol = "Sentence", infileRatingCol = "ManualLabel_HotEncoded",
                 algo="GBT", ngram_range = (1,1), max_df = 0.5, min_df = 3):
        self.additionalCols = featCols
        self.algo = algo
        #self.indir = "/home/gias/dev/opinion/papers/opinionvalue/SentiCR"
        self.vectorizer = getVectorizer(max_df, min_df, ngram_range) 
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


    def create_model_from_training_data(self):
        training_comments=[]
        training_ratings=[]
        print("Training classifier model..")
        for sentidata in self.training_data:
            comments = preprocess_text(sentidata.text, uid=sentidata.uid)
            training_comments.append(comments)
            training_ratings.append(sentidata.rating)
        X = hstack((self.vectorizer.fit_transform(training_comments),
                      self.train_df[self.additionalCols].values),
                      format='csr')
        #X_train = self.vectorizer.fit_transform(training_comments).toarray()
        X_train = X.toarray()
        Y_train = np.array(training_ratings)

        #Apply SMOTE to improve ratio of the minority class
        #smote_model = SMOTE(sampling_strategy=0.5, random_state=None, k_neighbors=10, m_neighbors=10, out_step=.0001, kind='regular', svm_estimator=None, n_jobs=1)
        
        #smote_model = SVMSMOTE(sampling_strategy=0.5, random_state=5000, k_neighbors=10, m_neighbors=10, out_step=.0001,n_jobs=1, svm_estimator=None)
        #smote_model = SMOTE(sampling_strategy=0.5)
        smote_model = SVMSMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=object, m_neighbors=object, out_step=.0001,n_jobs=1, svm_estimator=None)
        model= get_classifier(self.algo)
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
            self.train_df = fileutils.readExcel(self.trainingFile, sheetName, encoding = self.encoding).fillna(value={sentCol:"Hi"})
        else:
            self.train_df = pd.read_csv(self.trainingFile, encoding = self.encoding)
        # self.train_df.info()
        # self.train_df[sentCol] = self.train_df[sentCol].astype(str)
        for index, row in self.train_df.iterrows():
            text = row[sentCol]
            rating = row[ratingCol]
            # print(row['UID'])
            comments = SentimentData(text, rating, uid=row['UID'])
            oracle_data.append(comments)
        return  oracle_data

    def get_sentiment_polarity(self,text, additionalColVals):
        comment= preprocess_text(text)
        feature_vector= hstack((self.vectorizer.transform([comment]),
                      additionalColVals),
                      format='csr')
        feature_vector = feature_vector.toarray()
        sentiment_class=self.model.predict(feature_vector)
        return sentiment_class


class SupervisedDetectorNoBoW:
    def __init__(self, infileTraining, infileModel, featCols, training=True, encoding = 'ISO-8859-1',
                 infileSheetName = "Sheet1", infileSentCol = "Sentence", infileRatingCol = "ManualLabel_HotEncoded",
                 algo="GBT", ngram_range = (1,1), max_df = 0.5, min_df = 3):
        self.additionalCols = featCols
        self.algo = algo
        #self.indir = "/home/gias/dev/opinion/papers/opinionvalue/SentiCR"
        self.vectorizer = getVectorizer(max_df, min_df, ngram_range) 
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


    def create_model_from_training_data(self):
        training_comments=[]
        training_ratings=[]
        print("Training classifier model..")
        for sentidata in self.training_data:
            #comments = preprocess_text(sentidata.text)
            #training_comments.append(comments)
            training_ratings.append(sentidata.rating)
        X = self.train_df[self.additionalCols].values
        #X_train = self.vectorizer.fit_transform(training_comments).toarray()
        X_train = X#.toarray()
        Y_train = np.array(training_ratings)

        #Apply SMOTE to improve ratio of the minority class
        #smote_model = SMOTE(sampling_strategy=0.5, random_state=None, k_neighbors=10, m_neighbors=10, out_step=.0001, kind='regular', svm_estimator=None, n_jobs=1)
        
        #smote_model = SVMSMOTE(sampling_strategy=0.5, random_state=5000, k_neighbors=10, m_neighbors=10, out_step=.0001,n_jobs=1, svm_estimator=None)
        #smote_model = SMOTE(sampling_strategy=0.5)
        smote_model = SVMSMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=object, m_neighbors=object, out_step=.0001,n_jobs=1, svm_estimator=None)
        model= get_classifier(self.algo)
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
            comments = SentimentData(text, rating)
            oracle_data.append(comments)
        return  oracle_data

    def get_sentiment_polarity(self,text, additionalColVals):
        #comment= preprocess_text(text)
        feature_vector=  [additionalColVals]
        #feature_vector = feature_vector.toarray()
        sentiment_class=self.model.predict(feature_vector)
        return sentiment_class