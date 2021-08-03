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
 'throw','throws','lambda','endfor','endforeach','endif','endwhile','clone',
 u"'d", u"'s", u'abov', u'ani', u'becaus', u'befor', u'continu', u'could', u'deleg', u'doe', u'doubl', u'dure', 
 u'els', u'endwhil', u'extend', u'implement', u'includ', u'interfac', u'might', u'must', u"n't", u'namespac', 
 u'nativ', u'need', u'nowwhil', u'onc', u'onli', u'ourselv', u'overrid', u'packag', u'privat', u'protect', u'rais', 
 u'readon', u'requir', u'sha', u'sign', u'synchron', u'themselv', u'tri', u'veri', u'whi', u'wo', u'would', u'yourselv',
 u'el', u'rai'
]

pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
tok = WordPunctTokenizer()


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
    #stems = tokens
    return stems

def preprocess_text(text):
    if text is None: return []
    #comments = text.encode('ascii', 'ignore')
    text = text.lower()
    text = text.replace("\\", "")
    comments = text
    comments = Utils.expand_contractions(comments)
    comments = Utils.remove_url(comments)
    comments = Utils.replace_all(comments, Utils.emodict)
    #comments = tweet_cleaner_updated(comments)
    #comments = Utils.handle_negation(comments)
    #comments = detectSentimentDominentClause(comments)    return  comments
    return comments 

def tweet_cleaner_updated(text): # from: https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-2-333514854913
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()


#rootdir = r"C:\dev\opinion\papers\sentiplus"

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
            if label == 1:
                label = "p"
            elif label == -1:
                label = "n"
            elif label == 0:
                label = "o"
            else:
                label = "WTF"
            results.append(label)
        dfTest["DSOSE"] = (pd.Series(results)).values
        outfileResults = filename+"_Test_"+str(fold)+"_Results.csv"
        outfileResults = os.path.join(dirResult, outfileResults)
        dfTest.to_csv(outfileResults, index = False, encoding="ISO-8859-1")
    def trainTestSentiCRCustomized(self, algoName, ngram, outdir, featCols, filenames, parallelized=False):
        dirResult = os.path.join(outdir, "DSOSE_"+algoName)
        fileutils.make_dir(dirResult)
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

    def consolidateResults(self, algo, outdir, filenames, dirConsolidated):
        #algos = ["RF", "ADB", "GBT"]
        records = []
        dirResults = os.path.join(outdir, "DSOSE_"+algo)
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



class DSOSE(object):
    
    def __init__(self, rootdir):
        
        self.basedir = os.path.join(rootdir, "Hybrid")
        self.featCols =  [
                                'DsoLabelFullText_HotEncoded',
                                #'Pscore_FullText',
                                #'Nscore_FullText',
                                #'DsoLabelFullTextW2V_HotEncoded',
                                #'NscoreW2V_FullText',
                                #'PscoreW2V_FullText',
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
        
        self.outdir = os.path.join(self.basedir, "DSOSE")
        fileutils.make_dir(self.outdir)
        self.dirConsolidated = os.path.join(self.outdir, "consolidated")
        fileutils.make_dir(self.dirConsolidated)
        self.utils = CommonUtils()

    def prepareTrainTestFiles(self):
        self.utils.prepareTrainTestFiles(self.infile, self.outdir)
        
    def trainTestSupervisedDetector(self, algoName, parallelized=False, ngram=(1,2)):
        self.utils.trainTestSentiCRCustomized(algoName, ngram, self.outdir, self.featCols, self.filenames, parallelized)

    def consolidateResults(self, algo):
        self.utils.consolidateResults(algo, self.outdir, self.filenames, self.dirConsolidated)

    def computePerformanceOverallOfLearner(self, algo, learnerCol):
        self.utils.computePerformanceOverallOfLearner(algo, learnerCol, self.dirConsolidated, self.filenames)
            
    def computePerformancOfLearner(self, algo, learnerCol):
        self.utils.computePerformancOfLearner(algo, learnerCol, self.dirConsolidated, self.filenames)

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
        learnerCol = "DSOSE"
        algoSpec = get_classifier(algo)
        vect = getVectorizer(max_df=0.5, min_df=3, ngram_range=ngram)
        self.trainTestSupervisedDetector(algo, parallelized, ngram)
        self.consolidateResults(algo)

        print algoSpec 
        print "-"*80
        print "Vectorizer"
        print vect 
        print "-"*80
        
        print "Overall Performance"
        self.computePerformanceOverallOfLearner(algo, learnerCol)
        print "-"*80
        print "By File Performance"
        print "-"*80
        self.computePerformancOfLearner(algo, learnerCol)
        
        
class SentimentData:
    def __init__(self, text,rating):
        self.text = text
        self.rating =rating

def get_classifier(algo):
        if algo=="GBT":
            return GradientBoostingClassifier(learning_rate=0.1, n_estimators=500,max_depth=10, min_samples_split=100, 
                                         min_samples_leaf=20, subsample=0.85, random_state=10)
        if algo=="GBTSentiCR":
            return GradientBoostingClassifier()
        elif algo=="RF":
            return  RandomForestClassifier( n_estimators=150)
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
            comments = preprocess_text(sentidata.text)
            training_comments.append(comments)
            training_ratings.append(sentidata.rating)
        X = hstack((self.vectorizer.fit_transform(training_comments), self.train_df[self.additionalCols].values), format='csr')
        X_train = X.toarray()
        #Y_train = np.array(training_ratings)
        #X_train = self.train_df[self.additionalCols].values
        #X_train = self.vectorizer.fit_transform(training_comments).toarray()
        Y_train = np.array(training_ratings)
        

        #Apply SMOTE to improve ratio of the minority class
        #smote_model = SMOTE(sampling_strategy=0.5, random_state=None, k_neighbors=10, m_neighbors=10, out_step=.0001, kind='regular', svm_estimator=None, n_jobs=1)
        
        smote_model = SVMSMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=object, m_neighbors=object, out_step=.0001,n_jobs=1, svm_estimator=None)
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
        comment= preprocess_text(text)
        feature_vector= hstack((self.vectorizer.transform([comment]), additionalColVals), format='csr')
        feature_vector = feature_vector.toarray()
        #feature_vector = [additionalColVals]
        #feature_vector = feature_vector.toarray()
        
        sentiment_class=self.model.predict(feature_vector)
        return sentiment_class
