'''
Created on Mar 22, 2019

Add new features - whether not use it in a classifier

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
import spacy
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
import sentiplus.Hybrid.Utils as Utils
import gensim
from gensim.models import word2vec
from gensim.models import KeyedVectors
class Polarity(object):
    
    def __init__(self):
        indirSentiLexicons = r"C:\dev\opinion\papers\sentiplus\senti_lexicons"
        infileDsoSentimentLexicons = os.path.join(indirSentiLexicons, "dso_sentiment_lexicons.csv")
        self.dfDsoLex = pd.read_csv(infileDsoSentimentLexicons)
        self.positives, self.negatives, self.domainPositives, self.domainNegatives = self._load()
        #dfDsoLex.head()
        pass
    def _load(self):
        positives = set()
        negatives = set()
        domainPositives = set()
        domainNegatives = set()
        for index, row in self.dfDsoLex.iterrows():
            source = row["source"]
            lexicon = row["lexicon"]
            label = row["label"]
            lemma = row["lemma"]
            if source == "gradables"  or source == "cooc":
                if label == "p":
                    domainPositives.add(lemma)
                    domainPositives.add(lexicon)
                elif label == "n":
                    domainNegatives.add(lemma)
                    domainNegatives.add(lexicon)
            if label == "p":
                positives.add(lemma)
                positives.add(lexicon)
            elif label == "n":
                negatives.add(lemma)
                negatives.add(lexicon)       
        return positives, negatives, domainPositives, domainNegatives 

    

class PolarityScore(object):
    
    def __init__(self):
        self.infileDsoLexicons = os.path.join(Utils.dirSentimentLexicons, "dso_sentiment_lexicons.csv")
        self.dirSentistrengthLexicons = os.path.join(Utils.dirSentimentLexicons, "SentStrength_Data_Sept2011")
        self.infileEmotionLookup = os.path.join(Utils.dirSentimentLexicons, "EmoticonLookupTable.txt")
        self.dirWord2Vec = os.path.join(Utils.dirSentimentLexicons, "wordvec")
        self.infileWord2Vec = os.path.join(self.dirWord2Vec, "word2vec_model.pkl")
        self.modelWord2Vec = word2vec.Word2Vec.load(self.infileWord2Vec)
        self._load()
        self._loaEmoDict()
    def _loaEmoDict(self):
        self.emodict={}
        with open(self.infileEmotionLookup,"r") as emotable:
            emoticon_reader=csv.reader(emotable,delimiter='\t')
        
            #Hash words from dictionary with their values
            for rows in emoticon_reader:
                polar = rows[1]
                if polar == "NegativeSentiment":
                    self.emodict[rows[0]]= -1
                elif polar == "PositiveSentiment":
                    self.emodict[rows[0]]= 1
                else:  
                    self.emodict[rows[0]] = 0
        
    def _load(self):
        self.sentiLexicons = dict()
        infileSentistrengthEmotion = os.path.join(self.dirSentistrengthLexicons, "EmotionLookupTable.txt")
        with open(infileSentistrengthEmotion, "rb") as f:
            emotionReader = csv.reader(f, delimiter="\t")
            for row in emotionReader:
                score = row[1]
                if score > 0:
                    label = 'p'
                elif score == 0:
                    label = 'o'
                else:
                    label = 'n'
                self.sentiLexicons[row[0]] = {"label": label, "strength": score}
        self.sentiEmoticon = dict() 
        infileSentistrengthEmoticon = os.path.join(self.dirSentistrengthLexicons, "EmoticonLookupTable.txt")
        with open(infileSentistrengthEmoticon, "rb") as f:
            emotionReader = csv.reader(f, delimiter="\t")
            for row in emotionReader:
                score = row[1]
                if score > 0:
                    label = 'p'
                elif score == 0:
                    label = 'o'
                else:
                    label = 'n'
                self.sentiEmoticon[row[0]] = {"label": label, "strength": score}

        df = pd.read_csv(self.infileDsoLexicons, encoding="ISO-8859-1")
        self.dsoLexicons = dict()
        isnullExclude = pd.isnull(df["exclude"])
        for index, row in df.iterrows():
            strength = row["strength"]
            if strength != 's': continue
            if isnullExclude[index] == False:
                exclude = row["exclude"]
            else: 
                exclude = 0
            if exclude == 1: continue
            lexicon = row["lexicon"]
            lemma = row["lemma"]
            #if lemma not in self.dsoLexicons:
            #    self.dsoLexicons[lemma] = []
            if lexicon not in self.dsoLexicons:
                self.dsoLexicons[lexicon] = []
            #if strength != 's': continue
            self.dsoLexicons[lexicon].append({"label": row["label"], "strength": strength, "pos": row["pos"]})
            #self.dsoLexicons[lemma].append({"label": row["label"], "strength": strength, "pos": row["pos"]})
    

    def getPosMap(self, pos, poslemma):
        if poslemma == 'o':  return None
        pmap = {
            'V': 'v',
            'R': 'r',
            'J': 'a',
            'N': 'n'
            }
        for t, d in pmap.items():
            if pos.startswith(t):
                if d == poslemma:
                    return True
        
        return False 
    
    def getLabelFromToken(self, token):
        tokenOriginal = token["token_original"]
        tokenProcessed = token["token_processed"]
        negated = token["negated"]
        lemma = token["lemma"]
        if lemma is None: return None, None
        if len(lemma) == 0: return None, None
        pos = token["pos"]
        label = "o"
        isEmo = False
        if lemma == "PositiveSentiment":
            label = "p"
            isEmo = True
        elif lemma == "NegativeSentiment":
            label = "n"
            isEmo = True
        elif lemma in self.dsoLexicons:
            recs = self.dsoLexicons[lemma]
        elif tokenOriginal in self.dsoLexicons:
            recs = self.dsoLexicons[tokenOriginal]
        elif tokenProcessed in self.dsoLexicons:
            recs = self.dsoLexicons[tokenProcessed]
        else:
            recs = []
        if isEmo == False:
            accounted = False
            for rec in recs:
                strength = rec["strength"]
                if strength != 's': continue
                posLemma = rec["pos"]
                posMatched = self.getPosMap(pos, posLemma)
                if posMatched is None or posMatched == True:
                #if (pos.startswith('V') and pos == posLemma) or pos.startswith('V') == False:
                    label = rec["label"]
                    #print lemma, label, "dso", posLemma
                    accounted = True
                if accounted: break
        return label, negated

    def getLabelFromWord2Vec(self, token):
        tokenOriginal = token["token_original"]
        tokenProcessed = token["token_processed"]
        negated = token["negated"]
        lemma = token["lemma"]
        if lemma is None: return None, None
        if len(lemma) == 0: return None, None
        pos = token["pos"]
        label = "o"
        isEmo = False
        if lemma == "PositiveSentiment":
            label = "p"
            isEmo = True
        elif lemma == "NegativeSentiment":
            label = "n"
            isEmo = True
        elif lemma in self.dsoLexicons:
            recs = self.dsoLexicons[lemma]
        elif tokenOriginal in self.dsoLexicons:
            recs = self.dsoLexicons[tokenOriginal]
        elif tokenProcessed in self.dsoLexicons:
            recs = self.dsoLexicons[tokenProcessed]
        else:
            # find the closest match from word2vec model. look at the top 10 if at least 0.5 similarity
            found = False
            if pos.startswith('V') or pos.startswith('J') or pos.startswith('R'):
                try:
                    tokenOriginal = tokenOriginal.strip(',')
                    #tokenOriginal = tokenOriginal.lower()
                    similarWords = self.modelWord2Vec.wv.most_similar(positive = tokenOriginal)
                    top = 0
                    for item in similarWords:
                        word = item[0]
                        if word in self.dsoLexicons:
                            recs = self.dsoLexicons[word]
                            found = True
                            break
                        top += 1
                        if top > 3: break
                    if found == False:
                        recs = []
                except:
                    recs = []
            else:
                recs = []
        
        if isEmo == False:
            accounted = False
            for rec in recs:
                strength = rec["strength"]
                if strength != 's': continue
                posLemma = rec["pos"]
                posMatched = self.getPosMap(pos, posLemma)
                if posMatched is None or posMatched == True:
                #if (pos.startswith('V') and pos == posLemma) or pos.startswith('V') == False:
                    label = rec["label"]
                    #print lemma, label, "dso", posLemma
                    accounted = True
                if accounted: break
        return label, negated

    def computeScore(self, processedTokens):
        pscore = 0
        nscore = 0
        
        for i in processedTokens:
            token = processedTokens[i]
            label, negated = self.getLabelFromToken(token)
            if negated: 
                if label == "p":
                    nscore += 1
                elif label == "n":
                    pscore += 1
            else:
                if label == "p":
                    pscore += 1
                elif label == "n":
                    nscore += 1
        nscore = -nscore
        score = pscore + nscore
        return score, pscore, nscore
    
    def computeScoreFromPolarWordFromFirstOrLast(self, processedTokens, needsReversed=False):
        pscore = 0
        nscore = 0
        lProcessedTokens = list(processedTokens.keys())
        if needsReversed: lProcessedTokens = reversed(lProcessedTokens)
        for i in lProcessedTokens:
            token = processedTokens[i]     
            label, negated = self.getLabelFromToken(token)       
            if negated: 
                if label == "p":
                    nscore += 1
                elif label == "n":
                    pscore += 1
            else:
                if label == "p":
                    pscore += 1
                elif label == "n":
                    nscore += 1
            if pscore > 0 or nscore > 0:          
                break # we found the first/last polar sentiment word
        
        nscore = -nscore
        score = pscore + nscore
        #print i, score, pscore, nscore
        return score, pscore, nscore

    def computeScoreWord2Vec(self, processedTokens):
        pscore = 0
        nscore = 0
        for i in processedTokens:
            token = processedTokens[i]
            label, negated = self.getLabelFromWord2Vec(token)
            if negated: 
                if label == "p":
                    nscore += 1
                elif label == "n":
                    pscore += 1
            else:
                if label == "p":
                    pscore += 1
                elif label == "n":
                    nscore += 1
        nscore = -nscore
        score = pscore + nscore 
        return score, pscore, nscore
        
    
    
    def computeScoreFocused(self, deps):
        pass
    
    def computeScoreLastSentence(self, text):
        pass
    
    def computeScoreFirstSentence(self, text):
        pass
    

    

def computeLemmaPosDepUsingSpacy():
    indir = r"C:\dev\opinion\papers\sentiplus\Hybrid"
    infile = os.path.join(indir, "ConsolidatedWithBestTrainedClfsFromThreeSETools.pkl")
    outfileMeta = "ConsolidatedWithBestTrainedClfsFromThreeSETools_Meta.pkl"
    outfileMeta = os.path.join(indir, outfileMeta)
    with open(infile, 'rb') as f:
        #df = pd.read_excel(infile, encoding="ISO-8859-1")
        df = pickle.load(f)
    df_dict = df.to_dict(orient="records")
    data = []
    nlp = spacy.load('en_core_web_sm')
    count = 0
    for rec in df_dict:
        sent = rec["Sentence"]
        sent = sent.lstrip('"')
        sent = sent.rstrip('"')
        
        doc = nlp(sent)
        lemma = []
        for token in doc:
            dd = {}
            dd["text"] = token.text
            dd["lemma"] = token.lemma_
            dd["pos"] = token.pos_
            dd["tag"] = token.tag_
            dd["dep"] = token.dep_
            dd["alpha"] = token.is_alpha
            dd["stopword"] = token.is_stop
            lemma.append(dd)
        rec["LemmaPosDep"] = lemma
        data.append(rec)
        count += 1
        if count%100 == 0:
            print "Computed %i rows"%(count)
    dfo = pd.DataFrame.from_dict(data)
    with open(outfileMeta, 'wb') as f:
        pickle.dump(dfo, f)



class DependencyParser(object):
    
    def __init__(self):
        pass

    def detectPrimaryContents(self, text, processedText, deps):
        return text

class Features(object):
    
    def __init__(self):
        self.depParser = DependencyParser()
        self.polarityScore = PolarityScore()
        self.polarityWords = Polarity()
         
    def computeStoreMajorityPolarity(self, infile, outfile):
        def getPolarMap(label):
            if label == "p":
                return 1
            elif label == "n":
                return -1
            else:
                return 0
        df = pd.read_excel(infile, sheet_name="Sheet1", encoding="ISO-8859-1")
        majorityLabels = []
        majorityLabelsEncode = []
        for index, row in df.iterrows():
            dsose = getPolarMap(row["DSOSE"])
            scr = getPolarMap(row["SentiCR"])
            s4sd = getPolarMap(row["Senti4SD"])
            sse = getPolarMap(row["SentistrengthSE"])
            score = dsose + scr + s4sd + sse 
            if score > 0:
                majorityLabels.append("p")
                majorityLabelsEncode.append(1)
            elif score < 0:
                majorityLabels.append("n")
                majorityLabelsEncode.append(-1)
            else:
                majorityLabels.append("o")
                majorityLabelsEncode.append(0)
        df["Majority"] = (pd.Series(majorityLabels)).values
        df["Majority_HotEncoded"] = (pd.Series(majorityLabelsEncode)).values
        df.to_excel(outfile, sheet_name="Sheet1", encoding="ISO-8859-1")
            
    def getProcessedTokens(self, processedText, sentWithLemmaPOS):
        #print processedText
        lemmaPOS = dict()
        for sentlp in sentWithLemmaPOS: # change to spacy
            lemmaPOS[sentlp["text"]] = {"lemma": sentlp["lemma"], "pos": sentlp["tag"]}
        tokens = processedText.split()
        processedTokens = dict()
        for i, token in enumerate(tokens):
            if token.startswith("NOT_"):
                tokenOriginal = token.split("NOT_")[1]
                tokenProcessed = nlputils.remove_specialchars(tokenOriginal)
                negated = True 
            else:
                tokenOriginal = token
                tokenProcessed = nlputils.remove_specialchars(tokenOriginal)
                negated = False
                
            key = None
            if tokenOriginal in ["PositiveSentiment", "NegativeSentiment"]: # from emotion dict
                key = tokenOriginal
            if tokenOriginal in lemmaPOS:
                key = tokenOriginal
            elif tokenProcessed in lemmaPOS:
                key = tokenProcessed
            else:
                key = None  
            if key is not None:
                if key in ["PositiveSentiment", "NegativeSentiment"]:
                    lemma = key
                    l = key 
                    p = key
                else:
                    lemma = lemmaPOS[key]
                    l = lemma['lemma']
                    p = lemma['pos']
            else:
                l = None
                p = None
                    
            processedTokens[i] = {"token_original": tokenOriginal, "token_processed": tokenProcessed, "negated": negated, "lemma": l, "pos": p}
        return processedTokens  
    
    def getTextInfo(self, text, sentWithLemmaPOS, deps):
        #primaryText = self.depParser.detectPrimaryContents(text, sentWithLemmaPOS, deps)
        processedText = Utils.preprocess_text(text)
        processedTokens = self.getProcessedTokens(processedText, sentWithLemmaPOS)

    def getDsoLabel(self, score):
        if score > 0:
            label = 'p'
        elif score < 0:
            label = 'n'
        else:
            label = 'o'
        return label
    
    def hotEncodePolarLabel(self, label):
        if label == "p":
            return 1
        elif label == "n":
            return -1
        else:
            return 0
    def computeAndStorePolarityScoreForAll(self, infileMeta, outfile):
        #features = Features()
        
        with open(infileMeta, 'rb') as f:
            dfm = pickle.load(f)
        dfm_dict = dfm.to_dict(orient="records")
        data = dict()
        for rec in dfm_dict:
            #print rec 
            data[rec["UID"]] = rec
            #break
        recs = []
        recsm = []
        for row in dfm_dict:
            rec = dict()
            recm = dict()
            for k, v in row.items():
                if k != "LemmaPosDep" and k!='POME_RES':
                    rec[k] = v
                recm[k] = v
            uid = row["UID"]
            sent = row["Sentence"]
            sentWithLemmaPOS = row["LemmaPosDep"]
            processesedText = Utils.preprocess_text(sent)
            processedTokens = self.getProcessedTokens(processesedText, sentWithLemmaPOS)
            score, pscore, nscore = self.polarityScore.computeScore(processedTokens)
            
            rec["Pscore_FullText"] = pscore
            recm["Pscore_FullText"] = pscore 
            
            rec["Nscore_FullText"] = nscore
            recm["Nscore_FullText"] = nscore
            
            DsoLabel = self.getDsoLabel(score) 
            rec["DsoLabelFullText"] = DsoLabel
            recm["DsoLabelFullText"] = DsoLabel
            
            rec["DsoLabelFullText_HotEncoded"] = self.hotEncodePolarLabel(DsoLabel)
            recm["DsoLabelFullText_HotEncoded"] = self.hotEncodePolarLabel(DsoLabel)
            
            recs.append(rec)
            recsm.append(recm)
            #dataXls[uid] = rec
        dfn = pd.DataFrame.from_dict(recs)
        dfn.to_excel(outfile, sheet_name="Sheet1", encoding="ISO-8859-1", index=False) 
    
        dfm = pd.DataFrame.from_dict(recsm)
        with open(infileMeta, 'wb') as f:
            pickle.dump(dfm, f)
            
    def computeAndStorePolarityScoreWord2VecForAll(self, infileMeta, outfile):
        #features = Features()
        
        with open(infileMeta, 'rb') as f:
            dfm = pickle.load(f)
        dfm_dict = dfm.to_dict(orient="records")
        data = dict()
        for rec in dfm_dict:
            #print rec 
            data[rec["UID"]] = rec
            #break
        recs = []
        recsm = []
        for row in dfm_dict:
            rec = dict()
            recm = dict()
            for k, v in row.items():
                if k != "LemmaPosDep" and k!='POME_RES':
                    rec[k] = v
                recm[k] = v
            uid = row["UID"]
            sent = row["Sentence"]
            sentWithLemmaPOS = row["LemmaPosDep"]
            processesedText = Utils.preprocess_text(sent)
            processedTokens = self.getProcessedTokens(processesedText, sentWithLemmaPOS)
            score, pscore, nscore = self.polarityScore.computeScoreWord2Vec(processedTokens)
            
            rec["PscoreW2V_FullText"] = pscore
            recm["PscoreW2V_FullText"] = pscore 
            
            rec["NscoreW2V_FullText"] = nscore
            recm["NscoreW2V_FullText"] = nscore
            
            DsoLabel = self.getDsoLabel(score) 
            rec["DsoLabelFullTextW2V"] = DsoLabel
            recm["DsoLabelFullTextW2V"] = DsoLabel
            
            rec["DsoLabelFullTextW2V_HotEncoded"] = self.hotEncodePolarLabel(DsoLabel)
            recm["DsoLabelFullTextW2V_HotEncoded"] = self.hotEncodePolarLabel(DsoLabel)
            
            recs.append(rec)
            recsm.append(recm)
            #dataXls[uid] = rec
        dfn = pd.DataFrame.from_dict(recs)
        dfn.to_excel(outfile, sheet_name="Sheet1", encoding="ISO-8859-1", index=False) 
    
        dfm = pd.DataFrame.from_dict(recsm)
        with open(infileMeta, 'wb') as f:
            pickle.dump(dfm, f)

    
    def computeAndStoreUsingFirstLastPolarWord(self, infileMeta, outfile):
        with open(infileMeta, 'rb') as f:
            dfm = pickle.load(f)
        dfm_dict = dfm.to_dict(orient="records")
        recs = []
        recsm = []
        for row in dfm_dict:
            rec = dict()
            recm = dict()
            for k, v in row.items():
                if k != "LemmaPosDep" and k!='POME_RES':
                    rec[k] = v
                recm[k] = v
            uid = row["UID"]
            sent = row["Sentence"]
            sentWithLemmaPOS = row["LemmaPosDep"]
            processesedText = Utils.preprocess_text(sent)
            processedTokens = self.getProcessedTokens(processesedText, sentWithLemmaPOS)
            fscore, fpscore, fnscore = self.polarityScore.computeScoreFromPolarWordFromFirstOrLast(processedTokens, False)
            rec["Pscore_FirstPolar"] = fpscore 
            rec["Nscore_FirstPolar"] = fnscore 
            recm["Pscore_FirstPolar"] = fpscore 
            recm["Nscore_FirstPolar"] = fnscore 
            
            
            fDsoLabel = self.getDsoLabel(fscore)
            rec["DsoLabelFirstWord"] = fDsoLabel
            rec["DsoLabelFirstWord_HotEncoded"] = self.hotEncodePolarLabel(fDsoLabel)
            recm["DsoLabelFirstWord"] = fDsoLabel
            recm["DsoLabelFirstWord_HotEncoded"] = self.hotEncodePolarLabel(fDsoLabel)

            lscore, lpscore, lnscore = self.polarityScore.computeScoreFromPolarWordFromFirstOrLast(processedTokens, True)
            rec["Pscore_LastPolar"] = lpscore 
            rec["Nscore_LastPolar"] = lnscore
            recm["Pscore_LastPolar"] = lpscore 
            recm["Nscore_LastPolar"] = lnscore

            lDsoLabel = self.getDsoLabel(lscore) 
            rec["DsoLabelLastWord"] = lDsoLabel
            rec["DsoLabelLastWord_HotEncoded"] = self.hotEncodePolarLabel(lDsoLabel)
            recm["DsoLabelLastWord"] = lDsoLabel
            recm["DsoLabelLastWord_HotEncoded"] = self.hotEncodePolarLabel(lDsoLabel)
            recs.append(rec)
            recsm.append(recm)
        dfn = pd.DataFrame.from_dict(recs)
        dfn.to_excel(outfile, sheet_name="Sheet1", encoding="ISO-8859-1") 
        dfm = pd.DataFrame.from_dict(recsm)
        with open(infileMeta, 'wb') as f:
            pickle.dump(dfm, f)
            
    
    def getDiversityScore(self, text):
        text = text.lower()
        text = text.replace('\\', '')
        words = nltk.word_tokenize(text)
        data = dict()
        for w in words:
            data[w] = data.get(w, 0) + 1
        shannon = dm.shannon(data)
        return shannon
    
    def getDiversityScorePolarityOverall(self, text):
        words = word_tokenize(text)
        data = dict()
        #dataDomains = dict()
        for w in words:
            if w in self.polarityWords.positives or w in self.polarityWords.negatives:
                data[w] = data.get(w, 0) + 1
            #if w in self.polarityWords.domainPositives or w in self.polarityWords.domainNegatives:
            #    dataDomains[w] = dataDomains.get(w, 0) + 1
        shannon = dm.shannon(data)
        return shannon
    
    def getDiversityScorePolarityPositive(self, text):
        words = word_tokenize(text)
        data = dict()
        #dataDomains = dict()
        for w in words:
            if w in self.polarityWords.positives:
                data[w] = data.get(w, 0) + 1
            #if w in self.polarityWords.domainPositives or w in self.polarityWords.domainNegatives:
            #    dataDomains[w] = dataDomains.get(w, 0) + 1
        shannon = dm.shannon(data)
        return shannon
    
    def getDiversityScorePolarityNegative(self, text):
        words = word_tokenize(text)
        data = dict()
        #dataDomains = dict()
        for w in words:
            if w in self.polarityWords.negatives:
                data[w] = data.get(w, 0) + 1
            #if w in self.polarityWords.domainPositives or w in self.polarityWords.domainNegatives:
            #    dataDomains[w] = dataDomains.get(w, 0) + 1
        shannon = dm.shannon(data)
        return shannon
    
    def getDiversityScoreVerb(self, text):
        text = text.lower()
        tagged =  nltk.pos_tag(nltk.word_tokenize(text))
        data = dict()
        for tag in tagged:
            word, pos = tag[0], tag[1]
            if pos.startswith('V'):
                data[word] = data.get(word, 0) + 1
                
        shannon = dm.shannon(data)
        return shannon
    def getDiversityScoreAdjective(self, text):
        text = text.lower()
        tagged =  nltk.pos_tag(nltk.word_tokenize(text))
        data = dict()
        for tag in tagged:
            word, pos = tag[0], tag[1]
            if pos.startswith('J'):
                data[word] = data.get(word, 0) + 1
                
        shannon = dm.shannon(data)
        return shannon
    
    def getWordEmbeddingScoresSO(self, text):
        pass





class PrepareFeatureValuesForClassification(object):
    
    def __init__(self):
        self.features = Features()
    
    def convertFeatureValuesToNumeric(self, infile, outfile, colsToEncode):
        df = fileutils.readExcel(infile, "Sheet1", encoding="ISO-8859-1")
        data = dict()
        df_dict = df.to_dict(orient="records")
        for rec in df_dict:
            for col in colsToEncode:
                col_encoded = col+"_HotEncoded"
                val = rec[col]
                valEncoded = self.mapVals[val]
                rec[col_encoded] = valEncoded
            #print rec 
            data[rec["UID"]] = rec        
        records = []
        for index, row in df.iterrows():
            rec = data[row["UID"]]
            records.append(rec)
        dfn = pd.DataFrame.from_dict(records)
        dfn.to_excel(outfile, encoding="ISO-8859-1", sheet_name="Sheet1", index=False)
        #fileutils.writeExcel(self.outfile, "Sheet1", dfn, )
    
    def _encodeDiversityValues(self, score):
        val = math.ceil(score/2)
        val = int(val)
        return val
            
    def computeAndEncodeDiversityScores(self, infile, outfile):
        df = fileutils.readExcel(infile, "Sheet1", encoding="ISO-8859-1")
        data = dict()
        df_dict = df.to_dict(orient="records")
        recs = []
        for rec in df_dict:
            text = rec["Sentence"]
            shannonPolarOverall = self.features.getDiversityScorePolarityOverall(text)
            rec["ShannonPolarOverall"] = shannonPolarOverall

            shannonPolarOverall_Bin = self._encodeDiversityValues(shannonPolarOverall)
            rec["ShannonPolarOverallBin"] = shannonPolarOverall_Bin 

            shannonPolarPositive = self.features.getDiversityScorePolarityPositive(text)
            rec["ShannonPolarPositive"] = shannonPolarPositive
            shannonPolarPositive_Bin = self._encodeDiversityValues(shannonPolarPositive)
            rec["ShannonPolarPositiveBin"] = shannonPolarPositive_Bin

            shannonPolarNegative = self.features.getDiversityScorePolarityNegative(text)
            rec["ShannonPolarNegative"] = shannonPolarNegative
            shannonPolarNegative_Bin = self._encodeDiversityValues(shannonPolarNegative)
            rec["ShannonPolarNegativeBin"] = shannonPolarNegative_Bin

            shannonVerb = self.features.getDiversityScoreVerb(text)
            rec["ShannonVerb"] = shannonVerb
            shannonVerb_Bin = self._encodeDiversityValues(shannonVerb)
            rec["ShannonVerbBin"] = shannonVerb_Bin            

            shannonAdjective = self.features.getDiversityScoreAdjective(text)
            rec["ShannonAdjective"] = shannonAdjective
            shannonAdjective_Bin = self._encodeDiversityValues(shannonAdjective)
            rec["ShannonAdjectiveBin"] = shannonAdjective_Bin            
            recs.append(rec)   
        dfn = pd.DataFrame.from_dict(recs)
        dfn.to_excel(outfile, encoding="ISO-8859-1", sheet_name="Sheet1", index=False)

