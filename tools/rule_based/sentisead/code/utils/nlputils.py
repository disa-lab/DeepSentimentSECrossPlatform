'''
Created on Mar 24, 2014

@author: gias
'''
import re
import os
import string
from urlparse import urlparse, urlunsplit, urlsplit
import unicodedata
import nltk
from django.conf import settings

# taken from: http://stackoverflow.com/q/354038
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def num(str):
    try:
        return int(str)
    except ValueError:
        return float(str)


def remove_specialchars(word):
    if word is None or word == "":
        return word
    exclude = set(string.punctuation)    
    exclude.add('..')
    exclude.add('*')
    for c in exclude:
        word = word.strip(c)
    return word

def remove_specialchars_only_added(word, added):
    if word is None or word == "":
        return word
    exclude = set()    
    for a in added:
        exclude.add(a)
    for c in exclude:
        word = word.strip(c)
    return word

def getCustomPunctuations():
    added = ['#', '%', '@', '?', '=', 
        '(', ')', '>', '<',':', ',', 
        '*', '\'', '\\', '[',']',
        '||', '"', '$', '"', '_', '-']
    return added

def remove_specialchars_with_added(word, added):
    if word is None or word == "":
        return word
    exclude = set(string.punctuation)    
    exclude.add('..')
    for a in added:
        exclude.add(a)
    for c in exclude:
        word = word.strip(c)
    return word


def detect_url(sentence):
    # from: http://stackoverflow.com/questions/6883049/regex-to-find-urls-in-string-in-python
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', sentence)
    for i in range(len(urls)):
        urls[i] = remove_specialchars(urls[i])
        #if urls[i].endswith('\/'): urls[i] = urls[i][0:-2]
    if len(urls)>0: return urls
    else: # attempt # 2
        if sentence.startswith('www.'):
            if sentence.endswith('org') or sentence.endswith('net') or sentence.endswith('com'): 
                urls = ["http://"+sentence]
            else:
                urls = [sentence]
    return urls

def parse_url(aURL):
    try:
        o = urlparse(aURL)
        return o.netloc, o.port, o.path
    except:
        return None, None, None

def validUrl(aURL):
    s, u = False, None
    aURL = aURL.replace('\\', '//')
    try:
        o = urlsplit(aURL)
        if o.scheme == 'http' or o.scheme == 'https':
            the_url = urlunsplit((o))
            s = True
            u = the_url
            if o.port:
                print "the_url type = %s, content = %s" %(type(the_url), the_url)
    except ValueError:
        print "value error!"
        return False, None
    #if netloc is None:
    #    return False
    #if port: 
    #    return is_number(port)
    return s, u

    

def urlrstrip(url):
    #print "input url = ",url
    end_exclusion_chars = set(string.punctuation)
    end_exclusion_chars.add('/')
    for c in end_exclusion_chars:
        if url.endswith(c): url = url.rstrip(c)
    if url.endswith('/'): url = url.rstrip('/')
    #print "output url = ",url
    return url
    

def detect_emoticons(token):
    smileys = '^(:\(|:\)|:\-\)|:\-\(|\<\-\>)+$' # matches only the smileys ":)", ":-)" and ":(", ":-("
    m = re.match(smileys, token)
    if m is not None:
        return True
    return False

def get_api_name_from_url(the_url):
    the_url = urlrstrip(the_url)
    
    host_name, port, path = parse_url(the_url)
    if path:
        path = urlrstrip(path)
        s = path.split('/')
        #if path.endswith('/'):
        #    api_name = s[len(s) -2]
        #else:
        api_name = s[-1]
    else:
        api_name = host_name
    return api_name

APIHostingSites = {'start_one': ['github', 'sourceforge', 'freecode'], 
                   'start_two': ['code.google', 'bitbucket.org', 'com.googlecode'],
                   'end_two': ['maven.org', 'apache.org', 'sourceforge.net', 
                               'codehaus.org', 'googlecode.com', 'berlios.de', 
                               'github.com', 'github.io',
                               'java.net', 'google.com']
                   }

def url_from_hostingsites(netloc):
    if netloc.endswith('/'):
        netloc = netloc[0:len(netloc)-1]
    parts = netloc.split('.')
    for placeholder in APIHostingSites.keys():
        if placeholder == 'start_one':
            sites = APIHostingSites[placeholder]
            for site in sites:
                if parts[0] == site:
                    return True, site
        elif placeholder == 'start_two' and len(parts)>1:
            candidate = parts[0]+"."+parts[1]
            sites = APIHostingSites[placeholder]
            for site in sites:
                if site == candidate:
                    return True, site
        elif placeholder == 'end_two' and len(parts)>1:
            candidate = parts[-2]+"."+parts[-1]
            sites = APIHostingSites[placeholder]
            for site in sites:
                if site == candidate:
                    return True, site
    return False, netloc

def apiname_from_hostingsites(netloc):
    if netloc.endswith('/'):
        netloc = netloc[0:len(netloc)-1]
    parts = netloc.split('.')
    for placeholder in APIHostingSites.keys():
        if placeholder == 'start_one':
            sites = APIHostingSites[placeholder]
            for site in sites:
                if parts[0] == site:
                    name = " ".join(parts[1:])
                    return True, site, name
        elif placeholder == 'start_two' and len(parts)>1: # code.google.gson
            candidate = parts[0]+"."+parts[1]
            sites = APIHostingSites[placeholder]
            for site in sites:
                if site == candidate:
                    name = " ".join(parts[2:])
                    return True, site, name
        elif placeholder == 'end_two' and len(parts)>1: # xdoclet.sourceforge.com
            candidate = parts[-2]+"."+parts[-1]
            sites = APIHostingSites[placeholder]
            for site in sites:
                if site == candidate:
                    name = " ".join(parts[-3:-len(parts)+1])
                    return True, site, name
    return False, netloc, None



def split_by_specialchars(project_name):
    #specialChars = set(string.punctuation)
    parts = re.split(r'-|\_|\~|\+|\/|\.', project_name)
    return parts

def split_by_specialchars_with_added(token, added):
    #specialChars = set(string.punctuation)
    sps = '-|\_|\~|\+|\/|\.'
    for a in added:
        sps += '|' + '\\' + a 
    parts = re.split(sps, token)
    return parts


def split_by_specialchars_with_space(project_name):
    #specialChars = set(string.punctuation)
    parts = re.split(r'-|\_|\~|\+|\/|\.|\s', project_name)
    return parts

def split_by_specialchars_with_exclude(project_name, exclude = ['\.']):
    #specialChars = set(string.punctuation)
    separators = ['-', '_', '~','+', '/', '.']
    for e in exclude:
        if e in separators:
            separators.remove(e)
    
    parts = re.split('|'.join(re.escape(x) for x in separators), project_name)
    return parts

def split_by_specialchars_with_exclude_and_added(project_name, added, exclude = ['\.']):
    #specialChars = set(string.punctuation)
    separators = ['-', '_', '~','+', '/', '.']
    for a in added: separators.append(a)
    for e in exclude:
        if e in separators:
            separators.remove(e)
    
    parts = re.split('|'.join(re.escape(x) for x in separators), project_name)
    return parts

def get_apiname_from_jarref(name):
    jar = ".jar"
    start_index = name.index(jar)
    name = name[0:start_index]
    return name

def has_apiname_versionnum(name):
    check = re.findall(r'\d+',name)
    if check:
        return True
    else:
        return False

def get_apiname_without_versionnum(name):
  #  print "name = ",name
    match = re.match(r"([a-zA-Z\-\_\.\s]+)([0-9]+)", name, re.I)
    if match.group(0):
        api_name = match.group(1)
        api_name = " ".join(re.findall("[a-zA-Z]+", api_name))
        #return api_name
        api_name_v2 = " ".join(re.findall("[a-zA-Z]+[0-9]*[a-zA-Z]+", name))
        if api_name_v2:
            api_name = api_name_v2
        return api_name
    else:
        return name

# from: http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
 
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
 
    previous_row = xrange(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
 
    return previous_row[-1]


# This method is copied from github, from the repo of Bermi Ferrer Martinez
def unicodify(st):
    '''
    Convert the given string to normalized Unicode (i.e. combining characters such as accents are combined)
    If given arg is not a string, it's returned as is, and origType is 'noConversion'.
    @return a tuple with the unicodified string and the original string encoding.
    '''

    # Convert 'st' to Unicode
    if isinstance(st, unicode):
        origType = 'unicode'
    elif isinstance(st, str):
        try:
            st = st.decode('utf8')
            origType = 'utf8'
        except UnicodeDecodeError:
            try:
                st = st.decode('latin1')
                origType = 'latin1'
            except:
                raise UnicodeEncodeError('Given string %s must be either Unicode, UTF-8 or Latin-1' % repr(st))
    else:
        origType = 'noConversion'

    # Normalize the Unicode (to combine any combining characters, e.g. accents, into the previous letter)
    if origType != 'noConversion':
        st = unicodedata.normalize('NFKC', st)

    return st, origType

# This method is copied from github, from the repo of Bermi Ferrer Martinez
def deunicodify(unicodifiedStr, origType):
    '''
    Convert the given unicodified string back to its original type and encoding
    '''

    if origType == 'unicode':
        return unicodifiedStr

    return unicodifiedStr.encode(origType)

class NLTKInterface(object):
    def __init__(self):
        self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        self.wordnet = nltk.stem.wordnet.wordnet
        self.stanfordPOSModel = os.path.join(settings.STANFORD_POS_TAGGER['models'], 'english-bidirectional-distsim.tagger')
        self.stanfordPOSTaggerJar = settings.STANFORD_POS_TAGGER['jar']
        from nltk.tag.stanford import StanfordPOSTagger
        #self.stanfordPOSTagger = nltk.tag.stanford.POSTagger(self.stanfordPOSModel, self.stanfordPOSTaggerJar, encoding="UTF-8")
        self.stanfordPOSTagger = StanfordPOSTagger(self.stanfordPOSModel, self.stanfordPOSTaggerJar, encoding="UTF-8")
        self.stemmer = nltk.stem.PorterStemmer()
        #self.tagger = nltk.data.load(nltk.tag._POS_TAGGER)
        self.cmuDict = nltk.corpus.cmudict.dict()
    def get_stanford_pos(self, contents):
        tags = self.stanfordPOSTagger.tag(contents)
        tokens = list()
        pos = list()
        for i in range(len(tags)):
            tokens.append(tags[i][0])
            pos.append(tags[i][1])
        return tokens, pos
        
    #def lemmatize(self, contents = []): # the contents 
    #    lemmas = list()
    #    for line in contents:
    def get_wordnet_pos(self, treebank_tag):
        treebank_tag = treebank_tag.upper()
        if treebank_tag.startswith('J'):
            return self.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return self.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return self.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return self.wordnet.ADV
        else:
            return ''       
    
    def lemmatize(self, word, treebank_pos):
        #print "input word: '%s', pos: '%s'" %(word, treebank_pos)
        pos = self.get_wordnet_pos(treebank_pos)
        if pos != '':
            lemma = self.lemmatizer.lemmatize(word, pos)
        else:
            lemma = word
        #print "lemmatizing word: '%s' for the pos: '%s' with the wordnet pos: '%s'. Lemma = '%s'" %(word,treebank_pos, pos, lemma)
        return lemma
    
    def stemit(self, word):
        return self.stemmer.stem(word)
    
    def get_synset(self, token):
        return self.wordnet.synsets(token)

    # from : http://stackoverflow.com/questions/405161/detecting-syllables-in-a-word
    def nsyl(self, word):
      return [len(list(y for y in x if y[-1].isdigit())) for x in self.cmuDict[word.lower()]]

