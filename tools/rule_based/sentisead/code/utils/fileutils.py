'''
Created on Feb 20, 2014

@author: gias
'''
import os
import csv
import re
import json
import pandas as pd
from collections import defaultdict

def tree(): return defaultdict(tree)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def has_exponent(s):
    match_exp = re.match(r'(.*)E(.*)', s, re.M|re.I)
    if match_exp:
        return True
    else:
        return False
    return False

def get_filename_and_extension(filenameWithPath):
    fileName, fileExtension = os.path.splitext(filenameWithPath)
    return fileName, fileExtension

def jsondump(filename, data):
    #print "filename = %s"%(filename)
    #if '/' in filename: return
    if data is not None:
        file = open(filename, "w")
        json_data = json.dumps(data)
        file.write(json_data)
        file.close()
    else:
        print "no data to dump"
def jsonload(filename):
    fp = open(filename, "r")
    data = json.load(fp)
    return data

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_file_by_character(filename):
    contents = list()
    with open(filename) as f:
        while True:
            c = f.read(1)
            if not c:
                return contents
            else:
                contents.append(c)



def write_dict_to_csv(filename, labels, data, delim):
    c = csv.writer(open(filename, 'wb'), delimiter=delim)
    c.writerow(labels)
    for k in sorted(data.keys()):
        c.writerow([k, data[k]])

def write_lists_to_csv(filename, labels, data, delim):
    c = csv.writer(open(filename, 'wb'), delimiter=delim)
    c.writerow(labels)
    for d in data:
        #print "ad=",d
        c.writerow(d)

def read_csv_file(filename, delimiter, skip_header=False):
    afile = open(filename, 'r')
    reader = csv.reader(afile, delimiter='\t', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_ALL)
    if skip_header == True:
        next(reader, None)
    return reader

#def read_csv(filename, delimiter):
#    afile = open(filename, 'rb')
#    reader = csv.reader(afile, delimiter="\t", lineterminator='\n', quoting=csv.QUOTE_NONE)
#    return reader
    #contents = list()
    #for row in reader:
        #print num,":", row
    #    contents.append(row)
    #return contents
    #reader.close()
#    return reader


def get_dirlist(path):
    '''
    return the list of all dirs (not files) under the path
    '''
    print path
    dirs = list()
    for name in os.listdir(path):
        aname = os.path.join(path, name)
        if os.path.isdir(aname):
            dirs.append(aname)
    return dirs

def get_filelist(path):
    files = list()
    for name in os.listdir(path):
        aname = os.path.join(path, name)
        if os.path.isfile(aname):
            files.append(aname)
    return files
    
def get_filelist_nopath(path):
    files = list()
    for name in os.listdir(path):
        aname = os.path.join(path, name)
        if os.path.isfile(aname):
            files.append(name)
    return files

def get_filelist_nopath_noextension(path):
    files = list()
    for name in os.listdir(path):
        aname = os.path.join(path, name)
        if os.path.isfile(aname):
            name = os.path.splitext(name)[0]
            files.append(name)
    return files
    

def open_file(filename):
    try:
        f = open(filename, "rb")
        return 1, f
    except IOError:
        print "Error: the file ",filename," does not exist!"
    return 0, None

def readExcel(infile, sheetname, encoding = "utf-8"):
    xl = pd.ExcelFile(infile)
    df = xl.parse(sheetname)
    #df = pd.read_excel(infile, sheetname, encoding = encoding)
    return df

def writeExcel(infile, sheetname, df):
    writer = pd.ExcelWriter(infile)
    df.to_excel(writer,sheetname, index=False)
    writer.save()