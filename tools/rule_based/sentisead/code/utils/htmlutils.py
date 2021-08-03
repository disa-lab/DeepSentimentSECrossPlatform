'''
Created on 2013-01-01

@author: gias
'''
# -*- coding: UTF-8 -*-
from bs4 import BeautifulSoup, SoupStrainer
import re
import os
import shutil
import urllib2
import types
from django.utils.encoding import smart_unicode, smart_str



# from: http://code.activestate.com/recipes/466341-guaranteed-conversion-to-unicode-or-byte-string/
def safe_unicode(obj, *args):
    """ return the unicode representation of obj """
    try:
        return unicode(obj, *args)
    except UnicodeDecodeError:
        # obj is byte string
        ascii_text = str(obj).encode('string_escape')
        return unicode(ascii_text)

def safe_str(obj):
    """ return the byte string representation of obj """
    try:
        return str(obj)
    except UnicodeEncodeError:
        # obj is unicode
        return unicode(obj).encode('unicode_escape')

def get_all_html_rows_with_href(url):
    """Given a valid url, open the page referred to by it, find all html rows
    with href in it, form a list containing these links and return the list.
    Ex. <a href="http://www.example.com/page/index.html">Example Page</a>
    This is just a downloader. The preprocessing of the files should be in 
    another module."""
  
    try:
        page = urllib2.urlopen(url) 
    except IOError:
        print "Error : Could not get %s" % url

    content = page.read()
    links = SoupStrainer('a')
  
    return [tag for tag in BeautifulSoup(content, parseOnlyThese=links)]
  
def extract_from_tag(row_with_tag, option):
    """Given a tag, return the value of the field "option".
    Ex. Given, <a href="http://www.example.com/page/index.html">Example Page</a>
    and if the option is "href",
    return the string "http://www.example.com/page/index.html" """
  
    element = re.search(r'option=\"(.*)\"', str(row_with_tag))
    return element.group(1)

def get_string_from_tag(row_with_tag):
    """Given a row with a tag, return the string from it.
    Ex. Given, <a href="http://www.example.com/page/index.html">Example Page</a>
    return the string "Example Page" """
  
    row =  BeautifulSoup(str(row_with_tag))
    return u''.join([row.string for row in row.findAll(text = True)])
  
def get_tables_from_html_page(html_page):
    """Given an html page, return all html tables on the page and
    their contents, i.e., all <td ..>, <tr ..>"""
  
    #tables = (BeautifulSoup(html_page)).findChildren('table')
    tables = (BeautifulSoup(html_page)).find_all('table')
  
    return tables

def extract_row_data(table):
    """Given an html table from the overview, package and other pages,
    read each row and return the names, URL and description represented in each row
    """
    rows = table.find_all('tr')
  
    i = 0;
    tables = []

    for tr in rows:
        if len(tr.findAll('td')) != 2: continue
    
        name = tr.findAll('td')[0].find(text = True).rstrip().strip()
        if name:
            print(name)
        link = tr.findAll('td')[0].find('a')['href'].rstrip().strip()
        if link:
            print(link)
        description = ' '.join(text.replace('&nbsp;', '').rstrip().strip() for text in tr.findAll('td')[1].findAll(text = True))
        if description:
            print("desc found")
        #description = ' '.join(text.replace('&nbsp;', '').rstrip().strip() for text in tr.find_all('td')[1].find_all(text = True))
        
        if name is None:
            continue
        if link is None:
            continue;
        
       # print("link=>"+str(link))
        
#        print ("description = "+str(type(description))+"=>"+str(description))
#        if type(link) == type(None):
#            link = ' '
#        if type(description) == type(None):
#            description = ' '
        tables.append([])    
        tables[i].append(name)
        tables[i].append(link)
        tables[i].append(description)
        i = i + 1

    return tables

def extract_row_data_v2(table):
    """Given an html table from the overview, package and other pages,
    read each row and return the names, URL and description represented in each row
    """
    pattern_code = re.compile(r'(<[/]*code>)+')
    rows = table.find_all('tr')
  
    i = 0;
    tables = []

    for tr in rows:
        if len(tr.findAll('td')) != 2: continue
    
        name = tr.findAll('td')[0].find(text = True).rstrip().strip()
        if name:
            print(name)
        link = tr.findAll('td')[0].find('a')['href'].rstrip().strip()
        if link:
            print(link)
        desc = tr.find_all('td')[1]
        description = ' '.join(text.replace('&nbsp;', '').rstrip().strip() for text in desc.find_all(text = True))
        if description:
            print("desc found:")
        #codewords = desc.find_all('code')
        
        linkurl = ""
        linkcode = ""
        codewithlinks = []
        #links = desc.find_all('a')
        count=0
        for links in desc.find_all('a'):
            count = count + 1
            if links is not None:
                linkurl = links.get('href')
                if linkurl:
                    linkurl = linkurl.rstrip().strip()
                else:
                    continue
            if links.find('code'):
                acode = links.find('code').find(text=True)
                # this following if necessary to tackle these types of codewords:
                # <a href="http://docs.oracle.com/javase/6/docs/api/java/lang/CharSequence.html" title="interface in java.lang"><code></code>character sequence<code></code></a>
                if acode is None:
                    for content in links.contents:
                        if re.match(pattern_code, str(content)):
                            continue
                        else:
                            linkcode = str(content)
                    #acode = re.match(pattern_code, acode)
                    #linkcode = acode
                else:
                    linkcode = links.find('code').find(text=True).strip().strip()
                print "linkcode=",linkcode
                codewithlinks.append({"code":linkcode, "link":linkurl})
                
        
        #codelinks = ':'.join(str(links) for links in desc.find_all('a'))
        
        #codeword=""
        #for code in codewords:
        #    str_code = ':'.join(text for text in code.find_all(text=True))
        #    codeword = codeword+str_code+":"
        #codeword = codeword[0:len(codeword)-1]
        if count>1:
            print("link processed: ",count)
            count=0
        if name is None:
            continue
        if link is None:
            continue;
        print(name)
        

        tables.append([])    
        tables[i].append(name)
        tables[i].append(link)
        tables[i].append(description)
        #tables[i].append(codeword)
        #tables[i].append(linkcode)
        #tables[i].append(linkurl)
        tables[i].append(codewithlinks)
        #tables[i].append(codelinks)
        i = i + 1

    return tables


#def create_directory(dir):
#    if os.path.exists(dir):
#        choice = raw_input("Directory \'%s\' already exists. Overwrite? y/n [n]: " % dir)
#        if choice == 'y' or choice == 'yes':
#            removeall.removeall(dir)
#            shutil.rmtree(dir)
#        else:
#            print "Nothing Done"
#            exit(0)
# 
#    print "Creating directory ...", dir,
#    os.mkdir(dir), 
#    print " : OK"
  
def get_text(html_data):
    return ' '.join((data.replace('&nbsp;', '')).rstrip() for data in (BeautifulSoup(html_data)).findAll(text = True))
  
def get_string_from_list(list):
    return ' '.join(unicode(element) for element in list)
  
def get_contents(url):
    page = urllib2.urlopen(url);
    contents = page.read()   
    #contents = contents.decode('utf8')
    return contents

def clean_str(txt):
    #invalid = "-9999"
    len_txt = len(txt)
    valid_text = ""
    re1 = re.compile(r"[\w\.<>/{}[\]~`\(\)-\,\:\s+\=\&\;]+");
    #return ((''.join(text for text in (txt).re1.match(text))))
    for i in range(0, len_txt):
        if(re1.match(txt[i])):
            #print "txt["+str(i)+"]="+txt[i]
            valid_text = valid_text+txt[i]
    valid_text = ''.join(valid_text)
    valid_text = str(valid_text)
    t = valid_text.replace('&gt;', '>')
    t = t.replace('&lt;', '<')
    
    return t
        
    
def clean_code_in_words(code_in_words):
    len_codes = len(code_in_words)
    valid_code_in_words = ""
    re1 = re.compile(r'[\w\s\(\)\{\}[\]\:\,\.<>/{}\&]', re.IGNORECASE)
    for i in range(0, len_codes):
        if(re1.match(code_in_words[i])):
            valid_code_in_words = valid_code_in_words+code_in_words[i]
    valid_code_in_words = ''.join(valid_code_in_words)
    valid_code_in_words = str(valid_code_in_words)
    t = valid_code_in_words.replace('&gt', '>')
    t = t.replace('&lt', '<')
    return t    
    
def construct_package_hierarchy(hierarchy):
    mod = hierarchy.split("\n")
    for i in range(0, len(mod)):
        t = mod[i].lstrip(' ')
        t = mod[i].strip(' ')
        mod[i] = t
    out = '->'.join(mod)
    return out

def check_unicode(txt):
    try:
        txt =  smart_str(txt)
    except UnicodeEncodeError:
        txt = "" 

def process_text(text):
    utf_txt = check_unicode(text) 
    if  utf_txt == "":
        try:
            text = text.encode('ascii', 'ignore')
            return text
        except UnicodeDecodeError:
            text = ""
            return text
    else:
        return utf_txt 
        