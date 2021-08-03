'''
Created on Sep 3, 2014

@author: gias
'''

from __future__ import unicode_literals
import re
import unicodedata
import calendar
import datetime
import time

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

def utc_2_local(utc):
        #utc = datetime.datetime.utcnow().strftime(TIME_FORMAT)
        #print "utc_2_local: before convert:", utc
        timestamp =  calendar.timegm((datetime.datetime.strptime( utc, TIME_FORMAT)).timetuple())
        local = datetime.datetime.fromtimestamp(timestamp).strftime(TIME_FORMAT)
        #print "utc_2_local: after convert:", local
        return local

def utcNumToLocal(utc):
    return datetime.datetime.fromtimestamp(float(utc))

def find_all_substring_occurrence(substr, astr):
    return [m.start() for m in re.finditer(substr, astr)]


def smart_decode(s):
    if isinstance(s, unicode):
        return s
    elif isinstance(s, str):
        # Should never reach this case in Python 3
        return unicode(s, 'utf-8')
    else:
        return unicode(s)


def normalize(uni_str):
    '''Ensures that a unicode string does not have strange characters.
       Necessary with lxml because it does not handle encoding well...'''
    if uni_str is None:
        return None
    elif not isinstance(uni_str, unicode):
        uni_str = smart_decode(uni_str)
    return unicodedata.normalize('NFKD', uni_str)

def join_text(lines, line_breaks=True):
    if line_breaks:
        text = '\n'.join(normalize(line).strip() for line in lines)
    else:
        text = ' '.join(clean_breaks(normalize(line)) for line in lines)
    return text

def clean_for_re(original):
    return original.replace('\n', ' ')


def clean_breaks(original, clean_spaces=False):
    new_str = original.replace('\n', ' ').replace('\t', ' ').\
            replace('\r', '')

    if clean_spaces:
        size = len(new_str)
        while(True):
            new_str = new_str.replace('  ', ' ')
            new_size = len(new_str)
            if new_size == size:
                break
            else:
                size = new_size

    return new_str

def merge_lines(lines, line_breaks=True):
    if line_breaks:
        text = '\n'.join(normalize(line).strip() for line in lines)
    else:
        text = ' '.join(clean_breaks(normalize(line)) for line in lines)
    return text
