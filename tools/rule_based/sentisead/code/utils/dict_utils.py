'''
Created on Mar 12, 2014

@author: gias
'''
from collections import defaultdict
import collections

class OrderedSet(collections.Set):

    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)

def tree(): return defaultdict(tree)


class ExceptionPruneDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        return 0
