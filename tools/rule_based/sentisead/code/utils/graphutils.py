'''
Created on Aug 2, 2014

@author: gias
'''
import igraph

class SimpleGraph(object):
    
    def __init__(self):
        self.sampleGraph = {'A': ['B', 'C'],
             'B': ['C', 'D'],
             'C': ['D'],
             'D': ['C'],
             'E': ['F'],
             'F': ['C']}
    
    
    # the logic behind taken from: https://www.python.org/doc/essays/graphs/
    def find_first_path(self, graph, start, end, path = []):
        path = path + [start]
        if start == end:
            return path
        if not graph.has_key(start): return None
        for node in graph[start]: # traversing the set
            if node not in path: # we are onto a different key. avoiding the cycle.
                newPath = self.find_first_path(graph, node, end, path)
                if newPath: return newPath
        return None
    
    def find_all_paths(self, graph, start, end, path = []):
        path = path + [start]
        if start == end:
            return [path]
        if not graph.has_key(start): return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newPaths = self.find_all_paths(graph, node, end, path)
                for newPath in newPaths: paths.append(newPath)
        return paths
    
    def find_shortest_path(self, graph, start, end, path = []):
        #print "start = %s, end = %s"%(start, end)
        path = path + [start]
        if start == end: return path
        if not graph.has_key(start): return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newPath = self.find_shortest_path(graph, node, end, path)
                if newPath: 
                    if not shortest or len(shortest)>len(newPath): shortest = newPath
        return shortest

class IGraphs(object):
    def __init__(self):
        self.sampleGraph = {
                             'A': ['B', 'C'],
                             'B': ['C', 'D'],
                             'C': ['D'],
                             'D': ['C'],
                             'E': ['F'],
                             'F': ['C', 'G']
                             }
    
    def get_distinct_vertices(self, graph):
        self.num_id = dict()
        self.id_num = dict()
        index = 0
        for key in graph.keys():
            if key not in self.id_num:
                self.id_num[key] = index
                self.num_id[index] = key
                index += 1
            
            deps = graph[key]
            for dep in deps: 
                if dep not in self.id_num:
                    self.id_num[dep] = index
                    self.num_id[index] = dep
                    index += 1
        return self.id_num, self.num_id
    
    def create_igraph(self, graph):
        self.g = igraph.Graph(directed = True)
        self.get_distinct_vertices(graph)
        keys = self.num_id.keys()
        self.g.add_vertices(keys)
        for key in keys:
            self.g.vs["id"] = self.num_id[key]
        
        for key in graph.keys():
            key_num = self.id_num[key]
            deps = graph[key]
            for dep in deps:
                dep_num = self.id_num[dep]
                self.g.add_edges([(key_num, dep_num)])
        return self.g

    def get_shortest_path(self, v1, v2):
        v1 = self.id_num[v1]
        v2 = self.id_num[v2]
        results = list()
        sp = self.g.get_shortest_paths(v1, v2)
        for i, path in enumerate(sp):
            #print "path %i:"%(i),
            asp = list()
            for k in path:
                #print self.num_id[k],
                asp.append(self.num_id[k])
            results.append(asp)
        return results
        #print sp
    
    def delete_igraph(self):
        del self.g
        
 