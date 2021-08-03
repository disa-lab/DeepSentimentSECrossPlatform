'''
Created on Sep 3, 2014

@author: gias
'''
import utils.codeutils as cu

trace1 = r'''
org.codehaus.jackson.map.exc.UnrecognizedPropertyException: Unrecognized field "wrapper" (Class Wrapper), not marked as ignorable
 at [Source: java.io.StringReader@1198891; line: 1, column: 13] (through reference chain: Wrapper["wrapper"])
 at org.codehaus.jackson.map.exc.UnrecognizedPropertyException.from(UnrecognizedPropertyException.java:53)    
    '''
def test_java_stacktrace():
    (is_strace, conf) = cu.is_exception_trace_lines(trace1.splitlines())
    print "conf %f"%(conf)
    return is_strace

def run():
    test_java_stacktrace()
    