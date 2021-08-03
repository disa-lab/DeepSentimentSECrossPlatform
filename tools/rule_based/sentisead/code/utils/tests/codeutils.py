'''
Created on Sep 3, 2014

@author: gias
'''
from django.test import TestCase
import utils.codeutils as cu

class JavaElementFunctionsTest(TestCase):

    def test_java_name(self):
        to_test = [
            ('java.lang.String', 'java.lang.String', 'String'),
            ('String', 'String', 'String'),
            ('p1.Foo$Fa', 'p1.Foo.Fa', 'Fa'),
            ('p1.Foo$Fa<p2.String,int>', 'p1.Foo.Fa', 'Fa'),
            ('p1.Bar[[]]', 'p1.Bar', 'Bar'),
            ]
        for (original, fqn, simple) in to_test:
            (simple2, fqn2) = cu.clean_java_name(original)
            self.assertEqual(simple, simple2)
            self.assertEqual(fqn, fqn2)

class JaveStacktraceTest(TestCase):
    trace1 = r'''
org.codehaus.jackson.map.exc.UnrecognizedPropertyException: Unrecognized field "wrapper" (Class Wrapper), not marked as ignorable
 at [Source: java.io.StringReader@1198891; line: 1, column: 13] (through reference chain: Wrapper["wrapper"])
 at org.codehaus.jackson.map.exc.UnrecognizedPropertyException.from(UnrecognizedPropertyException.java:53)    
    '''
    def test_java_stacktrace(self):
        (is_strace, conf) = cu.is_exception_trace_lines(self.trace1.splitlines())
        print "conf %f"%(conf)
        self.assertTrue(is_strace)
        
            
class JavaSnippetTest(TestCase):
    snippet1 = r'''
This is a Java snippet:
public class Foo {
    public void main(String arg)
    {
      // Do something useful!
      System.out.println();
    }
}
    '''

    snippet2 = r'''
This is not a Java snippet:
BEGIN;
SELECT * FROM TABLE;
COMMIT;
END;
    '''

    snippet3 = r'''
This is an awfully commented Java snippet:
public class Foo { // Hello
    public void main(String arg) // Hello
    { // Hello
      // Do something useful!
      System.out.println(); /* Hello */
    } // Hello
} // Comment again!
    '''

    snippet4 = r'''
// This is a comment.
// Another one.
new Hello() {
    public void foo() {}
};
    '''

    snippet5 = r'''
/**
 * @param hello; = asdw3{ {}
 */
public List<String> main(String[] args) {
    System.out.println();
}
    '''

    def setUp(self):
        self.filters = [cu.SQLFilter(), cu.BuilderFilter()]

    def test_java_snippet1(self):
        (is_snippet, _) = cu.is_java_snippet(self.snippet1, self.filters)
        self.assertTrue(is_snippet)
        (is_snippet, _) = cu.is_java_snippet(self.snippet2, self.filters)
        self.assertFalse(is_snippet)
        (is_snippet, _) = cu.is_java_snippet(self.snippet3, self.filters)
        self.assertTrue(is_snippet)

    def test_snippet_classification(self):
        self.assertFalse(cu.is_class_body(self.snippet4))
        self.assertTrue(cu.is_class_body(self.snippet5))
