import unittest
import clean_data

class TestCleaning(unittest.TestCase):

    def testRemoveBrackets1(self):
        self.assertEqual(' Annos undeviginti natus exercitum', clean_data.remove_brackets('[1] Annos undeviginti natus exercitum'))

    def testRemoveBrackets2(self):
        self.assertEqual('libertatem vindicavi.   senatus decretis honorificis', clean_data.remove_brackets('libertatem vindicavi.  [Ob quae] senatus decretis honorificis'))

    def testRemoveBrackets3(self):
        self.assertEqual('test for the brackets  OK', clean_data.remove_brackets('test for the brackets (remove) OK'))

    def testRemoveArabicNumbers1(self):
        self.assertEqual('  I. ad Cornelium', clean_data.remove_arabic_numbers('115 116 I. ad Cornelium'))
        
    def testRemoveEndNote(self):
        self.assertEqual('gravis cruciatus adferente, obversis in Demetrium * * * Tacitus ', clean_data.remove_end_note('gravis cruciatus adferente, obversis in Demetrium * * * Tacitus The Latin Library The Classics Page'))
        

if __name__ == "__main__":
    unittest.main()