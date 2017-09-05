import unittest
import clean_data
import embedding

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

    def testLowerCase(self):
        self.assertEqual('the cat sat on the big cat sat the cat', clean_data.lower_case('The cat Sat on THE biG CAT saT tHE Cat'))
        
    def testCleanText(self):
        self.assertEqual('this text  has  text ', clean_data.clean_text('This TEXT [1] has 123 text The Latin Library'))
        
class TestReadingData(unittest.TestCase):
    
    def testLoadData1(self):
        data = embedding.read_data('test_library')
        self.assertEqual(['Rerum', 'gestarum', 'divi', 'Augusti', ',', 'quibus', 'orbem', 'terrarum', 'imperio', 'populi'], data[:10])
        self.assertEqual(126, len(data))

    def testGenerateDataSet(self):
        texts = embedding.read_data('test_library')
        int_text, freq_dist, word2int, int2word, vocabulary_size = embedding.generate_dataset(texts)
        self.assertEqual([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 0, 0, 0], int_text[:20])
        self.assertEqual([('HAPAX', 98), (',', 15), ('et', 4), ('in', 3), ('.', 2), ('cum', 2), ('ludere', 2)], freq_dist)
        self.assertEqual({'HAPAX': 0, ',': 1, 'et': 2, 'in': 3, '.': 4, 'cum': 5, 'ludere': 6}, word2int)
        self.assertEqual({0: 'HAPAX', 1: ',', 2: 'et', 3: 'in', 4: '.', 5: 'cum', 6: 'ludere'}, int2word)
        self.assertEqual(6, vocabulary_size)
        

if __name__ == "__main__":
    unittest.main()