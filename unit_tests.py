import unittest
import numpy as np
import clean_data
import embedding
import text_handling

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

    def testCleanWhiteSpace(self):
        self.assertEqual("Please don't hurt me.", clean_data.remove_white_space("Please \n don't \t hurt \x0b me."))        
        
    def testCleanWhiteSpace2(self):
        self.assertEqual("m. tvlli ciceronis epistvlarvm ad qvintvm fratrem liber primvs iiiiiiiv", clean_data.remove_white_space("    m. tvlli ciceronis epistvlarvm ad qvintvm fratrem liber primvs\n\n\n\n\niiiiiiiv\n\n\n  \n"))        
        
    def testCleanText(self):
        self.assertEqual('this text has text', clean_data.clean_text('This TEXT [1] has 123 text The Latin Library'))
        
class TestReadingData(unittest.TestCase):
    def setUp(self):
        self.text_data = text_handling.TextData()
    
    def testLoadData1(self):
        data = self.text_data.load_tokenized_data('test_library')
        self.assertEqual(['Rerum', 'gestarum', 'divi', 'Augusti', ',', 'quibus', 'orbem', 'terrarum', 'imperio', 'populi'], data[:10])
        self.assertEqual(126, len(data))

    def testGenerateDataSet(self):
        texts = self.text_data.load_tokenized_data('test_library')
        int_text, int2word, vocabulary_size = self.text_data.generate_tokenized_dataset(texts)
        self.assertEqual([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 0, 0, 0], int_text[:20])
        self.assertEqual({0: 'HAPAX', 1: ',', 2: 'et', 3: 'in', 4: '.', 5: 'cum', 6: 'ludere'}, int2word)
        self.assertEqual(6, vocabulary_size)
        
class TestWordEmbedding(unittest.TestCase):
    def setUp(self):
        self.text_data = text_handling.TextData()
        texts = self.text_data.load_tokenized_data('test_library')
        int_text, int2word, vocabulary_size = self.text_data.generate_tokenized_dataset(texts)
        self.embedding = embedding.WordEmbedding(int_text, int2word, vocabulary_size)
        np.random.seed(1)
        
    def testValidationSet(self):
        self.embedding.generate_similarity_validation_set()
        self.assertTrue(([80, 84, 33, 81, 93, 17, 36, 82, 69, 65, 92, 39, 56, 52, 51, 32] == self.embedding.validation_set).all())

if __name__ == "__main__":
    unittest.main()