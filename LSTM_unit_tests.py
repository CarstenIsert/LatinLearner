import unittest
import numpy as np
import LSTM_model

class TestSplitting(unittest.TestCase):

    def testStandardSplit(self):
        train_text, val_text, test_text = LSTM_model.split_text('Annos undeviginti natus exercitum et')
        self.assertEqual('Annos undeviginti natus exercitum', train_text)

if __name__ == "__main__":
    unittest.main()