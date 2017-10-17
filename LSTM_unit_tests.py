import unittest
import tensorflow as tf
import text_handling
import LSTM_model

class TestLSTMModel(unittest.TestCase):
    def setUp(self):
        self.text_data = text_handling.TextData()
        self.text = self.text_data.load_character_data('test_library')
        result = self.text_data.generate_character_dataset(self.text)
        self.int_text, self.int2char, self.char2int, self.character_set_size = result

    def testBatchGeneration(self):
        count = 0
        for x, y in LSTM_model.get_batches(self.int_text, 2, 3):
            self.assertEqual(x[0,1], y[0,0])
            self.assertEqual(y[1,1], x[1,2])
            count += 1
        
        self.assertEqual(count, len(self.int_text) // (2 * 3))

    def testPickTopN(self):
        prediction_softmax = [[0.01, 0.02, 0.4, 0.3, 0.2, 0.07]]
        int_char = LSTM_model.pick_top_n(prediction_softmax, 6, 3)
        self.assertIn(int_char, [2, 3, 4])

    def testSimpleModelTraining(self):
        model = LSTM_model.CharRNN(self.character_set_size, batch_size=10, num_steps=5,
                                   lstm_size=16, num_lstm_layers=1, 
                                   learning_rate=0.001)
        train_loss, val_loss = model.train(self.int_text, self.int_text, 1, 0.5)
        self.assertLess(train_loss, 3.5)
        self.assertLess(val_loss, 3.5)
        
        test_loss = model.evaluate(self.int_text)
        self.assertLess(test_loss, 3.5)

        # This should be exactly the latest checkpoint from the previous model
        checkpoint = tf.train.latest_checkpoint('checkpoints')
        samp_model = LSTM_model.CharRNN(self.character_set_size, lstm_size=16, num_lstm_layers=1, 
                                   sampling=True, checkpoint=checkpoint)
        result = samp_model.sample(3, self.character_set_size, self.int2char, self.char2int, 'et')
        self.assertEqual(len(result), 3)
        print(result)
        self.assertTrue(str.isprintable(result))

if __name__ == "__main__":
    unittest.main()