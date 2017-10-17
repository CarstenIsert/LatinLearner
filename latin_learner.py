import embedding
import text_handling
import LSTM_model
import argparse
import tensorflow as tf

if __name__ == '__main__':
    FLAGS = None

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='LSTM',
                        help='Define which model to use: LSTM, embedding')
    parser.add_argument('--inference', type=bool, default=False,
                        help='If true, restores latest checkpoint and starts the chatbot interface.')
    parser.add_argument('--checkpoint', type=str, default='none',
                        help="Checkpoint file location to use or 'latest' or 'none' for starting training from scratch.")
    FLAGS, unparsed = parser.parse_known_args()

    data = text_handling.TextData()

    if FLAGS.mode == 'LSTM':
        # TODO: Move the model parameters to the model
        batch_size = 100         # Sequences per batch (default 100)
        num_steps = 100          # Number of sequence steps per batch (default 100)
        lstm_size = 512         # Size of hidden layers in LSTMs
        num_lstm_layers = 1
        learning_rate = 0.001
        dropout_keep_prob = 0.5        
        epochs = 10
        
        sample_only = FLAGS.inference
        
        latin_text = data.load_character_data('small_library')
        integer_text, int2char_mapping, char2int_mapping, character_set_size = data.generate_character_dataset(latin_text)
        train_text, val_text, test_text = data.split_int_text(integer_text)

        if FLAGS.checkpoint == 'none':
            checkpoint = None
        elif FLAGS.checkpoint == 'latest':
            checkpoint = tf.train.latest_checkpoint('checkpoints')
        else:
            checkpoint = tf.train.load_checkpoint(FLAGS.checkpoint)
        print('Restoring from checkpoint: ', checkpoint)

        if sample_only == True:    
            model = LSTM_model.CharRNN(character_set_size, lstm_size=lstm_size, num_lstm_layers=num_lstm_layers, 
                                       sampling=True, checkpoint=checkpoint)

            samp = model.sample(100, character_set_size, int2char_mapping, char2int_mapping, prime='salve caesar!')
            print(samp)    
            while True:
                user_text = input("Prompt or exit: ")
                user_text = str.lower(user_text)
                if user_text == "exit":
                    break
                samp = model.sample(100, character_set_size, int2char_mapping, char2int_mapping, prime=user_text)
                print(samp)    
        else:
            model = LSTM_model.CharRNN(character_set_size, batch_size=batch_size, num_steps=num_steps,
                                       lstm_size=lstm_size, num_lstm_layers=num_lstm_layers, 
                                       learning_rate=learning_rate, checkpoint=checkpoint)
        
            model.train(train_text, val_text, 10, dropout_keep_prob)
            loss = model.evaluate(test_text)
            print("Test loss: ", loss)
    else:
        texts = data.load_tokenized_data('small_library')
        int_text, int2word, vocabulary_size = data.generate_tokenized_dataset(texts)
        model = embedding.WordEmbedding(int_text, int2word, vocabulary_size)
        model.process()
