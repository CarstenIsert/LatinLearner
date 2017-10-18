import numpy as np
import tensorflow as tf
import time
import math
from tqdm import tqdm

def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr. It will yield a different
       minibatch each time it is called.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # Get the number of characters per batch and number of batches we can make
    characters_per_batch = n_seqs * n_steps
    n_batches = len(arr)//characters_per_batch

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * characters_per_batch]
    
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:, n:n+n_steps]
        # The targets, shifted by one
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


def build_inputs(batch_size, num_steps):
    ''' Define TensorFlow placeholders for inputs, targets, and dropout with Tensorboard info
    
        Arguments
        ---------
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch
        
    '''
    with tf.name_scope('Input'):
        inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
        
    with tf.name_scope('Labels'):
        targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels')
    
    with tf.name_scope('Dropout'):
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    
    return inputs, targets, dropout_keep_prob


def build_lstm(lstm_size, num_lstm_layers, batch_size, dropout_keep_prob):
    ''' Build LSTM cell.
    
        Arguments
        ---------
        lstm_size: Size of the hidden layers in the LSTM cells
        num_lstm_layers: Number of LSTM layers
        batch_size: Batch size
        dropout_keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
        
        Returns:
        --------
        Drop
    '''
    def build_cell(idx, num_units, dropout_keep_prob):
        layer_name = 'BasicLSTM' + str(idx)
        with tf.name_scope(layer_name):
            lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, dropout_keep_prob=dropout_keep_prob)
        return lstm
        
    with tf.name_scope('StackedRNN'):
        cell = tf.contrib.rnn.MultiRNNCell([build_cell(idx, lstm_size, dropout_keep_prob) for idx in range(num_lstm_layers)])    
    
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    return cell, initial_state


def build_output(lstm_output, in_size, out_size):
    ''' Build a softmax layer, return the softmax output and logits.
    
        Arguments
        ---------
        lstm_output: List of output tensors from the LSTM layer
        in_size: Size of the input tensor, for example, size of the LSTM cells
        out_size: Size of this softmax layer
    
    '''
    # Reshape output so it's a bunch of rows, one row for each step for each sequence.
    # Concatenate lstm_output over axis 1 (the columns)
    with tf.name_scope('Concat'):
        seq_output = tf.concat(lstm_output, axis = 1)
        # Reshape seq_output to a 2D tensor with lstm_size columns
        x = tf.reshape(seq_output, [-1, in_size])
    
    # Connect the RNN outputs to a softmax layer
    with tf.variable_scope('Softmax'):
        # Create the weight and bias variables here
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    
    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and sequence
    with tf.name_scope('Logits'):
        logits = tf.matmul(x, softmax_w) + softmax_b
        tf.summary.histogram('logits', logits)
    
    # Use softmax to get the probabilities for predicted characters
    with tf.name_scope('Softmax_output'):
        out = tf.nn.softmax(logits, name='predictions')
    
    return out, logits

def build_loss(logits, targets, num_classes):
    ''' Calculate the loss from the logits and the targets.
    
        Arguments
        ---------
        logits: Logits from final fully connected layer
        targets: Targets for supervised learning
        num_classes: Number of classes in targets
        
    '''
    # One-hot encode targets and reshape to match logits, one row per sequence per step
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped =  tf.reshape(y_one_hot, logits.get_shape())
    
    # Softmax cross entropy loss
    with tf.name_scope('Loss'):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', loss)
    
    # TODO: Later add accuracy    
    return loss

def build_optimizer(loss, learning_rate, grad_clip):
    ''' Build optmizer for training, using gradient clipping.
    
        Arguments:
        ----------
        loss: Network loss
        learning_rate: Learning rate for optimizer
    
    '''
    # Optimizer for training, using gradient clipping to control exploding gradients
    with tf.name_scope('Optimizer'):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer


def pick_top_n(softmax_distribution, dist_size, top_n=5):
    ''' Choose the next character based on the top_n most likely characters
        using the probability distribution given in the softmax for the 
        top n entries.
    
        Arguments:
        ----------
        softmax_distribution: Softmax distribution / prediction for all characters
        dist_size: Size of the distribution to sample from
        top_n: Only sample from the top n candidates
        
        Returns:
        --------
        One character in integer format
    '''
    distribution = np.squeeze(softmax_distribution)
    distribution[np.argsort(distribution)[:-top_n]] = 0
    distribution = distribution / np.sum(distribution)
    int_char = np.random.choice(dist_size, 1, p=distribution)[0]
    return int_char


class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50, 
                       lstm_size=128, num_lstm_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False, checkpoint = None):
        ''' Set up the model in two main configurations: For training and for sampling.
            The main difference is if the sampling flag is set to true you should provide 
            a checkpoint and the input shape is set to 1, 1 to allow for single character
            input for prediction. Otherwise the batch_size and num_steps are used.
            
            Arguments:
            ----------
            checkpoint: If a checkpoint if provided, it will be used either for continuing to train the model
                or for sampling
        
        '''
    
        # When we're using this network for sampling later, we'll be passing in
        # one character at a time, so providing an option for that
        if sampling == True:
            self.batch_size, self.num_steps = 1, 1
        else:
            self.batch_size, self.num_steps = batch_size, num_steps

        tf.reset_default_graph()
        
        self.lstm_size = lstm_size
        
        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = build_inputs(self.batch_size, self.num_steps)

        # Build the LSTM cell
        cell, self.initial_state = build_lstm(self.lstm_size, num_lstm_layers, self.batch_size, self.keep_prob)

        ### Run the data through the RNN layers
        # First, one-hot encode the input tokens
        with tf.name_scope('One_hot_input'):
            x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        # Run each sequence step through the RNN with tf.nn.dynamic_rnn
        with tf.name_scope('RNN_Cells'): 
            rnn_outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
            tf.summary.histogram('rnn_out', rnn_outputs)
            self.final_state = state
        
        # Get softmax predictions and logits
        self.prediction, self.logits = build_output(rnn_outputs, lstm_size, num_classes)
        
        # Loss and optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, num_classes) 
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)

        self.saver = tf.train.Saver(max_to_keep=80)
        
        self.sess = tf.Session()
        
        if checkpoint == None:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, checkpoint)
        
        # TODO: Destructor close sessions!
        

    def train(self, train_text, val_text, epochs=10, dropout_keep_prob=0.5):
        ''' Train the model, save checkpoints and write Tensorboard information
        
            Arguments:
            ----------
            train_text: integer array representing the text used for training
            val_text: integer array representing the text used for validation
            epochs: Number of iterations through the whole training set
            dropout_keep_prob: Value used for dropout in the network
            
            Returns:
            --------
            lowest_train_loss: Lowest training loss
            lowest_val_loss: Lowest validation loss
        '''
        save_every_n_iterations = 50
        validate_every_n_iterations = 12
        print_every_n_iterations = 12
    
        lowest_train_loss = math.inf
        lowest_val_loss = math.inf
                
        # Setup for Tensorboard   
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./logs', self.sess.graph)

        counter = 0
        print('Started training...')
        for current_epoch in range(epochs):
            # Train network
            new_state = self.sess.run(self.initial_state)
            pbar = tqdm(total=len(train_text))
            for x, y in get_batches(train_text, self.batch_size, self.num_steps):
                counter += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: dropout_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = self.sess.run([self.loss, 
                                                          self.final_state, 
                                                          self.optimizer], 
                                                          feed_dict=feed)
                
                if batch_loss < lowest_train_loss: lowest_train_loss = batch_loss
                end = time.time()

                pbar.update(self.batch_size * self.num_steps)
                
                if (counter % print_every_n_iterations) == 0:
                    print('Epoch: {}/{}... '.format(current_epoch+1, epochs),
                          'Training Step: {}... '.format(counter),
                          'Training loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end-start)))
        
                # Save the model before doing the validation
                if (counter % save_every_n_iterations) == 0:
                    self.saver.save(self.sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.lstm_size))
                        
                # Validate and write Tensorboard Information
                if (counter % validate_every_n_iterations) == 0:
                    val_losses = []
                    val_state = self.sess.run(self.initial_state)
                    for x, y in get_batches(val_text, self.batch_size, self.num_steps):
                        feed = {self.inputs: x,
                                self.targets: y,
                                self.keep_prob: 1.,
                                self.initial_state: val_state}
                        summary, val_loss, val_state = self.sess.run([merged,
                                                                      self.loss, 
                                                                      self.final_state], 
                                                                      feed_dict=feed)
                        val_losses.append(val_loss)
                        train_writer.add_summary(summary, counter)
                            
                    val_loss = np.mean(val_losses)
                    if val_loss < lowest_val_loss: lowest_val_loss = val_loss
                    print("Val loss: {:.3f}".format(val_loss))
            
            pbar.close()
                                   
        print("Finished training...")
        self.saver.save(self.sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.lstm_size))
        train_writer.close()
        return lowest_train_loss, lowest_val_loss
    
    def evaluate(self, test_text):
        test_losses = []
        test_state = self.sess.run(self.initial_state)
        for x, y in get_batches(test_text, self.batch_size, self.num_steps):
            feed = {self.inputs: x,
                    self.targets: y,
                    self.keep_prob: 1.,
                    self.initial_state: test_state}
            test_loss, test_state = self.sess.run([self.loss, 
                                                   self.final_state], 
                                                   feed_dict=feed)
            test_losses.append(test_loss)
        
        test_loss = np.mean(test_losses)
        return test_loss

    def sample(self, n_samples, character_set_size, int2char_mapping, char2int_mapping, prime="salve!"):
        ''' Generate a text sample of specified size n_samples from the initialized model.
            Note that the network has to be reconfigured with a (1, 1) input size defined
            by the sampling=True parameter for the setup and the weights have to be reloaded.
            The sample will be generated up to a maximum length or shorter if the network
            generates some kind of stop token.
    
            Arguments
            ---------
            n_samples: Maximum number of characters to generate
            character_set_size: Number of characters used
            int2char_mapping: Dictionary on how to map an int representation to a char
            char2int_mapping: Dictionary on how to mapa character to an int used in the model
            prime: Initial input for the model to generate the text based upon. It should end with a stop token
                like . ! or ?

            Returns
            -------
            Generated text up to the max length or shorter when the model generates a stop token.
        '''
        samples = []

        new_state = self.sess.run(self.initial_state)
            
        # First feed in the characters in the primer to the network
        # To make the whole process more reasonable and let the network process
        # the whole input, we don't use the samples while processing the primer
        # The final input character should be a stop token like . ! or ?
        if prime[-1] not in '.!?': prime = prime + '.'
        
        for character in prime:
            x = np.zeros((1, 1))
            x[0,0] = char2int_mapping[character]
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = self.sess.run([self.prediction, self.final_state], 
                                              feed_dict=feed)
    
        # Now always use the last character to generate the next one.
        # If we encounter a stop item, we finish.
        int_character = pick_top_n(preds, character_set_size, 3)
        for _ in range(n_samples):
            x[0,0] = int_character
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = self.sess.run([self.prediction, self.final_state], 
                                              feed_dict=feed)

            int_character = pick_top_n(preds, character_set_size, 3)
            character = int2char_mapping[int_character]

            # Stop text generation whenever we find something like a marker
            # Do not append the punctuation
            if character in '.!;?':
                break

            samples.append(character)
                
        return ''.join(samples)
  
