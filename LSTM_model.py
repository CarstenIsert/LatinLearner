import numpy as np
import tensorflow as tf
import time
import math

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
    with tf.name_scope('input'):
        inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
        
    with tf.name_scope('labels'):
        targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels')
    
    with tf.name_scope('dropout'):
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    
    return inputs, targets, dropout_keep_prob


def build_lstm(lstm_size, num_lstm_layers, batch_size, dropout_keep_prob):
    ''' Build LSTM cell.
    
        Arguments
        ---------
        dropout_keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
        lstm_size: Size of the hidden layers in the LSTM cells
        num_lstm_layers: Number of LSTM layers
        batch_size: Batch size

    '''
    def build_cell(idx, num_units, dropout_keep_prob):
        layer_name = 'BasicLSTM' + str(idx)
        with tf.name_scope(layer_name):
            lstm = tf.contrib.rnn.BasicLSTMCell(num_units)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_keep_prob)
        return drop
        
# TODO: Need to fix this later. When launching in a python environment this works...
# Now we assume that we have TF 1.1 or greater...
#    if tf.__version__ == '1.0.0':
        ### Build the LSTM Cell
        # Use a basic LSTM cell
#        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell outputs
#        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_keep_prob)

        # Stack up multiple LSTM layers, for deep learning
        #cell = tf.contrib.rnn.MultiRNNCell([drop] * num_lstm_layers)
#    else:
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
    seq_output = tf.concat(lstm_output, axis = 1)
    # Reshape seq_output to a 2D tensor with lstm_size columns
    x = tf.reshape(seq_output, [-1, in_size])
    
    # Connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        # Create the weight and bias variables here
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    
    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and sequence
    with tf.name_scope('Logits'):
        logits = tf.matmul(x, softmax_w) + softmax_b
    
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
    
        Arguments:
        ----------
        softmax_distribution: Softmax distribution / prediction for all characters
        dist_size: Size of the distribution to sample from
        top_n: Only sample from the top n candidates
    '''
    distribution = np.squeeze(softmax_distribution)
    distribution[np.argsort(distribution)[:-top_n]] = 0
    distribution = distribution / np.sum(distribution)
    int_char = np.random.choice(dist_size, 1, p=distribution)[0]
    return int_char


class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50, 
                       lstm_size=128, num_lstm_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):
    
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
            tf.summary.histogram('One_hot', x_one_hot)
        
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
        

    def train(self, train_text, val_text, epochs=10, continue_training = False, dropout_keep_prob=0.5):
        ''' Train the model, save checkpoints and write Tensorboard information
        
            Arguments:
            
            Returns:
            --------
            lowest_train_loss: Lowest training loss
            lowest_val_loss: Lowest validation loss
        '''
        save_every_n_iterations = 20
        validate_every_n_iterations = 5
        print_every_n_iterations = 2
    
        saver = tf.train.Saver(max_to_keep=100)
                
        lowest_train_loss = math.inf
        lowest_val_loss = math.inf
                
        with tf.Session() as sess:
            # Setup for Tensorboard   
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('./logs', sess.graph)

            if continue_training == False:
                sess.run(tf.global_variables_initializer())
            else:
                checkpoint = tf.train.latest_checkpoint('checkpoints')
                saver.restore(sess, checkpoint)
        
            counter = 0
            print('Started training...')
            for current_epoch in range(epochs):
                # Train network
                new_state = sess.run(self.initial_state)
                for x, y in get_batches(train_text, self.batch_size, self.num_steps):
                    counter += 1
                    start = time.time()
                    feed = {self.inputs: x,
                            self.targets: y,
                            self.keep_prob: dropout_keep_prob,
                            self.initial_state: new_state}
                    batch_loss, new_state, _ = sess.run([self.loss, 
                                                         self.final_state, 
                                                         self.optimizer], 
                                                         feed_dict=feed)
                    
                    if batch_loss < lowest_train_loss: lowest_train_loss = batch_loss
                    end = time.time()
                
                    # TODO: Better progress solution! TQPM or so?!
                    if (counter % print_every_n_iterations) == 0:
                        print('Epoch: {}/{}... '.format(current_epoch+1, epochs),
                              'Training Step: {}... '.format(counter),
                              'Training loss: {:.4f}... '.format(batch_loss),
                              '{:.4f} sec/batch'.format((end-start)))
        
                    # Save the model before doing the validation
                    if (counter % save_every_n_iterations) == 0:
                        saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.lstm_size))
                        
                    # Validate and write Tensorboard Information
                    if (counter % validate_every_n_iterations) == 0:
                        val_losses = []
                        val_state = sess.run(self.initial_state)
                        for x, y in get_batches(val_text, self.batch_size, self.num_steps):
                            feed = {self.inputs: x,
                                    self.targets: y,
                                    self.keep_prob: 1.,
                                    self.initial_state: val_state}
                            summary, val_loss, val_state = sess.run([merged,
                                                                     self.loss, 
                                                                     self.final_state], 
                                                                     feed_dict=feed)
                            val_losses.append(val_loss)
                            train_writer.add_summary(summary, counter)
                            
                        val_loss = np.mean(val_losses)
                        if val_loss < lowest_val_loss: lowest_val_loss = val_loss
                        print("Val loss: {:.3f}".format(val_loss))
                                   
            print("Finished training...")
            saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.lstm_size))
            train_writer.close()
            return lowest_train_loss, lowest_val_loss

    def sample(self, n_samples, character_set_size, int2char_mapping, char2int_mapping, prime="salve"):
        ''' Generate a text sample of specified size from the specified model checkpoint.
            Note that the network has to be reconfigured with a (1, 1) input size defined
            by the sampling=True parameter for the setup and the weights have to be reloaded.
            The sample will be generated up to a maximum length or shorter if the network
            generates some kind of stop token.
    
            Arguments
            ---------
            n_samples: Maximum number of characters to generate
            prime: Initial input for the model to generate the text based upon
        
        '''
        samples = []

        # TODO: Only restore the model once for sampling!
        checkpoint = tf.train.latest_checkpoint('checkpoints')
        print("Resotring from checkpoint: ", checkpoint)
        saver = tf.train.Saver(max_to_keep=100)
        
        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            new_state = sess.run(self.initial_state)
            
            # First feed in the characters in the primer
            for character in prime:
                x = np.zeros((1, 1))
                x[0,0] = char2int_mapping[character]
                feed = {self.inputs: x,
                        self.keep_prob: 1.,
                        self.initial_state: new_state}
                preds, new_state = sess.run([self.prediction, self.final_state], 
                                             feed_dict=feed)
    
            int_character = pick_top_n(preds, character_set_size, 3)
            samples.append(int2char_mapping[int_character])
    
            # Now always use the last character to generate the next one.
            # If we encounter a stop item, we finish.
            for _ in range(n_samples):
                x[0,0] = int_character
                feed = {self.inputs: x,
                        self.keep_prob: 1.,
                        self.initial_state: new_state}
                preds, new_state = sess.run([self.prediction, self.final_state], 
                                             feed_dict=feed)
    
                int_character = pick_top_n(preds, character_set_size, 3)
                character = int2char_mapping[int_character]
    
                # Stop text generation whenever we find something like a marker
                # Do not append the punctuation
                if character in '.!;?':
                    break
    
                samples.append(character)
                
        return ''.join(samples)
  
