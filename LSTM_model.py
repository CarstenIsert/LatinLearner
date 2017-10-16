import numpy as np
import tensorflow as tf
import time
import text_handling

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

def split_text(input_text, training_percentage=0.9):
    '''Split the given text into the three mandatory sets for training, validation and testing
       The ratio is currently set at 90% training size and an even split for test and validation of the rest.  

       Arguments
       ---------
       input_text: The data to split
       training_percentage: The percentage of data which will go into the training set. 
    '''
    split_frac = training_percentage
    position_to_split = int(len(input_text) * split_frac)
    train_text, val_text = input_text[:position_to_split], input_text[position_to_split:]
    
    position_to_split = len(val_text) // 2
    val_text, test_text = val_text[:position_to_split], val_text[position_to_split:]
    
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_text.shape), 
          "\nValidation set: \t{}".format(val_text.shape),
          "\nTest set: \t\t{}".format(test_text.shape))
    return train_text, val_text, test_text
        
def build_inputs(batch_size, num_steps):
    ''' Define TensorFlow placeholders for inputs, targets, and dropout 
    
        Arguments
        ---------
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch
        
    '''
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels')
    
    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' Build LSTM cell.
    
        Arguments
        ---------
        keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
        lstm_size: Size of the hidden layers in the LSTM cells
        num_layers: Number of LSTM layers
        batch_size: Batch size

    '''
    def build_cell(num_units, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(num_units)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
        
# TODO: Need to fix this later. When launching in a python environment this works...
# Now we assume that we have TF 1.1 or greater...
#    if tf.__version__ == '1.0.0':
        ### Build the LSTM Cell
        # Use a basic LSTM cell
#        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell outputs
#        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        # Stack up multiple LSTM layers, for deep learning
        #cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
#    else:
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])    
    
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
    logits = tf.matmul(x, softmax_w) + softmax_b
    
    # Use softmax to get the probabilities for predicted characters
    out = tf.nn.softmax(logits, name='predictions')
    
    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):
    ''' Calculate the loss from the logits and the targets.
    
        Arguments
        ---------
        logits: Logits from final fully connected layer
        targets: Targets for supervised learning
        lstm_size: Number of LSTM hidden units
        num_classes: Number of classes in targets
        
    '''
    # One-hot encode targets and reshape to match logits, one row per sequence per step
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped =  tf.reshape(y_one_hot, logits.get_shape())
    
    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    
    # TODO: Later add accuracy
    
    return loss

def build_optimizer(loss, learning_rate, grad_clip):
    ''' Build optmizer for training, using gradient clipping.
    
        Arguments:
        loss: Network loss
        learning_rate: Learning rate for optimizer
    
    '''
    # Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, n_samples, lstm_size, prime="Gallia "):
    ''' Generate a text sample of specified size from the specified model checkpoint.
        Note that the network has to be reconfigured with a (1, 1) input size defined
        by the sampling=True parameter for the setup and the weights have to be reloaded.

        Arguments
        ---------
        checkpoint: Model checkpoint
        n_samples: Number of characters to generate
        lstm_size: Number of LSTM hidden units
        prime: Initial input for the model to generate the text based upon
    
    '''
    samples = [c for c in prime]
    model = CharRNN(character_set_size, lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = char2int_mapping[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, character_set_size)
        samples.append(int2char_mapping[c])

        for _ in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, character_set_size)
            samples.append(int2char_mapping[c])
        
    return ''.join(samples)


class CharRNN:
    # TODO: Why are the methods to build the network NOT in this class? Refactor!
    
    def __init__(self, num_classes, batch_size=64, num_steps=50, 
                       lstm_size=128, num_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):
    
        # When we're using this network for sampling later, we'll be passing in
        # one character at a time, so providing an option for that
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()
        
        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        # Build the LSTM cell
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        ### Run the data through the RNN layers
        # First, one-hot encode the input tokens
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        # Run each sequence step through the RNN with tf.nn.dynamic_rnn 
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        
        # Get softmax predictions and logits
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        
        # Loss and optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes) 
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)
        


batch_size = 50         # Sequences per batch (default 100)
num_steps = 20          # Number of sequence steps per batch (default 100)
lstm_size = 128         # Size of hidden layers in LSTMs
num_layers = 1          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob = 0.5         # Dropout keep probability        
epochs = 10             # Number of epochs, i.e. run through the whole training set

continue_training = True

save_every_n_iterations = 20
validate_every_n_iterations = 5
print_every_n_iterations = 2

data = text_handling.TextData()
latin_text = data.load_character_data('small_library')
integer_text, int2char_mapping, char2int_mapping, character_set_size = data.generate_character_dataset(latin_text)

train_text, val_text, test_text = split_text(integer_text)

model = CharRNN(character_set_size, batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:   
    if continue_training == False:
        sess.run(tf.global_variables_initializer())
    else:
        checkpoint = tf.train.latest_checkpoint('checkpoints')
        saver.restore(sess, checkpoint)

    counter = 0
    print('Started training...')
    for current_epoch in range(epochs):
        # Train network
        new_state = sess.run(model.initial_state)
        for x, y in get_batches(train_text, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss, 
                                                 model.final_state, 
                                                 model.optimizer], 
                                                 feed_dict=feed)
            
            end = time.time()
        
            # TODO: Better progress solution! TQPM or so?!
            if (counter % print_every_n_iterations) == 0:
                print('Epoch: {}/{}... '.format(current_epoch+1, epochs),
                      'Training Step: {}... '.format(counter),
                      'Training loss: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end-start)))
                
            if (counter % validate_every_n_iterations) == 0:
                val_losses = []
                val_state = sess.run(model.initial_state)
                for x, y in get_batches(val_text, batch_size, num_steps):
                    feed = {model.inputs: x,
                            model.targets: y,
                            model.keep_prob: 1.,
                            model.initial_state: val_state}
                    val_loss, val_state = sess.run([model.loss, 
                                                    model.final_state], 
                                                    feed_dict=feed)
                    val_losses.append(val_loss)
                print("Val loss: {:.3f}".format(np.mean(val_losses)))
                
            if (counter % save_every_n_iterations) == 0:
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    
    print("Finished training...")
    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    
checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 100, lstm_size, prime="Gallia ")
print(samp)