import numpy as np
import tensorflow.contrib.keras as K

class WordEmbedding:
    """ Class for building a model for embedding words.
    """
    window_size = 3
    embedding_vector_size = 300
    epochs = 900000
    size_of_similarity_set = 16

    def __init__(self, integer_text, int2word_mapping, vocabulary_size):
        self.integer_text = integer_text
        self.int2word_mapping = int2word_mapping
        self.vocabulary_size = vocabulary_size
    
    def generate_similarity_validation_set(self, window_size = 100):
        """ 
        We sample from the 'number_of_words' from the integer_text generated in the TextData
        class. As those are only integers, we don't actually need the data here.
        We sample from the top 'window_size' words as those are the ones with the highest frequencies.
        We will use this set to check during training which words are considered similar.
        """
        self.validation_set = np.random.choice(window_size, self.size_of_similarity_set, replace=False)

    def create_samples(self):
        sampling_table = K.preprocessing.sequence.make_sampling_table(self.vocabulary_size+1)
        couples, self.sample_labels = K.preprocessing.sequence.skipgrams(self.integer_text, self.vocabulary_size, window_size=self.window_size, sampling_table=sampling_table)
        # TODO: Review, understand... write test cases!
        word_target, word_context = zip(*couples)
        self.word_target = np.array(word_target, dtype="int32")
        self.word_context = np.array(word_context, dtype="int32")
        print(couples[:10], self.sample_labels[:10])

    def build(self):
        # Just feed one example into target and context
        input_target = K.layers.Input((1,))
        input_context = K.layers.Input((1,))
        
        embedding = K.layers.Embedding(self.vocabulary_size, self.embedding_vector_size, input_length=1, name='embedding')

        target = embedding(input_target)
        target_word_vector = K.layers.Reshape((self.embedding_vector_size, 1))(target)
        context = embedding(input_context)
        context_word_vector = K.layers.Reshape((self.embedding_vector_size, 1))(context)
        
        dot_product = K.layers.dot([target_word_vector, context_word_vector], axes=1, normalize=True)
        dot_product = K.layers.Reshape((1,))(dot_product)
        sigmoid_output_layer = K.layers.Dense(1, activation='sigmoid')(dot_product)

        self.model = K.models.Model(inputs=[input_target, input_context], outputs=sigmoid_output_layer)
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        
        # setup a cosine similarity operation which will be output in a secondary model
        # create a secondary validation model to train our similarity checks during training
        similarity = K.layers.dot([target_word_vector, context_word_vector], axes=0, normalize=True)
        self.validation_model = K.models.Model(inputs=[input_target, input_context], outputs=similarity)

    @staticmethod
    def _compute_similarity(self, valid_word_idx):
        similarity_scores = np.zeros((self.vocabulary_size,))
        input_target = np.zeros((1,))
        input_context = np.zeros((1,))
        input_target[0,] = valid_word_idx
        for i in range(self.vocabulary_size):
            input_context[0,] = i
            out = self.validation_model.predict_on_batch([input_target, input_context])
            similarity_scores[i] = out
        return similarity_scores

    def _check_similarity(self):
        for cnt in range(self.size_of_similarity_set):
            valid_word = self.int2word_mapping[self.validation_set[cnt]]
            number_of_nearest_neighbors = 6
            # TODO: Look up Python static methods and why I need it here...
            similarity_scores = self._compute_similarity(self, self.validation_set[cnt])
            nearest = (-similarity_scores).argsort()[1:number_of_nearest_neighbors + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(number_of_nearest_neighbors):
                close_word = self.int2word_mapping[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
        
    def optimize(self):
        input_target = np.zeros((1,))
        input_context = np.zeros((1,))
        output = np.zeros((1,))
        for cnt in range(self.epochs):
            idx = np.random.randint(0, len(self.sample_labels)-1)
            input_target[0,] = self.word_target[idx]
            input_context[0,] = self.word_context[idx]
            output[0,] = self.sample_labels[idx]
            loss = self.model.train_on_batch([input_target, input_context], output)
            if cnt % 100 == 0:
                print("Iteration {}, loss={}".format(cnt, loss))
            if cnt % 5000 == 1:
                self._check_similarity()

    def process(self):
        """ Main method to build and run the whole model.
        """
        self.generate_similarity_validation_set()
        self.create_samples()
        self.build()
        self.train()
        
    # TODO: Save network and provide option to load the network
    # TODO: Run model on AWS
    # TODO: Check out Tensorflow implementation
        

