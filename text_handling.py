import os
import nltk
import numpy as np

class TextData:
    """ 
    Class handling all the data / text handling issues.
    As several models need different input data (characters or tokens etc.)
    We can put the responsibility of this transformation into this class.
    """

    def load_tokenized_data(self, input_directory):
        """ 
        Load all the data from the given directory into a single list of tokens.
        That means that also punctuation etc. will be handled as a distinct token.
        """
        list_of_files = os.listdir(input_directory)
        tokens = []
        for file_name in list_of_files:
            infile = open(input_directory + '/' + file_name)
            try:
                single_file_text = infile.read()
                infile.close()
                single_file_tokens = nltk.word_tokenize(single_file_text)
                tokens += single_file_tokens
            except:
                print(file_name)
            
        return tokens

    def generate_tokenized_dataset(self, text):
        """
        Processes the given text into a dataset for a word2vec implementation
        by analyzing the frequency distribution and building a dataset for all words that occur 
        more than once. This dataset is consisting of the elements described in the return statement.
    
        Parameters
        ----------
        text : list of words
            The overall input to the embedding layer
    
        Returns
        ----------
        integer_text: list of int
            Each word from text will be replaced by an unique integer value.
        int2word_mapping: dictionary 
            Mapping each integer value to a specific word
        vocabulary_size : int
            The size of the vocabulary that will be generated.
            That means that the vocabulary_size most frequent words are used.
        """
        fdist = nltk.FreqDist(text)
        number_of_different_words = fdist.B()
        number_of_unique_words = len(fdist.hapaxes())
        vocabulary_size = number_of_different_words - number_of_unique_words
        frequency_distribution = [('HAPAX', number_of_unique_words)]
        HAPAX_INDEX = 0
        frequency_distribution.extend(fdist.most_common(vocabulary_size))
        
        word2int_mapping = dict()
        for word, _ in frequency_distribution:
            word2int_mapping[word] = len(word2int_mapping)
            
        integer_text = list()
        for word in text:
            if word in word2int_mapping:
                index = word2int_mapping[word]
            else:
                index = HAPAX_INDEX
            integer_text.append(index)
            
        int2word_mapping = dict(zip(word2int_mapping.values(), word2int_mapping.keys()))
        
        return integer_text, int2word_mapping, vocabulary_size


    def load_character_data(self, input_directory):
        """
        Load all the text data in the specified directory into one large string.
        """
        list_of_files = os.listdir(input_directory)
        output_text = ""
        for file_name in list_of_files:
            print("Processing file: ", file_name)
            infile = open(input_directory + '/' + file_name, 'r')
            file_text = infile.read()
            infile.close()
            output_text = output_text + " " + file_text
    
        return output_text
    
    def generate_character_dataset(self, text):
        """
        Generate an encoding of the text for each character as numpy array and provide the mapping to
        go back from the codes to the letters.
        """
        character_set = sorted(set(text))
        char2int_mapping = {character: idx for idx, character in enumerate(character_set)}
        int2char_mapping = dict(enumerate(character_set))
        character_set_size = len(character_set) 
        integer_text = np.array([char2int_mapping[character] for character in text], dtype=np.int32)
        
        return integer_text, int2char_mapping, char2int_mapping, character_set_size

    def split_int_text(self, input_text, training_percentage=0.9):
        """
        Split the given text into the three mandatory sets for training, validation and testing
        The ratio is currently set at 90% training size and an even split for test and validation of the rest.  
    
        Arguments
        ---------
        input_text: The text data to split
        training_percentage: The percentage of data which will go into the training set. 
        """
        split_frac = training_percentage
        position_to_split = int(len(input_text) * split_frac)
        train_text, val_text = input_text[:position_to_split], input_text[position_to_split:]
        
        position_to_split = len(val_text) // 2
        val_text, test_text = val_text[:position_to_split], val_text[position_to_split:]
        
        return train_text, val_text, test_text
            
        