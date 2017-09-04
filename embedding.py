import tensorflow as tf
import os
import nltk

def read_data(input_directory):
    listOfFiles = os.listdir(input_directory)
    tokens = []
    for fileName in listOfFiles:
        infile = open(input_directory + '/' + fileName)
        singleFileText = infile.read()
        infile.close()
        singleFileTokens = nltk.word_tokenize(singleFileText)
        tokens += singleFileTokens
        
    return tokens

def generate_dataset(texts):
    """Processes the given texts into a dataset for a word2vec implementation
    by analyzing the frequency distribution and building a dataset for all words that occur 
    more than once. This dataset is consisting of the elements described in the return statement.

    Parameters
    ----------
    texts : list of words
        The overall input to the embedding layer

    Returns
    ----------
    integer_text: list of int
        Each word from texts will be replaced by an unique integer value.
    frequency_distribution: list of tuples
        Containing the frequency distribution of the words from texts. 
        Additionally, a special token 'HAPAX' is introduced that counts the words occuring only once.
    word2int_mapping: dictionary
        Mapping each word to the designated integer value
    int2word_mapping: dictionary 
        Mapping each integer value to a specific word
    vocabulary_size : int
        The size of the vocabulary that will be generated.
        That means that the vocabulary_size most frequent words are used.
    """
    fdist = nltk.FreqDist(texts)
    number_of_different_words = fdist.B()
    number_of_unique_words = len(fdist.hapaxes())
    vocabulary_size = number_of_different_words - number_of_unique_words
    frequency_distribution = [('HAPAX', -1)]
    HAPAX_INDEX = 0
    frequency_distribution.extend(fdist.most_common(vocabulary_size))
    word2int_mapping = dict()
    for word, _ in frequency_distribution:
        word2int_mapping[word] = len(word2int_mapping)
    integer_text = list()
    hapax_count = 0
    for word in texts:
        if word in word2int_mapping:
            index = word2int_mapping[word]
        else:
            index = HAPAX_INDEX
            hapax_count += 1
        integer_text.append(index)
    frequency_distribution[0] = ('HAPAX', hapax_count)
    int2word_mapping = dict(zip(word2int_mapping.values(), word2int_mapping.keys()))
    return integer_text, frequency_distribution, word2int_mapping, int2word_mapping, vocabulary_size

def process_input():
    texts = read_data('test_library')
    int_text, freq_dist, word2int, int2word, vocabulary_size = generate_dataset(texts)


