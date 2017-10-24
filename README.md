# LatinLearner

Deep Learning project for the Latin Language with the goal to build a Latin Chatbot and improve the understanding of the Latin language.

## Purpose

This project has mainly educational purposes.
1. Learn the technologies required to build a deep learning system that can process Latin Language
2. Practice coding skills, github, testing etc.
3. Provide the Latin language model as a tool for use in school or university or on a website / app / etc.

## Data

To build a useful language model it is essential to have enough high quality input data about the specific language.

There are several sources for latin text on the internet.

* The [Latin Library](http://thelatinlibrary.com/index.html)
* [Perseus Database](http://www.perseus.tufts.edu/hopper/opensource/download)
* Many more

As I wanted to learn something on data collection and scraping, I developed the script [`scrape_latin_texts.py`](scrape_latin_texts.py) to download the data from the Latin Library and store those as text files. From the Perseus Database you can directly download a collection of latin and greek texts with English Translations. However, extensive postprocessing is required here which is adressed in the next section.

All the data will be stored in a directory called 'library' which you have to **generate with the scraping script before running the cleaning script**.

## Data Cleaning and Processing

The data from the scraping tool contains a lot of information that we don't want to have the neural network to learn or output to the user. Therefore, it is essential to clean this data. This is done by running the script [`clean_data.py`](clean_data.py).

The script can identify duplicates which might have been downloaded in several attempts of doing the scraping.
See the documentation of the function `find_duplicates` for details.

Additionally, the code will also remove certain parts of the files like arabic numbers which are used for marking paragraphs are information in square brackets or the references to the Latin Library at the end.

The output can then directly be used for processing with the [LSTM model](LSTM_model.py).

To make the text usable for the embeddings, you also need to remove all punctuation, which can be done by including a call to the respective function in the cleaning script.
Additionally, you need to concatenate all the small files into one large file by using `cat clean_library/*.txt > latin_texts.txt`.

## Models

### Char RNN
Based on the [Udacity AI Nanodegree Anna KaRNNa jupyter notebook](https://github.com/udacity/deep-learning/tree/master/intro-to-rnns) for text generation, I adopted the code to generate Latin text and put it into a [seperate Python file](LSTM_model.py). The code is also available in the [notebook](LatinLeaRNNr.ipynb). There is also a pretrained model available in the directory `models` which has been trained for several hours on a K80 GPU on AWS. Using the command line options in `latin_learner.py` you can either just load the model and generate text, continue training based on the existing model, or start training from scratch when you want to experience with the model.

### Embeddings

Based on the Tensorflow word2vec tutorial (see references), you can build an embedding space for Latin based on the texts from famous authors like Caesr, Cicero etc.

As mentioned in the tutorials page you need to compile the ops as follows:

```shell
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
```

On Mac, add `-undefined dynamic_lookup` to the g++ command.

(For an explanation of what this is doing, see the tutorial on [Adding a New Op to TensorFlow](https://www.tensorflow.org/how_tos/adding_an_op/#building_the_op_library). The flag `-D_GLIBCXX_USE_CXX11_ABI=0` is included to support newer versions of gcc. However, if you compiled TensorFlow from source using gcc 5 or later, you may need to exclude the flag.)
Then run using:

```shell
python word2vec.py \
  --train_data=latin_texts.txt \
  --eval_data=latin_questions.txt \
  --save_path=logs/
```

If you want to restore the latest pretrained embedding model and experiment with an interactive Python session on the data, you should add the following arguments:

```--training=False --checkpoint=latest
```

The embedding model of the Latin Language with Keras in the [embedding.py](embedding.py) file is based on the model from the Keras word2vec tutorial mentioned in the references. The results are not really convincing. 
I just wanted to experiment with the Keras version integrated into TensorFlow. It seems that not many people are actually using this. 


## Chatbot

Usually, you would need some kind of question answer pairs to build a chatbot. As this is not available for Latin, I used the Char RNN and primed it with the input text and then let it generate output text based on the respective input text.

## Results

### Char RNN

### Embeddings



## Future work / TODO



## Environment

* This package is using Tensorflow 1.3 and Keras which is included in Tensorflow.
* As there have been changes in the API of Keras this package is required.
* You can generate the environment to run this using the provided [environment file](environment.yml) and then run `conda env create -f environment.yml`
 
## Contributing

Anybody interested in working together on this please contact me through the mail in my github profile.

## License

Using GPL v3 for this project. Details see [LICENSE](LICENSE) file in the repo.

## References

* Tensorflow word2vec Tutorial https://www.tensorflow.org/tutorials/word2vec 
* and https://github.com/tensorflow/models/tree/master/tutorials/embedding
* Udacity AI Nanodegree Anna KaRNNa notebook: https://github.com/udacity/deep-learning/tree/master/intro-to-rnns
* Keras Examples: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
* Keras word2vec Tutorial: http://adventuresinmachinelearning.com/word2vec-keras-tutorial/
