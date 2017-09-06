# LatinLearner

Deep Learning project for the Latin Language with the ultimate goal to build a Latin Chatbot

## Purpose

This project has mainly educational purposes.
1. Learn the technologies required to build a deep learning system that can process Latin Language
2. Practice coding skills, github, testing etc.
3. Provide the Latin language model as a tool for use in school or university or on a website / app / Facebook etc.

## Data

To build a useful language model it is essential to have enough high quality input data about the specific language.

There are several sources for latin text on the internet.

* The [Latin Library](http://thelatinlibrary.com/index.html)
* [Perseus Database](http://www.perseus.tufts.edu/hopper/opensource/download)
* Many more

As I wanted to learn something on data collection and scraping, I developed the script [`scrape_latin_texts.py`](scrape_latin_texts.py) to download the data from the Latin Library and store those as text files. From the Perseus Database you can directly download a collection of latin and greek texts with English Translations. However, extensive postprocessing is required here which is adressed in the next section.

All the data will be stored in a directory called 'library' which you have to **generate before running the script**.

## Data Cleaning and Processing

The data from the scraping tool contains a lot of information that we don't want to have the neural network to learn or output to the user. Therefore, it is essential to clean this data. This is done by running the script [`clean_data.py`](clean_data.py).

The script can identify duplicates which might have been downloaded in several attempts of doing the scraping.
See the documentation of the function `find_duplicates` for details.

Additionally, the code will also remove certain parts of the files like arabic numbers which are used for marking paragraphs are information in square brackets or the references to the Latin Library at the end.

## Models

* First build an Embedding of the Latin Language with word2vec and visualize it
* Concept for further processing as we don't have explicit questions / answers or translations which can be easily build into a cost function.
* Planning to use a seq2seq model.
* Want to use Keras for this to get some experience with Keras with Tensorflow backend
* A text generation model can be used to generate text, maybe conditioned on some input text

## Chatbot

* Maybe build an iPhone App or something in WhatsApp, Alexa Skill etc. TBD

## Environment

* This package is using Tensorflow 1.3 and Keras which is included in Tensorflow.
* As there have been changes in the API of Keras this package is required.
* You can generate the environment to run this using the provided [environment file](environment.yml) and then run `conda env create -f environment.yml`
 
## Contributing

Anybody interested in working together on this please contact me through the mail in my github profile.

## License

Using GPL v3 for this project. Details see [LICENSE](LICENSE) file in the repo.

## References

* Keras word2vec Tutorial: http://adventuresinmachinelearning.com/word2vec-keras-tutorial/
* Keras Examples: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
* Tensorflow word2vec Tutorial https://www.tensorflow.org/tutorials/word2vec