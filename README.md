# LatinLearner
Deep Learning project for the Latin Language with the ultimate goal to build a Latin Chatbot

## Purpose

This project has mainly educational purposes.
1. Learn the technologies required to build a deep learning system that can talk in Latin
2. If this is working, provide the chatbot as a tool for use in school or university or on a website

## Data

To build a useful language model it is essential to have enough high quality input data about the specific language.

There are several sources for latin text on the internet.

* The Latin Library: http://thelatinlibrary.com/index.html
* Perseus Database: http://www.perseus.tufts.edu/hopper/opensource/download
* Many more

As I wanted to learn something on data collection and scraping, I developed a script to download the data from
the Latin Library and store those as text files. From the Perseus Database you can directly download a collection
of latin and greek texts with English Translations. However, extensive postprocessing is required here.

## Data Cleaning and Processing

1. Need to remove the header and footer of the documents.
2. Probably use NLTK to tokenize the whole text.

## Models

* Planning to use a seq2seq model.
* Want to use Keras for this

## Chatbot

* Maybe build an iPhone App or something in WhatsApp, Alexa Skill etc. TBD
 