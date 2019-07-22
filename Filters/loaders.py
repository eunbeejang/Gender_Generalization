""" This file is to load and clean any incoming dataset that we would like to send to our dataloader."""

""" Import statements"""
import pandas as pd
import re
import unidecode
import pprint
import nltk
from nltk.tokenize import sent_tokenize
pp = pprint.PrettyPrinter(indent=1)


""" Preprocessing function for IMDB dataset specifically """


def preprocess_IMDB(data):
    new_data = []
    for line in data:
        new_line = re.sub('<.*?>', ' ', line)  # remove HTML tags and replace with space
        sentences = sent_tokenize(new_line)  # sentence tokenization
        for sentence in sentences:
            new_sentence = re.sub('"', '', sentence)  # remove quotation marks
            new_sentence = unidecode.unidecode(
                new_sentence)  # removes accents and represents any unicode to closest ascii
            new_sentence = re.sub(r'\s+', ' ', new_sentence)  # Eliminate duplicate whitespaces
            # new_sentence = re.sub(r'[^\w\s]', '', new_sentence) # remove punctuation
            # new_sentence = new_sentence.lower() # convert to lower case
            wordcount = len(new_sentence.split())
            if new_sentence != '' and wordcount > 2:  # wordcount of 2 or lower doesn't make sense for coref
                new_data.append(new_sentence)
    return new_data


"""Preprocessing function for any gutenberg e-book. This breaks down the text into sentences."""


def preprocess_gutenberg(data):
    clean_sentences = []
    with open(data) as f:
        text = f.read()
    text = re.sub(r'(M\w{1,2})\.', r'\1', text)  # removes the '.' in Mr. and Mrs.
    sentences = sent_tokenize(text)
    new_sentences = [re.sub(r'\n+', ' ', s) for s in sentences]   # remove \n (line break)
    clean_s = [re.sub('"', '', quotation) for quotation in new_sentences] #remove quotation marks
    for sentence in clean_s:
        wordcount = len(sentence.split())
        if sentence != '' and wordcount > 2:   # wordcount of 2 or lower doesn't make sense for coref
            clean_sentences.append(sentence)
    return clean_sentences


""" load_gutenberg takes any gutenberg ebook in plain UTF-8 txt file and is sends to the preprocessing_
gutenberg function. """


def load_gutenberg(data_path):
    df = pd.read_csv(data_path)
    #df = data_path  # read in input_file
    sentences = [df['index'].tolist(),df['Sentences'].tolist(),df['Corpus'].tolist()]
    print("Size of Dataset: ", len(sentences))

    #df_clean = preprocess_gutenberg(df)  # cleaning sentences
    #print("Size of Dataset after clean: ", len(df_clean))

    #print("size of small subset for testing: ", len(df_clean))

    return sentences


""" load_IMDB takes the IMDB dataset, selects the first column and is sent 
to the preprocessing_IMDB function. """


def load_IMDB(data_path):
    # read in input_file
    df = pd.read_csv(data_path, delimiter='\t', encoding='utf-8', header=None, squeeze=True)

    print("Size of Dataset: ", len(df))
    print("Columns :", df.columns)

    df_clean = preprocess_IMDB(df[0])  # cleaning sentences
    print("Size of Dataset after clean: ", len(df_clean))

    return df_clean

# load general purpose text dataset that has one data text per line
def load_general(data_path):

    # read in input_file
    df = pd.read_csv(data_path, delimiter='\t', encoding='utf-8', header=None, squeeze=True)

    #info about the dataset
    print("Size of Dataset: ", len(df))

    return df