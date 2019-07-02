""" Labeling Attributes is a script to evaluate what created confusion amongst labelers in the Gender Generalizations
sentences or the Non Gender Generalization sentences.  Sentences are cleaned prior to counting words, symbols, female
pronouns, male pronouns and number of clusters. Following these counts we look into sentence labeling based of it source,
the variability of number of words and variability in symbols(remaining punctuation marks"""

# IMPORTS

import pandas as pd
import numpy as np
import re
import pprint
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk import ne_chunk, pos_tag
nltk.download('words')
from nltk.corpus import words as nltk_words
import allennlp
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive





class Label_Attr(object):

    def __init__(self, old_df, new_df):
        self. old_df = old_df
        self.new_df = new_df
        self. predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")

    def preprocess(self, data):
        clean_sentences = []

        for s in data:
            new_s = re.sub(r'\n+', ' ', s)                   # remove breakline
            new_s = re.sub(r'[^\w\/,:;.?\\\']', ' ', new_s)  # removes all weird characters except punctuations
            new_s = new_s.replace("//", "")                  # remove quotation marks
            new_s = re.sub(r'\d+.', '', new_s)               # removes number followed by a .
            new_s = re.sub(r'/\s\s+/g', ' ', new_s)
            new_s = new_s.replace('/', '')                   # removes forward slash
            new_s = new_s.replace('_', '')
            new_s = re.sub(r' +', ' ', new_s)                # removes extra spaces
            new_s = new_s.lstrip()                           # removes spaces at the beginning of a string
            new_s = new_s.rstrip()                           # removes spaces at the end of a string
            wordcount = len(new_s.split())
            if new_s != '' and wordcount > 5:  # sentences < 5 words are rejected
                clean_sentences.append(new_s)
            else:
                clean_sentences.append(np.nan)

        return clean_sentences

    def init_attr(self, old_sent, new_s):
        f_lst = ["she", "her", "hers", "herself"]
        m_lst = ["he", "him", "his", "himself"]
        f_cnt, m_cnt = 0, 0
        try:
            pred = predictor.predict(document=new_s)
            sym_pred = predictor.predict(document=old_sent)

            s_token = sym_pred['document']  # tokenized old sentences with symbols (before cleaning)
            token = pred['document']  # tokenized sentence
            word_length = sum(1 for i in token if i.isalpha() or i.isdigit())
            s_word_l = sum(1 for i in s_token if i.isalpha() or i.isdigit())
            symbol_cnt = len(s_token) - s_word_l
            num_cluster = len(pred['clusters'])
            for i in pred['clusters']:
                if any(c for c in i if ((c[0] == c[1]) and (token[c[0]]).lower() in f_lst)): f_cnt += 1
                if any(c for c in i if ((c[0] == c[1]) and (token[c[0]]).lower() in m_lst)): m_cnt += 1
            return [word_length, symbol_cnt, num_cluster, f_cnt, m_cnt]

        except:                  # to prevent coref bugs

            return [0, 0, 0, 0, 0]

    def conf_ctg(self, confidence, label):
        

    def df_build(self, ):


    def plotting(self,):


# For testing file, will connect to main.py once created
if __name__ == '__main__':
    input_path = '../Data/.csv'  ## set up for gutenberg
    final_candidates_filename = "master_finalcandidates_new"
    output_df = "master_allfilters_df"
    is_data_marked = False  # set to True if you want your final candidate sentences to contain html tags that will highlight the clusters
    # DATA IMPORT

    df = pd.read_csv(r'C:\Users\Y\Documents\MILA\Gender_Generalization\Data\Final_Labels.csv')
    data = [df['labelers'].tolist(), df['final label'].tolist(), df['confidence'].tolist(), df['sentence'].tolist(),
            df['source'].tolist()]

    dataloader = Dataloader(final_candidates_filename, filter_by_corpus, output_df)
    data = load_gutenberg(input_path)
    dataloader.filter_to_file(data, is_data_marked)
