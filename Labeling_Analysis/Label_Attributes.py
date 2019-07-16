""" Labeling Attributes is a script to evaluate what created confusion amongst labelers in the Gender Generalizations
sentences or the Non Gender Generalization sentences.  Sentences are cleaned prior to counting words, symbols, female
pronouns, male pronouns and number of clusters. Following these counts we look into sentence labeling based of it source
, the variability of number of words and variability in symbols(remaining punctuation marks"""

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
#import allennlp
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from collections import Counter

class Label_Attr(object):

    def __init__(self):
        #self.new_df = new_df
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
            if new_s != '' and wordcount > 5:                # sentences < 5 words are rejected
                clean_sentences.append(new_s)
            else:
                clean_sentences.append(np.nan)

        return clean_sentences

    def coref_attr(self, new_s, old_sent):
        """ This function will return word lentgh, number of symbols, number of coreference clusters, female and male
        mentions in a sentence. The TRY and EXCEPT are put in place for sentences that do not meet coreference
        requirements in their sentence. We preserve the sentence and set the labeling attributes for  word length,
        number of symbols, number of coreference clusters, female and male to zero since we depend on the coreference
        tokenization. """

        f_lst = ["she", "her", "hers", "herself"]            # female pronouns
        m_lst = ["he", "him", "his", "himself"]              # male pronouns
        f_cnt, m_cnt = 0, 0                                  # initial count
        try:
            pred = self.predictor.predict(document=new_s)         # Coref on clean sentences to get correct clusters
            sym_pred = self.predictor.predict(document=old_sent)  # Coref on old sentences to get tokens for symbol count

            s_token = sym_pred['document']                   # tokenized old sentences with symbols (before cleaning)
            token = pred['document']                         # tokenized clean sentence to do word count

            word_length = sum(1 for i in token if i.isalpha() or i.isdigit())    # word count on new sentences
            s_word_l = sum(1 for i in s_token if i.isalpha() or i.isdigit())     # word count on old sentences

            symbol_cnt = len(s_token) - s_word_l                                 # symbol count (Total tokens - words) = symbol
            num_cluster = len(pred['clusters'])                                  #  coref cluster count
            for i in pred['clusters']:
                if any(c for c in i if ((c[0] == c[1]) and (token[c[0]]).lower() in f_lst)): f_cnt += 1
                if any(c for c in i if ((c[0] == c[1]) and (token[c[0]]).lower() in m_lst)): m_cnt += 1
            return [word_length, symbol_cnt, num_cluster, f_cnt, m_cnt]

        except:                                              # to prevent coref bugs

            return [0, 0, 0, 0, 0]

    def conf_section(self, data):
        """ This function will return which sentences are 100 % confident, 66 % confident, 33 % confident and 0%
        confident with respect to gender generalization. This means 0 % confident is actually 100%confident that this
        is not a gender generalization."""

        conf_100 = [(data['Confidence'].iloc[i] == 1 and data['Final Label'].iloc[i] == 1) for i in range(0,len(data['Confidence']))]
        conf_66 = [data['Confidence'].iloc[i] < 1 and data['Final Label'].iloc[i] == 1 for i in range(0,len(data['Confidence']))]
        conf_33 = [data['Confidence'].iloc[i] < 1 and data['Final Label'].iloc[i] == 0 for i in range(0,len(data['Confidence']))]
        conf_0 = [data['Confidence'].iloc[i] == 1 and data['Final Label'].iloc[i] == 0 for i in range(0,len(data['Confidence']))]

        return [conf_100, conf_66, conf_33, conf_0]

    def count_ctg(self, data, ctg):
        """ This function will return groups of the amount of sentences which have different word lentgh, corpus origin,
        symbol count  and number of clusters. The goal is to see which of these affected the labeling the most."""

        if ctg == 'corpus':
            z = Counter(data['Corpus'])
            return z

        if ctg == 'word':
            y = np.count_nonzero((5 < data['Word Count']) & (data['Word Count'] <= 25))
            x = np.count_nonzero((25 < data['Word Count']) & (data['Word Count'] <= 50))
            w = np.count_nonzero((50 < data['Word Count']) & (data['Word Count'] <= 100))
            t = np.count_nonzero((100 < data['Word Count']) & (data['Word Count'] <= 170))
            a = np.count_nonzero(data['Word Count'] == 0)
            return y, x, w, t, a

        if ctg == 'symbol':
            y = np.count_nonzero((0 < data['Symbol Count'])&(data['Symbol Count'] <= 5))
            x = np.count_nonzero((5< data['Symbol Count'])&(data['Symbol Count'] <= 10))
            w = np.count_nonzero((10 < data['Symbol Count'])&(data['Symbol Count'] <= 15))
            t = np.count_nonzero((15 < data['Symbol Count'])&(data['Symbol Count'] <= 40))
            a = np.count_nonzero(data['Symbol Count'] ==0)
            return y,x,w,t,a

        if ctg == 'cluster':
            z = Counter(data['Number of Clusters'])
            return z

    def df_build(self, df):
        """ This function will return the dataframe with wanted attributes from different functions. This function will
        equally remove all unwanted values and build off the exisiting csv which is uploaded in the pipeline."""

        # SECTION FOR PREPROCESSING THE EXISTING CSV

        #df = pd.read_csv(data_path)                        # Read data path
        data = [df['labelers'].tolist(), df['final label'].tolist(), df['confidence'].tolist(), df['sentence'].tolist(),
                df['source'].tolist()]                     # Convert to lists for manipulation
        clean_s = self.preprocess(data[3])                      # Preprocess old sentences
        data_clean = {'Clean Sentence': clean_s, 'Old Sentence': data[3], 'Corpus': data[4], 'Final Label': data[1],
                      'Number of Labelers': data[0], 'Confidence': data[2]}    # Build new_df
        df_clean = pd.DataFrame(data_clean)

        new_df = df_clean.dropna()                             # Remove NaN from dataframe
        new_df['Corpus'] = new_df['Corpus'].replace(
            {'Business Hints for men and women': 'Business Hints for Men and Women', 'boy scouts ': 'boy scouts',
             'honey bee': 'honeybee', 'mind reading ': 'mind reading'})        # Remove typos in Corpus column

        # SECTION FOR ADDING THE NEW COLUMNS TO DATAFRAME
        sent_att = []
        a = new_df['Clean Sentence']
        b = new_df['Old Sentence']
        for x, y in zip(a, b):
            sent_att.append(self.coref_attr(x, y))
        print(a[:10])
        print(b[:10])
        print(sent_att[:10])
        w_len = [i[0] for i in sent_att]
        sym_c = [i[1] for i in sent_att]
        n_clus = [i[2] for i in sent_att]
        f = [i[3] for i in sent_att]
        m = [i[4] for i in sent_att]


        new_df['Word Count'] = w_len
        new_df['Symbol Count'] = sym_c
        new_df['Number of Clusters'] = n_clus
        new_df['Female Pronoun'] = f
        new_df['Male Pronoun'] = m
        #new_df.append(w_len)
        print(w_len[:10])

        return new_df

    def final_count(self, conf, data):
        """ This function will return the corpus, word lentgh, symbol count and cluster per different interval of
        confidence obtained by the labelers."""
        conf_list = []
        for i in range(0, len(conf)):
            if conf[i] == True:
                conf_list.append(data.iloc[i])              # iloc for full row of dataframe

        conf_pd = pd.DataFrame(conf_list)                   # make new TRUE dataframe for confidence of interest
        corpus = self.count_ctg(conf_pd, 'corpus')
        wordlength = self.count_ctg(conf_pd, 'word')
        symbolcount = self.count_ctg(conf_pd, 'symbol')
        cluster = self.count_ctg(conf_pd, 'cluster')


        print(wordlength)

        return corpus, wordlength, symbolcount, cluster

    def load_all(self,data):

        data = self.df_build(data)
        gb1 = [self.conf_section(data)][0][0]                # Get confidence for 100 Gender Generalization
        gb66 = [self.conf_section(data)][0][1]               # Get confidence for 66 Gender Generalization
        nb1 = [self.conf_section(data)][0][2]                # Get confidence for 100 Not Gender Generalization
        nb66 = [self.conf_section(data)][0][3]               # Get confidence for 66 Not Gender Generalization

        final_gb1 = self.final_count(gb1, data)              # Pass through final_count fn to apply to the confidence
        final_gb66 = self.final_count(gb66, data)
        final_nb1 = self.final_count(nb1, data)
        final_nb66 = self.final_count(nb66, data)

        #print("100 % gender generalization", final_gb1)
        #print("66 % gender generalization", final_gb66)
        #print("100 % not gender generalization", final_nb1)
        #print("66 % not gender generalization", final_nb66)

        print(final_gb1)
        print(gb1)

        return final_gb1, final_gb66, final_nb1, final_nb66




# For testing file, will connect to main.py once created
if __name__ == '__main__':
    #data_path = '../Data/NotClean_data_labels.csv'  ##
    data_path =  r'C:\Users\Y\Documents\MILA\Gender_Generalization\Data\NotClean_data_labels.csv'
    data = pd.read_csv(data_path)
    label_attr = Label_Attr()
    #data = label_attr.df_build(data)
    values = label_attr.load_all(data)

    # DATA IMPORT

    #data_path = pd.read_csv(r'C:\Users\Y\Documents\MILA\Gender_Generalization\Data\Final_Labels.csv')
    #data = [df['labelers'].tolist(), df['final label'].tolist(), df['confidence'].tolist(), df['sentence'].tolist(),
    #        df['source'].tolist()]

    #dataloader = Dataloader(final_candidates_filename, filter_by_corpus, output_df)
    #data = load_gutenberg(input_path)
    #dataloader.filter_to_file(data, is_data_marked)
