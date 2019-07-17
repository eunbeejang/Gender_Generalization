""" TO RUN THE CODE:

python3 structural_feat.py ../Data/Clean_data_labels_golden_fixed.csv

"""


from allennlp.predictors.predictor import Predictor
from collections import Counter
import pandas as pd
import numpy as np

import argparse
import re
import itertools
import os
import wget
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#from nltk.corpus import verbnet
#nltk.download('verbnet')

import tense
import modal
import adjective
import adverb
import semantics

sum_all = lambda x: sum(map(sum_all, x)) if isinstance(x, list) else x
flatten = lambda l: [item for sublist in l for item in sublist]

def clean_output(data):
  complete = list(set(flatten([[i[0] for i in j] for j in data])))
  lst = []
  for i in data:
    mini_lst = []
    for j in complete:   
      try:
        idx = [k[0] for k in i].index(j)
        mini_lst.append((j, i[idx][1]))
      except ValueError:
        mini_lst.append((j, 0))      
    lst.append(mini_lst)
  return lst




class structFeat(object):

    def __init__(self, file_path):

        pretrained_tree_path = './allennlp_pretrained/elmo-constituency-parser-2018.03.14.tar.gz'
        pretrained_coref_path = './allennlp_pretrained/allennlp_coref-model-2018.02.05.tar.gz'

        if not os.path.exists(pretrained_tree_path):
            tree_url = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz"
        if not os.path.exists(pretrained_coref_path):
            coref_url = "https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz"
            wget.download(coref_url, pretrained_coref_path)

        self.tree_predictor = Predictor.from_path(pretrained_tree_path)
        self.coref_predictor = Predictor.from_path(pretrained_coref_path)

        self.data = pd.read_csv(file_path)
        self.lemmatizer = WordNetLemmatizer() 
        self.NP_pattern = []


    def tree_parser(self, sent): # Parse const tree by depth -> returns (depth level, tree content)

        pred = self.tree_predictor.predict(sentence=sent)
        # pred.keys() >>> ['class_probabilities', 'spans', 'tokens', 'pos_tags', 'num_spans', 'hierplane_tree', 'trees']
        return ([([i[0], (''.join(i[1]).split())]) for i in list(self.parse(list(pred['trees'])))], pred) 


    def parse(self, string): # Parsing str as stack-pop

        stack = []
        for i, char in enumerate(string):
            if char == '(':
                stack.append(i)
            elif char == ')' and stack:
                start = stack.pop()
                yield (len(stack), string[start + 1: i])


    def load_all(self, which='trees'): # Load all coref and trees

        if which == 'coref':
            output = []
            for i in self.data['Clean Sentence']:
                try:
                    output.append(self.coref_predictor.predict(i))
                except: # when coref doesnt exists
                    output.append({'top_spans': [], 'predicted_antecedents': [0, 0, 0, 0], 
                        'document': word_tokenize(i), 'clusters': [[]]})
            return output

        elif which == 'trees':
            return [self.tree_parser(i) for i in self.data['Clean Sentence']] 


    def save_data(self, trees, coref, file_path):
        data = [[i, tense.get_tense(i),modal.get_modal(i),adjective.get_adj(i,j),adverb.get_adv(i,j)] for i,j in zip(trees,coref)]
        df = pd.DataFrame.from_records(data, columns=["Index","Tense", "Modal Type", "NP Pattern", "Adverb Exists"], index=False)
        df.to_csv(file_path, encoding='utf-8')


    def count_all(self, trees, coref, which): # counting the result of each analysis by their confidence score
        full = np.where((self.data['Confidence'] == 1.0000) & (self.data['Final Label'] == 1))[0]
        two_third = np.where((self.data['Confidence'] != 1.0000) & (self.data['Final Label'] == 1))[0]
        one_third = np.where((self.data['Confidence'] != 1.0000) & (self.data['Final Label'] == 0))[0]
        zero = np.where((self.data['Confidence'] == 1.0000) & (self.data['Final Label'] == 0))[0]
        if which == 'tense':
            return [Counter([tense.get_tense(trees[i]) for i in j]) for j in [full,two_third,one_third,zero]]
        elif which == 'modal':
            return [Counter(list(itertools.chain(*[modal.get_modal(trees[i]) for i in j]))) for j in [full,two_third,one_third,zero]]
        elif which == 'semantics':
            return [Counter([semantics.get_sem(trees[i]) for i in j]) for j in [full,two_third,one_third,zero]]
        elif which == 'adj':
            return [Counter([(adjective.get_adj(trees[i], coref[i])) for i in j]) for j in [full,two_third,one_third,zero]]
        elif which == 'adv':
            return [Counter([(adverb.get_adv(trees[i], coref[i])) for i in j]) for j in [full,two_third,one_third,zero]]
    

    def get_ratio(self, result_dict, which='by_group'):
        total = sum_all([[i for i in j.values()] for j in result_dict])
        ratio_output = []

        if which == 'by_total':
            for i in range(0, len(result_dict)):
              ratio_output.append([(j[0], j[1]/total) for j in result_dict[i].items()])

        elif which == 'by_group':
            for i in range(0, len(result_dict)):
                ratio_output.append([(j[0], j[1]/sum(result_dict[i].values())) for j in result_dict[i].items()])

        return ratio_output


    def analyze(self, save=False): # stores all results

        all_trees, all_coref = {}, {}
        tree_pickle_path = "./pickle/trees.p"
        coref_pickle_path = "./pickle/coref.p"


        # save const tree as pickle
        try:
            all_trees = pickle.load(open(tree_pickle_path, "rb"))
        except:
            pass
        try:
            assert (len(all_trees) ==  len(self.data))
        except (IOError, EOFError, AssertionError):
            all_trees = self.load_all()
            os.makedirs(os.path.dirname(tree_pickle_path), exist_ok=True)
            with open(tree_pickle_path, 'wb') as f:
                pickle.dump(all_trees, f)

        # save coref as pickle
        try:
            all_coref = pickle.load(open(coref_pickle_path, "rb"))
        except:
            pass
        try:
            assert (len(all_coref) ==  len(self.data))
        except (IOError, EOFError, AssertionError):
            all_coref = self.load_all('coref')
            os.makedirs(os.path.dirname(coref_pickle_path), exist_ok=True)
            with open(coref_pickle_path, 'wb') as f:
                pickle.dump(all_coref, f)  
        
        # save as csv
        if save == True:
            self.save_data(all_trees, all_coref,'structFeat_out.csv')


        tense_result = self.count_all(all_trees, all_coref, 'tense')
        modal_result = self.count_all(all_trees, all_coref, 'modal')
        ##semantics_result = self.count_all(all_trees, 'semantics')
        adj_result = self.count_all(all_trees, all_coref, 'adj')
        adv_result = self.count_all(all_trees, all_coref, 'adv')


        # returned ratio ->> (by group, by total)
        return [[clean_output(d) for d in k] for k in ([self.get_ratio(tense_result), self.get_ratio(modal_result), self.get_ratio(adj_result), self.get_ratio(adv_result)],
        [self.get_ratio(tense_result,'by_total'), self.get_ratio(modal_result,'by_total'), self.get_ratio(adj_result,'by_total'), self.get_ratio(adv_result,'by_total')])]

        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    analyzer = structFeat(args.filename)
    #ratio_result = analyzer.analyze()


    ratio_result = analyzer.analyze()

    print(ratio_result)
    
    ratio_result_pickle_path = "./pickle/ratio_result.p"
    os.makedirs(os.path.dirname(ratio_result_pickle_path), exist_ok=True)
    with open(ratio_result_pickle_path, 'wb') as f:
        pickle.dump(ratio_result, f) 
    





    