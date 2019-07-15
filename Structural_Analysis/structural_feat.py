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
from nltk.corpus import verbnet
#nltk.download('verbnet')

import tense
import modal
import adjective
import adverb
import semantics


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

        self.data = pd.read_csv(file_path)[:20]
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


    def count_all(self, trees, coref, which): # counting the result of each analysis by their confidence score
        full = np.where((self.data['Confidence'] == 1.0000) & (self.data['Final Label'] == 1))[0]
        two_third = np.where((self.data['Confidence'] != 1.0000) & (self.data['Final Label'] == 1))[0]
        one_third = np.where((self.data['Confidence'] != 1.0000) & (self.data['Final Label'] == 0))[0]
        zero = np.where((self.data['Confidence'] == 1.0000) & (self.data['Final Label'] == 0))[0]
        if which == 'tense':
            return("TENSE", [Counter([tense.get_tense(trees[i]) for i in j]) for j in [full,two_third,one_third,zero]])
        elif which == 'modal':
            return("MODAL", [Counter(list(itertools.chain(*[modal.get_modal(trees[i]) for i in j]))) for j in [full,two_third,one_third,zero]])
        elif which == 'semantics':
            return("SEMANTICS", [Counter([semantics.get_sem(trees[i]) for i in j]) for j in [full,two_third,one_third,zero]])
        elif which == 'adj':
            return ("ADJECTIVE", [Counter([(adjective.get_adj(trees[i], coref[i])) for i in j]) for j in [full,two_third,one_third,zero]])
        elif which == 'adv':
            return ("ADVERB", [Counter([(adverb.get_adv(trees[i], coref[i])) for i in j]) for j in [full,two_third,one_third,zero]])
        

    def save_data(self, trees, coref, which):
        if which == 'tense':
            return [tense.get_tense(i) for i in trees]
        elif which == 'modal':
            return [modal.get_modal(i) for i in trees]
        elif which == 'semantics':
            return [semantics.get_sem(i) for i in trees]
        elif which == 'adj':
            return [adjective.get_adj(i,j) for i,j in zip(trees,coref)]
        elif which == 'adv':
            return [adverb.get_adv(trees[i], coref[i]) for i,j in zip(trees,coref)]
        

    def analyze(self): # stores all results

        all_trees, all_coref = {}, {}

        tree_pickle_path = "./pickle/trees.p"
        coref_pickle_path = "./pickle/coref.p"

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
        

        tense_result = self.count_all(all_trees, all_coref, 'tense')
        modal_result = self.count_all(all_trees, all_coref, 'modal')
        ##semantics_result = self.count_all(all_trees, 'semantics')
        adj_result = self.count_all(all_trees, all_coref, 'adj')
        adv_result = self.count_all(all_trees, all_coref, 'adv')
        return (tense_result, modal_result, adj_result, adv_result)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    analyzer = structFeat(args.filename)
    result = analyzer.analyze()
    print(result)


if __name__ == '__main__':
    main()