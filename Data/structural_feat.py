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
            return("TENSE", [Counter([self.tense(trees[i]) for i in j]) for j in [full,two_third,one_third,zero]])
        elif which == 'modal':
            return("MODAL", [Counter(list(itertools.chain(*[self.modal(trees[i]) for i in j]))) for j in [full,two_third,one_third,zero]])
        elif which == 'semantics':
            return("SEMANTICS", [Counter([self.semantics(trees[i]) for i in j]) for j in [full,two_third,one_third,zero]])
        elif which == 'adj':
            return ("ADJECTIVE", [Counter([(self.adjective(trees[i], coref[i])) for i in j]) for j in [full,two_third,one_third,zero]])
        elif which == 'adv':
            return ("ADVERB", [Counter([(self.adverb(trees[i], coref[i])) for i in j]) for j in [full,two_third,one_third,zero]])
        

    #------------------------------ break into different obj later


    def tree_parser(self, sent): # Parse const tree by depth -> returns (depth level, tree content)

        pred = self.tree_predictor.predict(sentence=sent)
        # pred.keys() >>> ['class_probabilities', 'spans', 'tokens', 'pos_tags', 'num_spans', 'hierplane_tree', 'trees']
        return ([([i[0], (''.join(i[1]).split())]) for i in list(self.parse(list(pred['trees'])))], pred) 


    def parse(self, string): # Parsing str as stack-pop

        stack = []
        for i, c in enumerate(string):
            if c == '(':
                stack.append(i)
            elif c == ')' and stack:
                start = stack.pop()
                yield (len(stack), string[start + 1: i])


    #------------------------------ break into different obj later


    def tense(self, constTree):

        TENSE_DICT = {'VB':'base',
                    'VBD':'past tense',
                    'VBG':'present participle',
                    'VBN':'past participle',
                    'VBP':'present tense non-3rd.sg',
                    'VBZ':'present tense 3rd.sg'}

        # Future tense needs a separate lst because 'will' and 'shall' tagged as modals
        FUTURE_DICT = {'will':'future tense', 'shall':'future tense'} 
                    
        VP = [i[1] for i in constTree[0] if i[1][0] == 'VP'] # Extract only the verb phrases from the tree

        try: # Check if tense tag exists
            return (TENSE_DICT[[[re.sub('\(', '',j) for j in i if re.sub('\(', '',j) in TENSE_DICT][0] for i in VP][0]])      
        except:
            try: # Check if the future tag exists
                MD = [i[1][1].lower() for i in constTree[0] if i[1][0] == 'MD']
                return FUTURE_DICT[MD[0]]
            except:
                return '' # The sentence contains no tense information


    #------------------------------ break into different obj later


    def modal(self, constTree): # Analyze modals by their functions

        MODAL_DICT = {'can' : ['ability', 'suggestion', 'request'],
                    'could' : ['ability', 'request', 'suggestion'],
                    'may' : ['possibility', 'request', 'permission'],
                    'might' :['possibility'],
                    'must' : ['obligation', 'certainty'],
                    'should' : ['opinion', 'advice'],
                    'ought' : ['opinion', 'advice']}

        # These below are considered as modals when they are followed by 'to'                    
        VP_MODAL_DICT = {'able to' : ['ability'],
                    'have to' : ['obligation'],'has to' : ['obligation'],'had to' : ['obligation'],
                    'need to' : ['obligation'],'needs to' : ['obligation'],'needed to' : ['obligation']}
                    

        VP = [i[1] for i in constTree[0] if i[1][0] == 'VP']


        # List all MD tags in all VPs
        modal_found = [[re.sub(r'[\(\)]', '',j) for j in i if re.sub('\(', '',j) == 'MD'] for i in VP]

        # List all MODAL+'to' in all VPs
        # Convert to str first cus str search more efficient than list search
        word_str = [' '.join(map(str, k)) for k in [[re.sub(r'[\(\)]', '',j) 
                    for j in i if re.sub(r'[\(\)]', '',j).islower()] for i in VP]] 
        VP_modal_found = [[i.lower() for i in VP_MODAL_DICT.keys() if i.lower() in sub_lst] for sub_lst in word_str]


        # Extract actual word from MD list
        MD = [i[1][1].lower() for i in constTree[0] if i[1][0] == 'MD']
        # Store all possible functions for the word
        MD_intersect = list((set(MD)).intersection(set(MODAL_DICT))) 

        # Check for words in MODAL_DICT
        if (any(modal_found) and any((set(MD)).intersection(set(MODAL_DICT))) and len(MD_intersect) != 0):
            try:
                return MODAL_DICT[list((set(MD)).intersection(set(MODAL_DICT)))[0]]
            except:
                return ''

        # Check for words in VP_MODAL_DICT
        elif (not(any(modal_found)) and (any(VP_modal_found))):
            try:
                return VP_MODAL_DICT[VP_modal_found[0][0]]
            except:
                return VP_modal_found[0]
        else: return ''


    #------------------------------ break into different obj later

    #### THIS ONE IS ON HOLD, NOT ENOUGH INFO TO ANALYZE
    def semantics(self, constTree):
        pass
        """
        tree_token = [nltk.word_tokenize(i.replace("(", '')) for i in re.findall("\((.*?)\)", constTree[1]['trees'])]
        print(tree_token)
        vn_id = verbnet.classids([i[len(i)-1] for i in tree_token if (i[0] == 'VP' and i[1] != 'MD')][0].lower())
        print(vn_id)
        vn_frame = [i['description']['primary'] for i in verbnet.frames(vn_id)]
        print(vn_frame)
        exit()
        return ''
        
                for i in vn_frame:
                    pos_token_match = i.split('-')
                    

                    len(pos_token_match)
        """
    #------------------------------ break into different obj later


    def adjective(self, constTree, coref): # Categorize each coref cluster linked to a gendered pronoun

        output = []
        PRO_LIST = ["he", "she", "him", "her", "his", "hers", "himself", "herself"]

        for cluster in coref['clusters']: # List all linked mentions
            if any([((c[0] == c[1]) and (coref['document'][c[0]]).lower() in PRO_LIST) for c in cluster]):
                for j in range(cluster[0][0], cluster[0][1]+1):
                    if not(coref['document'][cluster[0][0]] in PRO_LIST):
                        output.append((constTree[1]['pos_tags'][j],coref['document'][j]))


        # Check and return if a particular structure exists
        if (len(output) >= 2 and output[0][0] == "DT" and output[1][0] == "NN"):
            return "DT", output[0][1].lower(), "NN(sg)" # 'the/a/an' + singular noun
        elif (len(output) >= 3 and output[0][0] == "DT" and output[1][0] == "NNS"):
            return "DT", output[0][1].lower(), "NNS(pl)" # 'the' + plural noun
        elif (len(output) >= 1 and output[0][0] == "NNS"):
            return "NNS(pl)" # plural noun
        elif (len(output) >= 1 and output[0][0] == "NNP"):
            return "NNP(proper)" # proper noun (NAMES)

        # Below checks the existance of adjectives in the mentions
        elif (len(output) >= 1 and any([i for i in output if (i[0] == "JJ" or i[0] == "JJR" or i[0] == "JJS")])):
            if any([i for i in output if (i[0] == "DT")]) and any([i for i in output if (i[0] == "NN")]):
                return ("DT", output[0][1].lower(), "JJ", "NN(sg)") # 'the/a/an' + adj + singular noun
            elif any([i for i in output if (i[0] == "DT")]) and any([i for i in output if (i[0] == "NNS")]):
                return ("DT", output[0][1].lower(), "JJ", "NNS(pl)") # 'the' + adj + plural noun
            elif all([i for i in output if (i[0] != "DT")]) and any([i for i in output if (i[0] == "NNS")]):
                return ("JJ", "NNS(pl)") # adj + plural noun
            else:
                return("OTHER JJ") # check for exceptions with adj
        else:
            return("OTHER") # all the other structures



    #------------------------------ break into different obj later


    def adverb(self, constTree, coref): # Counts the number of adverbs in each sent

        return len([(constTree[1]['pos_tags'][i], coref['document'][i]) 
            for i in range(0, len(constTree[1]['pos_tags'])) 
                if constTree[1]['pos_tags'][i] == "RB" 
                or constTree[1]['pos_tags'][i] == "RBR" 
                or constTree[1]['pos_tags'][i] == "RBS"])


    #------------------------------ break into different obj later


    def analyze(self): # stores all results

        all_trees, all_coref = {}, {}

        tree_pickle_path = "./pickle/trees.p"
        coref_pickle_path = "./pickle/coref.p"

        try:
            all_trees = pickle.load(open(tree_pickle_path, "rb"))
        except (IOError, EOFError):
            all_trees = self.load_all()
            os.makedirs(os.path.dirname(tree_pickle_path), exist_ok=True)
            with open(tree_pickle_path, 'wb') as f:
                pickle.dump(all_trees, f)

        try:
            all_coref = pickle.load(open(coref_pickle_path, "rb"))
        except (IOError, EOFError):
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