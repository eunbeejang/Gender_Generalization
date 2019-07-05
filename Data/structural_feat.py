from allennlp.predictors.predictor import Predictor
from collections import Counter
import pandas as pd
import numpy as np
import argparse
import re
import itertools
from nltk.corpus import verbnet
import nltk
from nltk.stem import WordNetLemmatizer 
nltk.download('verbnet')


class structFeat(object):
    def __init__(self, file_path):
        self.tree_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
        self.coref_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
        self.data = pd.read_csv(file_path)
        self.lemmatizer = WordNetLemmatizer() 
        self.NP_pattern = []

    def load_all(self, which='trees'): # load all sentences as constituency tree
        if which == 'coref':
            return [self.coref_predictor.predict(document=i) for i in self.data['Clean Sentence']]
        else:
            return [self.constituency_tree(i) for i in self.data['Clean Sentence']] 


    def analyze(self):
        all_trees = self.load_all()
        tense_result = self.count_all(all_trees, 'tense')
        modal_result = self.count_all(all_trees, 'modal')
        ##semantics_result = self.count_all(all_trees, 'semantics')
        all_coref = self.load_all('coref')
        adj_result = self.count_all(all_trees, all_coref, 'adj')
        adv_result = self.count_all(all_trees, all_coref, 'adv')
        
        print(tense_result)
        print(modal_result)
        print(adj_result)
        print(adv_result)

    def count_all(self, trees, coref, which):
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
    def constituency_tree(self, sent):
        pred = self.tree_predictor.predict(sentence=sent)
        # pred.keys() >> ['class_probabilities', 'spans', 'tokens', 'pos_tags', 'num_spans', 'hierplane_tree', 'trees']
        return ([([i[0], (''.join(i[1]).split())]) for i in list(self.parse(list(pred['trees'])))], pred) #(depth, tree content)

    def parse(self, string):
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
        FUTURE_DICT = {'will':'future tense', 'shall':'future tense'}
                    
        VP = [i[1] for i in constTree[0] if i[1][0] == 'VP']

        try:
            return (TENSE_DICT[[[re.sub('\(', '',j) for j in i if re.sub('\(', '',j) in TENSE_DICT][0] for i in VP][0]])      
        except:
            try: 
                MD = [i[1][1].lower() for i in constTree[0] if i[1][0] == 'MD']
                return FUTURE_DICT[MD[0]]
            except:
                return ''

    #------------------------------ break into different obj later

    def modal(self, constTree):
        MODAL_DICT = {'can' : ['ability', 'suggestion', 'request'],
                    'could' : ['ability', 'request', 'suggestion'],
                    'may' : ['possibility', 'request', 'permission'],
                    'might' :['possibility'],
                    'must' : ['obligation', 'certainty'],
                    'should' : ['opinion', 'advice'],
                    'ought' : ['opinion', 'advice']
                    }
        VP_MODAL_DICT = {'able to' : ['ability'],
                    'have to' : ['obligation'],'has to' : ['obligation'],'had to' : ['obligation'],
                    'need to' : ['obligation'],'needs to' : ['obligation'],'needed to' : ['obligation']
                    }

        VP = [i[1] for i in constTree[0] if i[1][0] == 'VP']

        modal_found = [[re.sub(r'[\(\)]', '',j) for j in i if re.sub('\(', '',j) == 'MD'] for i in VP]

        word_str = [' '.join(map(str, k)) for k in [[re.sub(r'[\(\)]', '',j) for j in i if re.sub(r'[\(\)]', '',j).islower()] for i in VP]]
        VP_modal_found = [[i.lower() for i in VP_MODAL_DICT.keys() if i.lower() in sub_lst] for sub_lst in word_str]

        MD = [i[1][1].lower() for i in constTree[0] if i[1][0] == 'MD']
        MD_intersect = list((set(MD)).intersection(set(MODAL_DICT))) # find the modal from MODAL_DICT

        if (any(modal_found) and any((set(MD)).intersection(set(MODAL_DICT))) and len(MD_intersect) != 0):
            try:
                return MODAL_DICT[list((set(MD)).intersection(set(MODAL_DICT)))[0]]
            except:
                return ''

        elif (not(any(modal_found)) and (any(VP_modal_found))):
            try:
                return VP_MODAL_DICT[VP_modal_found[0][0]]
            except:
                return VP_modal_found[0][0]        # CHECK WHEN RUNNING ON WHOLE DATA
        else: return ''


    #------------------------------ break into different obj later

    def semantics(self, constTree):

        tree_token = [nltk.word_tokenize(i.replace("(", '')) for i in re.findall("\((.*?)\)", constTree[1]['trees'])]
        print(tree_token)
        vn_id = verbnet.classids([i[len(i)-1] for i in tree_token if (i[0] == 'VP' and i[1] != 'MD')][0].lower())
        print(vn_id)
        vn_frame = [i['description']['primary'] for i in verbnet.frames(vn_id)]
        print(vn_frame)
        exit()
        return ''
        """
                for i in vn_frame:
                    pos_token_match = i.split('-')
                    

                    len(pos_token_match)
        """
    #------------------------------ break into different obj later


    def adjective(self, constTree, coref):

        output = []
        PRO_LIST = ["he", "she", "him", "her", "his", "hers", "himself", "herself"]
        for cluster in coref['clusters']:
            if any([((c[0] == c[1]) and (coref['document'][c[0]]).lower() in PRO_LIST) for c in cluster]):
                for j in range(cluster[0][0], cluster[0][1]+1):
                    if not(coref['document'][cluster[0][0]] in PRO_LIST):
                        output.append((constTree[1]['pos_tags'][j],coref['document'][j]))

        if (len(output) >= 2 and output[0][0] == "DT" and output[1][0] == "NN"):
            return "DT", output[0][1].lower(), "NN(sg)"
        elif (len(output) >= 3 and output[0][0] == "DT" and output[1][0] == "NNS"):
            return "DT", output[0][1].lower(), "NNS(pl)"
        elif (len(output) >= 1 and output[0][0] == "NNS"):
            return "NNS(pl)"
        elif (len(output) >= 1 and output[0][0] == "NNP"):
            return "NNP(proper)"
        elif (len(output) >= 1 and any([i for i in output if (i[0] == "JJ" or i[0] == "JJR" or i[0] == "JJS")])):
            if any([i for i in output if (i[0] == "DT")]) and any([i for i in output if (i[0] == "NN")]):
                return ("DT", output[0][1].lower(), "JJ", "NN(sg)")
            elif any([i for i in output if (i[0] == "DT")]) and any([i for i in output if (i[0] == "NNS")]):
                return ("DT", output[0][1].lower(), "JJ", "NNS(pl)")
            elif all([i for i in output if (i[0] != "DT")]) and any([i for i in output if (i[0] == "NNS")]):
                return ("JJ", "NNS(pl)")
            else:
                return("OTHER JJ")
        else:
            return("OTHER")

    #------------------------------ break into different obj later


    def adverb(self, constTree, coref):

        return len([(constTree[1]['pos_tags'][i], coref['document'][i]) 
            for i in range(0, len(constTree[1]['pos_tags'])) 
                if constTree[1]['pos_tags'][i] == "RB" 
                or constTree[1]['pos_tags'][i] == "RBR" 
                or constTree[1]['pos_tags'][i] == "RBS"])



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    analyzer = structFeat(args.filename)
    #print(analyzer.__dict__['self.data'][:10])
    all_trees = analyzer.analyze()



if __name__ == '__main__':
    main()