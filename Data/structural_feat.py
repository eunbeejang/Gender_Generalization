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
        self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
        self.data = pd.read_csv(file_path)[:10]
        self.lemmatizer = WordNetLemmatizer() 


    def load_all(self): # load all sentences as constituency tree
        return [self.constituency_tree(i) for i in self.data['Sentence']]

    def analyze(self):
        all_trees = self.load_all()
        #tense_result = self.count_all(all_trees, 'tense')
        #modal_result = self.count_all(all_trees, 'modal')
        semantics_result = self.count_all(all_trees, 'semantics')
        #print(tense_result)
        #print(modal_result)
        print(semantics_result)

    def count_all(self, trees, which):
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
        
    #------------------------------ break into different obj later
    def constituency_tree(self, sent):
        pred = self.predictor.predict(sentence=sent)
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
        tense_dict = {'VB':'base',
                    'VBD':'past tense',
                    'VBG':'present participle',
                    'VBN':'past participle',
                    'VBP':'present tense non-3rd.sg',
                    'VBZ':'present tense 3rd.sg'}
        future_dict = {'will':'future tense', 'shall':'future tense'}
                    
        VP = [i[1] for i in constTree[0] if i[1][0] == 'VP']

        try:
            return type(tense_dict[[[re.sub('\(', '',j) for j in i if re.sub('\(', '',j) in tense_dict][0] for i in VP][0]])      
        #print([sum(tense.dict.get(i) == j for j in all) for i in tense_dict.keys()])
        except:
            try: 
                MD = [i[1][1].lower() for i in constTree[0] if i[1][0] == 'MD']
                return future_dict[MD[0]]
            except:
                return ''

    #------------------------------ break into different obj later

    def modal(self, constTree):
        modal_dict = {'can' : ['ability', 'suggestion', 'request'],
                    'could' : ['ability', 'request', 'suggestion'],
                    'may' : ['possibility', 'request', 'permission'],
                    'might' :['possibility'],
                    'must' : ['obligation', 'certainty'],
                    'should' : ['opinion', 'advice'],
                    'ought' : ['opinion', 'advice']
                    }
        VP_modal_dict = {'able to' : ['ability'],
                    'have to' : ['obligation'],'has to' : ['obligation'],'had to' : ['obligation'],
                    'need to' : ['obligation'],'needs to' : ['obligation'],'needed to' : ['obligation']
                    }

        VP = [i[1] for i in constTree[0] if i[1][0] == 'VP']

        modal_found = [[re.sub(r'[\(\)]', '',j) for j in i if re.sub('\(', '',j) == 'MD'] for i in VP]

        word_str = [' '.join(map(str, k)) for k in [[re.sub(r'[\(\)]', '',j) for j in i if re.sub(r'[\(\)]', '',j).islower()] for i in VP]]
        VP_modal_found = [[i.lower() for i in VP_modal_dict.keys() if i.lower() in sub_lst] for sub_lst in word_str]

        MD = [i[1][1].lower() for i in constTree[0] if i[1][0] == 'MD']
        MD_intersect = list((set(MD)).intersection(set(modal_dict))) # find the modal from modal_dict

        if (any(modal_found) and any((set(MD)).intersection(set(modal_dict))) and len(MD_intersect) != 0):
            try:
                return modal_dict[list((set(MD)).intersection(set(modal_dict)))[0]]
            except:
                return ''

        elif (not(any(modal_found)) and (any(VP_modal_found))):
            try:
                return VP_modal_dict[VP_modal_found[0][0]]
            except:
                return "####" + VP_modal_found[0][0]        # CHECK WHEN RUNNING ON WHOLE DATA
        else: return ''


    #------------------------------ break into different obj later

    def semantics(self, constTree):
        tree_token = [nltk.word_tokenize(i.replace("(", '')) for i in re.findall("\((.*?)\)", constTree[0]['trees'])]
        vn_id = verbnet.classids([i[len(i)-1] for i in tree_token if (i[0] == 'VP' and i[1] != 'MD')][0])
        vn_frame = [i['description']['primary'] for i in verbnet.frames(vn_id)]
        for i in vn_frame:
            pos_token_match = i.split('-')
            





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    analyzer = structFeat(args.filename)
    #print(analyzer.__dict__['self.data'][:10])
    all_trees = analyzer.analyze()



if __name__ == '__main__':
    main()