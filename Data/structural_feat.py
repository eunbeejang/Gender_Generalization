from allennlp.predictors.predictor import Predictor
from collections import Counter
import pandas as pd
import numpy as np
import argparse
import re

class structFeat(object):
    def __init__(self, file_path):
        self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
        self.data = pd.read_csv(file_path)[:10]

    def load_all(self): # load all sentences as constituency tree
        return [self.constituency_tree(i) for i in self.data['Sentence']]

    def analyze(self):
        all_trees = self.load_all()
        #print(self.data['Sentence'][6])
        #self.tense(all_trees[6])
        #[self.tense(i) for i in all_trees]
        tense_result = self.count_tense(all_trees)

    #------------------------------ break into different obj later
    def constituency_tree(self, sent):
        pred = self.predictor.predict(sentence=sent)
        # pred.keys() >> ['class_probabilities', 'spans', 'tokens', 'pos_tags', 'num_spans', 'hierplane_tree', 'trees']
        return [([i[0], (''.join(i[1]).split())]) for i in list(self.parse(list(pred['trees'])))] #(depth, tree content)

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

        VP = [i[1] for i in constTree if i[1][0] == 'VP']
        try:
            return [[re.sub('\(', '',j) for j in i if re.sub('\(', '',j) in tense_dict][0] for i in VP][0]       
        #print([sum(tense.dict.get(i) == j for j in all) for i in tense_dict.keys()])
        except:
            #print("nan")
            return ''

    def count_tense(self, trees):
        print(trees[6])
        full = np.where((self.data['Confidence'] == 1.0000) & (self.data['Final Label'] == 1))[0]
        print(full)
        two_third = np.where((self.data['Confidence'] != 1.0000) & (self.data['Final Label'] == 1))[0]
        one_third = np.where((self.data['Confidence'] != 1.0000) & (self.data['Final Label'] == 0))[0]
        zero = np.where((self.data['Confidence'] == 1.0000) & (self.data['Final Label'] == 0))[0]
        print([Counter([self.tense(trees[i]) for i in j]) for j in [full,two_third,one_third,zero]])
        #print(Counter[ ifor i in full])





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    analyzer = structFeat(args.filename)
    all_trees = analyzer.analyze()
    #print(all_trees[:10])



if __name__ == '__main__':
    main()