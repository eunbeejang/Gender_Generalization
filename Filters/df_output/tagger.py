"""
    Input: (tok_sentence, word_range, GENDER_PRONOUNS)
        tok_sentence - one tokenized sentence ie. ["A","nurse","must","always","take","care","of","her","patients"]
        coref_range - a nested list of clusters with their ranges ie. [[[0,1],[7,7]]]
        GENDER_PRONOUNS - a list of gender pronouns ie. ['he','she','him','her','his','hers','himself','herself']

    Returns:
        sentence (detokenized) with <span> </span> tags at the start and end of the word range
        ie. ['"<span style='background-color:yellow"A', 'nurse</span>', 'must', 'always', 'take', 'care', 'of', '"<span style='background-color:yellow"her</span>', 'patients']

insert_highlight() and insert_brackets() will only tag a cluster if it contains a gender pronoun.
If there is more than one cluster with a gender pronoun, then it will tag the clusters in different colors/brackets.
Note: A max of 8 different colors and 6 different brackets are possible, feel free to add more if needed.
"""

from nltk.tokenize.treebank import TreebankWordDetokenizer


def insert_highlight(tok_sent, coref_range, GENDER_PRONOUNS=['he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself']):
    background_colors = ["yellow", "aqua", "lime", "magenta", "gold", "plum", "moccasin", "powderblue"]
    index = 0  #used to iterate through background_colors array so each cluster will be a different color
    for cluster in coref_range:
        if any([((c[0] == c[1]) and (tok_sent[c[0]]).lower() in GENDER_PRONOUNS) for c in cluster]): #check if cluster contains a gender pronoun
            for (start_index, end_index) in cluster:
                tok_sent[start_index]="<span style='background-color:"+background_colors[index]+"'>"+tok_sent[start_index]
                tok_sent[end_index]=tok_sent[end_index]+"</span>"
            index += 1
    return TreebankWordDetokenizer().detokenize(tok_sent)

def insert_brackets(tok_sent, coref_range, GENDER_PRONOUNS=['he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself']):
    start_bracket = ["<","[","{","<<","[[","{{"]
    end_bracket = [">","]","}",">>","]]","}}"]
    index = 0 #used to iterate through brackets array so each cluster will be within a different bracket
    for cluster in coref_range:
        if any([((c[0] == c[1]) and (tok_sent[c[0]]).lower() in GENDER_PRONOUNS) for c in cluster]): #check if cluster contains a gender pronoun
            for (start_index, end_index) in cluster:
                tok_sent[start_index]=start_bracket[index]+tok_sent[start_index]
                tok_sent[end_index]=tok_sent[end_index]+end_bracket[index]
            index += 1
    return TreebankWordDetokenizer().detokenize(tok_sent)