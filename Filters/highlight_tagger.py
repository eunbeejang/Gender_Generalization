"""
    Input: (tok_sentence, word_range, GENDER_PRONOUNS)
        tok_sentence - one tokenized sentence ie. ["A","nurse","must","always","take","care","of","her","patients"]
        word_range - a nested list of clusters with their ranges ie. [[[0,1],[7,7]]]
        GENDER_PRONOUNS - a list of gender pronouns ie. ['he','she','him','her','his','hers','himself','herself']

    Returns:
        sentence with <span> </span> tags at the start and end of the word range
        ie. ['"<span style='background-color:yellow"A', 'nurse</span>', 'must', 'always', 'take', 'care', 'of', '"<span style='background-color:yellow"her</span>', 'patients']

insert_tags() will only highlight a cluster if it contains a gender pronoun.
If there is more than one cluster with a gender pronoun, then it will highlight the clusters in different colors.
Note: A max of 8 different colors are possible (way more than currently needed), but if you want to add more colors,
go to https://www.w3schools.com/colors/colors_names.asp and add it to the background_colors array.
"""
def insert_tags(tok_sent, word_range, GENDER_PRONOUNS):
    background_colors = ["yellow", "aqua", "lime", "magenta", "gold", "plum", "moccasin", "powderblue"]
    index = 0  #used to iterate through background_colors array so each cluster will be a different color
    for cluster in word_range:
        if any([((c[0] == c[1]) and (tok_sent[c[0]]).lower() in GENDER_PRONOUNS) for c in cluster]): #check if cluster contains a gender pronoun
            for (start_index, end_index) in cluster:
                tok_sent[start_index]="<span style='background-color:"+background_colors[index]+"'>"+tok_sent[start_index]
                tok_sent[end_index]=tok_sent[end_index]+"</span>"
            index += 1
    return tok_sent