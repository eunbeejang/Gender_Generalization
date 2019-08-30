import nltk
from nltk import ne_chunk, pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
stemmer = PorterStemmer()
wn_lem = WordNetLemmatizer() # wn_lem.lemmatize()

"""
    Input: (data, word_range, which_type)
        data - one sentence
        word_range - a nested list of clusters with their ranges (ie. [[[0,0],[0,19]],[[19,19],[59,59]]])
        which_type - one of ("name", "pro", "term", "all"/nothing)

    return case: 
        1. if which_type == "name" and the function returns True >> THERE IS NAME LINK >> REMOVE
        2. if which_type == "pro" and function returns True >> ALL LINKS ARE PRONOUN >> REMOVE
        3. if which_type == "term" and function returns True >> THERE IS GENDER TERM LINK >> REMOVE
        4. if which_type == "all" or not given and function returns True >> one of three remove cases hold >> REMOVE
        5. if the function returns False >> NO NAME LINK and ALL LINKS ARE NOT PRONOUNS >> KEEP

"""


def check_remove(data, tok, word_range, which_type="all"):
    gen_fam_term = ["father", "mother", "son", "daughter", "husband", "wife", "brother", "sister",
                    "grandfather", "grandmother", "grandson", "granddaughter", "uncle", "aunt", "nephew", "niece"]
    gen_term = ["female", "male", "woman", "man", "girl", "boy"]
    pro_lst = ["he", "she", "him", "her", "his", "hers", "himself", "herself"]

    result = []
    for cluster in word_range:
        if (which_type == "name"):  # check if the cluster has name link

            # Check all the instances of human names in the sentence and build "name_lst"
            name_lst = []
            for sent_chunk in ne_chunk(pos_tag(tok)):
                if hasattr(sent_chunk, 'label'):
                    if (sent_chunk.label() == "PERSON"):
                        name_lst.append(' '.join(c[0] for c in sent_chunk))
                        # (print("TESTING", c[0]) for c in sent_chunk)
            result.append(any([((' '.join(w for w in tok[c[0]:c[1] + 1])) in name_lst)
                               for c in cluster]))

        elif (which_type == "pro"):  # check if the cluster has only pronoun links
            result.append(all([((c[0] == c[1]) and (tok[c[0]]).lower() in pro_lst) for c in cluster]))


        elif (which_type == "term"):  # check if the cluster has gendered term

            for c in cluster:
                for i in c:
                    word_disam = lesk(tok, tok[i], 'n')  # check definition assigned from word disambiguation
                    # if the word is a valid English word check if it's person word and the definition contains gendered meaning
                    if (word_disam is not None) and (word_disam.lexname() == "noun.person"):
                        # now looking at all nouns in the range but after ACL we can use dependency parsing and only look at the head noun
                        result.append(any([wn_lem.lemmatize(w) in (gen_fam_term + gen_term + pro_lst)
                                           and (x in (gen_fam_term + gen_term + pro_lst)
                                                for x in wn._morphy(w, wn.NOUN))
                                           # checks all possible morphological functions
                                           for w in word_tokenize(word_disam.definition())]
                                          + [tok[i] in (gen_fam_term + gen_term + pro_lst)]))
                    else:
                        result.append(False)
                else:
                    continue
        else:  # check all conditions at the same time
            result.append(any([check_remove(data, tok, word_range, which_type="name"),
                               check_remove(data, tok, word_range, which_type="pro"),
                               check_remove(data, tok, word_range, which_type="term")]))

    return any(result)


def filter_by_corpus(corpus, tok_corpus, coref_ranges, prev_result, which_type="all"):
    this_result = []
    for i in range(0, len(corpus)):
        if prev_result[i] == 1:
            if check_remove(corpus[i],tok_corpus[i], coref_ranges[i], which_type):
                this_result.append(0)
            else:
                this_result.append(1)
        else:
            this_result.append(0)
    return this_result
