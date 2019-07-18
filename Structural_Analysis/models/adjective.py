

def get_adj(constTree, coref): # Categorize each coref cluster linked to a gendered pronoun

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

