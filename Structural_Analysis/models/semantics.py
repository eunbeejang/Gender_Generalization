
#### THIS ONE IS ON HOLD, NOT ENOUGH INFO TO ANALYZE
def get_sem(constTree):
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